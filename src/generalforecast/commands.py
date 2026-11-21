# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The `gf model`, `gf context` and `gf admin` commands, and the connection plumbing.

Model commands read the hosted catalog; context commands call the live
context SDK methods; admin nouns bind to the SDK's identity and collection
operations or act on hosted predictors. Every command returns its result;
`generalforecast.render` writes it.

The SDK client, its transport and Arrow load inside the functions that use
them, so `gf --help` and argument errors stay fast.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from generalforecast import term
from generalforecast.command_specs import (
    ADMIN_PREDICTOR_NOUNS,
    COLLECTIONS,
    IDENTITY_GROUPS,
    NAME_FIELDS,
    admin_action,
)
from generalforecast.errors import InvalidInput, during

if TYPE_CHECKING:
    from generalforecast.client import Client
    from generalforecast.service.boundary import JsonObject
    from generalforecast.transport import ConnectionSettings

# The largest table a command reads from disk.
MAX_DATA_BYTES = 128 * 1024 * 1024


def require(condition: bool, message: str, *, hint: str | None = None) -> None:
    """Raises a local, actionable validation error unless `condition` holds."""
    if not condition:
        raise InvalidInput(message, hint=hint)


def read_object(path: str) -> dict:
    """Reads one bounded JSON/YAML object without reflecting input on failure."""
    import yaml

    from generalforecast.service.boundary import MAX_BYTES, json_object

    try:
        if path == "-":
            raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        else:
            with Path(path).expanduser().open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input exceeds 1 MiB")
        return json_object(yaml.safe_load(raw))
    except (ValueError, UnicodeError, yaml.YAMLError):
        raise InvalidInput(f"`{path}` is not one JSON or YAML object") from None


def read_data(path: str, root: Path):
    """Reads a bounded local table relative to the predictor directory."""
    from generalforecast.tabular import file_format, read_table

    given = Path(path).expanduser()
    target = given if given.is_absolute() else root / given
    with target.open("rb") as stream:
        raw = stream.read(MAX_DATA_BYTES + 1)
    require(len(raw) <= MAX_DATA_BYTES, "tabular input too large")
    return read_table(raw, file_format(target.name))


def connection_for(args: argparse.Namespace) -> ConnectionSettings:
    """The connection the SDK resolves: a profile, else GF_URL with GF_API_KEY or the `gf login` key.

    See `ConnectionSettings.resolve`; the CLI records where it came from.
    """
    from generalforecast.credentials import CONNECTION
    from generalforecast.transport import ConnectionSettings

    CONNECTION.clear()
    settings = ConnectionSettings.resolve()
    CONNECTION.update(
        server=settings.profile.url,
        source=settings.source or "profile",
        project_pending="1" if settings.project_from_key else "",
    )
    return settings


def client_for(
    args: argparse.Namespace,
    settings: ConnectionSettings | None = None,
    **options: Any,
) -> Client:
    """Uses named environment references; never persists credential values.

    `options` are the client's own, such as `wait_seconds` and `on_wait`.
    """
    from generalforecast.client import Client
    from generalforecast.credentials import SEEN

    settings = connection_for(args) if settings is None else settings
    token = settings.token
    # The transport's provider options, apart from the client's own.
    provider: dict[str, Any] = {
        "auth_mode": os.environ.get("GF_AUTH_MODE", "bearer"),
        "platform_token": os.environ.get("GF_PLATFORM_TOKEN"),
        "modal_key": os.environ.get("GF_MODAL_KEY"),
        "modal_secret": os.environ.get("GF_MODAL_SECRET"),
    }
    SEEN.update(
        value
        for value in (
            token,
            provider["platform_token"],
            provider["modal_key"],
            provider["modal_secret"],
        )
        if value
    )
    context = None
    ca, identity = os.environ.get("GF_TLS_CA"), os.environ.get("GF_TLS_IDENTITY")
    if ca or identity:
        import ssl

        try:
            context = ssl.create_default_context(cafile=ca)
            if identity:
                context.load_cert_chain(identity)
        except (OSError, ssl.SSLError):
            raise InvalidInput(
                "cannot load the TLS configuration",
                hint="check GF_TLS_CA and GF_TLS_IDENTITY",
            ) from None
    return Client(
        settings=settings,
        timeout=args.timeout,
        tls_context=context,
        **provider,
        **options,
    )


def upload_snapshot(client: Client, table, *, prefix: str = "table") -> dict:
    """Uploads the full canonical table through the SDK's authenticated transport.

    The snapshot is named `<prefix>-<first 12 hex digits of its digest>`.
    """
    from generalforecast.tabular import table_digest

    result = client.upload_snapshot(table, prefix=prefix)
    require(
        type(result) is dict
        and result.get("digest") == table_digest(table)
        and type(result.get("rows")) is int
        and result.get("rows") == table.num_rows,
        "snapshot integrity mismatch",
    )
    return result


def execute(args: argparse.Namespace, out: term.Output) -> dict | None:
    """Runs `gf model`, `gf context` or `gf admin`."""
    if args.command == "model":
        return _model(args, out)
    if args.command == "context":
        return _context(args, out)
    return _admin(args, out)


def _admin(args: argparse.Namespace, out: term.Output) -> dict | None:
    """`gf admin <noun> <action>`: API collections and hosted predictors."""
    noun = args.noun
    if noun in ADMIN_PREDICTOR_NOUNS:
        from generalforecast.predictor_commands import PredictorCommands

        return getattr(PredictorCommands(args, out), noun)()
    return _api(args, out)


def _api(args: argparse.Namespace, out: term.Output) -> dict:
    """An API-backed admin action, its NAME and `--file` bound to the request."""
    from generalforecast.service.contracts import name

    noun, spec = args.noun, admin_action(args.noun, args.action)
    payload: JsonObject = read_object(args.file) if spec.body else {}
    resource = name(args.name) if spec.target is not None else None
    if resource is not None:
        field = NAME_FIELDS[noun]
        require(
            payload.get(field, resource) == resource,
            f"the file's `{field}` is not `{resource}`",
            hint=f"drop `{field}` from the file, or pass its value as NAME",
        )
        payload[field] = resource
    if noun == "audit":
        require(args.after >= 0, "--after must be 0 or more")
        require(1 <= args.limit <= 100, "--limit must be from 1 to 100")
        payload = {"after": args.after, "limit": args.limit}
    if spec.destructive:
        words = [spec.name.capitalize(), noun, resource]
        question = " ".join(word for word in words if word) + "?"
        # revoke, rotate and delete all take a `d`: `Nothing was revoked.`
        done = f"{spec.name}d" if spec.name.endswith("e") else "changed"
        term.Prompter(out, yes=args.yes).confirm(
            question, declined=f"Nothing was {done}."
        )
    client = client_for(args)
    # The person's own words for the request: the command they typed.
    with during(f"`gf admin {noun} {spec.name}`"):
        if noun in IDENTITY_GROUPS:
            return client.identity.manage(
                f"{IDENTITY_GROUPS[noun]}.{spec.name}", payload
            )
        collection = COLLECTIONS[noun][1]
        sdk = getattr(client, collection)
        if spec.name == "list":
            return {collection: sdk.list()}
        return _wire(getattr(sdk, spec.name)(payload))


def _wire(value: Any) -> dict:
    """Returns the exact wire object behind an SDK snapshot DTO."""
    from generalforecast.service.sdk import Snapshot

    return value.to_dict() if isinstance(value, Snapshot) else value


def _context(args: argparse.Namespace, out: term.Output) -> dict:
    """`gf context create|append|label|show`, over the live context SDK methods."""
    import uuid

    from generalforecast.service.contracts import name

    ident = name(args.name)
    if args.action == "show":
        with during(f"the request for context {ident}"):
            return client_for(args).contexts.get(ident).to_dict()
    if args.action == "create":
        payload = read_object(args.file)
        require(payload.get("name") == ident, "the config's name differs from NAME")
        with during(f"the new context {ident}"):
            result = client_for(args).contexts.create(payload).to_dict()
        out.status("Created", f"context {ident}")
        return result
    table = read_data(args.file, Path.cwd())
    require(table.num_rows > 0, "the file has no rows")
    write_id = args.write_id or uuid.uuid4().hex
    require(args.id in table.column_names, f"column `{args.id}` is not in the file")
    client = client_for(args)
    if args.action == "label":
        require(
            args.target in table.column_names,
            f"column `{args.target}` is not in the file",
        )
        with during(f"the labels for {ident}"):
            result = client.contexts.label(
                ident, table, id=args.id, column=args.target, idempotency_key=write_id
            )
        out.status("Labeled", f"{term.plural(table.num_rows, 'row')} in {ident}")
    else:
        with during(f"the append to {ident}"):
            result = client.contexts.append(
                ident, table, id=args.id, idempotency_key=write_id
            )
        out.status("Appended", f"{term.plural(table.num_rows, 'row')} to {ident}")
    return _wire(result) if not isinstance(result, dict) else result


def _model(args: argparse.Namespace, out: term.Output) -> dict:
    """`gf model list|show`, over the hosted catalog."""
    catalog = model_registry(args)
    if args.action == "list":
        return {"models": [_model_info(info) for info in catalog.catalog(args.task)]}
    return _model_info(catalog.entry(args.model))


def _model_info(info: Any) -> dict:
    limits = info.limits
    return {
        "id": info.id,
        "task": info.task,
        "limits": {
            "context_rows": limits.context_rows,
            "inputs": limits.inputs,
        },
    }


def model_registry(args: argparse.Namespace) -> Any:
    """The hosted catalog, as `GET /models` lists it for this connection."""
    from generalforecast.catalog import Catalog

    client = client_for(args)
    with during("the model catalog"):
        return Catalog.from_reply(client.models.list())
