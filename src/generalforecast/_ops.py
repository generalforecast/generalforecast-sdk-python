# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Every operation, once, as a generator both clients drive.

An operation yields `_http.Request`s and gets their `Reply`s back, yields
`Sleep` to wait, `Say` to update the progress line, `Emit` to hand an item
to an iterator and `Notify` to call a caller's callback; it returns the
result. `Client` runs the steps in the calling thread and `AsyncClient`
awaits them, so the two behave alike.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import socket
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

from generalforecast import _data, _http
from generalforecast.errors import (
    AmbiguousReference,
    Error,
    InvalidInput,
    NotFound,
    Timeout,
    failed,
)
from generalforecast.types import (
    COLUMN_TYPES,
    NAME,
    PREDICTOR_FIELDS,
    SOURCE_KINDS,
    STANDARD_PERMISSIONS,
    TASKS,
    Context,
    ContextUse,
    Deploy,
    Environment,
    Estimate,
    Experiment,
    Holdout,
    Key,
    LargeContext,
    Limits,
    LogEntry,
    Logs,
    Me,
    Model,
    NewKey,
    Predictions,
    Predictor,
    PredictorInfo,
    Revision,
    Run,
    Slice,
    Source,
    SourceTable,
    Timing,
    TrainedOn,
    WriteReceipt,
    instant,
    load,
    loads,
    ref_of,
    spec_json,
)


@dataclass(frozen=True)
class Sleep:
    """Wait this many seconds."""

    seconds: float


@dataclass(frozen=True)
class Say:
    """Show this on the progress line, or clear it with None."""

    text: str | None


@dataclass(frozen=True)
class Emit:
    """Hand this item to the caller's iterator."""

    item: Any


@dataclass(frozen=True)
class Notify:
    """Call the caller's callback with this value."""

    callback: Callable[[Any], Any]
    value: Any


@dataclass
class State:
    """What a client remembers between calls: snapshots it knows the service holds."""

    snapshots: set[str] = field(default_factory=set)


Op = Generator[Any, Any, Any]

UNSET: Any = type("Unset", (), {"__repr__": lambda self: "UNSET"})()
"""Marks an argument left out, so `None` can mean clear."""

POLL_FIRST = 1.0
POLL_MAX = 5.0


def _poll(attempt: int) -> float:
    return min(POLL_FIRST * 1.5**attempt, POLL_MAX)


def name_of(value: Any, what: str) -> str:
    """The name a reference stands for, without a pinned version."""
    return split_ref(value, what)[0]


def split_ref(value: Any, what: str = "context") -> tuple[str, int | None]:
    """A reference's name and pinned version: `"adult@2"` is `("adult", 2)`."""
    if isinstance(value, Context):
        return value.name, value.version
    text = ref_of(value, what)
    name, sep, version = text.partition("@")
    if not sep:
        return name, None
    if not version.isdigit():
        raise InvalidInput(
            f"`{text}` pins no version",
            hint=f'write "{name}@2" for version 2, or "{name}" for the latest',
            param=what,
        )
    return name, int(version)


def _checked_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not NAME.match(name):
        raise InvalidInput(
            f"`{name}` is not a {what} name",
            hint="start with a letter, then lowercase letters, digits and hyphens, at most 64",
            param="name",
        )
    return name


def _duration(value: str | timedelta | None, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value))
    if found is None:
        raise InvalidInput(
            f"`{value}` is not a duration", hint="write 30s, 15m, 1h or 90d", param=what
        )
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[found[2]]
    return float(found[1]) * unit


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, int, float, bool)):
        return _jsonable(value.item())
    return value


def _suggest(word: str, options: Sequence[str]) -> str | None:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f"a similar name exists: `{close[0]}`" if close else None


def _machine() -> str:
    return (socket.gethostname().split(".")[0] or "unknown").lower()[:48]


def _not_done(what: str, timeout: float | None, hint: str) -> Timeout:
    shown = f" after {timeout:g}s" if timeout is not None else ""
    return Timeout(f"{what} is still going{shown}", hint=hint)


def _deadline(timeout: float | None) -> float | None:
    return None if timeout is None else time.monotonic() + timeout


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def whoami(server: str) -> Op:
    """Who the credential acts for."""
    reply = yield _http.auth_status()
    return me_of(reply.result or {}, server)


def me_of(result: Mapping[str, Any], server: str, expires: Any = None) -> Me:
    """A `Me` from an auth status or a device login's token reply."""
    principal = result.get("principal") or {}
    key = result.get("key") or {}
    workspace = result.get("workspace")
    if isinstance(workspace, Mapping):
        workspace = workspace.get("display_name") or workspace.get("id")
    project = result.get("project") or key.get("project") or principal.get("project")
    found = load(Key, key) if key else None
    token_expires = instant(expires) if expires is not None else None
    if token_expires is None and key.get("ephemeral"):
        token_expires = found.expires if found else None
    url = result.get("url") or (
        f"{server}/?project={project}" if project else server + "/"
    )
    return Me(
        email=result.get("email"),
        workspace=workspace,
        project=project,
        key=found,
        token_expires=token_expires,
        url=url,
    )


def device_login(sleep: Callable[[float], None], say: Callable[[str], None]) -> Op:
    """Runs the device flow (3.6) and returns the token reply.

    Raises:
        Timeout: When the code expires before it is approved.
        Unauthorized: When the login is denied in the browser.
    """
    reply = yield _http.device_login_start(_machine(), "notebook")
    grant = reply.result or {}
    for key in ("device_code", "user_code", "verification_uri"):
        if not isinstance(grant.get(key), str):
            raise InvalidInput("the server's device login is incomplete")
    say(f"Open {grant['verification_uri']} and enter the code {grant['user_code']}")
    interval = float(grant.get("interval") or 5)
    deadline = time.monotonic() + float(grant.get("expires_in") or 600)
    while True:
        if time.monotonic() >= deadline:
            raise Timeout(
                "the login code expired", hint="run gf.login() again, or pass key="
            )
        sleep(interval)
        try:
            reply = yield _http.device_login_token(grant["device_code"])
        except Error as error:
            if error.reason == "authorization_pending":
                continue
            if error.reason == "slow_down":
                interval = min(interval + 5, 60)
                continue
            if error.reason == "expired_token":
                raise Timeout(
                    "the login code expired", hint="run gf.login() again, or pass key="
                ) from None
            raise
        return reply.result or {}


def _source_payload(source: Any, table: str | None, sql: str | None) -> dict[str, Any]:
    if (table is None) == (sql is None):
        raise InvalidInput(
            "a context from a source reads one table= or one sql=",
            hint='pass table="dataset.table" or sql="select ..."',
        )
    out = {"source": name_of(source, "source")}
    out["table" if table is not None else "sql"] = table if table is not None else sql
    return out


def _types(columns: Mapping[str, str], what: str) -> dict[str, str]:
    out = {}
    for column, kind in columns.items():
        if kind not in COLUMN_TYPES:
            raise InvalidInput(
                f"`{kind}` is not a column type for `{column}`",
                hint="types are " + ", ".join(COLUMN_TYPES),
                param=what,
            )
        out[str(column)] = kind
    return out


def _upload_spec(rows: Any, body: bytes, digest: str, data: Any) -> dict[str, Any]:
    return {
        "format": "parquet",
        "bytes": len(body),
        "file": _data.file_name(data),
        "digest": digest,
        "rows": rows.num_rows,
    }


def context_create(
    name: str,
    *,
    data: Any = None,
    source: Any = None,
    table: str | None = None,
    sql: str | None = None,
    live: bool = False,
    columns: Mapping[str, str] | None = None,
    targets: Sequence[str] = (),
    id: str | None = None,
    time: str | None = None,
    freshness: str | timedelta = "15m",
    wait: bool = True,
) -> Op:
    """Creates a context from exactly one origin (1.2)."""
    _checked_name(name, "context")
    if isinstance(targets, str):
        targets = [targets]
    if data is not None and source is not None:
        raise InvalidInput(
            "a context has one origin: pass data= or source=, not both",
            hint="upload the rows with data=, or import them with source=",
        )
    if source is not None:
        if columns is not None or targets:
            raise InvalidInput(
                "columns= and targets= are for live contexts fed through the API",
                hint="a source's columns come from its table",
            )
        payload: dict[str, Any] = {
            "name": name,
            "kind": "live" if live else "snapshot",
            "source": _source_payload(source, table, sql),
            "id": id,
            "time": time,
        }
        if live:
            seconds = _duration(freshness, "freshness") or 0
            if not 60 <= seconds <= 86400:
                raise InvalidInput("freshness is between 1m and 24h", param="freshness")
            payload["freshness_seconds"] = seconds
        reply = yield _http.contexts_create(payload)
        found = load(Context, reply.result)
        return (yield from _settle(found, wait))
    if table is not None or sql is not None:
        raise InvalidInput(
            "table= and sql= read from a source", hint="pass source= with them"
        )
    if live:
        rows = _data.table(data) if data is not None else None
        if columns is None and rows is None:
            raise InvalidInput(
                "a live context fed through the API declares its columns",
                hint='pass columns={"plan": "category", ...}, or data= to infer them',
            )
        declared = (
            _types(columns, "columns")
            if columns is not None
            else _data.column_types(rows)  # type: ignore[arg-type]
        )
        for target in targets:
            if target not in declared:
                raise InvalidInput(
                    f"the target `{target}` is not one of the columns", param="targets"
                )
        reply = yield _http.contexts_create(
            {
                "name": name,
                "kind": "live",
                "columns": declared,
                "id": id,
                "time": time,
                "targets": list(targets),
            }
        )
        found = load(Context, reply.result)
        if rows is not None and rows.num_rows:
            yield from _write(name, "append", _records(rows), None, {})
            reply = yield _http.contexts_get(name)
            found = load(Context, reply.result)
        return found
    if data is None:
        raise InvalidInput(
            "a context needs an origin",
            hint="pass data= to upload rows, source= with table= or sql= to "
            "import them, or live=True with columns=",
        )
    if targets:
        raise InvalidInput("targets= is for live contexts fed through the API")
    rows = _data.table(data)
    body = _data.parquet(rows)
    digest = _data.digest(_data.stream(rows))
    payload = {
        "name": name,
        "kind": "snapshot",
        "data": _upload_spec(rows, body, digest, data),
        "id": id,
        "time": time,
    }
    if columns is not None:
        payload["columns"] = _types(columns, "columns")
    reply = yield _http.contexts_create(payload)
    found = yield from _upload(name, reply.result, body, rows.num_rows)
    return (yield from _settle(found, wait))


def _upload(name: str, result: Mapping[str, Any], body: bytes, rows: int) -> Op:
    """Sends an opened upload's bytes, to a signed URL or the bytes route."""
    found = load(Context, result)
    upload = result.get("upload")
    if found.unchanged or not isinstance(upload, Mapping):
        return found
    version = int(upload.get("version") or found.version or 1)
    url = upload.get("url")
    chunk = int(upload.get("chunk_bytes") or _data.UPLOAD_CHUNK_BYTES)
    total = len(body)
    yield Say(f"Uploading {name}  {rows:,} rows, {_data.size(total)}")
    offset = 0
    while offset < total:
        piece = body[offset : offset + chunk]
        if url:
            yield _http.upload_chunk(url, piece, offset, total)
            offset += len(piece)
        else:
            reply = yield _http.contexts_bytes(name, version, offset, piece)
            received = (reply.result or {}).get("received")
            offset = (
                int(received)
                if isinstance(received, int) and received > offset
                else offset + len(piece)
            )
    return replace(found, state="uploading" if found.state == "ready" else found.state)


def _settle(found: Context, wait: bool) -> Op:
    """Polls a context until it is ready, or raises when it failed."""
    if not wait:
        return found
    attempt = 0
    while found.state not in ("ready", "failed"):
        verb = "Importing" if found.state == "importing" else "Verifying"
        yield Say(f"{verb} {found.name}")
        yield Sleep(_poll(attempt))
        attempt += 1
        reply = yield _http.contexts_get(found.name)
        unchanged = found.unchanged
        found = load(Context, reply.result)
        if unchanged:
            found = replace(found, unchanged=True)
    if found.state == "failed":
        raise failed(f"context {found.name} failed", found.error, code="ContextFailed")
    return found


def context_upload(context: Any, data: Any, *, wait: bool = True) -> Op:
    """Uploads a new version of a snapshot context, unless its rows are unchanged."""
    name = name_of(context, "context")
    rows = _data.table(data)
    body = _data.parquet(rows)
    digest = _data.digest(_data.stream(rows))
    reply = yield _http.contexts_version(
        name, {"data": _upload_spec(rows, body, digest, data)}
    )
    found = yield from _upload(name, reply.result, body, rows.num_rows)
    return (yield from _settle(found, wait))


def context_refresh(context: Any, *, wait: bool = True) -> Op:
    """Imports a new version of a source snapshot context."""
    name = name_of(context, "context")
    reply = yield _http.contexts_refresh(name)
    return (yield from _settle(load(Context, reply.result), wait))


def _records(rows: Any) -> list[dict[str, Any]]:
    return [_jsonable(row) for row in rows.to_pylist()]


JSON_PART_BYTES = 900_000
JSON_PART_VALUES = 60_000


def json_parts(items: list[Any]) -> list[list[Any]]:
    """Items in parts under the JSON bounds: bytes and values both (3.10)."""
    parts: list[list[Any]] = []
    current: list[Any] = []
    size = values = 0
    for item in items:
        encoded = len(json.dumps(item, separators=(",", ":")))
        count = len(item) + 1 if isinstance(item, Mapping) else 1
        if current and (
            size + encoded > JSON_PART_BYTES or values + count > JSON_PART_VALUES
        ):
            parts.append(current)
            current, size, values = [], 0, 0
        current.append(item)
        size += encoded + 1
        values += count
    if current or not parts:
        parts.append(current)
    return parts


def _write(
    name: str,
    verb: str,
    items: list[Any],
    key: str | None,
    extra: Mapping[str, Any],
) -> Op:
    field_name = "rows" if verb == "append" else "labels"
    base = key or f"{verb}-{os.urandom(8).hex()}"
    parts = json_parts(items)
    total = sequence = 0
    for index, part in enumerate(parts):
        write_id = base if len(parts) == 1 else f"{base}-{index + 1}"
        reply = yield _http.contexts_write(
            name, verb, {"write_id": write_id, **extra, field_name: part}
        )
        result = reply.result or {}
        total += int(result.get("rows") or len(part))
        sequence = int(result.get("sequence") or sequence)
    return WriteReceipt(name, total, base, sequence)


def context_append(
    context: Any, data: Any, *, idempotency_key: str | None = None
) -> Op:
    """Appends rows to an API-fed live context."""
    name = name_of(context, "context")
    return (
        yield from _write(
            name, "append", _records(_data.table(data)), idempotency_key, {}
        )
    )


def context_label(
    context: Any, data: Any, *, target: str, idempotency_key: str | None = None
) -> Op:
    """Fills `target` for rows already in a live context, matched by its id column."""
    name = name_of(context, "context")
    reply = yield _http.contexts_get(name)
    found = load(Context, reply.result)
    if not found.id:
        raise InvalidInput(
            f"the context {name} declares no id column, so rows cannot be labeled",
            hint="create it with id= naming the column rows are matched on",
        )
    rows = _data.table(data)
    for column in (found.id, target):
        if column not in rows.column_names:
            raise InvalidInput(f"the labels have no column `{column}`", param="data")
    labels = [
        {"id": _jsonable(i), "value": _jsonable(v)}
        for i, v in zip(
            rows.column(found.id).to_pylist(),
            rows.column(target).to_pylist(),
            strict=True,
        )
    ]
    return (
        yield from _write(name, "label", labels, idempotency_key, {"target": target})
    )


def context_get(context: Any, *, version: int | None = None) -> Op:
    """Reads a context, its latest version or the one pinned."""
    name, pinned = split_ref(context)
    reply = yield _http.contexts_get(name, version if version is not None else pinned)
    return load(Context, reply.result)


def context_list() -> Op:
    """Lists the project's contexts."""
    reply = yield _http.contexts_list()
    return loads(Context, (reply.result or {}).get("contexts"))


def context_rows(context: Any, *, limit: int = 20, version: int | None = None) -> Op:
    """The first `limit` rows of a context."""
    name, pinned = split_ref(context)
    version = version if version is not None else pinned
    columns: Any = None
    rows: list[Any] = []
    after = None
    while len(rows) < limit:
        reply = yield _http.contexts_rows(
            name, limit=limit - len(rows), version=version, after=after
        )
        result = reply.result or {}
        columns = columns or result.get("columns")
        rows += list(result.get("rows") or ())
        after = result.get("next")
        if not after or not result.get("rows"):
            break
    return _data.frame(columns, rows[:limit])


def context_delete(context: Any) -> Op:
    """Deletes a context."""
    yield _http.contexts_delete(name_of(context, "context"))


def holdout_json(holdout: Any, time: str | None) -> dict[str, Any] | None:
    """An experiment's holdout as its setup sends it; left out, the last 20% by time when `time` is set."""
    if holdout is None:
        return {"kind": "time", "fraction": 0.2} if time else None
    if isinstance(holdout, str):
        holdout = Holdout.parse(holdout)
    if not isinstance(holdout, Holdout):
        raise InvalidInput('holdout is a string such as "20%" or a gf.Holdout')
    return holdout.to_json()


def _names(value: Any, what: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise InvalidInput(
            f"`{what}` is a list, not one string", hint=f'write {what}=["{value}"]'
        )
    return [str(item) for item in value]


def _task(task: str | None) -> str | None:
    if task is not None and task not in TASKS:
        raise InvalidInput(
            f"`{task}` is not a task",
            hint="tasks are " + ", ".join(TASKS),
            param="task",
        )
    return task


def setup(values: Mapping[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """An experiment's setup payload from `create`'s or `update`'s arguments.

    A field left out, or None, is not sent: an update changes only the
    fields given, and the contract has no way to clear one. `context` pins
    its version when it names one (`"adult@2"`).
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is UNSET or (value is None and key != "holdout"):
            continue
        if key == "context":
            name, version = split_ref(value)
            out["context"] = name
            if version is not None:
                out["version"] = version
            continue
        if key == "holdout":
            value = holdout_json(value, None if partial else values.get("time"))
        elif key in ("inputs", "exclude"):
            value = _names(value, key)
        elif key == "types":
            value = _types(value, "types")
        elif key == "task":
            value = _task(value)
        if value is None or (key == "exclude" and value == [] and not partial):
            continue
        out[key] = value
    return out


def experiment_create(
    values: Mapping[str, Any], models: Any, wait: bool, timeout: float | None
) -> Op:
    """Creates an experiment and starts run 1."""
    if values.get("name") is not None:
        _checked_name(values["name"], "experiment")
    payload = setup(values)
    if models is not None:
        payload["models"] = _names(models, "models")
    reply = yield _http.experiments_create(payload)
    found = load(Experiment, reply.result)
    if not wait or found.run is None:
        return found
    return (
        yield from experiment_wait(found.name, run=found.run.number, timeout=timeout)
    )


def experiment_estimate(values: Mapping[str, Any], models: Any) -> Op:
    """Prices an experiment's run 1 without starting it."""
    payload = setup(values)
    if models is not None:
        payload["models"] = _names(models, "models")
    payload["dry_run"] = True
    reply = yield _http.experiments_create(payload)
    result = reply.result or {}
    estimate = load(Estimate, result.get("estimate") or result)
    warnings = tuple(str(w) for w in result.get("warnings") or ())
    return replace(estimate, warnings=warnings) if warnings else estimate


def experiment_run(
    experiment: Any, models: Any, wait: bool, timeout: float | None
) -> Op:
    """Starts another run of an experiment."""
    name = name_of(experiment, "experiment")
    payload: dict[str, Any] = {}
    if models is not None:
        payload["models"] = _names(models, "models")
    reply = yield _http.experiments_run(name, payload)
    run = load(Run, reply.result)
    if wait:
        return (yield from experiment_wait(name, run=run.number, timeout=timeout))
    return (yield from experiment_get(name, run=run.number))


def experiment_update(experiment: Any, values: Mapping[str, Any]) -> Op:
    """Changes only the given setup fields."""
    payload = setup(values, partial=True)
    if not payload:
        raise InvalidInput("update changes at least one field")
    reply = yield _http.experiments_update(name_of(experiment, "experiment"), payload)
    return load(Experiment, reply.result)


def experiment_get(experiment: Any, *, run: int | None = None) -> Op:
    """Reads an experiment with its latest run, or run `run`."""
    reply = yield _http.experiments_get(name_of(experiment, "experiment"), run)
    return load(Experiment, reply.result)


def experiment_list(context: Any = None) -> Op:
    """Lists experiments, every page."""
    name = name_of(context, "context") if context is not None else None
    found: list[Experiment] = []
    after = None
    while True:
        reply = yield _http.experiments_list(name, after)
        result = reply.result or {}
        found += loads(Experiment, result.get("experiments"))
        after = result.get("next")
        if not after:
            return found


def _progress(run: Run) -> str:
    total = len(run.leaderboard) or len(run.models)
    finished = sum(1 for e in run.leaderboard if e.state in ("finished", "failed"))
    if run.state == "queued":
        return f"Queued run {run.number}  {total} models"
    return f"Running run {run.number}  {finished} of {total} models finished"


def experiment_wait(
    experiment: Any, *, run: int | None = None, timeout: float | None = None
) -> Op:
    """Polls until the run ends; raises when it failed."""
    name = name_of(experiment, "experiment")
    deadline = _deadline(timeout)
    attempt = 0
    while True:
        found = yield from experiment_get(name, run=run)
        current = found.run
        if current is None:
            return found
        if current.done:
            if current.state == "failed":
                raise failed(
                    f"run {current.number} of {name} failed",
                    current.error,
                    code="RunFailed",
                )
            return found
        if _past(deadline):
            raise _not_done(
                f"run {current.number} of {name}",
                timeout,
                f"gf.experiments.wait({name!r}) picks it up again",
            )
        yield Say(_progress(current))
        yield Sleep(_poll(attempt))
        attempt += 1
        run = current.number


def _run_number(name: str, run: int | None) -> Op:
    if run is not None:
        return run
    found = yield from experiment_get(name)
    if found.run is None:
        raise NotFound(f"the experiment {name} has no runs")
    return found.run.number


def experiment_cancel(experiment: Any, *, run: int | None = None) -> Op:
    """Cancels a run and returns the experiment with it."""
    name = name_of(experiment, "experiment")
    number = yield from _run_number(name, run)
    yield _http.experiments_cancel(name, number)
    return (yield from experiment_get(name, run=number))


def experiment_predictions(
    experiment: Any, *, run: int | None, model: str | None
) -> Op:
    """Every held-out prediction of a run, one row per held-out row and model."""
    name = name_of(experiment, "experiment")
    number = yield from _run_number(name, run)
    columns: Any = None
    rows: list[Any] = []
    after = None
    while True:
        reply = yield _http.experiments_rows(name, number, model=model, after=after)
        result = reply.result or {}
        columns = columns or result.get("columns")
        rows += list(result.get("rows") or result.get("predictions") or ())
        after = result.get("next")
        if not after:
            return _data.frame(columns, rows)


def experiment_slices(experiment: Any, *, run: int | None, model: str | None) -> Op:
    """A run's error by slice, worst first as the service orders it."""
    name = name_of(experiment, "experiment")
    number = yield from _run_number(name, run)
    reply = yield _http.experiments_slices(name, number, model=model)
    return loads(Slice, (reply.result or {}).get("slices"))


def experiment_delete(experiment: Any) -> Op:
    """Deletes an experiment and its runs."""
    yield _http.experiments_delete(name_of(experiment, "experiment"))


def predictor_spec(name: str, values: Mapping[str, Any]) -> Predictor:
    """A `Predictor` from keyword arguments, naming any argument it does not take."""
    for key in values:
        if key not in PREDICTOR_FIELDS:
            raise InvalidInput(
                f"gf.Predictor() has no argument `{key}`",
                hint=_suggest(key, PREDICTOR_FIELDS),
                param=key,
            )
    try:
        return Predictor(name, **values)
    except TypeError as error:
        raise InvalidInput(f"gf.Predictor(): {error}") from None


def _env_wait(name: str, env: str, revision: int | None, timeout: float | None) -> Op:
    deadline = _deadline(timeout)
    attempt = 0
    while True:
        reply = yield _http.predictors_environment(name, env)
        found = load(Environment, {"name": env, **(reply.result or {})})
        current = revision is None or found.revision == revision
        if current and found.state == "ready":
            return found
        if current and found.state in ("failed", "stopped"):
            raise failed(
                f"{name} revision {found.revision} failed to start in {env}",
                found.error,
                code="DeployFailed",
            )
        if _past(deadline):
            raise _not_done(
                f"deploying {name} to {env}",
                timeout,
                f"gf.predictors.get({name!r}) shows its environments",
            )
        shown = revision if revision is not None else found.revision
        yield Say(f"Deploying {name} revision {shown} to {env}  starting a replica")
        yield Sleep(_poll(attempt))
        attempt += 1


def _deployed(deploy: Deploy, wait: bool, timeout: float | None) -> Op:
    if not wait or deploy.action == "unchanged":
        return deploy
    number = deploy.revision.number if deploy.revision else None
    env = yield from _env_wait(deploy.predictor, deploy.env, number, timeout)
    return replace(deploy, environment=env)


def predictor_create(
    name: str, env: str, wait: bool, timeout: float, values: Mapping[str, Any]
) -> Op:
    """Creates a predictor from its fields and deploys it to `env`."""
    spec = predictor_spec(name, values)
    reply = yield _http.predictors_create(
        {"name": name, "spec": spec.to_json(), "env": env}
    )
    yield from _deployed(_deploy_of(reply.result, name, env), wait, timeout)
    return (yield from predictor_get(name))


def _deploy_of(result: Any, name: str, env: str) -> Deploy:
    result = dict(result or {})
    result.setdefault("predictor", name)
    result.setdefault("env", env)
    revision = result.get("revision")
    if isinstance(revision, Mapping) and isinstance(revision.get("spec"), Mapping):
        revision = dict(revision)
        revision["spec"] = {"name": name, **revision["spec"]}
        result["revision"] = revision
    return load(Deploy, result)


def predictor_update(
    predictor: Any, env: str, wait: bool, timeout: float, values: Mapping[str, Any]
) -> Op:
    """Makes a revision of the given fields over the serving spec and deploys it."""
    name = name_of(predictor, "predictor")
    if not values:
        raise InvalidInput("update changes at least one field")
    spec = spec_json(values, partial=True)
    reply = yield _http.predictors_update(name, {"spec": spec, "env": env})
    yield from _deployed(_deploy_of(reply.result, name, env), wait, timeout)
    return (yield from predictor_get(name))


def predictor_deploy(
    specs: Sequence[Predictor], env: str, dry_run: bool, wait: bool, timeout: float
) -> Op:
    """Deploys each spec in order, idempotently; one that fails stops the rest."""
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, Predictor):
            raise InvalidInput("deploy takes gf.Predictor specs", param="specs")
        if spec.name in seen:
            raise InvalidInput(f"the predictor {spec.name} is named twice")
        seen.add(spec.name)
    out: list[Deploy] = []
    for spec in specs:
        reply = yield _http.predictors_deploy(
            spec.name, {"spec": spec.to_json(), "env": env, "dry_run": dry_run}
        )
        deploy = _deploy_of(reply.result, spec.name, env)
        if not dry_run:
            deploy = yield from _deployed(deploy, wait, timeout)
        out.append(deploy)
    return out


def predictor_get(predictor: Any) -> Op:
    """Reads a predictor."""
    reply = yield _http.predictors_get(name_of(predictor, "predictor"))
    return load(PredictorInfo, reply.result)


def predictor_list() -> Op:
    """Lists predictors."""
    reply = yield _http.predictors_list()
    return loads(PredictorInfo, (reply.result or {}).get("predictors"))


def predictor_revisions(predictor: Any) -> Op:
    """Lists a predictor's revisions, newest first as the service orders them."""
    name = name_of(predictor, "predictor")
    reply = yield _http.predictors_revisions(name)
    out = []
    for item in (reply.result or {}).get("revisions") or ():
        item = dict(item)
        if isinstance(item.get("spec"), Mapping):
            item["spec"] = {"name": name, **item["spec"]}
        out.append(load(Revision, item))
    return out


WAITING = {
    "queued": "waiting for a GPU",
    "starting": "waiting for a GPU (cold start)",
    "running": "predicting",
}


def _answered(request: _http.Request, label: str) -> Op:
    """Sends a prediction part and follows a 202 until it is answered."""
    reply = yield request
    while reply.status == 202:
        status = reply.result or {}
        state = str(status.get("state") or "starting")
        yield Say(f"{label}  {WAITING.get(state, state)}")
        wait = status.get("retry_after") or reply.headers.get("retry-after") or 1
        yield Sleep(min(max(float(wait), 0.0), 30.0))
        location = reply.headers.get("location") or str(status.get("id") or "")
        if not location:
            raise InvalidInput("the service accepted a prediction without saying where")
        reply = yield _http.prediction_get(location)
    return reply


def _id_name(row: Mapping[str, Any]) -> str | None:
    for key in row:
        if key not in ("label", "value", "probabilities", "quantiles"):
            return key
    return None


def prediction_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Prediction rows as a table: `id`, `label`, `p_<class>`, `value`, `q_<level>`."""
    import pyarrow as pa

    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        key = _id_name(row)
        if key is not None:
            item["id"] = row[key]
        if "label" in row:
            item["label"] = row["label"]
        for cls, p in (row.get("probabilities") or {}).items():
            item[f"p_{cls}"] = p
        if "value" in row:
            item["value"] = row["value"]
        for level, v in (row.get("quantiles") or {}).items():
            item[f"q_{level}"] = v
        out.append(item)
    return pa.Table.from_pylist(out)


def predictions_of(reply: _http.Reply) -> Predictions:
    """A prediction reply, JSON or Arrow, as `Predictions`."""
    result = dict(reply.result or {})
    table = (
        reply.table
        if reply.table is not None
        else prediction_table(result.get("predictions") or ())
    )
    usage = result.get("usage") or {}

    def used(key: str, default: Any) -> Any:
        return usage.get(key, result.get(key, default))

    context = result.get("context")
    timing = result.get("timing")
    gpu = used("gpu_seconds", None)
    return Predictions(
        predictor=result.get("predictor"),
        revision=result.get("revision"),
        environment=result.get("environment"),
        model=str(result.get("model") or ""),
        task=result.get("task"),
        rows=int(used("rows", table.num_rows)),
        credits=float(used("credits", 0.0)),
        usd=float(used("usd", 0.0)),
        request_id=reply.request_id,
        table=table,
        gpu_seconds=float(gpu) if gpu is not None else None,
        timing=load(Timing, timing) if isinstance(timing, Mapping) else None,
        context=load(ContextUse, context) if isinstance(context, Mapping) else None,
        notes=tuple(str(n) for n in result.get("notes") or ()),
    )


def combined(parts: Sequence[Predictions]) -> Predictions:
    """The parts of one prediction, in order, as one."""
    import pyarrow as pa

    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    gpu = [p.gpu_seconds for p in parts if p.gpu_seconds is not None]
    timings = [p.timing for p in parts if p.timing is not None]
    return replace(
        first,
        rows=sum(p.rows for p in parts),
        credits=round(sum(p.credits for p in parts), 6),
        usd=round(sum(p.usd for p in parts), 9),
        table=pa.concat_tables([p.table for p in parts], promote_options="permissive"),
        gpu_seconds=sum(gpu) if gpu else None,
        timing=Timing(
            sum(t.queued_seconds for t in timings),
            sum(t.cold_start_seconds for t in timings),
            sum(t.compute_seconds for t in timings),
        )
        if timings
        else None,
        notes=tuple(dict.fromkeys(n for p in parts for n in p.notes)),
    )


def _scored(
    queries: Any,
    request: Callable[[Any], _http.Request],
    label: str,
    series: Sequence[str] = (),
) -> Op:
    rows = _data.table(queries, what="queries")
    parts = _data.split(rows, series=series)
    answers = []
    for index, part in enumerate(parts):
        shown = (
            label if len(parts) == 1 else f"{label}  part {index + 1} of {len(parts)}"
        )
        yield Say(shown)
        reply = yield from _answered(request(part), shown)
        answers.append(predictions_of(reply))
    return combined(answers)


def predictor_predict(predictor: Any, queries: Any, *, env: str, as_of: Any) -> Op:
    """Scores queries with a predictor, in parts by bytes."""
    name = name_of(predictor, "predictor")
    payload: dict[str, Any] = {"env": env}
    if as_of is not None:
        payload["as_of"] = (
            as_of.isoformat() if isinstance(as_of, datetime) else str(as_of)
        )
    return (
        yield from _scored(
            queries,
            lambda part: _http.predictors_predict(name, payload, part),
            f"Predicting with {name} in {env}",
        )
    )


def predictor_move(
    predictor: Any, env: str, verb: str, payload: Mapping[str, Any], wait: bool
) -> Op:
    """Promotes or rolls back an environment, then waits for it."""
    name = name_of(predictor, "predictor")
    reply = yield _http.predictors_move(name, env, verb, payload)
    found = load(Environment, {"name": env, **(reply.result or {})})
    if not wait:
        return found
    return (yield from _env_wait(name, env, found.revision, None))


def predictor_logs(predictor: Any, env: str, params: Mapping[str, Any]) -> Op:
    """Reads a page of an environment's logs."""
    query = {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in params.items()
    }
    if query.get("level") not in (None, "info", "warn", "error"):
        raise InvalidInput("level is info, warn or error", param="level")
    reply = yield _http.predictors_logs(name_of(predictor, "predictor"), env, query)
    result = reply.result or {}
    return Logs(
        tuple(loads(LogEntry, result.get("logs"))),
        result.get("next"),
        result.get("previous"),
    )


def predictor_delete(predictor: Any) -> Op:
    """Deletes a predictor."""
    yield _http.predictors_delete(name_of(predictor, "predictor"))


def snapshot(state: State, data: Any, *, what: str = "context") -> Op:
    """Uploads a table once as an Arrow snapshot named by its digest; returns the digest.

    Refuses a table past the snapshot ceiling at once, before any request.
    """
    rows = _data.table(data, what=what)
    body = _data.stream(rows)
    _data.check_snapshot(body, rows.num_rows, what=what)
    digest = _data.digest(body)
    if digest in state.snapshots:
        return digest, rows.num_rows
    try:
        yield _http.snapshot_probe(digest)
    except NotFound:
        yield Say(
            f"Uploading the {what}  {rows.num_rows:,} rows, {_data.size(len(body))}"
        )
        yield _http.snapshot_put(digest, body)
    state.snapshots.add(digest)
    return digest, rows.num_rows


def model_predict(
    state: State,
    model: str,
    *,
    context: Any,
    queries: Any,
    target: str,
    task: str | None,
    fields: Mapping[str, Any],
) -> Op:
    """Scores queries with a model, its context sent once by digest."""
    for key in fields:
        if key not in PREDICTOR_FIELDS or key in ("context", "model", "target", "task"):
            raise InvalidInput(
                f"models.predict() has no argument `{key}`",
                hint=_suggest(key, PREDICTOR_FIELDS),
                param=key,
            )
    if isinstance(context, Context):
        if not context.digest:
            raise InvalidInput(
                f"the context {context.name} has no snapshot digest", param="context"
            )
        digest, rows = context.digest, context.rows or 0
    else:
        digest, rows = yield from snapshot(state, context)
    payload = {
        "model": model,
        "target": target,
        **({"task": _task(task)} if task else {}),
        "context": {"snapshot": digest},
        **spec_json(fields, partial=False),
    }
    series = tuple(fields.get("series") or ())
    return (
        yield from _scored(
            queries,
            lambda part: _http.predict(payload, part),
            f"Predicting with {model}  {rows:,} context rows",
            series,
        )
    )


def model_of(data: Mapping[str, Any]) -> Model:
    """A `Model` from a catalog entry, whichever route sent it."""
    full = str(data.get("id") or data.get("name") or data.get("model") or "")
    resolved = str(data.get("model") or "")
    digest = data.get("digest")
    if digest is None and "@sha256:" in resolved:
        digest = "sha256:" + resolved.split("@sha256:", 1)[1]
    full = full.split("@sha256:", 1)[0]
    short = data.get("short") or full.split("/")[-1]
    kind = data.get("kind") or ("fine-tune" if "@ft" in full else "catalog")
    trained = data.get("trained_on")
    limits = data.get("limits")
    derived = data.get("derived") or {}
    return Model(
        id=full,
        short=str(short),
        task=data.get("task"),
        family=data.get("family") or (str(short) if data.get("short") else None),
        kind=kind,
        base=data.get("base") or derived.get("base"),
        trained_on=load(TrainedOn, trained) if isinstance(trained, Mapping) else None,
        hostable=bool(data.get("hostable", True)),
        digest=digest,
        architecture=dict(data.get("architecture") or {}),
        limits=load(Limits, limits) if isinstance(limits, Mapping) else None,
    )


def model_list(task: str | None, all: bool) -> Op:
    """Lists models."""
    reply = yield _http.models_list(_task(task), all)
    return [model_of(m) for m in (reply.result or {}).get("models") or ()]


def model_get(model: str, task: str | None) -> Op:
    """Reads one model."""
    reply = yield _http.models_get(model, _task(task))
    return model_of(reply.result or {})


CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "bigquery": ("project", "dataset"),
    "gcs": ("bucket", "prefix"),
    "snowflake": ("account", "warehouse", "database", "role"),
    "databricks": ("host", "warehouse", "client_id"),
    "s3": ("role_arn", "region", "bucket", "prefix"),
    "postgres": ("host", "port", "database", "user", "tls"),
}


def source_create(
    name: str, kind: str, secret: str | None, config: Mapping[str, Any]
) -> Op:
    """Connects a source, checking its config keys for its kind (3.8)."""
    _checked_name(name, "source")
    if kind not in SOURCE_KINDS:
        raise InvalidInput(
            f"`{kind}` is not a source kind",
            hint="kinds are " + ", ".join(SOURCE_KINDS),
            param="kind",
        )
    allowed = CONFIG_KEYS[kind]
    for key in config:
        if key not in allowed:
            raise InvalidInput(
                f"a {kind} source has no setting `{key}`",
                hint=_suggest(key, allowed) or "its settings are " + ", ".join(allowed),
                param=key,
            )
    payload: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "config": {k: str(v) for k, v in config.items()},
    }
    if secret is not None:
        payload["secret"] = secret
    reply = yield _http.sources_create(payload)
    return load(Source, reply.result)


def source_get(source: Any) -> Op:
    """Reads a source."""
    reply = yield _http.sources_get(name_of(source, "source"))
    return load(Source, reply.result)


def source_list() -> Op:
    """Lists sources."""
    reply = yield _http.sources_list()
    return loads(Source, (reply.result or {}).get("sources"))


def source_test(source: Any) -> Op:
    """Tests a source's connection now."""
    reply = yield _http.sources_action(name_of(source, "source"), "test", {})
    return load(Source, reply.result)


def source_tables(source: Any, prefix: str | None) -> Op:
    """Lists a source's tables."""
    reply = yield _http.sources_tables(name_of(source, "source"), prefix)
    return loads(SourceTable, (reply.result or {}).get("tables"))


def source_preview(source: Any, table: str | None, sql: str | None) -> Op:
    """Twenty rows of a source's table or query."""
    if (table is None) == (sql is None):
        raise InvalidInput("preview reads one table= or one sql=")
    payload = {"table": table} if table is not None else {"sql": sql}
    reply = yield _http.sources_action(name_of(source, "source"), "preview", payload)
    result = reply.result or {}
    return _data.frame(result.get("columns"), result.get("rows"))


def source_delete(source: Any) -> Op:
    """Deletes a source."""
    yield _http.sources_action(name_of(source, "source"), "delete", {})


KEY_ID = re.compile(r"^key[-_][0-9A-Za-z]+$")


def key_id(key: Any) -> Op:
    """A key's id from its id, its name or the key itself."""
    if isinstance(key, Key):
        return key.id
    if not isinstance(key, str) or not key:
        raise InvalidInput("a key is its id, its name or a gf.Key", param="key")
    if KEY_ID.match(key):
        return key
    found = yield from key_list()
    matches = [k for k in found if k.name == key]
    if not matches:
        raise NotFound(
            f"no key named `{key}`",
            hint=_suggest(key, [k.name for k in found]) or "gf.keys.list() shows them",
        )
    if len(matches) > 1:
        raise AmbiguousReference(
            f"{len(matches)} keys are named `{key}`", hint="pass the key's id instead"
        )
    return matches[0].id


def key_list() -> Op:
    """Lists the project's keys."""
    reply = yield _http.keys_list()
    return loads(Key, (reply.result or {}).get("keys"))


def _new_key(result: Mapping[str, Any]) -> NewKey:
    token = result.get("token")
    if not isinstance(token, str):
        raise InvalidInput("the service sent a new key without its token")
    return NewKey(load(Key, result), token)


def key_create(
    name: str, permissions: Sequence[str], expires: str | timedelta | None
) -> Op:
    """Creates a key; `expires=None` never expires."""
    _checked_name(name, "key")
    if isinstance(permissions, str):
        permissions = [permissions]
    payload = {
        "name": name,
        "permissions": list(permissions or STANDARD_PERMISSIONS),
        "expires_in": _duration(expires, "expires"),
    }
    reply = yield _http.keys_create(payload)
    return _new_key(reply.result or {})


def key_action(key: Any, verb: str) -> Op:
    """Rotates, revokes or deletes a key."""
    found = yield from key_id(key)
    reply = yield _http.keys_action(found, verb)
    if verb == "rotate":
        return _new_key(reply.result or {})
    if verb == "revoke":
        # The reply says only that it is revoked; the key reads as the list shows it.
        keys = yield from key_list()
        return next((k for k in keys if k.id == found), Key(found, state="revoked"))
    return None


def estimate_of(result: Mapping[str, Any]) -> Estimate:
    """An `Estimate` with its large-context line, when the price has one."""
    estimate = load(Estimate, result)
    large = result.get("large_context")
    if isinstance(large, Mapping):
        estimate = replace(estimate, large_context=load(LargeContext, large))
    return estimate


def predict_estimate(
    model: str, task: str | None, context_rows: int, queries: int
) -> Op:
    """Prices a prediction without making it (3.4), the large-context line included."""
    payload: dict[str, Any] = {
        "model": model,
        "context_rows": context_rows,
        "queries": queries,
    }
    if task:
        payload["task"] = _task(task)
    reply = yield _http.estimate(payload)
    return estimate_of(reply.result or {})
