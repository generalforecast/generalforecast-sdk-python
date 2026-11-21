# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""A fake General Forecast service, built from the contract's examples.

It serves the routes of `api/openapi.yaml` over real HTTP on loopback,
keeps its resources in memory, and records every request, so tests drive
the SDK end to end. Its canned answers (the catalog, a leaderboard, auth
status, logs, slices) are the contract's examples, and every request the
SDK sends and every reply the fake gives is checked against the route's
schemas (`tests/contract.py`): a test fails when they part ways.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

import pyarrow as pa

from generalforecast import _data
from tests import contract

TOKEN = "gf_test_0123456789abcdefghijklmnopqrstuv"
CONSOLE = "https://app.generalforecast.com"
NOW = "2026-11-21T09:12:04Z"
EPOCH = 1795252324.0  # NOW, in Unix seconds
DAY = 86400.0
ARROW = "application/vnd.apache.arrow.stream"

ORIGIN = contract.example("contexts.get")["origin"]
LEADERBOARD = contract.example("experiments.get")["run"]["leaderboard"]
# The times of the surface spec's leaderboard sample.
for _entry, _seconds in zip(LEADERBOARD, (38.2, 51.0, 44.0), strict=False):
    _entry["seconds"] = _seconds
# The recommended sets of section 3.5, run when an experiment names no models.
RECOMMENDED = {
    "classification": ["kumo-medium", "tabicl-v2", "mitra-2", "kumo-small"],
    "regression": ["kumo-medium", "tabicl-v2", "mitra-2", "kumo-small"],
    "forecast": ["chronos-2", "timestral-209m"],
}


def digest(text: str) -> str:
    """A well-formed digest standing for `text`."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def origin(surface: str = "notebook") -> dict[str, Any]:
    """Where a resource was made, as the contract writes it."""
    return {
        **ORIGIN,
        "surface": surface,
        "notebook": ORIGIN["notebook"] if surface == "notebook" else None,
    }


def model(
    full: str,
    short: str,
    task: str,
    *,
    family: str | None = None,
    context_rows: int = 100000,
    queries: int = 100,
    inputs: int = 10000,
) -> dict[str, Any]:
    """A catalog entry as the plane serves one, shaped as the contract's `models.list` example.

    `limits` are the model's own, the engine catalog's `limits` of it, and
    the plane names no warm shapes.
    """
    entry = contract.example("models.list")["models"][0]
    pinned = digest(full)
    entry.update(
        model=f"{full}@{pinned}", name=full, short=short, task=task, family=family or short,
        repository=full, limits={"context_rows": context_rows, "queries": queries, "inputs": inputs},
        warm_shapes=[],
    )  # fmt: skip
    entry["bounds"] = {**entry["bounds"], "context_rows": context_rows, "query_rows": queries,
                       "features": inputs}  # fmt: skip
    entry["artifact"] = {**entry["artifact"], "repository": full, "digest": pinned}
    if task != "classification":
        entry["trained_on"] = None
    return entry


CATALOG = [
    model("general-forecast/tabicl-v2-classifier", "tabicl-v2", "classification", family="tabicl"),
    model("general-forecast/tabicl-v2-regressor", "tabicl-v2", "regression", family="tabicl"),
    model("general-forecast/kumo-tabular-medium-classifier", "kumo-medium", "classification",
          family="kumo", context_rows=60000),
    model("general-forecast/kumo-tabular-medium-regressor", "kumo-medium", "regression",
          family="kumo", context_rows=60000),
    model("general-forecast/kumo-tabular-small-classifier", "kumo-small", "classification",
          family="kumo", context_rows=60000),
    model("general-forecast/kumo-tabular-small-regressor", "kumo-small", "regression",
          family="kumo", context_rows=60000),
    model("general-forecast/kumo-tabular-large-classifier", "kumo-large", "classification",
          family="kumo", context_rows=60000),
    model("general-forecast/mitra-2-classifier", "mitra-2", "classification", context_rows=5120),
    model("general-forecast/mitra-2-regressor", "mitra-2", "regression", context_rows=5120),
    model("general-forecast/chronos-2", "chronos-2", "forecast", queries=8, inputs=8),
    model("general-forecast/timestral-209m", "timestral-209m", "forecast", queries=8, inputs=8),
]  # fmt: skip


def entry(full: str) -> dict[str, Any]:
    """A catalog entry by full id or short name; classification for a short name with several."""
    return next(m for m in CATALOG if full in (m["name"], m["short"]))


class Refusal(Exception):
    """An error reply."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        given = {k: v for k, v in extra.items() if v is not None}
        self.status, self.body = status, {"code": code, "message": message, **given}


@dataclass
class Call:
    """One request as the fake saw it."""

    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    payload: Any = None
    body: bytes = b""
    table: pa.Table | None = None
    envelope: dict[str, Any] | None = None


@dataclass
class Fake:
    """The service's state; tests read and change it directly."""

    url: str = ""
    project: str = "acme"
    calls: list[Call] = field(default_factory=list)
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    snapshots: dict[str, pa.Table] = field(default_factory=dict)
    experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictors: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    writes: list[dict[str, Any]] = field(default_factory=list)
    chunk_bytes: int = 4096
    # Contexts and snapshots upload to a signed object-store URL, query string
    # and all, as the hosted service's do; False answers `url: null`, as a
    # plane without object storage does, and takes the chunks at the bytes routes.
    signed_uploads: bool = True
    # Bytes at the end of every chunk a signed upload session does not keep,
    # answering 308 with a `Range` that says what it holds.
    chunk_loss: int = 0
    polls_until_done: int = 2
    accept_predictions: int = 0
    # False answers `available: false` to `sources.list`, as a plane that connects no sources.
    sources_available: bool = True
    fail: dict[str, int] = field(default_factory=dict)
    # The reasons of the 429s a handler answers, one a call, before it answers.
    throttle: dict[str, list[str]] = field(default_factory=dict)
    pending_logins: int = 1
    deploy_ends: str = "ready"
    training: dict[str, dict[str, Any]] = field(default_factory=dict)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    run_ends: str = "finished"
    errors: list[str] = field(default_factory=list)
    # The key the tests call with never expires, as a key made with `expires=None`.
    key: dict[str, Any] = field(default_factory=lambda: {
        "id": "key-0f1e2d3c4b5a69788796", "name": "notebook-maya-mbp", "expires_at": None})  # fmt: skip
    catalog: list[dict[str, Any]] = field(default_factory=lambda: list(CATALOG))
    log_entries: list[dict[str, Any]] = field(
        default_factory=lambda: contract.example("environments.logs")["logs"]
    )
    source_errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Every console page the fake linked to, fetched once it stops.
    links: set[str] = field(default_factory=set)

    def calls_to(self, method: str, pattern: str) -> list[Call]:
        """The calls whose path matches `pattern`."""
        return [
            c
            for c in self.calls
            if c.method == method and re.fullmatch(pattern, c.path)
        ]

    def link(self, part: str = "") -> str:
        """A console page, as the plane links one: the project in the query, the page in the fragment."""
        found = f"{self.url}/?project={self.project}" + (f"#/{part}" if part else "")
        self.links.add(found)
        return found

    def context_json(self, name: str, version: int | None = None) -> dict[str, Any]:
        ctx = self.contexts[name]
        out = {k: v for k, v in ctx.items() if not k.startswith("_")}
        if version is not None:
            found = [v for v in ctx["_versions"] if v["version"] == version]
            if not found:
                raise Refusal(
                    404, "NotFound", f"context {name} has no version {version}"
                )
            out.update(found[0])
        return out

    def handle(self, call: Call) -> tuple[int, Any, dict[str, str]]:
        for method, pattern, handler in ROUTES:
            if call.method != method:
                continue
            found = re.fullmatch(pattern, call.path)
            if found:
                key = handler.__name__
                if self.fail.get(key):
                    self.fail[key] -= 1
                    raise Refusal(503, "Unavailable", "the service is busy")
                if self.throttle.get(key):
                    reason = self.throttle[key].pop(0)
                    standing = (
                        {"limit": 1, "actual": 1} if reason != "rate_limited" else {}
                    )
                    raise Refusal(429, "CapacityExceeded", f"refused: {reason}", reason=reason,
                                  **standing)  # fmt: skip
                return handler(self, call, *found.groups())
        raise Refusal(404, "NotFound", f"no route {call.method} {call.path}")


def ok(
    result: Any, status: int = 200, **headers: str
) -> tuple[int, Any, dict[str, str]]:
    return status, result, headers


def _named(count: int) -> list[dict[str, Any]]:
    return [
        {"name": f"c{i}", "type": "number", "missing": 0, "distinct": None}
        for i in range(count)
    ]


def _columns(table: pa.Table) -> list[dict[str, Any]]:
    out = []
    for name in table.column_names:
        column = table.column(name)
        kind = column.type
        type_ = (
            "number" if pa.types.is_integer(kind) or pa.types.is_floating(kind)
            else "boolean" if pa.types.is_boolean(kind) else "category"
        )  # fmt: skip
        out.append({"name": name, "type": type_, "missing": column.null_count,
                    "distinct": len(set(column.to_pylist()))})  # fmt: skip
    return out


def auth_status(fake: Fake, call: Call) -> Any:
    status = contract.example("auth.status")
    status.update(
        principal={"kind": "api-key", "id": fake.key["id"], "project": fake.project},
        key={**fake.key, "project": fake.project}, email="maya@acme.com", machine="maya-mbp",
        workspace={"id": "acme", "display_name": "Acme Analytics"}, url=fake.link(),
    )  # fmt: skip
    return ok(status)


def auth_logout(fake: Fake, call: Call) -> Any:
    return ok({"revoked": fake.key["id"]})


def device_start(fake: Fake, call: Call) -> Any:
    assert call.payload["client"] in ("cli", "notebook")
    fake.login = dict(call.payload)  # type: ignore[attr-defined]
    grant = contract.example("device.start")
    grant.update(
        user_code="WXHG-PTRB",
        verification_uri=f"{fake.url}/device",
        verification_uri_complete=f"{fake.url}/device?code=WXHG-PTRB",
    )
    fake.links |= {grant["verification_uri"], grant["verification_uri_complete"]}
    return ok(grant)


def device_token(fake: Fake, call: Call) -> Any:
    if fake.pending_logins > 0:
        fake.pending_logins -= 1
        raise Refusal(400, "Unauthorized", "waiting for browser approval",
                      reason="authorization_pending")  # fmt: skip
    login = fake.login  # type: ignore[attr-defined]
    notebook = login["client"] == "notebook"
    expires = 1795295640.0 if notebook else 1803029524.0
    token = contract.example("device.token")
    token.update(
        access_token=TOKEN, expires_at=expires, project=fake.project,
        key={"id": "key-0f1e2d3c4b5a69788796", "name": f"{login['client']}-{login['machine']}",
             "project": fake.project, "expires_at": expires, "ephemeral": notebook},
        email="maya@acme.com", workspace={"id": "acme", "display_name": "Acme Analytics"},
        url=fake.link(),
    )  # fmt: skip
    return ok(token)


def _new_context(fake: Fake, payload: dict[str, Any], state: str) -> dict[str, Any]:
    name = payload["name"]
    source = payload.get("source")
    ctx = {
        "name": name, "kind": payload["kind"], "state": state, "version": None, "versions": 0,
        "digest": None, "rows": None, "columns": [], "id": payload.get("id"),
        "time": payload.get("time"),
        "source": {"kind": "bigquery", "table": None, "sql": None, **source} if source else None,
        "freshness_seconds": payload.get("freshness_seconds"), "read_at": None, "error": None,
        "predictors": [], "experiments": [], "origin": origin(), "created": NOW, "updated": NOW,
        "url": fake.link(f"data/contexts/{name}"), "_versions": [], "_rows": [],
    }  # fmt: skip
    if payload.get("columns"):
        ctx["columns"] = [{"name": k, "type": v, "missing": 0, "distinct": None}
                          for k, v in payload["columns"].items()]  # fmt: skip
    fake.contexts[name] = ctx
    return ctx


def _open_upload(fake: Fake, name: str, data: dict[str, Any]) -> Any:
    ctx = fake.contexts[name]
    if ctx["digest"] == data.get("digest"):
        return ok({**fake.context_json(name), "unchanged": True})
    version = ctx["versions"] + 1
    fake.uploads[name] = {"version": version, "bytes": data["bytes"], "data": b"",
                          "digest": data.get("digest")}  # fmt: skip
    ctx["state"] = "uploading"
    ctx["file"] = data.get("file")
    out = fake.context_json(name)
    url = (
        f"{fake.url}{_signed(f'/uploads/contexts/{name}', name, version)}"
        if fake.signed_uploads
        else None
    )
    out["upload"] = {"version": version, "url": url, "chunk_bytes": fake.chunk_bytes}
    return ok(out)


def _signed(path: str, name: str, version: int) -> str:
    """An upload URL as an object store signs one: the session rides in its query."""
    return f"{path}?uploadType=resumable&upload_id={_signed_session(name, version)}"


def _signed_session(name: str, version: int) -> str:
    return f"{name}-{version}"


def contexts_create(fake: Fake, call: Call) -> Any:
    payload = call.payload
    if payload["name"] in fake.contexts:
        raise Refusal(409, "Conflict", f"a context named {payload['name']} exists",
                      reason="already_exists")  # fmt: skip
    if payload.get("data"):
        _new_context(fake, payload, "uploading")
        return _open_upload(fake, payload["name"], payload["data"])
    if payload.get("source"):
        ctx = _new_context(fake, payload, "importing")
        ctx["_polls"] = fake.polls_until_done
        if payload["kind"] == "snapshot":
            ctx["_final"] = {"version": 1, "versions": 1, "digest": digest("customers@1"), "rows": 48210,
                             "columns": _named(9)}  # fmt: skip
        else:
            ctx["_final"] = {"rows": 12804, "columns": _named(11), "read_at": NOW}
        return ok(fake.context_json(payload["name"]))
    ctx = _new_context(fake, payload, "ready")
    ctx["rows"] = 0
    return ok(fake.context_json(payload["name"]))


def contexts_version(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.contexts:
        raise Refusal(404, "NotFound", f"no context {name}")
    return _open_upload(fake, name, call.payload["data"])


def contexts_bytes(fake: Fake, call: Call, name: str, version: str) -> Any:
    upload = fake.uploads[name]
    assert int(version) == upload["version"]
    assert int(call.query["offset"]) == len(upload["data"]), "chunks arrive in order"
    return ok({"received": _receive(fake, name, call.body)})


def _content_range(call: Call) -> tuple[int, int]:
    """A chunk's first byte and the upload's total, from its `Content-Range: bytes first-last/total`."""
    found = re.fullmatch(
        r"bytes (\d+)-(\d+)/(\d+)", call.headers.get("content-range", "")
    )
    assert found and int(found[2]) - int(found[1]) + 1 == len(call.body), (
        "Content-Range names the chunk"
    )
    return int(found[1]), int(found[3])


def _session(fake: Fake, call: Call, held: int, session: str) -> bytes:
    """What a signed resumable session keeps of a chunk: all but `chunk_loss` of its bytes.

    The chunk must start where the session holds bytes up to, as its last
    308's `Range` said.
    """
    assert call.query == {"uploadType": "resumable", "upload_id": session}, (
        "a signed upload URL keeps its query string"
    )
    assert "authorization" not in call.headers, "an upload URL carries no credential"
    start, _ = _content_range(call)
    assert start == held, (
        f"a chunk starts at byte {held}, where the session holds up to"
    )
    return call.body[: max(1, len(call.body) - fake.chunk_loss)]


def _resumed(held: int, total: int) -> Any:
    """A session's answer: 308 with the bytes it holds while it waits for more, 200 once it holds them all."""
    if held < total:
        return 308, None, {"Range": f"bytes=0-{held - 1}"}
    return ok(None)


def contexts_signed_upload(fake: Fake, call: Call, name: str) -> Any:
    """A chunk sent to a context's signed upload URL, the object store's and not an API route."""
    upload = fake.uploads[name]
    session = _signed_session(name, upload["version"])
    kept = _session(fake, call, len(upload["data"]), session)
    return _resumed(_receive(fake, name, kept), upload["bytes"])


def _receive(fake: Fake, name: str, chunk: bytes) -> int:
    """Adds a chunk to a context's open upload, verifying the version on its last byte as the plane does.

    The plane hashes the table it was sent (`table_digest`), whatever
    digest the upload announced, and drops a version whose rows are the
    latest's.
    """
    upload = fake.uploads[name]
    upload["data"] += chunk
    if len(upload["data"]) == upload["bytes"]:
        table = _data.read_parquet(upload["data"])
        found = _data.table_digest(table)
        if upload["digest"] != found:
            fake.errors.append(f"context {name}: the upload announced {upload['digest']}, its table is {found}")  # fmt: skip
        ctx = fake.contexts[name]
        if found == ctx["digest"]:
            ctx.update(state="ready")
            return len(upload["data"])
        ctx.update(state="verifying", version=upload["version"], versions=upload["version"],
                   digest=found, rows=table.num_rows, columns=_columns(table),
                   updated=NOW, _polls=1)  # fmt: skip
        ctx["_versions"].append({"version": upload["version"], "digest": found,
                                 "rows": table.num_rows})  # fmt: skip
        ctx["_rows"] = table.to_pylist()
    return len(upload["data"])


def contexts_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.contexts:
        raise Refusal(404, "NotFound", f"no context named `{name}`")
    ctx = fake.contexts[name]
    if ctx.get("_polls"):
        ctx["_polls"] -= 1
        if ctx["state"] == "importing":
            ctx["rows"] = 31200 if ctx["_polls"] else ctx["_final"]["rows"]
        if not ctx["_polls"]:
            ctx.update(ctx.pop("_final", {}), state="ready")
    version = call.query.get("version")
    return ok(fake.context_json(name, int(version) if version else None))


def contexts_list(fake: Fake, call: Call) -> Any:
    return ok({"contexts": [fake.context_json(n) for n in fake.contexts]})


def contexts_refresh(fake: Fake, call: Call, name: str) -> Any:
    ctx = fake.contexts[name]
    version = (ctx["version"] or 0) + 1
    ctx["state"], ctx["_polls"] = "importing", fake.polls_until_done
    ctx["_final"] = {
        "version": version,
        "versions": version,
        "digest": digest(f"{name}@{version}"),
        "rows": 48377,
    }
    return ok(fake.context_json(name))


def contexts_rows(fake: Fake, call: Call, name: str) -> Any:
    ctx = fake.contexts[name]
    limit = int(call.query.get("limit", 20))
    start = int(call.query.get("after", 0))
    names = [c["name"] for c in ctx["columns"]]
    rows = [
        {n: r.get(n) for n in names}
        for r in ctx["_rows"][start : start + min(limit, 3)]
    ]
    after = start + len(rows)
    return ok({"columns": ctx["columns"], "rows": rows,
               "next": str(after) if after < len(ctx["_rows"]) else None})  # fmt: skip


def contexts_write(fake: Fake, call: Call, name: str, verb: str) -> Any:
    fake.writes.append({"context": name, "verb": verb, **call.payload})
    items = call.payload.get("rows") or call.payload.get("labels") or []
    return ok({"context": name, "write_id": call.payload["write_id"], "operation": verb,
               "rows": len(items), "sequence": 1900 + len(fake.writes),
               "accepted_at": 1761400100.0, "durably_accepted": True})  # fmt: skip


def contexts_delete(fake: Fake, call: Call, name: str) -> Any:
    if fake.contexts[name]["predictors"]:
        raise Refusal(409, "Conflict", f"the predictor {fake.contexts[name]['predictors'][0]} serves {name}",
                      reason="context_in_use")  # fmt: skip
    del fake.contexts[name]
    return ok({"name": name, "deleted": True})


def snapshots_put(fake: Fake, call: Call) -> Any:
    digest = call.query["digest"]
    if not call.body:
        if digest not in fake.snapshots:
            raise Refusal(404, "NotFound", f"no snapshot {digest}")
        return ok({"digest": digest, "rows": fake.snapshots[digest].num_rows})
    table = _data.read_stream(call.body)
    found = _data.table_digest(table)
    if found != digest:
        fake.errors.append(
            f"PUT /snapshots?digest={digest} sent a table whose digest is {found}"
        )
        raise Refusal(422, "ValidationError", f"the table's digest is {found}, not {digest}",
                      param="digest")  # fmt: skip
    fake.snapshots[digest] = table
    return ok({"digest": digest, "rows": table.num_rows}, 201)


def _entries(models: list[str]) -> list[dict[str, Any]]:
    """Finished leaderboard entries for these models, the contract's example where it has one."""
    known = {e["short"]: e for e in LEADERBOARD}
    out = []
    for name in models:
        found = entry(name)
        base = known.get(found["short"]) or dict(LEADERBOARD[-1], score=0.9)
        out.append(dict(base, model=found["name"], short=found["short"]))
    return out


def _run(
    fake: Fake, experiment: str, number: int, models: list[str], state: str
) -> dict[str, Any]:
    return {
        "experiment": experiment, "number": number, "state": state, "stale": False,
        "models": [entry(m)["name"] for m in models], "metric": "auroc", "lower_is_better": False,
        "held_rows": 6513, "context_rows": 26048, "repeats": 1,
        "leaderboard": [dict(e, state="queued", score=None, interval=None) for e in _entries(models)],
        "rows": 0, "credits": 0.0, "usd": 0.0, "created": NOW, "started": None, "finished": None,
        "origin": origin(), "error": None, "url": fake.link(f"experiments/{experiment}?run={number}"),
        "_polls": fake.polls_until_done,
    }  # fmt: skip


def _experiment_json(fake: Fake, name: str, number: int | None) -> dict[str, Any]:
    exp = fake.experiments[name]
    runs = exp["_runs"]
    out = {k: v for k, v in exp.items() if not k.startswith("_")}
    run = runs[(number or len(runs)) - 1] if runs else None
    if run is not None and run["state"] in ("queued", "running"):
        run["_polls"] -= 1
        if run["_polls"] <= 0:
            count = len(run["leaderboard"])
            run.update(state="finished", leaderboard=_entries(run["models"]), rows=6513 * count,
                       credits=13.0 * count, usd=0.013 * count, started="2026-11-21T09:10:52Z",
                       finished=NOW)  # fmt: skip
        else:
            run["state"] = "running"
            if len(run["leaderboard"]) > 1:
                run["leaderboard"][0] = _entries(run["models"])[0]
    out["run"] = (
        {k: v for k, v in run.items() if not k.startswith("_")} if run else None
    )
    out["runs"] = [{"number": r["number"], "state": r["state"], "models": r["models"],
                    "best": r.get("_best", "tabicl-v2"), "score": r.get("_score", 0.931),
                    "origin": origin(), "created": NOW} for r in runs]  # fmt: skip
    return out


def experiments_create(fake: Fake, call: Call) -> Any:
    payload = dict(call.payload)
    context = payload["context"]
    if context not in fake.contexts:
        raise Refusal(404, "NotFound", f"no context named `{context}`")
    columns = [c["name"] for c in fake.contexts[context]["columns"]]
    if columns and payload["target"] not in columns:
        raise Refusal(422, "ValidationError", f"the context {context} has no column `{payload['target']}`",
                      hint="a similar column exists: `income`", param="target")  # fmt: skip
    task = payload.get("task") or "classification"
    models = payload.pop("models", None) or RECOMMENDED[task]
    if payload.pop("dry_run", False):
        return ok({"setup": payload, "estimate": {"calls": 198, "rows": 19539, "credits": 39.1, "usd": 0.0391},
                   "warnings": []})  # fmt: skip
    name = payload.get("name") or f"{context}-{payload['target']}"
    fake.experiments[name] = {
        "name": name, "title": payload.get("title") or "", "context": context,
        "context_version": payload.get("version") or fake.contexts[context]["version"],
        "target": payload["target"], "task": task,
        "id": payload.get("id"), "time": payload.get("time"), "inputs": payload.get("inputs") or [],
        "types": payload.get("types") or {}, "sql": payload.get("sql"),
        "holdout": payload.get("holdout") or {"kind": "random", "fraction": 0.2},
        "metric": "auroc", "origin": origin(), "created": NOW, "updated": NOW,
        "url": fake.link(f"experiments/{name}"),
        "_runs": [_run(fake, name, 1, models, "queued")], "_setup": payload,
    }  # fmt: skip
    out = _experiment_json(fake, name, 1)
    return ok(out)


def experiments_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.experiments:
        raise Refusal(404, "NotFound", f"no experiment named `{name}`")
    run = call.query.get("run")
    return ok(_experiment_json(fake, name, int(run) if run else None))


def experiments_list(fake: Fake, call: Call) -> Any:
    names = [n for n, e in fake.experiments.items()
             if not call.query.get("context") or e["context"] == call.query["context"]]  # fmt: skip
    start = int(call.query.get("after", 0))
    page = names[start : start + 1]
    found = []
    for n in page:
        item = _experiment_json(fake, n, None)
        if item["run"]:
            item["run"]["leaderboard"] = []
        found.append(item)
    after = start + 1
    return ok(
        {"experiments": found, "next": str(after) if after < len(names) else None}
    )


def experiments_update(fake: Fake, call: Call, name: str) -> Any:
    exp = fake.experiments[name]
    exp["_updates"] = [*exp.get("_updates", []), call.payload]
    for key, value in call.payload.items():
        if key in exp:
            exp[key] = value
    for run in exp["_runs"]:
        run["stale"] = True
    return ok(_experiment_json(fake, name, None))


def experiments_run(fake: Fake, call: Call, name: str) -> Any:
    exp = fake.experiments[name]
    models = call.payload.get("models") or ["tabicl-v2"]
    run = _run(fake, name, len(exp["_runs"]) + 1, models, "queued")
    exp["_runs"].append(run)
    return ok({k: v for k, v in run.items() if not k.startswith("_")})


def experiments_cancel(fake: Fake, call: Call, name: str, number: str) -> Any:
    run = fake.experiments[name]["_runs"][int(number) - 1]
    run.update(state="cancelled", _polls=0, credits=4.0, usd=0.004)
    run["leaderboard"] = [e if e["state"] == "finished" else dict(e, state="cancelled")
                          for e in run["leaderboard"]]  # fmt: skip
    return ok({k: v for k, v in run.items() if not k.startswith("_")})


def experiments_rows(fake: Fake, call: Call, name: str, number: str) -> Any:
    after = int(call.query.get("after", 0))
    held = contract.example("experiments.rows")
    held["rows"] = [dict(held["rows"][0], row=after + i) for i in range(2)]
    held["model"] = entry(call.query.get("model") or "tabicl-v2")["name"]
    held["next"] = "2" if after == 0 else None
    return ok(held)


def experiments_slices(fake: Fake, call: Call, name: str, number: str) -> Any:
    found = contract.example("experiments.slices")
    found["slices"].append({"column": "education", "value": "HS-grad", "rows": 2096,
                            "score": 0.902, "delta": -0.029})  # fmt: skip
    return ok(found)


def experiments_delete(fake: Fake, call: Call, name: str) -> Any:
    del fake.experiments[name]
    return ok({"name": name, "deleted": True})


def _resolved(fake: Fake, spec: dict[str, Any]) -> dict[str, Any]:
    found = next(
        (m for m in fake.catalog if spec["model"] in (m["short"], m["name"])), None
    )
    if found is None:
        raise Refusal(404, "NotFound", f"no model named `{spec['model']}`", param="spec.model",
                      hint="a similar model exists: `kumo-medium`; `gf models list` shows them all")  # fmt: skip
    name, _, pinned = str(spec["context"]).partition("@")
    ctx = fake.contexts.get(name)
    if ctx is None:
        raise Refusal(404, "NotFound", f"no context named `{name}`")
    return {"model": found["name"], "short": found["short"],
            "model_digest": found["artifact"]["digest"],
            "base_digest": (found.get("derived") or {}).get("base_digest"),
            "task": spec.get("task") or found["task"],
            "context": name, "context_version": int(pinned) if pinned else ctx["version"],
            "context_digest": ctx["digest"],
            "context_rows": ctx["rows"] if ctx["kind"] == "snapshot" else None,
            "inputs": [c["name"] for c in ctx["columns"] if c["name"] != spec["target"]],
            "types": {}}  # fmt: skip


def _deploy(
    fake: Fake, name: str, spec: dict[str, Any], env: str, dry_run: bool, create: bool
) -> Any:
    resolved = _resolved(fake, spec)
    existing = fake.predictors.get(name)
    if existing is None:
        action, number = "created", 1
        changes = _changes(None, spec, resolved)
    else:
        serving = existing["environments"].get(env, {}).get("revision")
        current = existing["_revisions"][serving - 1] if serving else None
        if current is not None and current["resolved"] == resolved:
            return ok({"predictor": name, "env": env, "action": "unchanged", "changes": [],
                       "revision": current, "environment": _shown(existing["environments"][env])})  # fmt: skip
        action, number = "updated", len(existing["_revisions"]) + 1
        changes = _changes(current, spec, resolved)
    revision = {"predictor": name, "number": number, "status": "pending" if dry_run else "building",
                "lock_digest": digest(json.dumps({"spec": spec, "resolved": resolved}, sort_keys=True)),
                "spec": spec, "resolved": resolved, "origin": origin(), "created": NOW}  # fmt: skip
    environment = {"name": env, "revision": number, "state": "starting", "generation": number,
                   "calls_today": 0, "p50_seconds": None, "updated": NOW,
                   "url": fake.link(f"predictors/{name}"), "_polls": fake.polls_until_done,
                   "_final": fake.deploy_ends}  # fmt: skip
    if not dry_run:
        if existing is None:
            existing = fake.predictors[name] = {
                "name": name, "_revisions": [], "environments": {}, "created": NOW,
                "url": fake.link(f"predictors/{name}"), "origin": origin(),
            }  # fmt: skip
            ctx = fake.contexts[resolved["context"]]
            ctx["predictors"] = [*ctx["predictors"], name]
        existing["_revisions"].append(revision)
        previous = existing["environments"].get(env)
        environment["_previous"] = previous["revision"] if previous else None
        existing["environments"][env] = environment
    if dry_run:
        environment = (existing or {}).get("environments", {}).get(env)
    return ok({"predictor": name, "env": env, "action": action, "changes": changes,
               "revision": revision, "environment": _shown(environment) if environment else None})  # fmt: skip


def _shown(item: dict[str, Any]) -> dict[str, Any]:
    """A stored resource without the fake's own bookkeeping."""
    return {k: v for k, v in item.items() if not k.startswith("_")}


def _changes(
    current: dict[str, Any] | None, spec: dict[str, Any], resolved: dict[str, Any]
) -> list[dict[str, Any]]:
    """What a spec changes, as the contract's example writes it: the model by short name, the context pinned."""
    before = dict(current["spec"]) if current else {}
    after = dict(spec)
    if current:
        was = current["resolved"]
        before.update(
            model=was["short"], context=f"{was['context']}@{was['context_version']}"
        )
    after.update(
        model=resolved["short"],
        context=f"{resolved['context']}@{resolved['context_version']}",
    )
    keys = dict.fromkeys(["model", "context", "target", *before, *after])
    return [{"field": k, "before": None if before.get(k) is None else _text(before[k]),
             "after": None if after.get(k) is None else _text(after[k])}
            for k in keys if before.get(k) != after.get(k)]  # fmt: skip


def _text(value: Any) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def predictors_create(fake: Fake, call: Call) -> Any:
    name = call.payload["name"]
    if name in fake.predictors:
        raise Refusal(
            409, "Conflict", f"a predictor named {name} exists", reason="already_exists"
        )
    return _deploy(fake, name, call.payload["spec"], call.payload["env"], False, True)


def predictors_deploy(fake: Fake, call: Call, name: str) -> Any:
    p = call.payload
    return _deploy(fake, name, p["spec"], p["env"], bool(p.get("dry_run")), False)


def predictors_update(fake: Fake, call: Call, name: str) -> Any:
    existing = fake.predictors[name]
    env = call.payload["env"]
    serving = existing["environments"][env]["revision"]
    spec = dict(existing["_revisions"][serving - 1]["spec"])
    for key, value in call.payload["spec"].items():
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = value
    return _deploy(fake, name, spec, env, False, False)


def _predictor_json(fake: Fake, name: str) -> dict[str, Any]:
    p = fake.predictors[name]
    production = p["environments"].get("production") or next(
        iter(p["environments"].values())
    )
    revision = p["_revisions"][production["revision"] - 1]
    envs = {k: _shown(v) for k, v in p["environments"].items()}
    return {"name": name, "spec": revision["spec"], "revision": revision["number"],
            "environments": envs, "origin": p["origin"], "created": NOW, "url": p["url"],
            "resolved": revision["resolved"]}  # fmt: skip


def predictors_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.predictors:
        raise Refusal(404, "NotFound", f"no predictor named `{name}`")
    return ok(_predictor_json(fake, name))


def predictors_list(fake: Fake, call: Call) -> Any:
    return ok({"predictors": [_predictor_json(fake, n) for n in fake.predictors]})


def predictors_revisions(fake: Fake, call: Call, name: str) -> Any:
    return ok({"revisions": list(reversed(fake.predictors[name]["_revisions"]))})


def predictors_environment(fake: Fake, call: Call, name: str, env: str) -> Any:
    environment = fake.predictors[name]["environments"][env]
    if environment.get("_polls"):
        environment["_polls"] -= 1
        if not environment["_polls"]:
            environment["state"] = environment.get("_final", "ready")
            revisions = fake.predictors[name]["_revisions"]
            revision = revisions[environment["revision"] - 1]
            if revision["status"] == "building":
                revision["status"] = (
                    "ready" if environment["state"] == "ready" else "failed"
                )
            if environment["state"] == "failed":
                environment["error"] = {
                    "message": "the replica exited loading the model",
                    "request_id": "r-8f2c41d0",
                    "serving_revision": environment.get("_previous"),
                }
    return ok(_shown(environment))


def predictors_move(fake: Fake, call: Call, name: str, env: str, verb: str) -> Any:
    p = fake.predictors[name]
    target = p["environments"].get(env)
    if call.payload.get("revision"):
        number = call.payload["revision"]
    elif verb == "promote":
        number = p["environments"][call.payload["from_env"]]["revision"]
    else:
        number = target["_previous"]
    p["environments"][env] = {
        "name": env, "revision": number, "state": "starting",
        "generation": (target or {}).get("generation", 0) + 1, "calls_today": 0,
        "p50_seconds": None, "updated": NOW, "url": p["url"], "_polls": 1,
        "_previous": (target or {}).get("revision"),
    }  # fmt: skip
    return ok(_shown(p["environments"][env]))


def predictors_logs(fake: Fake, call: Call, name: str, env: str) -> Any:
    limit = int(call.query.get("limit", 100))
    entries = fake.log_entries[:limit]
    return ok(
        {
            "logs": entries,
            "next": entries[-1]["id"] if entries else None,
            "previous": None,
        }
    )


def predictors_delete(fake: Fake, call: Call, name: str) -> Any:
    del fake.predictors[name]
    return ok({"name": name, "deleted": True})


def _answer(fake: Fake, call: Call, predictor: str | None) -> Any:
    assert call.table is not None, "queries arrive as Arrow"
    assert call.envelope is not None and call.envelope["api_version"] == "gf/v1"
    if fake.accept_predictions > 0:
        fake.accept_predictions -= 1
        return ok({"id": "pred-0123456789abcdef01234567", "state": "starting", "created": 1795252324.0,
                   "retry_after": 1}, 202,
                  Location=f"/v1/projects/{fake.project}/predictions/pred-0123456789abcdef01234567")  # fmt: skip
    rows = call.table.num_rows
    payload = call.envelope["payload"]
    classes = ["<=50K", ">50K"]
    snapshot = fake.snapshots.get((payload.get("context") or {}).get("snapshot"))
    if snapshot is not None and payload.get("target") in snapshot.column_names:
        classes = sorted(
            {str(v) for v in snapshot.column(payload["target"]).to_pylist()}
        )
    share = 1 / len(classes)
    predictions = [{"id": i, "label": classes[0],
                    "probabilities": {c: share for c in classes}} for i in range(rows)]  # fmt: skip
    # The model decides the task, as gf-server's does; a short name is a classifier's here.
    task = next(
        (m["task"] for m in fake.catalog if m["name"] == payload.get("model")),
        "classification",
    )
    if task == "regression":
        predictions = [{"value": 1.5, "quantiles": {str(q): q for q in payload.get("quantiles", [])}}
                       for _ in range(rows)]  # fmt: skip
    if task == "forecast":
        predictions = [{"value": 2.0, "quantiles": {f"{q:g}": q for q in payload.get("quantiles", [])}}
                       for _ in range(rows)]  # fmt: skip
    fake.last_prediction = payload  # type: ignore[attr-defined]
    held = (payload.get("context") or {}).get("snapshot")
    return ok({
        "model": f"{CATALOG[0]['name']}@{CATALOG[0]['artifact']['digest']}", "task": "classification",
        "predictor": predictor, "revision": 1 if predictor else None,
        "environment": payload.get("env") if predictor else None,
        "context": {"source": "snapshot", "name": None, "version": None,
                    "digest": held or digest(str(predictor)), "rows": 26048, "as_of": None},
        "usage": {"calls": -(-rows // 100), "rows": rows, "credits": 0.002 * rows,
                  "usd": 0.000002 * rows, "gpu_seconds": None},
        "timing": {"queued_seconds": 0.1, "cold_start_seconds": 0.0, "compile_seconds": 0.0,
                   "compute_seconds": 0.4},
        "notes": [], "predictions": predictions,
    })  # fmt: skip


def predictors_predict(fake: Fake, call: Call, name: str) -> Any:
    # The environment rides in the query; the payload holds only `as_of`, and
    # gf-server refuses any other field as the engine does.
    assert set(call.query) <= {"env"}, (
        "a predictor call names its environment in the query"
    )
    if call.envelope is not None and not call.envelope["payload"].keys() <= {"as_of"}:
        raise Refusal(422, "ValidationError", "unknown fields are not supported")
    return _answer(fake, call, name)


def predict(fake: Fake, call: Call) -> Any:
    if "task" in call.envelope["payload"]:
        # As gf-server: the model, resolved for the target column, decides the task.
        raise Refusal(
            422, "ValidationError", "task is not a field: the model determines the task"
        )
    digest = call.envelope["payload"]["context"]["snapshot"]
    if digest not in fake.snapshots:
        raise Refusal(404, "NotFound", f"no snapshot {digest}")
    return _answer(fake, call, None)


def predictions_get(fake: Fake, call: Call, prediction: str) -> Any:
    raise AssertionError("handled by the server, which replays the original")


LARGE_CONTEXT_ROWS = 31250  # a million cells at 32 inputs


def estimate(fake: Fake, call: Call) -> Any:
    queries = call.payload["queries"]
    calls = -(-queries // 100)
    out = {
        "calls": calls,
        "rows": queries,
        "credits": 0.002 * queries,
        "usd": 0.000002 * queries,
    }
    if call.payload["context_rows"] > LARGE_CONTEXT_ROWS:
        out["large_context"] = {
            "gpu_seconds": 1.5 * calls,
            "credits": 3.0 * calls,
            "usd": 0.003 * calls,
        }
    return ok(out)


def models_list(fake: Fake, call: Call) -> Any:
    task = call.query.get("task")
    return ok({"models": [m for m in fake.catalog if not task or m["task"] == task]})


def models_get(fake: Fake, call: Call, name: str) -> Any:
    name = unquote(name)
    task = call.query.get("task")
    found = [
        m
        for m in fake.catalog
        if name in (m["name"], m["short"]) and (not task or m["task"] == task)
    ]
    if not found:
        raise Refusal(404, "NotFound", f"no model named `{name}`")
    if len(found) > 1:
        raise Refusal(422, "AmbiguousReference", f"`{name}` names models for several tasks",
                      hint="pass task=")  # fmt: skip
    return ok(found[0])


def sources_list(fake: Fake, call: Call) -> Any:
    return ok({"sources": [_shown(s) for s in fake.sources.values()],
               "available": fake.sources_available})  # fmt: skip


def sources_create(fake: Fake, call: Call) -> Any:
    p = call.payload
    tables = contract.example("sources.tables")["tables"]
    fake.sources[p["name"]] = {"name": p["name"], "kind": p["kind"],
                               "access": "secret" if p.get("secret") else "service_account",
                               "state": "connected", "config": p["config"], "secret_expires": None,
                               "tested_at": NOW, "tables": len(tables), "contexts": 0,
                               "error": None, "origin": origin("cli"),
                               "created": NOW, "url": fake.link(f"data/sources/{p['name']}"),
                               "_secret": p.get("secret")}  # fmt: skip
    return ok(_shown(fake.sources[p["name"]]))


def sources_get(fake: Fake, call: Call, name: str) -> Any:
    return ok(_shown(fake.sources[name]))


def sources_action(fake: Fake, call: Call, name: str, verb: str) -> Any:
    if verb == "preview":
        return ok(contract.example("sources.preview"))
    if verb == "delete":
        del fake.sources[name]
        return ok({"name": name, "deleted": True})
    if name in fake.source_errors:
        refused = dict(fake.source_errors[name])
        raise Refusal(422, refused.pop("code"), refused.pop("message"), **refused)
    return sources_get(fake, call, name)


def sources_tables(fake: Fake, call: Call, name: str) -> Any:
    return ok(contract.example("sources.tables"))


NEW_TOKEN = "gf_live_3kT9qR2mV8xPzL4wN7bY1cH6aaaaaaaa"
# The signed-in account the fake's keys are made by.
OWNER = {"issuer": "https://accounts.google.com", "subject": "108234567890123456789",
         "email": "maya@acme.com"}  # fmt: skip
ROTATED_TOKEN = "gf_live_Q8wE5tY1uI3oP7aS2dF6gH0jbbbbbbbb"


def keys_list(fake: Fake, call: Call) -> Any:
    return ok({"keys": list(fake.keys.values())})


def keys_create(fake: Fake, call: Call) -> Any:
    p = call.payload
    key_id = f"key-{len(fake.keys) + 1:020x}"
    fake.keys[key_id] = {"id": key_id, "project": fake.project, "name": p["name"],
                         "permissions": p["permissions"], "created_at": EPOCH,
                         "expires_at": None if p["expires_in"] is None else EPOCH + p["expires_in"],
                         "revoked_at": None, "state": "active", "prefix": NEW_TOKEN[:7],
                         "last_used_at": None, "owner": OWNER, "service": False, "legacy": False,
                         "machine": None}  # fmt: skip
    return ok({**fake.keys[key_id], "token": NEW_TOKEN})


def keys_action(fake: Fake, call: Call, key: str, verb: str) -> Any:
    if key not in fake.keys:
        raise Refusal(404, "NotFound", f"no key {key}")
    if verb == "revoke":
        fake.keys[key].update(revoked_at=EPOCH - 3600, state="revoked")
        return ok({"id": key, "revoked": True})
    if verb == "rotate":
        fake.keys[key].update(prefix=ROTATED_TOKEN[:7], last_used_at=None)
        return ok({**fake.keys[key], "token": ROTATED_TOKEN})
    if not fake.keys[key]["revoked_at"]:
        raise Refusal(409, "Conflict", f"key {fake.keys[key]['name']} is active", reason="key_active",
                      param="id", hint="revoke it first")  # fmt: skip
    del fake.keys[key]
    return ok({"id": key, "deleted": True})


def _snapshot_json(fake: Fake, name: str) -> dict[str, Any]:
    return {k: v for k, v in fake.training[name].items() if not k.startswith("_")}


def _store_snapshot(fake: Fake, name: str, table: pa.Table, metadata: dict[str, Any],
                    tags: list[str]) -> dict[str, Any]:  # fmt: skip
    digest = _data.table_digest(table)
    existing = fake.training.get(name)
    if existing is not None and existing["digest"] not in (None, digest):
        raise Refusal(409, "Conflict", f"snapshot {name} exists; a snapshot never changes",
                      param="name")  # fmt: skip
    arrow = {"double": "double", "int64": "int64", "bool": "bool", "string": "string"}
    fake.training[name] = {
        "name": name, "state": "ready", "digest": digest, "bytes": table.nbytes,
        "rows": table.num_rows,
        "columns": [{"name": f.name, "type": arrow.get(str(f.type), str(f.type))} for f in table.schema],
        "metadata": {**metadata, "origin": {"kind": "uploaded"}} if metadata.get("target") else None,
        "tags": sorted({*tags, "uploaded"}), "error": None, "expires": None,
        "created": 1761200000.0, "origin": origin(),
    }  # fmt: skip
    return fake.training[name]


def snapshots_post(fake: Fake, call: Call) -> Any:
    if call.table is not None:
        query = dict(call.query)
        name = query.pop("name")
        _store_snapshot(fake, name, call.table, query, [])
        return ok(_snapshot_json(fake, name))
    p = call.payload
    url = (
        f"{fake.url}{_signed(f'/uploads/snapshots/{p["name"]}', p['name'], 1)}"
        if fake.signed_uploads
        else None
    )
    fake.training[p["name"]] = {
        "name": p["name"], "state": "open", "digest": None, "bytes": p["bytes"], "rows": None,
        "columns": None, "metadata": p["metadata"], "tags": sorted({*p.get("tags", []), "uploaded"}),
        "error": None, "expires": 1761286400.0, "created": 1761200000.0, "origin": origin(),
        "url": url, "method": "PUT", "chunk_bytes": fake.chunk_bytes, "_data": b"",
        "_tags": p.get("tags", []),
    }  # fmt: skip
    return ok(_snapshot_json(fake, p["name"]))


def _received_snapshot(fake: Fake, name: str, chunk: bytes) -> int:
    """Adds a chunk to a snapshot's open upload, verifying it on its last byte."""
    snap = fake.training[name]
    snap["_data"] += chunk
    if len(snap["_data"]) == snap["bytes"]:
        table = _data.read_parquet(snap["_data"])
        meta = {k: v for k, v in snap["metadata"].items() if k != "origin"}
        _store_snapshot(fake, name, table, meta, snap["_tags"])
        fake.training[name]["state"], fake.training[name]["_polls"] = "verifying", 1
    return len(snap["_data"])


def snapshots_upload(fake: Fake, call: Call, name: str) -> Any:
    """A chunk sent to a snapshot's upload URL, the resumable object store's and not an API route."""
    snap = fake.training[name]
    kept = _session(fake, call, len(snap["_data"]), _signed_session(name, 1))
    return _resumed(_received_snapshot(fake, name, kept), snap["bytes"])


def snapshots_bytes(fake: Fake, call: Call, name: str) -> Any:
    """A chunk of a snapshot's upload sent to the plane, which gave it no signed URL."""
    snap = fake.training[name]
    assert snap["url"] is None, "a snapshot with a signed URL takes its chunks there"
    start, total = _content_range(call)
    if start != len(snap["_data"]) or total != snap["bytes"]:
        raise Refusal(
            422,
            "ValidationError",
            f"the next chunk starts at byte {len(snap['_data'])}",
        )
    received = _received_snapshot(fake, name, call.body)
    return ok({"name": name, "received": received, "bytes": snap["bytes"]})


def snapshots_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.training:
        raise Refusal(404, "NotFound", f"no snapshot named `{name}`")
    snap = fake.training[name]
    if snap.get("_polls"):
        snap["_polls"] -= 1
        if not snap["_polls"]:
            snap["state"] = "ready"
    return ok(_snapshot_json(fake, name))


def snapshots_list(fake: Fake, call: Call) -> Any:
    tag = call.query.get("tag")
    found = [
        _snapshot_json(fake, n)
        for n, v in fake.training.items()
        if not tag or tag in v["tags"]
    ]
    return ok({"snapshots": found})


def snapshots_update(fake: Fake, call: Call, name: str) -> Any:
    fake.training[name]["tags"] = sorted({*call.payload["tags"]})
    return ok(_snapshot_json(fake, name))


def snapshots_delete(fake: Fake, call: Call, name: str) -> Any:
    fake.training[name]["state"] = "deleted"
    return ok(_snapshot_json(fake, name))


def synthetic_post(fake: Fake, call: Call) -> Any:
    p = dict(call.payload)
    name, count = p["name"], p.get("count", 1)
    fake.batches[name] = {
        "name": name, "tag": f"synthetic-{name}", "count": count, "seed": p.get("seed", 7),
        "snapshots": [f"{name}-{i}" for i in range(count)],
        "options": {k: v for k, v in p.items() if k != "tags"}, "state": "pending",
        "counts": {"pending": count, "ready": 0, "rejected": 0},
        "tables": [{"name": f"{name}-{i}", "state": "pending", "error": None} for i in range(count)],
        "created": 1761200000.0, "origin": origin(), "url": fake.link("fine-tunes"),
        "_polls": 1,
    }  # fmt: skip
    created = ("name", "tag", "count", "snapshots", "state", "created", "origin", "url")
    return ok({k: fake.batches[name][k] for k in created})


def synthetic_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.batches:
        raise Refusal(404, "NotFound", f"no synthetic batch named `{name}`")
    batch = fake.batches[name]
    if batch["_polls"]:
        batch["_polls"] -= 1
    else:
        batch.update(
            state="done", counts={"pending": 0, "ready": batch["count"], "rejected": 0}
        )
        for table in batch["tables"]:
            table["state"] = "ready"
    return ok({k: v for k, v in batch.items() if not k.startswith("_")})


def _run_json(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def runs_post(fake: Fake, call: Call) -> Any:
    p = call.payload
    versions = fake.runs.setdefault(p["name"], [])
    if versions and versions[0]["base"] != entry(p["base"])["name"]:
        raise Refusal(409, "Conflict", f"{p['name']} trains from {versions[0]['base']}")
    version = len(versions) + 1
    data = []
    items = [(None, d) for d in p.get("data") or ()]
    items += [
        (phase["name"], d) for phase in p.get("phases", []) for d in phase["data"]
    ]
    for phase, item in items:
        pinned = {"phase": phase} if phase else {}
        if isinstance(item, dict) and "context" in item:
            ctx = fake.contexts[item["context"]]
            if "target" not in p:
                raise Refusal(422, "ValidationError", "a run on a context names its target",
                              param="target")  # fmt: skip
            if p["target"] not in [c["name"] for c in ctx["columns"]]:
                raise Refusal(422, "ValidationError", f"the context {ctx['name']} has no column "
                              f"`{p['target']}`", param="target")  # fmt: skip
            data.append({"context": item["context"], "version": item.get("version") or ctx["version"],
                         "name": f"context-{item['context']}", "digest": ctx["digest"], **pinned})  # fmt: skip
        elif isinstance(item, dict):
            data += [{"name": n, "digest": v["digest"], **pinned} for n, v in fake.training.items()
                     if item["tag"] in v["tags"]]  # fmt: skip
        else:
            data.append(
                {"name": item, "digest": fake.training[item]["digest"], **pinned}
            )
    run = {
        "model": f"{fake.project}/{p['name']}@ft{version}", "name": p["name"], "version": version,
        "state": "queued", "reason": None, "attempt": 0, "base": entry(p["base"])["name"],
        "data": data, "data_digest": digest(str(data)),
        "phases": [{**phase, "data": [d for d in data if d.get("phase") == phase["name"]]}
                   for phase in p.get("phases", [])],
        "architecture": [{"stages": [], "init": None, **a} for a in p.get("architecture", [])],
        "warnings": [], "progress": None, "method": p["method"],
        "evaluation": {**contract.example("runs.get")["evaluation"], **p.get("evaluation", {})},
        "budget": p.get("budget"), "seed": p.get("seed", 0), "tags": p.get("tags", []),
        "plan": p.get("plan", {"steps": sum(ph["plan"]["steps"] for ph in p.get("phases", []))}),
        "created": 1761200000.0, "updated": 1761200000.0, "usage": None, "published": None,
        "origin": origin(), "url": fake.link(f"fine-tunes/{p['name']}"),
        "_polls": fake.polls_until_done, "_events": [
            {"at": 1761200000.0, "kind": "created", "body": {"name": p["name"], "base": p["base"]}},
            {"at": 1761200000.0, "kind": "state", "body": {"state": "queued", "reason": None}}],
        **({"target": p["target"]} if "target" in p else {}),
        "_payload": p,
    }  # fmt: skip
    versions.append(run)
    return ok(_run_json(run))


def _run_for(fake: Fake, segment: str) -> dict[str, Any]:
    name, _, version = segment.partition("@ft")
    versions = fake.runs.get(name)
    if not versions:
        raise Refusal(404, "NotFound", f"no run named `{segment}`")
    return versions[int(version) - 1] if version else versions[-1]


def _advance(fake: Fake, run: dict[str, Any]) -> None:
    if run["state"] in ("finished", "failed", "cancelled"):
        return
    run["_polls"] -= 1
    steps = run["plan"].get("steps", 100)
    if run["attempt"] == 0:
        run["attempt"] = 1
        run["_events"] += [
            {"at": 1761200002.0, "kind": "attempt", "body": {"attempt": 1}},
            {"at": 1761200010.0, "kind": "state", "body": {"state": "training", "reason": None}},
            {"at": 1761200030.0, "kind": "calibration", "body": {"base_loss": 0.47, "start_loss": 0.47}},
        ]  # fmt: skip
    if run["_polls"] > 0:
        # The attempt is still running, so its seconds are pending, not yet charged.
        run.update(state="training", progress={"step": steps // 2, "loss": 0.45},
                   usage=_usage(60.0, charged=0.0))  # fmt: skip
        run["_events"].append({"at": 1761200060.0, "kind": "metrics",
                               "body": {"steps": steps // 2, "loss": 0.45, "validation_loss": 0.43}})  # fmt: skip
        return
    if fake.run_ends == "failed":
        run.update(state="failed", reason="the trainer ran out of memory")
        run["_events"] += [
            {"at": 1761200090.0, "kind": "failure", "body": {"error": "oom", "message": "the trainer ran out of memory"}},
            {"at": 1761200090.0, "kind": "state", "body": {"state": "failed", "reason": "oom"}},
        ]  # fmt: skip
        return
    run.update(state="finished", progress={"step": steps, "loss": 0.41},
               usage=_usage(130.0),
               published={"model": run["model"], "digest": digest(run["model"]), "kept_step": steps,
                          "validation_loss": 0.398, "base_validation_loss": 0.421,
                          "report": {"data": {"rows": 900, "base": 0.421, "kept": 0.398,
                                              "gain": 0.023, "ci95": [0.01, 0.03],
                                              "metrics": {"base": {"auroc": 0.81}, "kept": {"auroc": 0.84}}}}})  # fmt: skip
    run["_events"] += [
        {"at": 1761200120.0, "kind": "metrics", "body": {"steps": steps, "loss": 0.41, "validation_loss": 0.398}},
        {"at": 1761200130.0, "kind": "published", "body": {"model": run["model"]}},
        {"at": 1761200130.0, "kind": "state", "body": {"state": "finished", "reason": None}},
    ]  # fmt: skip


def _usage(seconds: float, charged: float | None = None) -> dict[str, float]:
    charged = seconds if charged is None else charged
    return {"gpu_seconds": seconds, "charged_gpu_seconds": charged, "voided_gpu_seconds": 0.0,
            "credits": 2.0 * seconds}  # fmt: skip


def runs_get(fake: Fake, call: Call, segment: str) -> Any:
    run = _run_for(fake, segment)
    _advance(fake, run)
    return ok(_run_json(run))


def runs_list(fake: Fake, call: Call) -> Any:
    found = [_run_json(r) for versions in fake.runs.values() for r in versions
             if not call.query.get("tag") or call.query["tag"] in r["tags"]]  # fmt: skip
    return ok({"runs": found})


def runs_lineage(fake: Fake, call: Call, segment: str) -> Any:
    run = _run_for(fake, segment)
    return ok({"model": run["model"], "events": run["_events"]})


def runs_cancel(fake: Fake, call: Call, segment: str) -> Any:
    run = _run_for(fake, segment)
    run["state"] = "cancelled"
    if call.payload.get("save"):
        run["published"] = {"model": run["model"], "digest": digest(run["model"] + "-saved"), "kept_step": 10,
                            "validation_loss": 0.44, "base_validation_loss": 0.421, "report": None}  # fmt: skip
    return ok(_run_json(run))


P = "/v1/projects/acme"
N = r"([^/]+)"
ROUTES: list[tuple[str, str, Callable[..., Any]]] = [
    ("GET", "/v1/auth/status", auth_status),
    ("POST", "/v1/auth/logout", auth_logout),
    ("POST", "/v1/device-logins", device_start),
    ("POST", "/v1/device-logins/token", device_token),
    ("POST", f"{P}/contexts", contexts_create),
    ("GET", f"{P}/contexts", contexts_list),
    ("GET", f"{P}/contexts/{N}", contexts_get),
    ("POST", f"{P}/contexts/{N}/versions", contexts_version),
    ("POST", f"{P}/contexts/{N}/versions/(\\d+)/bytes", contexts_bytes),
    ("POST", f"{P}/contexts/{N}/refresh", contexts_refresh),
    ("GET", f"{P}/contexts/{N}/rows", contexts_rows),
    ("POST", f"{P}/contexts/{N}/(append|label)", contexts_write),
    ("POST", f"{P}/contexts/{N}/delete", contexts_delete),
    ("PUT", f"{P}/snapshots", snapshots_put),
    ("POST", f"{P}/snapshots", snapshots_post),
    ("GET", f"{P}/snapshots", snapshots_list),
    ("POST", f"{P}/snapshots/synthetic", synthetic_post),
    ("GET", f"{P}/snapshots/synthetic/{N}", synthetic_get),
    ("GET", f"{P}/snapshots/{N}", snapshots_get),
    ("PUT", f"/uploads/snapshots/{N}", snapshots_upload),
    ("POST", f"{P}/snapshots/{N}/bytes", snapshots_bytes),
    ("PUT", f"/uploads/contexts/{N}", contexts_signed_upload),
    ("POST", f"{P}/snapshots/{N}/update", snapshots_update),
    ("POST", f"{P}/snapshots/{N}/delete", snapshots_delete),
    ("POST", f"{P}/runs", runs_post),
    ("GET", f"{P}/runs", runs_list),
    ("GET", f"{P}/runs/{N}", runs_get),
    ("GET", f"{P}/runs/{N}/lineage", runs_lineage),
    ("POST", f"{P}/runs/{N}/cancel", runs_cancel),
    ("POST", f"{P}/experiments", experiments_create),
    ("GET", f"{P}/experiments", experiments_list),
    ("GET", f"{P}/experiments/{N}", experiments_get),
    ("POST", f"{P}/experiments/{N}/update", experiments_update),
    ("POST", f"{P}/experiments/{N}/runs", experiments_run),
    ("POST", f"{P}/experiments/{N}/runs/(\\d+)/cancel", experiments_cancel),
    ("GET", f"{P}/experiments/{N}/runs/(\\d+)/rows", experiments_rows),
    ("GET", f"{P}/experiments/{N}/runs/(\\d+)/slices", experiments_slices),
    ("POST", f"{P}/experiments/{N}/delete", experiments_delete),
    ("POST", f"{P}/predictors", predictors_create),
    ("GET", f"{P}/predictors", predictors_list),
    ("GET", f"{P}/predictors/{N}", predictors_get),
    ("POST", f"{P}/predictors/{N}/deploy", predictors_deploy),
    ("POST", f"{P}/predictors/{N}/update", predictors_update),
    ("GET", f"{P}/predictors/{N}/revisions", predictors_revisions),
    ("GET", f"{P}/predictors/{N}/environments/{N}", predictors_environment),
    (
        "POST",
        f"{P}/predictors/{N}/environments/{N}/(promote|rollback)",
        predictors_move,
    ),
    ("GET", f"{P}/predictors/{N}/environments/{N}/logs", predictors_logs),
    ("POST", f"{P}/predictors/{N}/delete", predictors_delete),
    ("POST", f"{P}/predictors/{N}/predict", predictors_predict),
    ("POST", f"{P}/predict", predict),
    ("POST", f"{P}/estimate", estimate),
    ("GET", f"{P}/models", models_list),
    ("GET", f"{P}/models/(.+)", models_get),
    ("GET", f"{P}/sources", sources_list),
    ("POST", f"{P}/sources", sources_create),
    ("GET", f"{P}/sources/{N}", sources_get),
    ("GET", f"{P}/sources/{N}/tables", sources_tables),
    ("POST", f"{P}/sources/{N}/(test|preview|delete)", sources_action),
    ("GET", f"{P}/keys", keys_list),
    ("POST", f"{P}/keys", keys_create),
    ("POST", f"{P}/keys/{N}/(rotate|revoke|delete)", keys_action),
]


# What each route's request envelope names, as the plane binds it
# (`gfi.http.routes.bind`): its resource kind, and its name as a literal, a
# path parameter (`{name}`), or the payload's `name` (`PAYLOAD`), any name
# when the payload has none, as an experiment named by no one.
PAYLOAD = "payload.name"
RESOURCES: dict[str, tuple[str, str]] = {
    "auth.logout": ("workspace", "auth"),
    "contexts.create": ("context", PAYLOAD),
    "contexts.upload": ("context", "{name}"),
    "contexts.refresh": ("context", "{name}"),
    "contexts.append": ("context", "{name}"),
    "contexts.label": ("context", "{name}"),
    "contexts.delete": ("context", "{name}"),
    "experiments.create": ("experiment", PAYLOAD),
    "experiments.update": ("experiment", "{name}"),
    "experiments.run": ("experiment", "{name}"),
    "experiments.cancel": ("experiment", "{name}"),
    "experiments.delete": ("experiment", "{name}"),
    "predictors.create": ("predictor", PAYLOAD),
    "predictors.update": ("predictor", "{name}"),
    "predictors.deploy": ("predictor", "{name}"),
    "predictors.delete": ("predictor", "{name}"),
    "predictors.predict": ("predictor", "{name}"),
    "environments.promote": ("predictor", "{name}"),
    "environments.rollback": ("predictor", "{name}"),
    "predict": ("predictor", "ad-hoc"),
    "estimate": ("estimate", "estimate"),
    "sources.create": ("source", PAYLOAD),
    "sources.test": ("source", "{source}"),
    "sources.preview": ("source", "{source}"),
    "sources.delete": ("source", "{source}"),
    "keys.create": ("api-key", "new"),
    "keys.rotate": ("api-key", "{key}"),
    "keys.revoke": ("api-key", "{key}"),
    "keys.delete": ("api-key", "{key}"),
    "snapshots.create": ("snapshot", PAYLOAD),
    "snapshots.update": ("snapshot", "{snapshot}"),
    "snapshots.delete": ("snapshot", "{snapshot}"),
    "snapshots.create_synthetic": ("snapshot", PAYLOAD),
    "runs.create": ("run", PAYLOAD),
    "runs.cancel": ("run", "{run}"),
}


def resource_problems(
    op: contract.Operation, path: str, envelope: dict[str, Any]
) -> list[str]:
    """How a request envelope's `resource` departs from what the route binds (`RESOURCES`)."""
    if op.id not in RESOURCES:
        return [f"{op.id}: the fake names no resource for the route's envelope"]
    kind, rule = RESOURCES[op.id]
    if rule == PAYLOAD:
        named = envelope["payload"].get("name")
    elif rule.startswith("{"):
        named = op.bound(path)[rule[1:-1]]
    else:
        named = rule
    sent = envelope["resource"]
    if sent["kind"] != kind or named not in (None, sent["name"]):
        return [f"{op.id}: resource route/envelope mismatch, {sent} for {kind}/{named}"]
    return []


def request_problems(
    fake: Fake, op: contract.Operation, call: Call, rid: str
) -> list[str]:
    """How a request departs from what the plane takes, which it refuses with 422.

    Its query and headers keep to the route's parameters, its body to the
    route's media types, and its envelope names the fake's project, the
    request's id and the resource the route binds.
    """
    problems = contract.parameter_problems(op, "query", call.query)
    problems += contract.parameter_problems(op, "header", call.headers)
    problems += contract.media_problems(
        op, call.headers.get("content-type", ""), call.body
    )
    if call.envelope is not None:
        sent = call.envelope
        if sent.get("api_version") != "gf/v1" or sent.get("request_id") != rid:
            problems.append(f"{op.id}: the envelope names another version or request")
        if sent.get("project") != fake.project:
            problems.append(
                f"{op.id}: project path/envelope mismatch, {sent.get('project')!r}"
            )
        problems += resource_problems(op, call.path, sent)
    return [f"{op.id} request: {p}" for p in problems]


def reply_problems(
    op: contract.Operation, call: Call, status: int, data: Any
) -> list[str]:
    """How a request's body, or the fake's reply to it, departs from the route's schemas."""
    sent = call.envelope if call.envelope is not None else call.payload
    if call.table is not None and sent is not None:
        # Queries sent as Arrow: the envelope rides in the stream's metadata without them.
        sent = {
            **sent,
            "payload": {
                **sent["payload"],
                "queries": json.loads(
                    json.dumps(call.table.slice(0, 1).to_pylist(), default=str)
                ),
            },
        }
    problems = []
    if sent is not None:
        problems += [
            f"{op.id} request: {p}" for p in contract.request_problems(op, sent)
        ]
    problems += [
        f"{op.id} reply {status}: {p}"
        for p in contract.reply_problems(op, status, data)
    ]
    return problems


def _accepts_arrow(headers: Mapping[str, str]) -> bool:
    accept = headers.get("accept", "")
    return any(part.split(";")[0].strip() == ARROW for part in accept.split(","))


def _arrow_answer(data: dict[str, Any]) -> bytes:
    """Predictions as the engine answers Arrow: rows `id`, `label`, `p_<class>`,
    `value` and `q_<level>`, and the reply envelope, its result without
    `predictions`, as JSON under `gf.result` in the schema metadata."""
    result = dict(data["result"])
    rows = []
    for prediction in result.pop("predictions"):
        row = {"id": prediction["id"]} if "id" in prediction else {}
        if "label" in prediction:
            row["label"] = prediction["label"]
        row.update(
            {f"p_{c}": p for c, p in (prediction.get("probabilities") or {}).items()}
        )
        if "value" in prediction:
            row["value"] = prediction["value"]
        row.update(
            {f"q_{q}": v for q, v in (prediction.get("quantiles") or {}).items()}
        )
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata(
        {"gf.result": json.dumps({**data, "result": result})}
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def serve(fake: Fake) -> Callable[[], None]:
    """Starts the fake on loopback; returns a function that stops it."""
    pending: dict[str, tuple[int, Any, dict[str, str]]] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.reply()

        def do_POST(self) -> None:
            self.reply()

        def do_PUT(self) -> None:
            self.reply()

        def do_DELETE(self) -> None:
            self.reply()

        def reply(self) -> None:
            parts = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            headers = {k.lower(): v for k, v in self.headers.items()}
            query = dict(parse_qsl(parts.query, keep_blank_values=True))
            if parts.path in PAGES:
                self.send(*_page(self.command, parts.path, query))
                return
            call = Call(self.command, parts.path, query, headers, body=body)
            rid = headers.get("x-gf-request-id", "r-none")
            status, result, extra = 500, None, {}
            upload = parts.path.startswith("/uploads/")
            op = None if upload else contract.operation(self.command, parts.path)
            envelope = not upload and not parts.path.startswith("/v1/device-logins")
            try:
                if (
                    parts.path.startswith(P)
                    and headers.get("x-gf-project") != fake.project
                ):
                    raise Refusal(400, "ValidationError", "X-GF-Project is missing")
                if envelope and headers.get("authorization") != f"Bearer {TOKEN}":
                    raise Refusal(401, "Unauthorized", "not logged in")
                kind = headers.get("content-type", "")
                if kind == ARROW and body and self.command == "POST":
                    call.table = pa.ipc.open_stream(body).read_all()
                    meta = call.table.schema.metadata or {}
                    if b"gf.request" in meta:
                        call.envelope = json.loads(meta[b"gf.request"])
                        call.payload = call.envelope["payload"]
                elif kind == "application/json" and body:
                    value = json.loads(body)
                    if envelope:
                        call.envelope, call.payload = value, value.get("payload")
                    else:
                        call.payload = value
                fake.calls.append(call)
                if op is None and not upload:
                    raise Refusal(
                        404, "NotFound", f"no route {self.command} {parts.path}"
                    )
                refused = [] if op is None else request_problems(fake, op, call, rid)
                if refused:
                    fake.errors.extend(refused)
                    raise Refusal(422, "ValidationError", refused[0])
                prediction = re.fullmatch(f"{P}/predictions/(.+)", parts.path)
                if prediction:
                    status, result, extra = pending.pop(prediction[1])
                else:
                    status, result, extra = fake.handle(call)
                    if status == 202:
                        held = re.sub(r".*/", "", extra["Location"])
                        pending[held] = fake.handle(call)
                data = {"api_version": "gf/v1", "project": fake.project, "request_id": rid,
                        "resource": {"kind": "project", "name": "project"}, "result": result}  # fmt: skip
                if not envelope:
                    data = result
            except Refusal as refusal:
                status, data = refusal.status, {"error": refusal.body}
                if envelope:
                    data = {"api_version": "gf/v1", "project": fake.project, "request_id": rid,
                            "resource": {"kind": "project", "name": fake.project}, **data}  # fmt: skip
            except Exception as error:  # noqa: BLE001
                fake.errors.append(f"{type(error).__name__}: {error}")
                status, data = (
                    500,
                    {"error": {"code": "Internal", "message": str(error)}},
                )
            if op is not None:
                fake.errors += reply_problems(op, call, status, data)
            elif not upload:
                fake.errors.append(
                    f"{self.command} {parts.path} is not a route of api/openapi.yaml"
                )
            if status == 503:  # the contract: every 503 carries Retry-After
                extra = {**extra, "Retry-After": "0"}
            raw, kind = json.dumps(data).encode(), "application/json"
            if status == 200 and call.table is not None and _accepts_arrow(headers):
                raw, kind = _arrow_answer(data), ARROW
            self.send(status, raw, kind, extra)

        def send(
            self, status: int, raw: bytes, kind: str, extra: Mapping[str, str]
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_port}"

    def stop() -> None:
        fake.errors += _unreachable(fake.links)
        server.shutdown()
        server.server_close()
        thread.join()

    return stop


# The console's pages the fake links to, and the query each takes: the
# project, a versioned asset's `v`, and a device login's code.
PAGES = {"/": {"project", "v"}, "/device": {"code"}}


def _page(
    method: str, path: str, query: Mapping[str, str]
) -> tuple[int, bytes, str, dict[str, str]]:
    """A console page as the plane serves it: HTML, or 422 for a query it does not take."""
    if method != "GET":
        return 405, b"", "text/plain", {}
    if not set(query) <= PAGES[path]:
        return (
            422,
            b'{"code":"ValidationError","message":"unexpected query"}',
            "application/json",
            {},
        )
    return (
        200,
        b"<!doctype html><title>General Forecast</title>",
        "text/html; charset=utf-8",
        {},
    )


def _unreachable(links: set[str]) -> list[str]:
    """The console links that do not open a page."""
    import httpx

    problems = []
    with httpx.Client(trust_env=False) as client:
        for link in sorted(links):
            answer = client.get(link)
            if answer.status_code != 200 or not answer.headers[
                "content-type"
            ].startswith("text/html"):
                problems.append(f"the link {link} answers {answer.status_code}")
    return problems
