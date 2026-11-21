# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""A fake General Forecast service, built from the surface spec's examples.

It serves the routes of section 3 over real HTTP on loopback, keeps its
resources in memory, and records every request, so tests drive the SDK
end to end. Shapes follow the spec's JSON examples; when engine PR 1's
`api/openapi.yaml` lands, its examples replace these.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

TOKEN = "gf_test_0123456789abcdefghijklmnopqrstuv"
CONSOLE = "https://app.generalforecast.com"
NOW = "2026-11-21T09:12:04Z"
ARROW = "application/vnd.apache.arrow.stream"

ORIGIN = {
    "surface": "notebook",
    "notebook": "jupyter",
    "client": "generalforecast/0.2.3",
    "by": "maya@acme.com",
}

CATALOG = [
    {"id": "general-forecast/tabicl-v2-classifier", "short": "tabicl-v2", "task": "classification",
     "family": "tabicl-v2", "kind": "catalog", "hostable": True, "digest": "sha256:b14107",
     "limits": {"context_rows": 100000, "queries": 100, "inputs": 500},
     "trained_on": {"rows": 100000, "inputs": 500, "classes": 10}, "architecture": {}},
    {"id": "general-forecast/tabicl-v2-regressor", "short": "tabicl-v2", "task": "regression",
     "family": "tabicl-v2", "kind": "catalog", "hostable": True, "digest": "sha256:b14108",
     "limits": {"context_rows": 100000, "queries": 100, "inputs": 500}},
    {"id": "general-forecast/kumo-tabular-medium-classifier", "short": "kumo-medium",
     "task": "classification", "family": "kumo", "kind": "catalog", "hostable": True,
     "digest": "sha256:d0f1d0", "limits": {"context_rows": 60000, "queries": 100, "inputs": 32}},
]  # fmt: skip

LEADERBOARD = [
    {"model": "general-forecast/tabicl-v2-classifier", "short": "tabicl-v2", "state": "finished",
     "score": 0.931, "interval": [0.925, 0.937],
     "metrics": {"auroc": 0.931, "log loss": 0.287, "accuracy": 0.862},
     "seconds": 38.2, "rows": 6513, "credits": 13.0, "usd": 0.013, "error": None},
    {"model": "general-forecast/kumo-tabular-medium-classifier", "short": "kumo-medium",
     "state": "finished", "score": 0.929, "interval": [0.923, 0.935], "metrics": {"auroc": 0.929},
     "seconds": 51.0, "rows": 6513, "credits": 13.0, "usd": 0.013, "error": None},
    {"model": "general-forecast/mitra-2-classifier", "short": "mitra-2", "state": "finished",
     "score": 0.918, "interval": [0.911, 0.925], "metrics": {"auroc": 0.918},
     "seconds": 44.0, "rows": 6513, "credits": 13.0, "usd": 0.013, "error": None},
]  # fmt: skip


class Refusal(Exception):
    """An error reply."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.body = status, {"code": code, "message": message, **extra}


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
    polls_until_done: int = 2
    accept_predictions: int = 0
    fail: dict[str, int] = field(default_factory=dict)
    pending_logins: int = 1
    deploy_ends: str = "ready"
    errors: list[str] = field(default_factory=list)

    def calls_to(self, method: str, pattern: str) -> list[Call]:
        """The calls whose path matches `pattern`."""
        return [
            c
            for c in self.calls
            if c.method == method and re.fullmatch(pattern, c.path)
        ]

    def link(self, part: str) -> str:
        return f"{CONSOLE}/?project={self.project}#/{part}"

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
                return handler(self, call, *found.groups())
        raise Refusal(404, "NotFound", f"no route {call.method} {call.path}")


def ok(
    result: Any, status: int = 200, **headers: str
) -> tuple[int, Any, dict[str, str]]:
    return status, result, headers


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
    return ok({
        "authenticated": True,
        "principal": {"kind": "api-key", "id": "key-0f1e2d3c4b5a69788796", "project": fake.project},
        "key": {"id": "key-0f1e2d3c4b5a69788796", "name": "notebook-maya-mbp",
                "project": fake.project, "expires_at": 1795000000.0},
        "email": "maya@acme.com",
        "workspace": {"id": "acme", "display_name": "Acme Analytics"},
    })  # fmt: skip


def auth_logout(fake: Fake, call: Call) -> Any:
    return ok({"revoked": "key-0f1e2d3c4b5a69788796"})


def device_start(fake: Fake, call: Call) -> Any:
    assert call.payload["client"] == "notebook"
    return ok({
        "device_code": "gf_dc_abc", "user_code": "WXHG-PTRB",
        "verification_uri": f"{CONSOLE}/device",
        "verification_uri_complete": f"{CONSOLE}/device?code=WXHG-PTRB",
        "expires_in": 600, "interval": 1,
    })  # fmt: skip


def device_token(fake: Fake, call: Call) -> Any:
    if fake.pending_logins > 0:
        fake.pending_logins -= 1
        raise Refusal(400, "Unauthorized", "waiting for browser approval",
                      reason="authorization_pending")  # fmt: skip
    return ok({
        "access_token": TOKEN, "token_type": "bearer", "expires_at": "2026-11-21T21:14:00Z",
        "key": {"id": "key-0f1e2d3c4b5a69788796", "name": "notebook-maya-mbp",
                "project": fake.project, "expires_at": "2026-11-21T21:14:00Z", "ephemeral": True},
        "email": "maya@acme.com", "workspace": {"id": "acme", "display_name": "Acme Analytics"},
        "project": fake.project, "url": f"{CONSOLE}/?project={fake.project}",
    })  # fmt: skip


def _new_context(fake: Fake, payload: dict[str, Any], state: str) -> dict[str, Any]:
    name = payload["name"]
    source = payload.get("source")
    ctx = {
        "name": name, "kind": payload["kind"], "state": state, "version": None, "versions": 0,
        "digest": None, "rows": None, "columns": [], "id": payload.get("id"),
        "time": payload.get("time"),
        "source": {"kind": "bigquery", "table": None, "sql": None, **source} if source else None,
        "freshness_seconds": payload.get("freshness_seconds"), "read_at": None, "error": None,
        "predictors": [], "experiments": [], "origin": ORIGIN, "created": NOW, "updated": NOW,
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
    out = fake.context_json(name)
    out["upload"] = {"version": version, "url": None, "chunk_bytes": fake.chunk_bytes}
    return ok(out)


def contexts_create(fake: Fake, call: Call) -> Any:
    payload = call.payload
    if payload["name"] in fake.contexts:
        raise Refusal(409, "Conflict", f"a context named {payload['name']} exists",
                      reason="already_exists")  # fmt: skip
    if payload.get("data"):
        _new_context(fake, payload, "uploading")
        return _open_upload(fake, payload["name"], payload["data"])
    if payload.get("source") and payload["kind"] == "snapshot":
        ctx = _new_context(fake, payload, "importing")
        ctx["_polls"] = fake.polls_until_done
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
    upload["data"] += call.body
    if len(upload["data"]) == upload["bytes"]:
        table = pq.read_table(io.BytesIO(upload["data"]))
        ctx = fake.contexts[name]
        ctx.update(state="verifying", version=upload["version"], versions=upload["version"],
                   digest=upload["digest"], rows=table.num_rows, columns=_columns(table),
                   _polls=1)  # fmt: skip
        ctx["_versions"].append({"version": upload["version"], "digest": upload["digest"],
                                 "rows": table.num_rows})  # fmt: skip
        ctx["_rows"] = table.to_pylist()
    return ok({"received": len(upload["data"])})


def contexts_get(fake: Fake, call: Call, name: str) -> Any:
    if name not in fake.contexts:
        raise Refusal(404, "NotFound", f"no context named `{name}`")
    ctx = fake.contexts[name]
    if ctx.get("_polls"):
        ctx["_polls"] -= 1
        if not ctx["_polls"]:
            if ctx["state"] == "importing":
                ctx.update(version=1, versions=1, digest="sha256:91ac", rows=48210)
            ctx["state"] = "ready"
    version = call.query.get("version")
    return ok(fake.context_json(name, int(version) if version else None))


def contexts_list(fake: Fake, call: Call) -> Any:
    return ok({"contexts": [fake.context_json(n) for n in fake.contexts]})


def contexts_refresh(fake: Fake, call: Call, name: str) -> Any:
    ctx = fake.contexts[name]
    ctx["state"], ctx["_polls"] = "importing", fake.polls_until_done
    return ok(fake.context_json(name))


def contexts_rows(fake: Fake, call: Call, name: str) -> Any:
    ctx = fake.contexts[name]
    limit = int(call.query.get("limit", 20))
    start = int(call.query.get("after", 0))
    names = [c["name"] for c in ctx["columns"]]
    rows = [
        [r.get(n) for n in names] for r in ctx["_rows"][start : start + min(limit, 3)]
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
    import hashlib

    assert "sha256:" + hashlib.sha256(call.body).hexdigest() == digest
    fake.snapshots[digest] = pa.ipc.open_stream(call.body).read_all()
    return ok({"digest": digest, "rows": fake.snapshots[digest].num_rows}, 201)


def _run(
    fake: Fake, experiment: str, number: int, models: list[str], state: str
) -> dict[str, Any]:
    return {
        "experiment": experiment, "number": number, "state": state, "stale": False,
        "models": models, "metric": "auroc", "lower_is_better": False,
        "held_rows": 6513, "context_rows": 26048, "repeats": 1,
        "leaderboard": [dict(e, state="queued", score=None, interval=None) for e in LEADERBOARD[: len(models)]],
        "rows": 0, "credits": 0.0, "usd": 0.0, "created": NOW, "started": None, "finished": None,
        "origin": {"surface": "notebook"}, "error": None, "_polls": fake.polls_until_done,
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
            run.update(state="finished", leaderboard=LEADERBOARD[:count], rows=6513 * count,
                       credits=13.0 * count, usd=0.013 * count, finished=NOW)  # fmt: skip
        else:
            run["state"] = "running"
            run["leaderboard"][0] = LEADERBOARD[0]
    out["run"] = (
        {k: v for k, v in run.items() if not k.startswith("_")} if run else None
    )
    out["runs"] = [{"number": r["number"], "state": r["state"], "models": r["models"],
                    "best": "tabicl-v2", "score": 0.931, "origin": {"surface": "notebook"},
                    "created": NOW} for r in runs]  # fmt: skip
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
    models = payload.pop("models", None) or ["tabicl-v2", "kumo-medium", "mitra-2"]
    if payload.pop("dry_run", False):
        return ok({"setup": payload, "estimate": {"calls": 198, "rows": 19539, "credits": 39.1, "usd": 0.0391},
                   "warnings": []})  # fmt: skip
    name = payload.get("name") or f"{context}-{payload['target']}"
    fake.experiments[name] = {
        "name": name, "title": payload.get("title"), "context": context,
        "context_version": payload.get("version") or fake.contexts[context]["version"],
        "target": payload["target"], "task": payload.get("task") or "classification",
        "id": payload.get("id"), "time": payload.get("time"), "inputs": payload.get("inputs") or [],
        "types": payload.get("types") or {}, "sql": payload.get("sql"),
        "holdout": payload.get("holdout") or {"kind": "random", "fraction": 0.2},
        "metric": "auroc", "origin": ORIGIN, "created": NOW, "updated": NOW,
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
            item["run"].pop("leaderboard", None)
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
    run.update(state="cancelled", _polls=0)
    return ok({k: v for k, v in run.items() if not k.startswith("_")})


def experiments_rows(fake: Fake, call: Call, name: str, number: str) -> Any:
    after = int(call.query.get("after", 0))
    rows = [{"row": after + i, "model": call.query.get("model") or "tabicl-v2", "label": "<=50K",
             "p_>50K": 0.1} for i in range(2)]  # fmt: skip
    return ok({"rows": rows, "next": "2" if after == 0 else None})


def experiments_slices(fake: Fake, call: Call, name: str, number: str) -> Any:
    return ok({"slices": [
        {"column": "marital_status", "value": "Never-married", "rows": 2140, "score": 0.884, "delta": -0.047},
        {"column": "education", "value": "HS-grad", "rows": 2096, "score": 0.902, "delta": -0.029},
    ]})  # fmt: skip


def experiments_delete(fake: Fake, call: Call, name: str) -> Any:
    del fake.experiments[name]
    return ok({"name": name, "deleted": True})


def _resolved(fake: Fake, spec: dict[str, Any]) -> dict[str, Any]:
    entry = next((m for m in CATALOG if spec["model"] in (m["short"], m["id"])), None)
    if entry is None:
        raise Refusal(404, "NotFound", f"no model named `{spec['model']}`",
                      hint="a similar model exists: `kumo-medium`")  # fmt: skip
    name, _, pinned = str(spec["context"]).partition("@")
    ctx = fake.contexts.get(name)
    if ctx is None:
        raise Refusal(404, "NotFound", f"no context named `{name}`")
    return {"model": entry["id"], "short": entry["short"], "model_digest": entry["digest"],
            "task": spec.get("task") or entry["task"], "context": name,
            "context_version": int(pinned) if pinned else ctx["version"],
            "context_digest": ctx["digest"],
            "inputs": [c["name"] for c in ctx["columns"] if c["name"] != spec["target"]],
            "types": {}}  # fmt: skip


def _deploy(
    fake: Fake, name: str, spec: dict[str, Any], env: str, dry_run: bool, create: bool
) -> Any:
    resolved = _resolved(fake, spec)
    existing = fake.predictors.get(name)
    if existing is None:
        action, changes, number = "created", [], 1
    else:
        serving = existing["environments"].get(env, {}).get("revision")
        current = existing["_revisions"][serving - 1] if serving else None
        if current is not None and current["resolved"] == resolved:
            return ok({"predictor": name, "env": env, "action": "unchanged", "changes": [],
                       "revision": current, "environment": existing["environments"][env]})  # fmt: skip
        action, number = "updated", len(existing["_revisions"]) + 1
        before = current["spec"] if current else {}
        changes = [{"field": k, "before": str(before.get(k)) if before.get(k) is not None else None,
                    "after": str(v) if v is not None else None}
                   for k, v in spec.items() if before.get(k) != v]  # fmt: skip
    revision = {"number": number, "spec": spec, "resolved": resolved,
                "origin": {"surface": "notebook"}, "created": NOW}  # fmt: skip
    environment = {"name": env, "revision": number, "state": "starting", "generation": number,
                   "calls_today": 0, "p50_seconds": None, "updated": NOW,
                   "url": fake.link(f"predictors/{name}"), "_polls": fake.polls_until_done,
                   "_final": fake.deploy_ends, "error": None}  # fmt: skip
    if not dry_run:
        if existing is None:
            existing = fake.predictors[name] = {
                "name": name, "_revisions": [], "environments": {}, "created": NOW,
                "url": fake.link(f"predictors/{name}"),
            }  # fmt: skip
            ctx = fake.contexts[resolved["context"]]
            ctx["predictors"] = [*ctx["predictors"], name]
        existing["_revisions"].append(revision)
        previous = existing["environments"].get(env)
        environment["_previous"] = previous["revision"] if previous else None
        existing["environments"][env] = environment
    shown = {k: v for k, v in environment.items() if not k.startswith("_")}
    return ok({"predictor": name, "env": env, "action": action, "changes": changes,
               "revision": revision, "environment": shown})  # fmt: skip


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
    envs = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in p["environments"].items()}  # fmt: skip
    return {"name": name, "spec": revision["spec"], "revision": revision["number"],
            "environments": envs, "created": NOW, "url": p["url"],
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
            if environment["state"] == "failed":
                environment["error"] = "the replica exited loading the model"
    return ok({k: v for k, v in environment.items() if not k.startswith("_")})


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
    return ok(
        {k: v for k, v in p["environments"][env].items() if not k.startswith("_")}
    )


def predictors_logs(fake: Fake, call: Call, name: str, env: str) -> Any:
    return ok({"entries": [
        {"at": "2026-11-21T10:04:12Z", "level": "info", "source": "predict", "code": None,
         "message": "100 rows  0.92s", "request_id": "r-3f1a92c0", "fields": {"rows": 100}},
    ], "next": "c2", "previous": None})  # fmt: skip


def predictors_delete(fake: Fake, call: Call, name: str) -> Any:
    del fake.predictors[name]
    return ok({"name": name, "deleted": True})


def _answer(fake: Fake, call: Call, predictor: str | None) -> Any:
    assert call.table is not None, "queries arrive as Arrow"
    assert call.envelope is not None and call.envelope["api_version"] == "gf/v1"
    if fake.accept_predictions > 0:
        fake.accept_predictions -= 1
        return ok({"id": "pred-0123456789abcdef01234567", "state": "starting", "created": 1761400100.0,
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
    if payload.get("task") == "regression":
        predictions = [{"value": 1.5, "quantiles": {str(q): q for q in payload.get("quantiles", [])}}
                       for _ in range(rows)]  # fmt: skip
    if payload.get("task") == "forecast":
        predictions = [{"value": 2.0, "quantiles": {f"{q:g}": q for q in payload.get("quantiles", [])}}
                       for _ in range(rows)]  # fmt: skip
    fake.last_prediction = payload  # type: ignore[attr-defined]
    return ok({
        "model": "general-forecast/tabicl-v2-classifier@sha256:7a01", "task": "classification",
        "predictor": predictor, "revision": 1 if predictor else None,
        "environment": payload.get("env") if predictor else None,
        "context": {"rows": 26048, "used": "exact", "core": None},
        "usage": {"rows": rows, "credits": 0.002 * rows, "usd": 0.000002 * rows, "gpu_seconds": None},
        "timing": {"queued_seconds": 0.1, "cold_start_seconds": 0.0, "compute_seconds": 0.4},
        "notes": [], "predictions": predictions,
    })  # fmt: skip


def predictors_predict(fake: Fake, call: Call, name: str) -> Any:
    return _answer(fake, call, name)


def predict(fake: Fake, call: Call) -> Any:
    digest = call.envelope["payload"]["context"]["snapshot"]
    if digest not in fake.snapshots:
        raise Refusal(404, "NotFound", f"no snapshot {digest}")
    return _answer(fake, call, None)


def predictions_get(fake: Fake, call: Call, prediction: str) -> Any:
    raise AssertionError("handled by the server, which replays the original")


def estimate(fake: Fake, call: Call) -> Any:
    queries = call.payload["queries"]
    calls = -(-queries // 100)
    return ok(
        {
            "calls": calls,
            "rows": queries,
            "credits": 0.002 * queries,
            "usd": 0.000002 * queries,
        }
    )


def models_list(fake: Fake, call: Call) -> Any:
    task = call.query.get("task")
    return ok({"models": [m for m in CATALOG if not task or m["task"] == task]})


def models_get(fake: Fake, call: Call, name: str) -> Any:
    name = unquote(name)
    task = call.query.get("task")
    found = [
        m
        for m in CATALOG
        if name in (m["id"], m["short"]) and (not task or m["task"] == task)
    ]
    if not found:
        raise Refusal(404, "NotFound", f"no model named `{name}`")
    if len(found) > 1:
        raise Refusal(422, "AmbiguousReference", f"`{name}` names models for several tasks",
                      hint="pass task=")  # fmt: skip
    return ok(found[0])


def sources_list(fake: Fake, call: Call) -> Any:
    return ok({"sources": list(fake.sources.values())})


def sources_create(fake: Fake, call: Call) -> Any:
    p = call.payload
    fake.sources[p["name"]] = {"name": p["name"], "kind": p["kind"], "state": "ready",
                               "config": p["config"], "secret_expires": None,
                               "tested_at": NOW, "error": None, "created": NOW,
                               "url": fake.link(f"data/sources/{p['name']}"),
                               "_secret": p.get("secret")}  # fmt: skip
    return ok(
        {k: v for k, v in fake.sources[p["name"]].items() if not k.startswith("_")}
    )


def sources_get(fake: Fake, call: Call, name: str) -> Any:
    return ok({k: v for k, v in fake.sources[name].items() if not k.startswith("_")})


def sources_action(fake: Fake, call: Call, name: str, verb: str) -> Any:
    if verb == "preview":
        return ok({"columns": ["customer_id", "plan"], "rows": [["C-100481", "team"]]})
    if verb == "delete":
        del fake.sources[name]
        return ok({"name": name, "deleted": True})
    return sources_get(fake, call, name)


def sources_tables(fake: Fake, call: Call, name: str) -> Any:
    return ok({"tables": [{"name": "acme.analytics.customers", "rows": 48210, "bytes": 1200000,
                           "updated": NOW}]})  # fmt: skip


def keys_list(fake: Fake, call: Call) -> Any:
    return ok({"keys": list(fake.keys.values())})


def keys_create(fake: Fake, call: Call) -> Any:
    p = call.payload
    key_id = f"key-{len(fake.keys) + 1:020x}"
    now = time.time()
    fake.keys[key_id] = {"id": key_id, "project": fake.project, "name": p["name"],
                         "permissions": p["permissions"], "created_at": now,
                         "expires_at": None if p["expires_in"] is None else now + p["expires_in"],
                         "revoked_at": None}  # fmt: skip
    return ok(
        {**fake.keys[key_id], "token": "gf_live_3kT9qR2mV8xPzL4wN7bY1cH6aaaaaaaa"}
    )


def keys_action(fake: Fake, call: Call, key: str, verb: str) -> Any:
    if key not in fake.keys:
        raise Refusal(404, "NotFound", f"no key {key}")
    if verb == "revoke":
        fake.keys[key]["revoked_at"] = 1763700100.0
        return ok(fake.keys[key])
    if verb == "rotate":
        return ok(
            {**fake.keys[key], "token": "gf_live_Q8wE5tY1uI3oP7aS2dF6gH0jbbbbbbbb"}
        )
    if not fake.keys[key]["revoked_at"]:
        raise Refusal(409, "Conflict", "only an expired or revoked key is deleted")
    del fake.keys[key]
    return ok({"id": key, "deleted": True})


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
            call = Call(
                self.command,
                parts.path,
                dict(parse_qsl(parts.query)),
                headers,
                body=body,
            )
            rid = headers.get("x-gf-request-id", "r-none")
            status, result, extra = 500, None, {}
            envelope = not parts.path.startswith("/v1/device-logins")
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
                    call.envelope = json.loads(meta[b"gf.request"])
                    call.payload = call.envelope["payload"]
                elif kind == "application/json" and body:
                    value = json.loads(body)
                    if envelope:
                        assert (
                            value["api_version"] == "gf/v1"
                            and value["request_id"] == rid
                        )
                        assert value["project"] in (fake.project, None)
                        call.envelope, call.payload = value, value["payload"]
                    else:
                        call.payload = value
                fake.calls.append(call)
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
                status, data = (
                    refusal.status,
                    {"error": {**refusal.body, "request_id": rid}},
                )
            except Exception as error:  # noqa: BLE001
                fake.errors.append(f"{type(error).__name__}: {error}")
                status, data = (
                    500,
                    {"error": {"code": "Internal", "message": str(error)}},
                )
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
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
        server.shutdown()
        server.server_close()
        thread.join()

    return stop
