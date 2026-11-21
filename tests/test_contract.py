# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The SDK and the fake keep to `api/openapi.yaml`, the engine's contract."""

import re

import pytest

from generalforecast import _ops
from generalforecast.types import (
    Context,
    Deploy,
    Environment,
    Experiment,
    Key,
    LogEntry,
    PredictorInfo,
    Revision,
    Slice,
    Source,
    SourceTable,
    load,
    loads,
)
from tests import contract, fake


def test_every_route_the_fake_serves_is_a_route_of_the_contract():
    for method, pattern, handler in fake.ROUTES:
        if pattern.startswith("/uploads/"):
            continue  # an upload URL is the object store's, outside the API
        path = re.sub(
            r"\(\[\^/\]\+\)|\(\\d\+\)|\(\.\+\)|\((\w+)\|[\w|]+\)",
            lambda m: m[1] or "x1",
            pattern,
        )
        assert contract.operation(method, path), f"{handler.__name__}: {method} {path}"


@pytest.mark.parametrize(
    ("operation", "read"),
    [
        ("contexts.get", lambda r: load(Context, r)),
        ("contexts.list", lambda r: loads(Context, r["contexts"])),
        ("experiments.get", lambda r: load(Experiment, r)),
        ("experiments.list", lambda r: loads(Experiment, r["experiments"])),
        ("experiments.slices", lambda r: loads(Slice, r["slices"])),
        ("predictors.get", lambda r: load(PredictorInfo, r)),
        ("predictors.create", lambda r: load(Deploy, r)),
        ("revisions.list", lambda r: loads(Revision, r["revisions"])),
        ("environments.get", lambda r: load(Environment, r)),
        ("environments.logs", lambda r: loads(LogEntry, r["logs"])),
        ("sources.get", lambda r: load(Source, r)),
        ("sources.tables", lambda r: loads(SourceTable, r["tables"])),
        ("keys.list", lambda r: loads(Key, r["keys"])),
        ("models.list", lambda r: [_ops.model_of(m) for m in r["models"]]),
        ("estimate", _ops.estimate_of),
    ],
)
def test_the_sdk_reads_the_contracts_examples(operation, read):
    found = read(contract.example(operation))
    assert found


def test_the_sdk_reads_a_model_and_a_prediction_as_the_contract_writes_them():
    model = _ops.model_of(contract.example("models.list")["models"][0])
    assert (
        model.id == "general-forecast/tabicl-v2-classifier"
        and model.short == "tabicl-v2"
    )
    assert model.digest.startswith("sha256:") and model.limits.queries == 100
    result = contract.example("predictions.get")
    reply = _ops._http.Reply(200, result, {}, "r-1")
    predictions = _ops.predictions_of(reply)
    assert predictions.rows == 2 and predictions.context.rows == 1840
    assert predictions.context.used == "exact"


# Request examples the contract gives as what a route refuses.
REFUSED = {("predictors.predict", "declared"), ("predictors.predict", "context"),
           ("snapshots.create_synthetic", "unknown_field"), ("snapshots.update", "digest")}  # fmt: skip


def test_the_contracts_examples_keep_to_its_schemas_but_for_its_known_defects():
    broken = []
    for op in contract.operations():
        for status, reply in op.body.get("responses", {}).items():
            content = (
                contract._resolved(reply).get("content", {}).get("application/json", {})
            )
            for name, example in content.get("examples", {}).items():
                if status.isdigit() and contract.reply_problems(
                    op, int(status), example["value"]
                ):
                    broken.append((op.id, name))
        request = op.body.get("requestBody")
        content = (
            contract._resolved(request or {})
            .get("content", {})
            .get("application/json", {})
        )
        for name, example in content.get("examples", {}).items():
            if (op.id, name) not in REFUSED and contract.request_problems(
                op, example["value"]
            ):
                broken.append((op.id, name))
    assert broken == []
