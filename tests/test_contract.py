# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The SDK and the fake keep to `api/openapi.yaml`, the engine's contract."""

import re

import httpx
import pandas as pd
import pytest

import generalforecast as gf
from generalforecast import _http, _ops
from generalforecast.types import (
    Context,
    Deploy,
    Environment,
    Experiment,
    Key,
    Limits,
    LogEntry,
    PredictorInfo,
    Revision,
    Slice,
    Source,
    SourceTable,
    UploadLimits,
    load,
    loads,
)
from tests import contract, fake
from tests.fake import TOKEN
from tests.fake import _unreachable as fake_links_unreachable


def served():
    """The contract's route of each of the fake's API routes, with its handler."""
    for method, pattern, handler in fake.ROUTES:
        if pattern.startswith("/uploads/"):
            continue  # an upload URL is the object store's, outside the API
        path = re.sub(
            r"\(\[\^/\]\+\)|\(\\d\+\)|\(\.\+\)|\((\w+)\|[\w|]+\)",
            lambda m: m[1] or "x1",
            pattern,
        )
        yield contract.operation(method, path), handler


def test_every_route_the_fake_serves_is_a_route_of_the_contract():
    for op, handler in served():
        assert op is not None, handler.__name__


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
    assert model.digest.startswith("sha256:") and model.limits == Limits(100000, 10000)
    result = contract.example("predictions.get")
    reply = _ops._http.Reply(200, result, {}, "r-1")
    predictions = _ops.predictions_of(reply)
    assert predictions.rows == 2 and predictions.context.rows == 1840
    assert predictions.context.used == "exact"
    assert predictions.calls == 1


def test_the_sdk_reads_the_services_limits_and_what_each_environment_serves():
    me = _ops.me_of(contract.example("auth.status"), fake.CONSOLE)
    assert me.trains is True
    assert me.limits == UploadLimits(
        snapshot_bytes=33554432, snapshot_rows=100000, upload_bytes=8589934592,
        upload_rows=10000000, csv_bytes=268435456, chunk_bytes=8388608,
        request_bytes=1048576,
    )  # fmt: skip
    environment = load(Environment, contract.example("environments.get"))
    assert environment.resolved.short == "kumo-medium"
    assert environment.resolved.context_version == 8


# Request examples the contract gives as what a route refuses.
def test_the_contracts_examples_keep_to_its_schemas():
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
            # An example the contract marks `x-gf-invalid` shows a refusal.
            if not example.get("x-gf-invalid") and contract.request_problems(
                op, example["value"]
            ):
                broken.append((op.id, name))
    assert broken == []


def test_the_fakes_envelope_resources_are_the_contracts():
    """Each route's resource in `fake.RESOURCES` is what the contract's request examples send."""
    for op in contract.operations():
        if op.id not in fake.RESOURCES:
            continue
        kind, rule = fake.RESOURCES[op.id]
        examples = op.media().get("application/json", {}).get("examples", {})
        for name, example in examples.items():
            sent = example["value"]
            assert sent["resource"]["kind"] == kind, (op.id, name)
            if rule == fake.PAYLOAD and "name" in sent["payload"]:
                assert sent["resource"]["name"] == sent["payload"]["name"], (
                    op.id,
                    name,
                )
            elif rule != fake.PAYLOAD and not rule.startswith("{"):
                assert sent["resource"]["name"] == rule, (op.id, name)


def test_every_enveloped_route_the_fake_serves_names_its_resource():
    for op, handler in served():
        enveloped = op.body.get("x-gf-envelope", True) and (
            "application/json" in op.media()
            or op.id in ("predict", "predictors.predict")
        )
        assert not enveloped or op.id in fake.RESOURCES, handler.__name__


def _refused(fake_service, request):
    """Sends a request as the SDK would; the fake's refusal, which it also records."""
    connection = _http.Connection(fake_service.url, TOKEN, "acme", "test", retries=0)
    transport = _http.Transport(connection)
    try:
        with pytest.raises(gf.InvalidInput) as error:
            transport.send(request)
    finally:
        transport.close()
    assert fake_service.errors, "the fake records what it refused"
    fake_service.errors.clear()
    return error.value.message


def test_the_fake_refuses_what_the_plane_refuses(fake):
    # A query parameter the route does not declare, or a value its schema refuses.
    assert "`name` is not declared" in _refused(
        fake, _http.Request("GET", "/runs", "run", "runs", params={"name": "credit"})
    )
    assert "`state`" in _refused(
        fake, _http.Request("GET", "/runs", "run", "runs", params={"state": "done"})
    )
    # An envelope that names another resource than the route binds.
    assert "resource route/envelope mismatch" in _refused(
        fake, _http.Request("POST", "/keys", "api-key", "ci", payload={"name": "ci"})
    )
    assert "resource route/envelope mismatch" in _refused(
        fake, _http.Request("POST", "/estimate", "model", "estimate", payload={})
    )
    # A body of a media type the route does not take.
    assert "not `text/plain`" in _refused(
        fake,
        _http.Request(
            "POST", "/contexts/adult/delete", "context", "adult", body=b"{}",
            content_type="text/plain",
        ),
    )  # fmt: skip
    # An envelope without the project, which no client sends.
    with pytest.raises(gf.InvalidInput, match="no project chosen"):
        _http.Connection(fake.url, TOKEN, None, "test").envelope(
            _http.auth_logout(), "r-1"
        )
    answer = httpx.post(
        f"{fake.url}/v1/auth/logout",
        headers={"Authorization": f"Bearer {TOKEN}", "X-GF-Request-ID": "r-1",
                 "Content-Type": "application/json"},
        json={"api_version": "gf/v1", "project": None, "request_id": "r-1",
              "resource": {"kind": "workspace", "name": "auth"}, "payload": {}},
        trust_env=False,
    )  # fmt: skip
    assert answer.status_code == 422 and "project path/envelope mismatch" in answer.text
    fake.errors.clear()


def test_every_console_link_opens_a_page(fake):
    gf.contexts.create("adult", data=pd.DataFrame({"a": [1, 2]}))
    assert gf.contexts.get("adult").url in fake.links
    assert fake_links_unreachable(fake.links) == []
    assert fake_links_unreachable({f"{fake.url}/?project=acme&page=2"}) == [
        f"the link {fake.url}/?project=acme&page=2 answers 422"
    ]
