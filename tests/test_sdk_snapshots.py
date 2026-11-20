# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Snapshots are listed, read and deleted by name like every other collection, found by tag, and generated in synthetic batches."""

import inspect

import pytest

from generalforecast import _api
from generalforecast import _operations as ops
from tests.test_sdk_client import run

SNAPSHOT = {
    "name": "credit-2026-01",
    "state": "ready",
    "digest": "sha256:" + "5" * 64,
    "bytes": 10,
    "rows": 2,
    "columns": [{"name": "y", "type": "int64"}],
    "metadata": {"target": "y", "task": "classification"},
    "tags": [],
    "error": None,
    "expires": None,
    "created": 1.0,
}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_snapshots_are_listed_read_and_deleted(wire, asynchronous):
    snapshot = SNAPSHOT["name"]
    wire.expect("GET", "/snapshots", {"snapshots": [SNAPSHOT]})
    wire.expect("GET", f"/snapshots/{snapshot}", SNAPSHOT)
    deleted = {**SNAPSHOT, "state": "deleted"}
    wire.expect("POST", f"/snapshots/{snapshot}/delete", deleted, payload={})

    async def body(client):
        found = []
        for call in (
            lambda: client.snapshots.list(),
            lambda: client.snapshots.get(snapshot),
            lambda: client.snapshots.delete(snapshot),
        ):
            value = call()
            found.append(await value if asynchronous else value)
        return found

    listed, got, gone = run(asynchronous, wire.url, body)
    assert listed == [SNAPSHOT]
    assert got == SNAPSHOT
    assert gone == deleted


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_snapshots_tags_are_replaced_and_found(wire, asynchronous):
    snapshot = SNAPSHOT["name"]
    tagged = {**SNAPSHOT, "tags": ["credit", "stage:signal"]}
    wire.expect(
        "POST",
        f"/snapshots/{snapshot}/update",
        tagged,
        payload={"tags": ["stage:signal", "credit"]},
    )
    wire.expect("GET", "/snapshots?tag=stage%3Asignal", {"snapshots": [tagged]})
    cleared = {**SNAPSHOT, "tags": []}
    wire.expect("POST", f"/snapshots/{snapshot}/update", cleared, payload={"tags": []})

    async def body(client):
        found = []
        for call in (
            lambda: client.snapshots.update(snapshot, tags=["stage:signal", "credit"]),
            lambda: client.snapshots.list(tag="stage:signal"),
            lambda: client.snapshots.update(snapshot, tags=[]),
        ):
            value = call()
            found.append(await value if asynchronous else value)
        return found

    updated, listed, emptied = run(asynchronous, wire.url, body)
    assert updated == tagged
    assert listed == [tagged]
    assert emptied == cleared


def test_a_tag_is_the_lists_only_query():
    every = ops.snapshot_list().send(None)
    assert isinstance(every, ops.Call) and every.params is None
    tagged = ops.snapshot_list("stage:signal")
    step = tagged.send(None)
    assert isinstance(step, ops.Call)
    assert (step.method, step.path) == ("GET", "/snapshots")
    assert step.params == {"tag": "stage:signal"}
    with pytest.raises(StopIteration) as done:
        tagged.send({"snapshots": [SNAPSHOT]})
    assert done.value.value == [SNAPSHOT]


def test_an_update_sends_only_tags_as_a_retried_write():
    step = ops.snapshot_update("credit-2026-01", ("credit",)).send(None)
    assert isinstance(step, ops.Call)
    assert (step.method, step.path) == ("POST", "/snapshots/credit-2026-01/update")
    assert step.payload == {"tags": ["credit"]}
    assert step.retry_class == "write"


BATCH = "credit-warmup"
TABLES = [f"{BATCH}-{i}" for i in range(3)]
CREATED = {
    "name": BATCH,
    "tag": f"synthetic-{BATCH}",
    "count": 3,
    "snapshots": TABLES,
    "state": "pending",
}


def batch(*states, error="every draw was trivial to predict after 20 attempts"):
    """The batch as `snapshots.synthetic` reads it, its tables in `states`."""
    return {
        **CREATED,
        "state": "pending" if "pending" in states else "done",
        "seed": 0,
        "options": {"name": BATCH, "count": 3, "seed": 0, "task": "classification"},
        "tables": [
            {
                "name": table,
                "state": state,
                "error": error if state == "rejected" else None,
            }
            for table, state in zip(TABLES, states, strict=True)
        ],
        "counts": {s: states.count(s) for s in ("pending", "ready", "rejected")},
    }


def drive(operation, replies):
    """Runs an operation, answering each Call with the next reply; returns the steps and result."""
    steps, value = [], None
    try:
        while True:
            step = operation.send(value)
            steps.append(step)
            value = replies.pop(0)
    except StopIteration as stop:
        assert not replies, replies
        return steps, stop.value


def test_a_batchs_options_go_as_given_and_its_defaults_are_left_out():
    assert ops.synthetic_spec(BATCH) == {"name": BATCH}
    assert ops.synthetic_spec(BATCH, count=1, seed=None, rows=None) == {"name": BATCH}
    payload = ops.synthetic_spec(
        BATCH,
        count=64,
        seed=0,
        tags=("stage:warmup",),
        task="classification",
        rows={"min": 1024, "max": 8192},
        columns=24,
        categorical={"share": {"min": 0, "max": 0.5}, "max_categories": 50},
        missing={"rate": 0.05, "patterns": ["random", "censored"]},
        reject=["trivial"],
    )
    assert payload == {
        "name": BATCH,
        "count": 64,
        "seed": 0,
        "task": "classification",
        "rows": {"min": 1024, "max": 8192},
        "columns": 24,
        "categorical": {"share": {"min": 0, "max": 0.5}, "max_categories": 50},
        "missing": {"rate": 0.05, "patterns": ["random", "censored"]},
        "reject": ["trivial"],
        "tags": ["stage:warmup"],
    }
    # The payload is the spec's, which the service checks the same way.
    assert _api.SyntheticBatchCreate.from_payload(payload).to_payload() == payload
    shaped = ops.synthetic_spec("credit-shaped", schema_from="credit-2026-01")
    assert shaped == {"name": "credit-shaped", "schema_from": "credit-2026-01"}


def test_an_unknown_option_or_a_range_without_its_max_is_refused_by_the_spec():
    with pytest.raises(_api.PayloadError) as unknown:
        _api.SyntheticBatchCreate.from_payload(ops.synthetic_spec(BATCH, depth=4))
    assert unknown.value.param == "depth"
    with pytest.raises(_api.PayloadError) as half:
        _api.SyntheticBatchCreate.from_payload(
            ops.synthetic_spec(BATCH, rows={"min": 64})
        )
    assert half.value.param == "rows"


def test_a_batch_is_created_as_a_retried_write_and_returned_at_once():
    payload = ops.synthetic_spec(BATCH, count=3, task="classification")
    steps, result = drive(ops.synthetic_create(payload), [CREATED])
    (step,) = steps
    assert isinstance(step, ops.Call)
    assert (step.method, step.path) == ("POST", "/snapshots/synthetic")
    assert (step.kind, step.resource) == ("snapshot", BATCH)
    assert step.payload == payload
    assert step.retry_class == "write"
    assert result == CREATED


@pytest.mark.parametrize("asynchronous", [False, True])
def test_create_synthetic_returns_the_pending_batch_without_reading_it(
    wire, asynchronous
):
    payload = {
        "name": BATCH,
        "count": 3,
        "seed": 0,
        "task": "classification",
        "rows": {"min": 1024, "max": 8192},
        "tags": ["stage:warmup"],
    }
    # One request: the wire refuses any read the client would send after it.
    wire.expect("POST", "/snapshots/synthetic", CREATED, payload=payload)
    result = run(
        asynchronous,
        wire.url,
        lambda client: client.snapshots.create_synthetic(
            BATCH,
            count=3,
            seed=0,
            tags=["stage:warmup"],
            task="classification",
            rows={"min": 1024, "max": 8192},
        ),
    )
    assert result == CREATED
    assert result["snapshots"] == TABLES


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_batch_is_read_by_its_name(wire, asynchronous):
    read = batch("ready", "rejected", "pending")
    wire.expect("GET", f"/snapshots/synthetic/{BATCH}", read)
    result = run(asynchronous, wire.url, lambda c: c.snapshots.synthetic(BATCH))
    assert result == read
    assert _api.SyntheticBatch.from_payload(result).to_payload() == result


def test_both_clients_take_a_batch_alike():
    from generalforecast.async_client import Snapshots as AsyncSnapshots
    from generalforecast.client import Snapshots

    for method in ("create_synthetic", "synthetic"):
        sync, awaited = getattr(Snapshots, method), getattr(AsyncSnapshots, method)
        assert inspect.iscoroutinefunction(awaited)
        assert list(inspect.signature(sync, eval_str=True).parameters.values()) == (
            list(inspect.signature(awaited, eval_str=True).parameters.values())
        )
        assert (sync.__doc__ or "").strip()
    parameters = inspect.signature(Snapshots.create_synthetic).parameters
    assert {k: p.default for k, p in parameters.items() if k != "self"} == {
        "name": inspect.Parameter.empty,
        "count": 1,
        "seed": None,
        "schema_from": None,
        "tags": None,
        "options": inspect.Parameter.empty,
    }


def test_an_origin_is_uploaded_a_contexts_or_a_synthetic_tables_new_shape():
    origin = {
        "kind": "synthetic",
        "batch": BATCH,
        "seed": 1,
        "generator": "scm-1",
        "options": {"rows": 2048, "columns": 24},
    }
    metadata = {"target": "y", "task": "classification", "origin": origin}
    decoded = _api.SnapshotMetadata.from_payload(metadata)
    assert isinstance(decoded.origin, _api.SyntheticOrigin)
    assert decoded.to_payload() == metadata
    uploaded = _api.SnapshotMetadata.from_payload(
        {"target": "y", "task": "regression", "origin": {"kind": "uploaded"}}
    )
    assert isinstance(uploaded.origin, _api.UploadedOrigin)
    context = {"kind": "context", "context": "churn-history", "as_of": None}
    from_context = _api.SnapshotMetadata.from_payload(
        {"target": "y", "task": "regression", "origin": context}
    )
    assert isinstance(from_context.origin, _api.ContextOrigin)
    assert from_context.origin.to_payload() == context
    with pytest.raises(_api.PayloadError) as old:
        _api.SnapshotMetadata.from_payload(
            {
                "target": "y",
                "task": "regression",
                "origin": {
                    "kind": "synthetic",
                    "prior": "mlp",
                    "version": "1",
                    "seed": 0,
                },
            }
        )
    assert old.value.param == "origin"


def test_a_batch_name_leaves_room_for_its_tag_and_table_names():
    longest = "b" * 54
    assert _api.SyntheticBatchCreate.from_payload({"name": longest}).name == longest
    with pytest.raises(_api.PayloadError) as long:
        _api.SyntheticBatchCreate.from_payload({"name": longest + "b"})
    assert long.value.param == "name"
    # Its tag is a tag, and its last table's name a name.
    _api._decode_RunTag(f"synthetic-{longest}", "tag")
    _api._decode_Name(f"{longest}-255", "name")
