"""Predictions past one call's queries go as several calls and come back as one."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import generalforecast as gf
from generalforecast import _operations as ops
from generalforecast.batches import query_batches
from generalforecast.catalog import ModelInfo
from generalforecast.contracts import MAX_QUERIES
from generalforecast.service.contracts import ServiceError
from generalforecast.transport import Profile
from tests.test_hosted import MODEL, MODEL_ID

ROWS = 25_001


def client():
    """What an operation reads of its client, with the model already in the catalog."""
    return SimpleNamespace(
        project="test",
        profile=Profile("https://service.example", "test"),
        _models_seen={MODEL_ID: ModelInfo.from_reply(MODEL)},
    )


def scored(call):
    """A classification answer for a call: each query's label is its `x`."""
    rows = call.payload["queries"]
    return {
        "model": MODEL_ID,
        "task": "classification",
        "predictor": None if call.resource == "ad-hoc" else call.resource,
        "predictions": [
            {"label": row["x"], "probabilities": {"0": 0.5, "1": 0.5}} for row in rows
        ],
    }


def drive(operation, answer):
    """Runs an operation's steps here: work in line, each call answered by `answer`."""
    calls, value, failure = [], None, None
    while True:
        try:
            step = (
                operation.send(value) if failure is None else operation.throw(failure)
            )
        except StopIteration as stop:
            return stop.value, calls
        value, failure = None, None
        if isinstance(step, ops.Work):
            value = step.function(*step.args, **dict(step.kwargs))
        elif isinstance(step, ops.Call):
            calls.append(step)
            try:
                value = answer(step)
            except gf.Error as error:
                failure = error
        else:
            raise TypeError(f"unexpected step {step!r}")


def queries(rows=ROWS):
    return pa.table({"x": list(range(rows))})


def context():
    return pa.table({"x": [0, 1, 2, 3], "y": [0, 1, 0, 1]})


def test_more_queries_than_one_call_go_as_parts_and_come_back_in_order():
    result, calls = drive(
        ops.predict(
            client(),
            model=MODEL_ID,
            target="y",
            context=context(),
            queries=queries(),
            request_id="r-batch",
        ),
        scored,
    )
    assert [len(call.payload["queries"]) for call in calls] == [
        MAX_QUERIES,
        MAX_QUERIES,
        ROWS - 2 * MAX_QUERIES,
    ]
    assert [call.request_id for call in calls] == [
        "r-batch-1",
        "r-batch-2",
        "r-batch-3",
    ]
    assert all(call.payload["context"] == calls[0].payload["context"] for call in calls)
    assert result.table.num_rows == ROWS
    assert result.table.column("label").to_pylist() == list(range(ROWS))


def test_one_call_goes_under_its_own_request_id():
    _, calls = drive(
        ops.predict(
            client(),
            model=MODEL_ID,
            target="y",
            context=context(),
            queries=queries(MAX_QUERIES),
            request_id="r-one",
        ),
        scored,
    )
    assert [call.request_id for call in calls] == ["r-one"]


def test_parts_without_an_id_share_a_fresh_one():
    _, calls = drive(
        ops.predict(
            client(), model=MODEL_ID, target="y", context=context(), queries=queries()
        ),
        scored,
    )
    base = calls[0].request_id.rsplit("-", 1)[0]
    assert base.startswith("r-")
    assert [call.request_id for call in calls] == [f"{base}-{n}" for n in (1, 2, 3)]


def test_a_part_the_service_refuses_names_its_rows():
    def answer(call):
        if call.request_id.endswith("-2"):
            error = ServiceError("Internal", "the worker stopped", 500)
            error.remote = True
            raise error
        return scored(call)

    with pytest.raises(gf.Error) as raised:
        drive(
            ops.predictor_predict(
                client(),
                "churn",
                queries(),
                context=None,
                as_of=None,
                env=None,
                request_id="r-parts",
            ),
            lambda call: (
                {"task": "classification"} if call.method == "GET" else answer(call)
            ),
        )
    assert raised.value.operation == "query rows 10,001 to 20,000 of 25,001"


def test_a_declared_predictor_scores_any_number_of_rows_in_order():
    seen = []

    def answer(call):
        if call.method == "GET":
            seen.append(call.path)
            return {"task": "classification"}
        return scored(call)

    result, calls = drive(
        ops.predictor_predict(
            client(), "churn", queries(), context=None, as_of=None, env="staging"
        ),
        answer,
    )
    posts = [call for call in calls if call.method == "POST"]
    assert [len(call.payload["queries"]) for call in posts] == [
        MAX_QUERIES,
        MAX_QUERIES,
        ROWS - 2 * MAX_QUERIES,
    ]
    assert all(call.params == {"env": "staging"} for call in posts)
    assert seen == ["/predictors/churn"]
    assert result.table.column("label").to_pylist() == list(range(ROWS))


def test_a_declared_predictor_under_one_call_is_not_read_first():
    _, calls = drive(
        ops.predictor_predict(
            client(), "churn", queries(3), context=None, as_of=None, env=None
        ),
        scored,
    )
    assert [call.method for call in calls] == ["POST"]


def test_a_forecasters_series_stay_whole_and_rows_return_in_order():
    series = ["a"] * 6000 + ["b"] * 6000 + ["c"] * 10
    table = pa.table({"s": series, "x": list(range(len(series)))})
    batches = query_batches(table, ("s",), MAX_QUERIES)
    assert [len(rows) for rows in batches] == [6000, 6010]
    assert np.concatenate(batches).tolist() == list(range(len(series)))
    interleaved = pa.table({"s": ["a", "b"] * 6000, "x": list(range(12000))})
    batches = query_batches(interleaved, ("s",), MAX_QUERIES)
    assert [
        set(interleaved.take(rows).column("s").to_pylist()) for rows in batches
    ] == [
        {"a"},
        {"b"},
    ]


def test_parts_from_different_revisions_are_a_conflict():
    def answer(call):
        if call.method == "GET":
            return {"task": "classification"}
        result = scored(call)
        result["revision"] = int(call.request_id.rsplit("-", 1)[1])
        return result

    with pytest.raises(gf.Error) as raised:
        drive(
            ops.predictor_predict(
                client(), "churn", queries(), context=None, as_of=None, env=None
            ),
            answer,
        )
    assert raised.value.code == "Conflict"
