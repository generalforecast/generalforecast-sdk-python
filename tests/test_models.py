# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import generalforecast as gf
from generalforecast import _data


def frame(rows):
    return pd.DataFrame(
        {
            "x": np.arange(rows, dtype=float),
            "y": ["a" if i % 2 else "b" for i in range(rows)],
        }
    )


def test_list_and_get_return_the_same_model_shape(fake):
    models = gf.models.list(task="classification")
    assert {m.short for m in models} == {
        "tabicl-v2",
        "kumo-medium",
        "kumo-small",
        "kumo-large",
        "mitra-2",
    }
    one = gf.models.get("tabicl-v2", task="classification")
    assert one == next(m for m in models if m.id == one.id)
    assert (
        one.kind == "catalog"
        and one.limits.queries == 100
        and one.trained_on.rows == 100000
    )
    with pytest.raises(gf.AmbiguousReference):
        gf.models.get("tabicl-v2")
    with pytest.raises(gf.NotFound):
        gf.models.get("kumo-med")


def test_predict_uploads_the_context_once_by_digest(fake):
    context, queries = frame(50), frame(5).drop(columns="y")
    out = gf.models.predict("tabicl-v2", context=context, queries=queries, target="y")
    assert out.rows == 5 and out.predictor is None
    puts = fake.calls_to("PUT", r".*/snapshots")
    assert [bool(c.body) for c in puts] == [False, True]
    digest = puts[0].query["digest"]
    assert fake.last_prediction["context"] == {"snapshot": digest}
    gf.models.predict("tabicl-v2", context=context, queries=queries, target="y")
    assert len(fake.calls_to("PUT", r".*/snapshots")) == 2
    other = gf.Client()
    other.models.predict("tabicl-v2", context=context, queries=queries, target="y")
    assert [bool(c.body) for c in fake.calls_to("PUT", r".*/snapshots")][2:] == [False]
    other.close()


def test_predict_takes_predictor_fields_and_refuses_others(fake):
    gf.models.predict(
        "chronos-2",
        context=pd.DataFrame(
            {"sku": ["a"] * 4, "t": [1, 2, 3, 4], "units": [1.0, 2, 3, 4]}
        ),
        queries=pd.DataFrame({"sku": ["a"], "t": [5]}),
        target="units",
        task="forecast",
        time="t",
        series=["sku"],
        horizon=1,
        quantiles=[0.1, 0.9],
    )
    sent = fake.last_prediction
    assert (
        sent["series"] == ["sku"]
        and sent["horizon"] == 1
        and sent["task"] == "forecast"
    )
    with pytest.raises(gf.InvalidInput) as error:
        gf.models.predict("m", context=frame(2), queries=frame(1), target="y", horizn=3)
    assert error.value.hint == "a similar name exists: `horizon`"


def test_a_context_past_the_snapshot_ceiling_fails_before_any_request(
    fake, monkeypatch
):
    monkeypatch.setattr(_data, "SNAPSHOT_MAX_BYTES", 1000)
    before = len(fake.calls)
    with pytest.raises(gf.CapacityExceeded) as error:
        gf.models.predict("tabicl-v2", context=frame(500), queries=frame(1), target="y")
    assert error.value.limit == 1000 and error.value.actual > 1000
    assert "gf.contexts.create" in error.value.hint
    assert len(fake.calls) == before


def test_estimators_fit_uploads_and_fail_fast(fake, monkeypatch):
    X, y = frame(40).drop(columns="y"), frame(40)["y"]
    clf = gf.Classifier("tabicl-v2").fit(X, y)
    assert (
        repr(clf)
        == "Classifier(model='tabicl-v2')  # 40 context rows uploaded; nothing trained"
    )
    assert (
        repr(gf.Classifier())
        == "Classifier(model='tabicl-v2')  # fit uploads rows; nothing is trained"
    )
    probabilities = clf.predict_proba(X.head(3))
    assert probabilities.shape == (3, 2)
    assert clf.usage_.rows == 3 and clf.last_usage_.calls == 1
    priced = clf.estimate(X)
    assert (priced.calls, priced.rows) == (1, 40)
    assert fake.calls_to("POST", r".*/estimate")[0].payload == {
        "model": "tabicl-v2",
        "context_rows": 40,
        "queries": 40,
        "task": "classification",
    }
    monkeypatch.setattr(_data, "SNAPSHOT_MAX_BYTES", 100)
    calls = len(fake.calls)
    with pytest.raises(gf.CapacityExceeded):
        gf.Classifier().fit(X, y)
    assert len(fake.calls) == calls
    with pytest.raises(gf.InvalidInput):
        gf.Classifier().predict(X)


def test_split_keeps_series_whole_and_bounds_bytes():
    table = pa.table({"s": ["a", "b", "a", "c", "b"], "v": [1.0, 2, 3, 4, 5]})
    parts = _data.split(table, max_bytes=40, series=["s"])
    seen = [set(p.column("s").to_pylist()) for p in parts]
    assert all(len(s) >= 1 for s in seen)
    for key in ("a", "b", "c"):
        assert sum(key in s for s in seen) == 1
    assert sum(p.num_rows for p in parts) == 5
    wide = pa.table({"x": list(range(10_000))})
    parts = _data.split(wide, max_bytes=20_000)
    assert len(parts) > 1 and all(len(_data.stream(p)) <= 20_000 for p in parts)
    assert pa.concat_tables(parts).equals(wide)


def test_the_same_rows_hash_alike_whatever_holds_them(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    digests = {
        _data.digest(_data.stream(_data.table(v)))
        for v in (
            df,
            pa.Table.from_pandas(df),
            str(path),
            {"a": [1, 2], "b": ["x", "y"]},
        )
    }
    assert len(digests) == 1
    with pytest.raises(gf.InvalidInput):
        _data.table(tmp_path / "t.xlsx")
    with pytest.raises(gf.InvalidInput):
        _data.table(42)


def test_json_parts_count_values_as_well_as_bytes(monkeypatch):
    from generalforecast import _ops

    monkeypatch.setattr(_ops, "JSON_PART_VALUES", 10)
    rows = [{"a": i, "b": i, "c": i, "d": i} for i in range(5)]
    parts = _ops.json_parts(rows)
    assert [len(p) for p in parts] == [2, 2, 1]
    monkeypatch.undo()
    assert [len(p) for p in _ops.json_parts(list(range(250)), 100)] == [100, 100, 50]
