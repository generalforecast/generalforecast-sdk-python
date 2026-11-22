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
    # `limits` are the model's own, as the engine's catalog gives them.
    assert one.kind == "catalog" and one.limits == gf.types.Limits(100000, 10000)
    assert one.digest and one.digest.startswith("sha256:")
    assert one.trained_on.rows == 100000
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


# A frame with a missing number, dates and categories, and the digest the
# engine's `table_digest` gives it, the same for the Parquet a context upload
# sends and the Arrow stream a held snapshot is put as.
DATED = {
    "id": ["a", "b", "c"],
    "amount": [1.5, float("nan"), 3.0],
    "day": ["2026-01-01", "2026-01-02", "2026-01-03"],
    "plan": ["x", "y", "x"],
}
DIGEST = "sha256:2c88ad7e2570a4fd2a332e9cda3556a652e0236fd60e8900cc78fc82a89947a7"
# The engine's digest of the table under `..._whatever_lies_under_its_nulls`.
ENGINE_NULLS = "sha256:282daaa4f8bf80e2d9d9d60ad697c3d907038c66b0e03f8f965f764508237969"


def dated() -> pd.DataFrame:
    rows = pd.DataFrame(DATED)
    return rows.assign(
        day=pd.to_datetime(rows["day"]), plan=rows["plan"].astype("category")
    )


def test_a_table_hashes_as_the_engine_hashes_what_the_plane_reads():
    rows = _data.normalized(_data.table(dated()))
    assert _data.table_digest(_data.table(dated())) == DIGEST
    assert _data.table_digest(_data.read_parquet(_data.parquet(rows))) == DIGEST
    assert _data.table_digest(_data.read_stream(_data.stream(rows))) == DIGEST
    assert str(rows.schema.field("day").type) == "timestamp[us, tz=UTC]"
    assert rows.column("plan").type == pa.string()


def test_a_table_has_one_digest_whatever_lies_under_its_nulls():
    # The engine's own case: a NaN made null stays NaN beneath in an Arrow
    # stream, and Parquet reads a zero back. Rows 1 and 3 are null.
    values = np.array([1.5, np.nan, 3.0, 0.0]).tobytes()
    table = pa.table(
        {
            "x": pa.Array.from_buffers(
                pa.float64(), 4, [pa.py_buffer(bytes([0b0101])), pa.py_buffer(values)]
            ),
            "s": pa.array(["a", None, "c", "d"]),
            "i": pa.array([1, None, 3, 4]),
            "b": pa.array([True, None, False, True]),
            "t": pa.array([1, None, 3, 4], pa.timestamp("us", tz="UTC")),
        }
    )
    sent = _data.read_stream(_data.stream(table))
    read = _data.read_parquet(_data.parquet(sent))

    def beneath(table: pa.Table) -> bytes:
        return table.column("x").chunk(0).buffers()[1].to_pybytes()[8:16]

    assert beneath(sent) != beneath(read)
    assert _data.table_digest(sent) == _data.table_digest(read) == ENGINE_NULLS
    assert beneath(_data.normalized(sent)) == bytes(8)


def test_contexts_and_predictions_name_a_table_by_the_engines_digest(fake):
    gf.contexts.create("dated", data=dated())
    created = fake.calls_to("POST", r".*/contexts")[0].payload
    assert created["data"]["digest"] == DIGEST == fake.contexts["dated"]["digest"]
    gf.models.predict(
        "tabicl-v2",
        context=dated(),
        queries=dated().drop(columns="plan"),
        target="plan",
    )
    assert fake.calls_to("PUT", r".*/snapshots")[0].query["digest"] == DIGEST


def test_a_table_the_service_cannot_hold_is_refused_before_it_is_sent():
    with pytest.raises(gf.InvalidInput, match="`span`"):
        _data.normalized(_data.table(pd.DataFrame({"span": pd.to_timedelta([1], "s")})))
    with pytest.raises(gf.InvalidInput, match="infinity"):
        _data.normalized(_data.table(pd.DataFrame({"x": [float("inf")]})))
    with pytest.raises(gf.InvalidInput, match="cannot be held"):
        _data.normalized(
            _data.table(pd.DataFrame({"t": pd.to_datetime([1], unit="ns")}))
        )
    empty = _data.normalized(_data.table(pd.DataFrame({"x": [None, None]})))
    assert empty.column("x").type == pa.string()


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
        and sent["model"] == "general-forecast/chronos-2"
        and "task" not in sent
    )
    with pytest.raises(gf.InvalidInput) as error:
        gf.models.predict("m", context=frame(2), queries=frame(1), target="y", horizn=3)
    assert error.value.hint == "a similar name exists: `horizon`"


def test_a_context_past_the_services_snapshot_limits_fails_before_it_is_sent(fake):
    # The limits are the service's, as `auth.status` says them, asked once.
    fake.limits["snapshot_bytes"] = 1000
    with pytest.raises(gf.CapacityExceeded) as error:
        gf.models.predict("tabicl-v2", context=frame(500), queries=frame(1), target="y")
    assert error.value.limit == 1000 and error.value.actual > 1000
    assert "gf.contexts.create" in error.value.hint
    fake.limits.update(snapshot_bytes=1 << 25, snapshot_rows=100)
    with gf.Client() as other, pytest.raises(gf.CapacityExceeded) as error:
        other.models.predict(
            "tabicl-v2", context=frame(101), queries=frame(1), target="y"
        )
    assert (error.value.limit, error.value.actual) == (100, 101)
    assert not fake.calls_to("PUT", r".*/snapshots")
    assert len(fake.calls_to("GET", r"/v1/auth/status")) == 2


def test_estimators_fit_uploads_and_fail_fast(fake):
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
        "inputs": X.shape[1],
        "task": "classification",
    }
    fake.limits["snapshot_bytes"] = 100
    puts = len(fake.calls_to("PUT", r".*/snapshots"))
    with gf.Client() as other, pytest.raises(gf.CapacityExceeded):
        gf.Classifier(client=other).fit(X, y)
    assert len(fake.calls_to("PUT", r".*/snapshots")) == puts
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
        _data.table_digest(_data.table(v))
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
    assert [len(p) for p in _ops.json_parts(rows, 1000)] == [2, 2, 1]
    assert [len(p) for p in _ops.json_parts(rows, 40)] == [1, 1, 1, 1, 1]
    monkeypatch.undo()
    parts = _ops.json_parts(list(range(250)), 1000, 100)
    assert [len(p) for p in parts] == [100, 100, 50]
