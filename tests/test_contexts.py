# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import generalforecast as gf


def adult(rows=200):
    return pd.DataFrame(
        {
            "age": [20 + i % 50 for i in range(rows)],
            "workclass": ["State-gov" if i % 3 else "Private" for i in range(rows)],
            "income": ["<=50K" if i % 4 else ">50K" for i in range(rows)],
        }
    )


@pytest.mark.parametrize("signed", [True, False], ids=["signed-url", "bytes-route"])
def test_create_uploads_in_chunks_and_waits_until_ready(fake, signed):
    fake.chunk_bytes, fake.signed_uploads = 512, signed
    ctx = gf.contexts.create("adult", data=adult())
    assert ctx.state == "ready" and ctx.version == 1 and ctx.rows == 200
    assert [c.name for c in ctx.columns] == ["age", "workclass", "income"]
    assert ctx.url.endswith("#/data/contexts/adult")
    assert repr(ctx) == "Context adult  snapshot v1  200 rows  3 columns"
    created = fake.calls_to("POST", r".*/contexts")[0].payload
    assert created["kind"] == "snapshot" and created["data"]["format"] == "parquet"
    assert created["data"]["digest"].startswith("sha256:")
    if signed:
        # The signed URL's query names the upload session, so every chunk keeps it.
        chunks = fake.calls_to("PUT", r"/uploads/contexts/adult")
        assert {c.query["upload_id"] for c in chunks} == {"adult-1"}
        offsets = [
            int(c.headers["content-range"].split()[1].split("-")[0]) for c in chunks
        ]
    else:
        chunks = fake.calls_to("POST", r".*/versions/1/bytes")
        offsets = [int(c.query["offset"]) for c in chunks]
    assert len(chunks) > 1 and all(len(c.body) <= 512 for c in chunks)
    assert offsets[:2] == [0, 512]


def test_the_same_rows_again_are_unchanged_and_send_no_bytes(fake):
    gf.contexts.create("adult", data=adult())
    sent = len(fake.calls_to("PUT", r"/uploads/.*"))
    again = gf.contexts.upload("adult", adult())
    assert again.unchanged and again.version == 1
    assert repr(again).endswith("unchanged")
    assert len(fake.calls_to("PUT", r"/uploads/.*")) == sent
    changed = gf.contexts.upload(again, adult(300))
    assert changed.version == 2 and not changed.unchanged


def test_upload_creates_a_context_it_does_not_find(fake):
    first = gf.contexts.upload("adult", adult())
    assert first.state == "ready" and first.version == 1 and first.rows == 200
    assert fake.calls_to("POST", r".*/contexts")[0].payload["kind"] == "snapshot"
    again = gf.contexts.upload("adult", adult())
    assert again.unchanged and again.version == 1
    assert len(fake.calls_to("POST", r".*/contexts")) == 1


def test_a_path_is_read_and_named(fake, tmp_path):
    path = tmp_path / "adult.csv"
    adult().to_csv(path, index=False)
    ctx = gf.contexts.create("adult", data=str(path))
    assert (
        fake.calls_to("POST", r".*/contexts")[0].payload["data"]["file"] == "adult.csv"
    )
    assert ctx.file == "adult.csv"


def test_a_source_import_polls_until_ready(fake):
    ctx = gf.contexts.create(
        "customers",
        source="bigquery-acme",
        table="acme.analytics.customers",
        id="customer_id",
    )
    assert ctx.state == "ready" and ctx.rows == 48210
    payload = fake.calls_to("POST", r".*/contexts")[0].payload
    assert payload["source"] == {
        "source": "bigquery-acme",
        "table": "acme.analytics.customers",
    }
    assert payload["id"] == "customer_id" and payload["kind"] == "snapshot"
    refreshed = gf.contexts.refresh("customers")
    assert refreshed.state == "ready"


def test_live_contexts_from_a_source_and_from_the_api(fake):
    gf.contexts.create(
        "events",
        source="snowflake-prod",
        sql="select 1",
        live=True,
        freshness="5m",
        time="event_at",
    )
    events = fake.calls_to("POST", r".*/contexts")[0].payload
    assert events["kind"] == "live" and events["freshness_seconds"] == 300
    assert events["source"] == {"source": "snowflake-prod", "sql": "select 1"}
    signups = gf.contexts.create(
        "signups",
        live=True,
        columns={"plan": "category", "seats": "number", "converted": "boolean"},
        id="account_id",
        targets=["converted"],
    )
    assert signups.kind == "live"
    receipt = gf.contexts.append(
        "signups", pd.DataFrame({"account_id": [1, 2], "plan": ["team", "free"]})
    )
    assert receipt.rows == 2 and receipt.sequence > 0
    # A write carries at most 100 rows: more go in parts, each its own write id.
    many = gf.contexts.append(
        "signups",
        pd.DataFrame({"account_id": range(250), "plan": ["team"] * 250}),
        idempotency_key="rows-1",
    )
    assert many.rows == 250
    assert [len(w["rows"]) for w in fake.writes[-3:]] == [100, 100, 50]
    assert [w["write_id"] for w in fake.writes[-3:]] == [
        "rows-1-1",
        "rows-1-2",
        "rows-1-3",
    ]
    with pytest.raises(gf.InvalidInput):
        gf.contexts.append("signups", pd.DataFrame({"account_id": []}))
    labeled = gf.contexts.label(
        signups,
        pd.DataFrame({"account_id": [1], "converted": [True]}),
        target="converted",
        idempotency_key="labels-1",
    )
    assert labeled.write_id == "labels-1"
    assert fake.writes[-1]["labels"] == [{"id": 1, "value": True}]


def test_seeding_a_live_context_infers_columns_and_appends(fake):
    gf.contexts.create("seeded", live=True, data=adult(5), id="age")
    payload = fake.calls_to("POST", r".*/contexts")[0].payload
    assert payload["columns"] == {
        "age": "number",
        "workclass": "category",
        "income": "category",
    }
    assert len(fake.writes[0]["rows"]) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"data": adult(), "source": "bq"},
        {"source": "bq"},
        {"source": "bq", "table": "t", "sql": "s"},
        {"table": "t"},
        {"live": True},
        {"data": adult(), "targets": ["income"]},
        {"source": "bq", "table": "t", "live": True, "freshness": "30s"},
    ],
)
def test_create_takes_exactly_one_origin(fake, kwargs):
    with pytest.raises(gf.InvalidInput) as error:
        gf.contexts.create("x", **kwargs)
    assert error.value.recoverable
    assert not fake.calls_to("POST", r".*/contexts")


def test_get_pins_versions_and_rows_page(fake):
    gf.contexts.create("adult", data=adult())
    gf.contexts.upload("adult", adult(300))
    assert gf.contexts.get("adult@1").version == 1
    assert gf.contexts.get("adult", version=1).rows == 200
    assert gf.contexts.get("adult").version == 2
    rows = gf.contexts.rows("adult", limit=5)
    assert list(rows.columns) == ["age", "workclass", "income"] and len(rows) == 5
    assert [c.name for c in gf.contexts.list()] == ["adult"]
    with pytest.raises(gf.InvalidInput):
        gf.contexts.get("adult@latest")


def test_delete_is_refused_while_a_predictor_serves_it(fake):
    gf.contexts.create("adult", data=adult())
    fake.contexts["adult"]["predictors"] = ["adult-income"]
    with pytest.raises(gf.Conflict) as error:
        gf.contexts.delete("adult")
    assert error.value.reason == "context_in_use" and error.value.request_id
    fake.contexts["adult"]["predictors"] = []
    gf.contexts.delete("adult")
    with pytest.raises(gf.NotFound):
        gf.contexts.get("adult")


def test_context_waits_check_every_poll_seconds(fake, monkeypatch):
    from generalforecast import _ops

    slept = []
    real = _ops._poll
    monkeypatch.setattr(
        _ops, "_poll", lambda a, p=None, *rest: slept.append(real(a, p)) or 0
    )
    gf.contexts.create(
        "customers", source="bigquery-acme", table="acme.analytics.customers", poll=0.5
    )
    gf.contexts.refresh("customers", poll=0.5)
    assert slept and set(slept) == {0.5}
    with pytest.raises(gf.InvalidInput) as error:
        gf.contexts.upload("adult", adult(), poll=-1)
    assert error.value.param == "poll"
