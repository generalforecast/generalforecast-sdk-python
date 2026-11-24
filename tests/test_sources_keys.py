# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

from datetime import timedelta

import pytest

import generalforecast as gf


def test_sources_take_the_config_keys_of_their_kind(fake):
    source = gf.sources.create(
        "pg-replica",
        kind="postgres",
        secret="pw",
        host="10.2.0.4",
        port="5432",
        database="app",
    )
    assert (
        source.kind == "postgres" and source.state == "connected" and source.tested_at
    )
    assert source.access == "secret" and source.tables == 3 and source.contexts == 0
    sent = fake.calls_to("POST", r".*/sources")[0].payload
    assert sent["secret"] == "pw" and sent["config"]["port"] == "5432"
    with pytest.raises(gf.InvalidInput) as error:
        gf.sources.create("bq", kind="bigquery", projcet="acme")
    assert error.value.hint == "a similar name exists: `project`"
    with pytest.raises(gf.InvalidInput):
        gf.sources.create("x", kind="mysql")
    listed = gf.sources.list()
    assert [s.name for s in listed] == ["pg-replica"] and listed.available
    assert gf.sources.test(source).name == "pg-replica"
    tables = gf.sources.tables("pg-replica", prefix="customers")
    assert (tables.location, tables.more) == ("acme-prod.analytics", False)
    assert [(t.name, t.type, t.readable) for t in tables][2] == (
        "customers_model",
        "MODEL",
        False,
    )
    assert len(tables) == 3 and tables[0].name == "customers"
    preview = gf.sources.preview("pg-replica", table="acme.analytics.customers")
    assert list(preview.columns) == ["customer_id", "churned"]
    with pytest.raises(gf.InvalidInput):
        gf.sources.preview("pg-replica")
    gf.sources.delete("pg-replica")
    assert gf.sources.list() == []


def test_keys_by_id_or_name(fake):
    new = gf.keys.create("ci-deploy")
    assert new.token.startswith("gf_live_") and "token" not in repr(new)
    assert new.key.state == "active" and new.key.expires is not None
    assert new.key.owner.email == "maya@acme.com" and new.key.owner.subject
    assert not new.key.service and not new.key.legacy and new.key.machine is None
    sent = fake.calls_to("POST", r".*/keys")[0].payload
    assert sent["expires_in"] == 90 * 86400 and "models:read" in sent["permissions"]
    never = gf.keys.create("forever", expires=None, permissions=["predictors:predict"])
    assert never.key.expires is None
    gf.keys.create("week", expires=timedelta(days=7))
    rotated = gf.keys.rotate("ci-deploy")
    assert rotated.token != new.token
    with pytest.raises(gf.Conflict):
        gf.keys.delete(new.key)
    revoked = gf.keys.revoke(new.key.id)
    assert revoked.state == "revoked"
    gf.keys.delete("ci-deploy")
    assert {k.name for k in gf.keys.list()} == {"forever", "week"}
    with pytest.raises(gf.NotFound) as error:
        gf.keys.revoke("forevr")
    assert error.value.hint == "a similar name exists: `forever`"
    with pytest.raises(gf.InvalidInput):
        gf.keys.create("bad", expires="soon")
    # A permission only a browser session uses is not a key's to hold.
    with pytest.raises(gf.InvalidInput) as error:
        gf.keys.create("sso", permissions=["sso:read"])
    assert error.value.param == "permissions"


def test_prepare_answers_what_to_set_up_for_a_kind(fake):
    setup = gf.sources.prepare("bigquery")
    assert setup.kind == "bigquery"
    assert setup.identity.email.startswith("gfws-")
    assert [(g.role, g.on) for g in setup.grants] == [
        ("roles/bigquery.dataViewer", "dataset"),
        ("roles/bigquery.jobUser", "project"),
        ("roles/bigquery.readSessionUser", "project"),
    ]
    assert "billing_project" in setup.steps[1]
    assert setup.oidc is None and setup.policy is None
    assert fake.calls[-1].payload == {"kind": "bigquery"}
    with pytest.raises(gf.InvalidInput) as error:
        gf.sources.prepare("postgres")
    assert error.value.param == "kind"


def test_a_preview_is_its_columns_and_sample(fake):
    from tests import samples

    samples._sources(fake)
    frame = gf.sources.preview("bigquery-acme", table="customers")
    assert list(frame.columns) == ["customer_id", "churned"]
    assert frame.to_dict("records") == [
        {"customer_id": "C-100481", "churned": "false"},
        {"customer_id": "C-100482", "churned": "true"},
    ]
