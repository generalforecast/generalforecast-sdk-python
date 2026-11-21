# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import generalforecast as gf


@pytest.fixture
def adult(fake):
    frame = pd.DataFrame(
        {"age": [30, 40, 50, 60], "income": ["<=50K", ">50K", "<=50K", ">50K"]}
    )
    gf.contexts.create("adult", data=frame)
    return fake


def test_create_deploys_and_waits_until_ready(adult):
    p = gf.predictors.create(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    assert p.environments["production"].state == "ready"
    assert p.spec.model == "tabicl-v2" and p.resolved.short == "tabicl-v2"
    assert repr(p).startswith(
        "Predictor adult-income  revision 1  tabicl-v2, target income"
    )
    sent = adult.calls_to("POST", r".*/predictors")[0].payload
    assert sent == {
        "name": "adult-income",
        "spec": {"context": "adult", "model": "tabicl-v2", "target": "income"},
        "env": "production",
    }
    with pytest.raises(gf.Conflict):
        gf.predictors.create(
            "adult-income", context="adult", model="tabicl-v2", target="income"
        )


def test_a_misspelt_field_fails_locally_with_the_nearest_name(adult):
    with pytest.raises(gf.InvalidInput) as error:
        gf.predictors.create(
            "churn", context="adult", modle="kumo-medium", target="income"
        )
    assert error.value.message == "gf.Predictor() has no argument `modle`"
    assert error.value.hint == "a similar name exists: `model`"
    with pytest.raises(gf.InvalidInput):
        gf.Predictor("Bad Name", context="adult", model="m", target="t")
    with pytest.raises(gf.InvalidInput):
        gf.Predictor("ok", context="adult", model="m", target="t", inputs="age")


def test_deploy_is_idempotent_and_reports_changes(adult):
    spec = gf.Predictor(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    first = gf.predictors.deploy(spec)
    assert [d.action for d in first] == ["created"]
    assert first[0].environment.state == "ready"
    resolved = first[0].revision.resolved
    assert resolved.context_version == 1 and resolved.context_rows == 4
    assert resolved.model_digest.startswith("sha256:") and resolved.base_digest is None
    assert first[0].revision.status == "building"
    assert first[0].revision.lock_digest.startswith("sha256:")
    (revision,) = gf.predictors.revisions("adult-income")
    assert revision.status == "ready"
    assert revision.lock_digest == first[0].revision.lock_digest
    again = gf.predictors.deploy(spec)
    assert again[0].action == "unchanged" and again[0].revision.number == 1
    plan = gf.predictors.deploy(
        gf.Predictor(
            "adult-income", context="adult", model="kumo-medium", target="income"
        ),
        dry_run=True,
    )
    assert plan[0].action == "updated"
    assert (
        plan[0].changes[0].field == "model"
        and plan[0].changes[0].after == "kumo-medium"
    )
    assert len(adult.predictors["adult-income"]["_revisions"]) == 1
    with pytest.raises(gf.InvalidInput):
        gf.predictors.deploy(spec, spec)


def test_update_merges_fields_and_none_clears(adult):
    gf.predictors.create(
        "adult-income",
        context="adult",
        model="tabicl-v2",
        target="income",
        exclude=["age"],
    )
    p = gf.predictors.update("adult-income", model="kumo-medium", exclude=None)
    assert p.revision == 2 and p.spec.model == "kumo-medium" and p.spec.exclude == ()
    sent = adult.calls_to("POST", r".*/update")[0].payload
    assert sent["spec"] == {"model": "kumo-medium", "exclude": None}
    revisions = gf.predictors.revisions(p)
    assert [r.number for r in revisions] == [2, 1]
    assert (
        revisions[0].resolved.model == "general-forecast/kumo-tabular-medium-classifier"
    )


def test_a_failed_start_is_not_recoverable(adult):
    gf.predictors.create(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    adult.deploy_ends = "failed"
    with pytest.raises(gf.Error) as error:
        gf.predictors.update("adult-income", model="kumo-medium")
    assert error.value.code == "DeployFailed" and not error.value.recoverable
    assert (
        error.value.message == "adult-income revision 2 failed to start in production"
    )
    assert error.value.note == "the replica exited loading the model"
    assert error.value.hint == (
        "production still serves revision 1; "
        "see gf.predictors.logs('adult-income', level='error')"
    )
    assert error.value.request_id == "r-8f2c41d0"


def test_predict_sends_arrow_in_parts_and_adds_usage(adult, monkeypatch):
    from generalforecast import _data

    gf.predictors.create(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    monkeypatch.setattr(_data, "QUERY_PART_BYTES", 600)
    queries = pd.DataFrame({"age": list(range(100))})
    out = gf.predictors.predict("adult-income", queries)
    calls = adult.calls_to("POST", r".*/predictors/adult-income/predict")
    assert len(calls) > 1
    assert sum(c.table.num_rows for c in calls) == 100
    # The environment is a query parameter; the payload carries no `env`.
    assert {c.query["env"] for c in calls} == {"production"}
    assert all("env" not in c.envelope["payload"] for c in calls)
    assert all(
        c.headers["content-type"] == "application/vnd.apache.arrow.stream"
        and c.headers["accept"].startswith("application/vnd.apache.arrow.stream")
        for c in calls
    )
    # Each part is answered as Arrow, its envelope under `gf.result`.
    assert out.model.startswith("general-forecast/") and out.predictor == "adult-income"
    assert out.request_id == calls[0].headers["x-gf-request-id"]
    assert out.rows == 100 and out.credits == pytest.approx(0.2)
    assert out.calls == len(calls)
    assert list(out.to_pandas().columns) == ["id", "label", "p_<=50K", "p_>50K"]
    assert out.timing.compute_seconds == pytest.approx(0.4 * len(calls))
    assert out.context.used == "exact"


def test_a_prediction_accepted_for_later_is_waited_for(adult, capsys):
    gf.predictors.create(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    adult.accept_predictions = 1
    client = gf.Client(progress=True)
    out = client.predictors.predict("adult-income", pd.DataFrame({"age": [1, 2]}))
    assert out.rows == 2
    [poll] = adult.calls_to("GET", r".*/predictions/pred-.*")
    # The poll asks to be held until the prediction is answered.
    assert poll.headers["prefer"] == "wait=25"
    assert "waiting for a GPU (cold start)" in capsys.readouterr().err
    client.close()


def test_promote_rollback_logs_and_delete(adult):
    gf.predictors.create(
        "adult-income", context="adult", model="tabicl-v2", target="income"
    )
    gf.predictors.create(
        "churn", context="adult", model="tabicl-v2", target="income", env="staging"
    )
    promoted = gf.predictors.promote("churn")
    assert (
        promoted.name == "production"
        and promoted.state == "ready"
        and promoted.revision == 1
    )
    gf.predictors.update("adult-income", model="kumo-medium")
    back = gf.predictors.rollback("adult-income")
    assert back.revision == 1
    logs = gf.predictors.logs("adult-income", level="error", limit=2)
    assert logs.entries[0].request_id == "r-3abb2401f119428db148927c6a0d358c"
    assert logs.entries[1].level == "warn" and logs.entries[1].at.year == 2026
    assert logs.next == "1795034671207000"
    query = adult.calls_to("GET", r".*/logs")[0].query
    assert query == {"limit": "2", "level": "error"}
    assert {p.name for p in gf.predictors.list()} == {"adult-income", "churn"}
    gf.predictors.delete("churn")
    with pytest.raises(gf.NotFound):
        gf.predictors.get("churn")
