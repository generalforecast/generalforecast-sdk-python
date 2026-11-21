# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

from datetime import date

import pandas as pd
import pytest

import generalforecast as gf
from generalforecast.types import Holdout

SAMPLE = """\
Experiment adult-income  run 1  classification, target income, 6,513 held-out rows
     Model        AUROC  Interval        Time  Cost
  1  tabicl-v2    0.931  0.925 to 0.937  38s   $0.013
  2  kumo-medium  0.929  0.923 to 0.935  51s   $0.013
  3  mitra-2      0.918  0.911 to 0.925  44s   $0.013
  https://app.generalforecast.com/?project=acme#/experiments/adult-income"""

MODELS = ["kumo-medium", "tabicl-v2", "mitra-2"]


@pytest.fixture
def adult(fake):
    frame = pd.DataFrame(
        {"age": [30, 40, 50, 60], "income": ["<=50K", ">50K", "<=50K", ">50K"]}
    )
    gf.contexts.create("adult", data=frame)
    return fake


def test_create_waits_and_prints_the_leaderboard_as_the_spec_shows(adult):
    exp = gf.experiments.create(
        context="adult", target="income", holdout="20%", models=MODELS
    )
    assert exp.run.state == "finished"
    assert str(exp) == SAMPLE and repr(exp) == SAMPLE
    assert exp.best.short == "tabicl-v2" and exp.best.model.startswith(
        "general-forecast/"
    )
    payload = adult.calls_to("POST", r".*/experiments")[0].payload
    assert payload == {
        "context": "adult",
        "target": "income",
        "holdout": {"kind": "random", "fraction": 0.2},
        "models": MODELS,
    }
    html = exp._repr_html_()
    assert (
        "font-weight:600" in html
        and "tabicl-v2" in html
        and "View in the console" in html
    )
    assert exp._mime_() == ("text/html", html)
    assert "Running run 1" not in html


def test_the_leaderboard_drops_columns_to_fit(adult):
    exp = gf.experiments.create(context="adult", target="income", models=MODELS)
    from generalforecast import _render

    narrow = _render.leaderboard(exp.run, width=36)
    assert narrow[0].split() == ["Model", "AUROC", "Cost"]
    assert repr(exp.run).splitlines()[1].split()[:2] == ["1", "tabicl-v2"]


def test_estimate_is_a_dry_run(adult):
    estimate = gf.experiments.estimate(context="adult", target="income", models=MODELS)
    assert repr(estimate) == "Estimate(calls=198, rows=19539, credits=39.1, usd=0.0391)"
    assert adult.calls_to("POST", r".*/experiments")[0].payload == {
        "context": "adult",
        "target": "income",
        "models": MODELS,
        "dry_run": True,
    }
    assert not adult.experiments


def test_holdout_defaults_to_the_last_20_percent_by_time_when_time_is_declared(adult):
    gf.experiments.create(context="adult@1", target="income", time="age", wait=False)
    payload = adult.calls_to("POST", r".*/experiments")[0].payload
    assert payload["holdout"] == {"kind": "time", "fraction": 0.2}
    assert payload["version"] == 1 and payload["time"] == "age"


def test_run_update_cancel_and_reads(adult):
    exp = gf.experiments.create(context="adult", target="income", models=MODELS)
    second = gf.experiments.run(exp, models=["kumo-large"])
    assert second.run.number == 2 and second.run.state == "finished"
    # Only the fields given go; None leaves a field as it is, as the contract has it.
    updated = gf.experiments.update(
        "adult-income", holdout="20% stratified", title=None
    )
    assert updated.run.stale
    assert adult.experiments["adult-income"]["_updates"][0] == {
        "holdout": {"kind": "random", "fraction": 0.2, "stratify": True},
    }
    third = gf.experiments.run("adult-income", wait=False)
    cancelled = gf.experiments.cancel(third)
    assert cancelled.run.state == "cancelled" and cancelled.run.number == 3
    rows = gf.experiments.predictions("adult-income", model="tabicl-v2")
    assert len(rows) == 4 and "label" in rows.columns
    slices = gf.experiments.slices(exp, model=exp.best.model)
    assert slices[0].column == "marital_status" and slices[0].delta == -0.047
    assert [e.name for e in gf.experiments.list(context="adult")] == ["adult-income"]
    gf.experiments.delete(exp)
    with pytest.raises(gf.NotFound):
        gf.experiments.get("adult-income")


def test_wait_times_out_and_a_failed_run_is_not_recoverable(adult):
    adult.polls_until_done = 100
    with pytest.raises(gf.Timeout) as timeout:
        gf.experiments.create(context="adult", target="income", timeout=0)
    assert timeout.value.recoverable and "wait" in timeout.value.hint
    run = adult.experiments["adult-income"]["_runs"][0]
    run.update(state="failed", error="the engine ran out of memory", _polls=0)
    with pytest.raises(gf.Error) as failed:
        gf.experiments.wait("adult-income")
    assert not failed.value.recoverable and failed.value.code == "RunFailed"
    assert "out of memory" in failed.value.message


def test_a_service_refusal_keeps_its_hint(adult):
    with pytest.raises(gf.InvalidInput) as error:
        gf.experiments.create(context="adult", target="incme")
    assert error.value.message == "the context adult has no column `incme`"
    assert error.value.hint == "a similar column exists: `income`"
    assert error.value.param == "target"


@pytest.mark.parametrize(
    "text, holdout",
    [
        ("20%", Holdout("random", 0.2)),
        ("20% stratified", Holdout("random", 0.2, stratify=True)),
        ("20% by time", Holdout("time", 0.2)),
        ("25% by region", Holdout("group", 0.25, column="region")),
        ("1000 rows", Holdout("random", rows=1000)),
        ("until 2025-06-01", Holdout("time", until=date(2025, 6, 1))),
        ("none", Holdout("none")),
    ],
)
def test_holdout_strings_parse(text, holdout):
    assert Holdout.parse(text) == holdout
    assert str(holdout) == text


@pytest.mark.parametrize("text", ["", "20", "by time", "120%", "20% sideways"])
def test_bad_holdouts_are_refused(text):
    with pytest.raises(gf.InvalidInput):
        Holdout.parse(text)
