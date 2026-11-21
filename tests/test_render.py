# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

from generalforecast import _render, types
from generalforecast.types import (
    Column,
    Context,
    Environment,
    Predictor,
    PredictorInfo,
    load,
)


def test_context_reprs():
    ctx = Context(
        "adult",
        version=1,
        versions=1,
        rows=32561,
        columns=tuple(Column(f"c{i}", "number") for i in range(15)),
        url="https://app.generalforecast.com/?project=acme#/data/contexts/adult",
    )
    assert repr(ctx) == "Context adult  snapshot v1  32,561 rows  15 columns"
    html = ctx._repr_html_()
    assert html.count("<tr>") == 16 and "View in the console" in html
    live = Context("events", kind="live", state="importing", rows=None)
    assert repr(live) == "Context events  live  importing"


def test_html_escapes_what_the_service_sent():
    ctx = Context("x", columns=(Column("<script>", "text"),), url='https://a/"><b>')
    html = ctx._repr_html_()
    assert "<script>" not in html and "&lt;script&gt;" in html and '"><b>' not in html


def test_predictor_reprs():
    info = PredictorInfo(
        "adult-income",
        spec=Predictor(
            "adult-income", context="adult", model="kumo-medium", target="income"
        ),
        revision=2,
        environments={
            "production": Environment(
                "production", 2, "ready", calls_today=1204, p50_seconds=0.9
            ),
            "staging": Environment("staging", 2, "ready"),
        },
    )
    assert repr(info).splitlines() == [
        "Predictor adult-income  revision 2  kumo-medium, target income",
        "  production  revision 2  ready  1,204 calls today",
        "  staging     revision 2  ready  0 calls today",
    ]
    assert "0.90s" in info._repr_html_()


def test_formats():
    assert [_render.seconds(v) for v in (0.21, 6.8, 41, 72)] == [
        "0.21s",
        "6.8s",
        "41s",
        "1m12s",
    ]
    assert [types.money(v) for v in (0.013, 0.0391, 12.5, 0)] == [
        "$0.013",
        "$0.039",
        "$12.50",
        "$0",
    ]
    assert (
        _render.metric_name("auroc") == "AUROC"
        and _render.metric_name("log loss") == "Log loss"
    )


def test_results_load_leniently():
    ctx = load(
        Context,
        {"name": "a", "freshness_seconds": 300, "created": 1761200000.0, "new": 1},
    )
    assert ctx.freshness.total_seconds() == 300 and ctx.created.tzinfo is not None
    assert (
        load(Context, {"name": "a", "created": "2026-11-21T09:12:04Z"}).created.hour
        == 9
    )
    assert (
        load(Context, {"name": "a", "error": {"message": "bad file"}}).error
        == "bad file"
    )
