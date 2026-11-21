# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The estimators: fit uploads once by digest, predict sends queries, usage adds up, scikit-learn takes them."""

import re

import numpy as np
import pandas as pd
import pytest

import generalforecast as gf

HOSTED = "Hosted, in-context: fit uploads your rows and trains nothing; predict reads them as context."


def adult(rows=120):
    return pd.DataFrame(
        {
            "age": [20 + i % 50 for i in range(rows)],
            "hours": [float(30 + i % 20) for i in range(rows)],
            "workclass": ["State-gov" if i % 3 else "Private" for i in range(rows)],
        }
    )


def income(rows=120):
    return pd.Series(["<=50K" if i % 4 else ">50K" for i in range(rows)], name="income")


def uploads(fake):
    """The snapshots sent with their bytes, not the probes that found one already there."""
    return [c for c in fake.calls_to("PUT", r".*/snapshots") if c.body]


@pytest.mark.parametrize("cls", [gf.Classifier, gf.Regressor, gf.Forecaster])
def test_every_estimator_says_what_fit_does(cls):
    assert cls.__doc__.startswith(HOSTED)
    estimator = (
        cls(target="units", time="week", horizon=4) if cls is gf.Forecaster else cls()
    )
    assert repr(estimator).endswith("  # fit uploads rows; nothing is trained")


def test_fit_uploads_once_and_each_predict_prints_and_adds_its_usage(fake, capsys):
    clf = gf.Classifier("tabicl-v2").fit(adult(), income())
    assert (
        repr(clf)
        == "Classifier(model='tabicl-v2')  # 120 context rows uploaded; nothing trained"
    )
    assert len(uploads(fake)) == 1
    labels = clf.predict(adult(250))
    assert labels.shape == (250,) and set(labels) <= {"<=50K", ">50K"}
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert re.fullmatch(
        r"Predicted 250 rows with tabicl-v2-classifier from 120 context rows: "
        r"3 calls, 0\.5 credits \(\$0\.0005\) in [\d.]+s",
        line,
    ), line
    assert clf.last_usage_ == gf.types.Estimate(3, 250, 0.5, 0.0005)
    clf.predict_proba(adult(10))
    assert clf.usage_.calls == 4 and clf.usage_.rows == 260
    assert clf.last_usage_.rows == 10
    capsys.readouterr()
    # The same rows again send nothing; the service already holds them.
    gf.Classifier("tabicl-v2", verbose=False).fit(adult(), income()).predict(adult(5))
    assert len(uploads(fake)) == 1 and capsys.readouterr().err == ""
    sent = fake.calls_to("POST", r".*/predict")[-1]
    assert sent.table.num_rows == 5 and sent.table.column_names == [
        "age",
        "hours",
        "workclass",
    ]


def test_estimate_prices_a_predict_and_names_the_large_context_line(fake):
    clf = gf.Classifier(verbose=False).fit(adult(), income())
    assert clf.estimate(adult(250)) == gf.types.Estimate(3, 250, 0.5, 0.0005)
    big = gf.Regressor(verbose=False).fit(adult(40000), np.arange(40000.0))
    priced = big.estimate(adult(100))
    assert priced.large_context == gf.types.LargeContext(1.5, 3.0, 0.003)
    assert fake.calls_to("POST", r".*/estimate")[-1].payload == {
        "model": "tabicl-v2", "task": "regression", "context_rows": 40000, "queries": 100,
        "inputs": 3,
    }  # fmt: skip
    assert len(fake.calls_to("POST", r".*/predict")) == 0


def test_regressor_and_forecaster(fake):
    reg = gf.Regressor(quantiles=[0.1, 0.9], verbose=False).fit(
        adult(), np.arange(120.0)
    )
    assert reg.predict(adult(3)).tolist() == [1.5, 1.5, 1.5]
    assert reg.predict_quantiles(adult(2)).tolist() == [[0.1, 0.9], [0.1, 0.9]]
    history = pd.DataFrame(
        {"week": pd.date_range("2026-01-05", periods=20, freq="7D").repeat(2),
         "sku": ["a", "b"] * 20, "units": np.arange(40.0)}
    )  # fmt: skip
    future = pd.DataFrame(
        {
            "week": pd.date_range("2026-05-25", periods=4, freq="7D").repeat(2),
            "sku": ["a", "b"] * 4,
        }
    )
    fc = gf.Forecaster(
        target="units", time="week", series=["sku"], horizon=4, verbose=False
    ).fit(history)
    assert (
        repr(fc)
        == "Forecaster(model='chronos-2', target='units')  # 40 context rows uploaded; nothing trained"
    )
    assert fc.predict(future).tolist() == [0.5] * 8
    assert fc.forecast(future).shape == (8, 3)
    sent = fake.calls_to("POST", r".*/predict")[-1].payload
    # The task picks the catalog model; the route itself takes no task.
    assert "task" not in sent
    assert (sent["model"], sent["series"], sent["horizon"], sent.get("frequency")) == (
        "general-forecast/chronos-2",
        ["sku"],
        4,
        None,
    )
    assert fc.estimate(future).rows == 8
    sent = fake.calls_to("POST", r".*/estimate")[-1].payload
    assert (sent["task"], sent["inputs"]) == ("forecast", 2)


def test_scikit_learn_clones_tags_and_cross_validates_without_sending_rows_twice(fake):
    from sklearn.base import clone, is_classifier, is_regressor
    from sklearn.model_selection import GridSearchCV, cross_val_score

    client = gf.Client()
    clf = gf.Classifier("tabicl-v2", verbose=False, client=client)
    copy = clone(clf)
    assert copy.get_params() == clf.get_params() and copy.client is client
    assert is_classifier(clf) and is_regressor(gf.Regressor())
    assert copy.set_params(model="kumo-medium").model == "kumo-medium"
    with pytest.raises(gf.InvalidInput):
        copy.set_params(modle="x")
    scores = cross_val_score(clf, adult(), income(), cv=3)
    assert scores.shape == (3,) and np.all((0 <= scores) & (scores <= 1))
    assert len(uploads(fake)) == 3
    # The same folds again, as a grid search runs them, send no rows.
    search = GridSearchCV(clf, {"model": ["tabicl-v2", "kumo-medium"]}, cv=3).fit(
        adult(), income()
    )
    assert search.best_params_["model"] in ("tabicl-v2", "kumo-medium")
    assert len(uploads(fake)) == 4, (
        "three folds already held, and the refit on every row"
    )
    client.close()


def _forecast() -> np.ndarray:
    history = pd.DataFrame(
        {"week": pd.date_range("2026-01-05", periods=20, freq="7D").repeat(2),
         "sku": ["a", "b"] * 20, "units": np.arange(40.0)}
    )  # fmt: skip
    future = history.drop(columns="units").tail(4)
    return gf.Forecaster(target="units", time="week", series=["sku"], horizon=4,
                         verbose=False).fit(history).predict(future)  # fmt: skip


@pytest.mark.parametrize(
    "predict",
    [
        lambda: gf.models.predict(
            "tabicl-v2", context=adult().assign(income=income()), queries=adult(3),
            target="income",
        ),
        lambda: gf.Classifier("tabicl-v2", verbose=False).fit(adult(), income()).predict(adult(3)),
        _forecast,
    ],
    ids=["models.predict", "Classifier", "Forecaster"],
)  # fmt: skip
def test_every_predict_waits_for_a_prediction_accepted_for_later(fake, predict):
    # The hosted plane answers 202 with where to poll while a model warms up.
    fake.accept_predictions = 1
    assert len(predict()) > 0
    assert fake.calls_to("GET", r".*/predictions/pred-.*")
