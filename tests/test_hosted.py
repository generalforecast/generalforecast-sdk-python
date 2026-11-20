# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Every prediction, check and evaluation runs on the service: the routes gf sends."""

import json

import numpy as np
import pytest

import generalforecast as gf
from generalforecast import predictor_commands
from generalforecast._operations import snapshot_name
from generalforecast.catalog import Catalog, ModelInfo
from generalforecast.client import Client
from generalforecast.service.sdk import table
from generalforecast.tables import read_frame
from generalforecast.tabular import arrow_bytes, table_digest
from generalforecast.transport import Profile
from generalforecast.validation import bytes_digest
from tests.test_cli_output import gf as run

MODEL_ID = "general-forecast/tabicl-v2-classifier"
MODEL = {
    "model": MODEL_ID + "@sha256:" + "a" * 64,
    "name": MODEL_ID,
    "task": "classification",
    "limits": {"context_rows": 100000, "queries": 100, "inputs": 32},
}
MANIFEST = (
    "name: churn\n"
    f"model: {MODEL_ID}\n"
    "target: y\n"
    "id: id\n"
    "context:\n  snapshot: rows.csv\n"
)
LOCK = {
    "lock_version": 2,
    "digest": "sha256:" + "b" * 64,
    "manifest": "sha256:" + "d" * 64,
    "predictor": {
        "name": "churn",
        "model": MODEL_ID,
        "model_digest": "sha256:" + "a" * 64,
        "task": "classification",
        "target": "y",
        "context": {"kind": "snapshot", "rows": 6},
    },
    "model": {"files": {}},
}
OUTSIDE = "local serving is not part of gf"


@pytest.fixture
def churn(tmp_path):
    root = tmp_path / "churn"
    root.mkdir()
    rows = ["id,x,y"] + [f"{i},{i / 10},{i % 2}" for i in range(6)]
    (root / "rows.csv").write_text("\n".join(rows) + "\n")
    (root / "gf.yaml").write_text(MANIFEST)
    return root


@pytest.fixture
def hosted(wire, tmp_path, monkeypatch):
    """The CLI's connection: the wire fixture by GF_URL, with no saved credentials."""
    env = {
        "GF_URL": wire.url,
        "GF_API_KEY": "fixture-key",
        "GF_PROJECT": "test",
        "GF_CONFIG_PATH": str(tmp_path / "config" / "config.toml"),
        "GF_KEYCHAIN": "0",
    }
    for key in ("GF_PROFILE", "GF_TOKEN", "GF_ENV"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(predictor_commands.time, "sleep", lambda seconds: None)
    return env


def uploaded(root):
    """The name and digest the snapshot of `root`'s manifest uploads under."""
    rows = table(read_frame(root, "rows.csv"))
    return snapshot_name("churn", bytes_digest(arrow_bytes(rows))), table_digest(rows)


def client(wire):
    return Client(profile=Profile(wire.url, "test"), api_key="fixture-key")


def test_check_uploads_the_snapshot_and_writes_the_lock_the_service_answers(
    wire, hosted, churn, monkeypatch
):
    name, _ = uploaded(churn)
    answer = {
        "lock": LOCK,
        "model": MODEL_ID,
        "inferred": {"task": "classification", "inputs": ["x"], "id": "id"},
        "changes": [],
    }
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        answer,
        payload={"manifest": MANIFEST, "snapshot": name, "previous": None},
    )
    code, stdout, stderr = run("check", str(churn), monkeypatch=monkeypatch, env=hosted)
    assert code == 0 and stdout == "", stderr
    assert json.loads((churn / "gf.lock").read_text()) == LOCK
    assert "Resolved churn  classification on " + MODEL_ID in stderr
    assert "Locked gf.lock" in stderr
    assert [path for path, _ in wire.requests] == [
        "/v1/projects/test/snapshots",
        "/v1/projects/test/predictors/churn/check",
    ]

    # Checked again with the lock in place, --locked verifies it unchanged.
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        answer,
        payload={"manifest": MANIFEST, "snapshot": name, "previous": LOCK},
    )
    code, _, stderr = run(
        "check", "--locked", str(churn), monkeypatch=monkeypatch, env=hosted
    )
    assert code == 0 and "Verified gf.lock" in stderr


def test_a_changed_lock_fails_check_locked_and_lists_the_change(
    wire, hosted, churn, monkeypatch
):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    name, _ = uploaded(churn)
    moved = {**LOCK, "digest": "sha256:" + "e" * 64}
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        {
            "lock": moved,
            "model": MODEL_ID,
            "inferred": {},
            "changes": [["~", "target", "y", "z"]],
        },
        payload={"manifest": MANIFEST, "snapshot": name, "previous": LOCK},
    )
    code, _, stderr = run(
        "--json", "check", "--locked", str(churn), monkeypatch=monkeypatch, env=hosted
    )
    assert code == 1
    assert json.loads(stderr)["error"]["message"] == "gf.lock would change"
    assert json.loads((churn / "gf.lock").read_text()) == LOCK


def evaluation(state, scored, metrics=None):
    return {
        "evaluation": "ev-1",
        "state": state,
        "held": 2,
        "task": "classification",
        "metric": "accuracy",
        "models": [
            {
                "model": MODEL_ID,
                "state": state,
                "metrics": metrics or {},
                "interval": None,
                "scored": scored,
                "total": 2,
                "seconds": None,
                "error": None,
            }
        ],
        "created": 1795035000.0,
        "started": 1795035001.0,
        "finished": None,
    }


def test_eval_starts_a_hosted_evaluation_and_polls_it_to_its_scores(
    wire, hosted, churn, monkeypatch
):
    name, _ = uploaded(churn)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        {"evaluation": "ev-1", "state": "queued"},
        payload={"manifest": MANIFEST, "snapshot": name},
    )
    wire.expect("GET", "/predictors/churn/evaluations/ev-1", evaluation("running", 1))
    done = evaluation("succeeded", 2, {"accuracy": 0.5, "log_loss": 0.69})
    wire.expect("GET", "/predictors/churn/evaluations/ev-1", done)
    code, stdout, stderr = run("eval", str(churn), monkeypatch=monkeypatch, env=hosted)
    assert code == 0, stderr
    assert "Evaluated churn  2 held-out rows" in stderr
    assert "accuracy" in stdout and "0.500" in stdout and "0.690" in stdout


def test_a_failed_evaluation_is_an_error_naming_the_model(
    wire, hosted, churn, monkeypatch
):
    wire.expect("POST", "/snapshots")
    name, _ = uploaded(churn)
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        {"evaluation": "ev-1", "state": "queued"},
        payload={
            "manifest": MANIFEST,
            "snapshot": name,
            "models": ["general-forecast/x"],
        },
    )
    failed = evaluation("failed", 0)
    failed["models"][0]["error"] = "the target has one class"
    wire.expect("GET", "/predictors/churn/evaluations/ev-1", failed)
    code, _, stderr = run(
        "--json",
        "eval",
        str(churn),
        "--model",
        "general-forecast/x",
        monkeypatch=monkeypatch,
        env=hosted,
    )
    assert code == 1
    error = json.loads(stderr)["error"]
    assert error["message"] == "evaluation ev-1 failed"
    assert error["because"] == f"{MODEL_ID}: the target has one class"


def test_the_sdk_evaluates_and_waits(wire, churn):
    name, _ = uploaded(churn)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        {"evaluation": "ev-1", "state": "queued"},
        payload={"manifest": MANIFEST, "snapshot": name},
    )
    done = evaluation("succeeded", 2, {"accuracy": 1.0})
    wire.expect("GET", "/predictors/churn/evaluations/ev-1", done)
    with client(wire) as hosted:
        result = hosted.predictors.evaluate(churn, poll_interval=0)
    assert result["state"] == "succeeded"
    assert result["models"][0]["metrics"] == {"accuracy": 1.0}


def test_push_sends_the_lock_the_service_wrote_and_the_snapshot(wire, churn):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    _, digest = uploaded(churn)
    wire.expect("GET", "/predictors/churn", "NotFound", status=404)
    wire.expect("POST", "/predictors", {"name": "churn"}, payload={"name": "churn"})
    wire.expect("POST", "/snapshots")
    revision = {
        "predictor": "churn",
        "number": 1,
        "predictor_digest": LOCK["digest"],
        "model": MODEL_ID,
        "created_at": 1795035000.0,
        "status": "pending",
    }
    manifest = {
        "name": "churn",
        "model": MODEL_ID,
        "target": "y",
        "id": "id",
        "context": {"snapshot": "rows.csv"},
    }
    wire.expect(
        "POST",
        "/predictors/churn/revisions",
        revision,
        payload={"lock": LOCK, "snapshot": digest, "manifest": manifest},
    )
    environment = {
        "predictor": "churn",
        "name": "staging",
        "revision": None,
        "generation": 0,
        "status": "pending",
    }
    wire.expect("GET", "/predictors/churn/environments/staging", "NotFound", status=404)
    wire.expect(
        "POST",
        "/predictors/churn/environments",
        environment,
        payload={"name": "staging"},
    )
    wire.expect(
        "POST",
        "/predictors/churn/environments/staging/promote",
        {**environment, "revision": 1, "generation": 1},
        payload={"revision": 1, "expected_generation": 0},
    )
    with client(wire) as hosted:
        pushed = hosted.predictors.push(churn, env="staging")
    assert pushed.number == 1


def test_push_without_a_lock_says_to_check_first(churn):
    with (
        pytest.raises(gf.Error) as raised,
        Client(
            profile=Profile("https://service.example", "test"), api_key="fixture-key"
        ) as hosted,
    ):
        hosted.predictors.push(churn)
    assert raised.value.code == "LockRequired"
    assert "run `gf check` first" in raised.value.message


@pytest.mark.parametrize(
    "argv",
    [
        ("serve",),
        ("serve", "./churn", "--port", "9000"),
        ("--json", "serve"),
        ("model", "pull", MODEL_ID),
        ("admin", "node", "init"),
        ("--timeout", "5", "admin", "router", "serve"),
    ],
)
def test_commands_outside_gf_print_one_plain_line(argv, monkeypatch):
    code, stdout, stderr = run(*argv, monkeypatch=monkeypatch)
    assert code == 2 and stdout == ""
    assert stderr.count("\n") == 1 and stderr.startswith(OUTSIDE)
    assert "gf push" in stderr and not stderr.startswith("error")


def test_predict_refuses_a_predictor_directory_and_points_at_push(churn, monkeypatch):
    code, stdout, stderr = run(
        "predict", str(churn), "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 1 and stdout == ""
    assert stderr == (
        f"error: `{churn}` is a predictor directory; predictions run on the hosted "
        "service\n"
        f"help: push it with `gf push {churn}`, then run `gf predict churn --queries "
        "FILE`\n"
    )
    code, _, stderr = run(
        "predict", MODEL_ID, "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 1 and f"pass --model {MODEL_ID}" in stderr
    with pytest.raises(gf.InvalidInput, match="predictor directory"):
        gf.Predictor(str(churn))
    with pytest.raises(gf.InvalidInput, match="predictor directory"):
        gf.Predictor(churn)


def test_predict_with_a_model_needs_a_target_and_context(monkeypatch):
    code, _, stderr = run(
        "predict", "--model", MODEL_ID, "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith("error: --model needs a target")


def test_predict_with_a_model_sends_the_ad_hoc_route(
    wire, hosted, tmp_path, monkeypatch
):
    (tmp_path / "context.csv").write_text("x,y\n1,0\n2,1\n")
    (tmp_path / "queries.csv").write_text("x\n3\n")
    wire.expect("GET", "/models/tabicl-v2-classifier", MODEL)
    wire.expect(
        "POST",
        "/predict",
        {
            "model": MODEL_ID,
            "task": "classification",
            "predictions": [{"label": 1, "probabilities": {"0": 0.25, "1": 0.75}}],
        },
        payload={
            "model": MODEL_ID,
            "target": "y",
            "context": [{"x": 1, "y": 0}, {"x": 2, "y": 1}],
            "queries": [{"x": 3}],
            "inputs": ["x"],
        },
    )
    code, stdout, stderr = run(
        "predict",
        "--model",
        MODEL_ID,
        "--target",
        "y",
        "--context",
        str(tmp_path / "context.csv"),
        "--queries",
        str(tmp_path / "queries.csv"),
        monkeypatch=monkeypatch,
        env=hosted,
    )
    assert code == 0, stderr
    assert stdout == "label,p_0,p_1\n1,0.25,0.75\n"


def test_the_classifier_fits_nothing_here_and_predicts_on_the_service(wire):
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = [0, 1, 0]
    wire.expect("GET", "/models/tabicl-v2-classifier", MODEL)
    wire.expect(
        "POST",
        "/predict",
        {
            "model": MODEL_ID,
            "task": "classification",
            "predictions": [{"label": 1, "probabilities": {"0": 0.2, "1": 0.8}}],
        },
        payload={
            "model": MODEL_ID,
            "target": "target",
            "context": [
                {"f0": 1.0, "f1": 2.0, "target": 0},
                {"f0": 3.0, "f1": 4.0, "target": 1},
                {"f0": 5.0, "f1": 6.0, "target": 0},
            ],
            "queries": [{"f0": 7.0, "f1": 8.0}],
            "inputs": ["f0", "f1"],
            "outputs": ["label", "probabilities"],
        },
    )
    with client(wire) as hosted:
        model = gf.Classifier(client=hosted).fit(X, y)
        assert list(model.classes_) == [0, 1]
        assert model.predict_proba(np.array([[7.0, 8.0]])).tolist() == [[0.2, 0.8]]
    assert model.get_params()["model"] == MODEL_ID


def test_more_training_rows_than_one_request_point_at_pushing_a_context(wire):
    small = {**MODEL, "limits": {"context_rows": 2, "queries": 100, "inputs": 32}}
    wire.expect("GET", "/models/tabicl-v2-classifier", small)
    with client(wire) as hosted, pytest.raises(gf.CapacityExceeded) as raised:
        gf.Classifier(client=hosted).fit(np.array([[1.0], [2.0], [3.0]]), [0, 1, 0])
    assert raised.value.message == "3 training rows exceed one request"
    assert "gf push" in raised.value.hint and "context" in raised.value.hint
    with pytest.raises(gf.InvalidInput, match="not fitted"):
        gf.Regressor(client=hosted).predict(np.array([[1.0]]))


def test_the_catalog_reads_ids_tasks_and_limits():
    catalog = Catalog.from_reply([MODEL])
    assert catalog.entry(MODEL_ID) == ModelInfo(
        MODEL_ID, "classification", catalog.limits(MODEL_ID)
    )
    assert catalog.limits(MODEL_ID).queries == 100
    with pytest.raises(gf.InvalidInput) as raised:
        catalog.canonical("tabicl-v2-classifier")
    assert raised.value.hint == f"use {MODEL_ID}"
    with pytest.raises(gf.NotFound):
        catalog.canonical("general-forecast/unknown")
    assert catalog.default("classification") == MODEL_ID
    assert catalog.default("forecast") == "general-forecast/timesfm-3"


def test_init_writes_a_manifest_from_the_table_and_the_hosted_catalog(
    tmp_path, monkeypatch
):
    root = tmp_path / "churn"
    root.mkdir()
    rows = ["id,x,y"] + [f"{i},{i / 10},{i % 2}" for i in range(40)]
    (root / "rows.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(
        "generalforecast.commands.model_registry",
        lambda args: Catalog.from_reply([MODEL]),
    )
    code, _, stderr = run("init", "--yes", str(root), monkeypatch=monkeypatch)
    assert code == 0, stderr
    import yaml

    written = yaml.safe_load((root / "gf.yaml").read_text())
    assert written["name"] == "churn" and written["target"] == "y"
    assert written["model"] == MODEL_ID and written["id"] == "id"
    assert written["context"]["snapshot"] == "rows.csv"


def test_the_package_has_no_local_serving_or_model_code():
    import importlib.util

    for name in ("native", "nn", "models", "runtime", "serve", "engine", "predictors"):
        assert importlib.util.find_spec(f"generalforecast.{name}") is None, name
    assert not hasattr(gf, "engine")


class Recorded:
    """A client that records the ad hoc predictions sent to it and answers each."""

    def __init__(self, entry, answer):
        self.entry, self.answer, self.sent = entry, answer, []
        self._models_seen = {}
        self.models = self

    def get(self, model):
        return self.entry

    def predict(self, model, target, context, queries, **options):
        import pyarrow as pa

        self.sent.append((model, target, context, queries, options))
        rows = self.answer(queries)
        return type("Result", (), {"table": pa.Table.from_pylist(rows)})()


def test_the_forecaster_forecasts_each_series_on_the_service():
    import pandas as pd

    days = pd.date_range("2025-01-01", periods=5, freq="D")
    history = pd.DataFrame(
        {
            "date": list(days) * 2,
            "sku": ["a"] * 5 + ["b"] * 5,
            "units": [float(i) for i in range(10)],
        }
    )
    forecaster = {**MODEL, "name": "general-forecast/timesfm-3", "task": "forecast"}
    levels = {"0.1": 1.0, "0.5": 2.0, "0.9": 3.0}
    hosted = Recorded(
        forecaster, lambda queries: [{"quantiles": levels}] * len(queries)
    )
    model = gf.Forecaster(
        target="units", time="date", series="sku", horizon=2, client=hosted
    ).fit(history)
    grid = model.future_grid()
    assert grid["sku"].tolist() == ["a", "a", "b", "b"]
    assert [str(t.date()) for t in grid["date"]] == [
        "2025-01-06",
        "2025-01-07",
        "2025-01-06",
        "2025-01-07",
    ]
    assert model.forecast().tolist() == [[1.0, 2.0, 3.0]] * 4
    assert model.predict(grid).tolist() == [2.0] * 4
    sent_model, target, context, queries, options = hosted.sent[0]
    assert sent_model == "general-forecast/timesfm-3" and target == "units"
    assert context.num_rows == 10 and queries.num_rows == 4
    assert options["series"] == ["sku"] and options["horizon"] == 2
    assert options["quantiles"] == [0.1, 0.5, 0.9]


def test_the_regressor_reports_values_and_quantiles_from_the_service():
    regressor = {
        **MODEL,
        "name": "general-forecast/tabicl-v2-regressor",
        "task": "regression",
    }

    def answer(queries):
        return [{"value": 1.5, "quantiles": {"0.1": 1.0, "0.9": 2.0}}] * len(queries)

    hosted = Recorded(regressor, answer)
    model = gf.Regressor(client=hosted).fit(np.array([[1.0], [2.0]]), [1.0, 2.0])
    assert model.predict(np.array([[3.0]])).tolist() == [1.5]
    assert model.predict_quantiles(np.array([[3.0]]), [0.1, 0.9]).tolist() == [
        [1.0, 2.0]
    ]
    with pytest.raises(gf.InvalidInput, match="Regressor predicts numbers"):
        model.fit(np.array([[1.0], [2.0]]), ["a", "b"])
