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
NORMALIZED = {
    "name": "churn",
    "model": MODEL_ID,
    "target": "y",
    "id": "id",
    "context": {"snapshot": "rows.csv"},
}
CONTEXT = {
    "kind": "snapshot",
    "rows": 6,
    "query_rows": 0,
    "digest": "sha256:" + "c" * 64,
}
KEPT = {"name": "churn-" + "c" * 12, "digest": "sha256:" + "c" * 64, "rows": 6}
# What a check with `deploy` answers for the held-out churn directory: every
# labeled row, 6, under the lock and manifest without the holdout.
DEPLOYED_LOCK = {
    **LOCK,
    "digest": "sha256:" + "f" * 64,
    "predictor": {**LOCK["predictor"], "context": {"kind": "snapshot", "rows": 6}},
}
DEPLOYED = {
    "lock": DEPLOYED_LOCK,
    "manifest": NORMALIZED,
    "context": {**CONTEXT, "digest": "sha256:" + "9" * 64},
    "snapshot": {
        "name": "churn-" + "9" * 12,
        "digest": "sha256:" + "9" * 64,
        "rows": 6,
    },
    "held_out": 2,
}
HELD = "  holdout: {fraction: 0.34, seed: 0}\n"
OUTSIDE = "local serving is not part of gf"


def checked(lock=LOCK, *, changes=(), snapshot=None, inferred=None, deployed=False):
    """A check's answer: the lock, what was inferred and, when kept, the stored rows.

    With `deployed`, a check with `deploy` answers it too: `DEPLOYED`, or the
    value given.
    """
    extra = (
        {}
        if deployed is False
        else {"deployed": DEPLOYED if deployed is True else deployed}
    )
    return {
        **extra,
        "lock": lock,
        "model": MODEL_ID,
        "inferred": {"task": "classification", "inputs": ["x"], "id": "id"}
        if inferred is None
        else inferred,
        "changes": [list(change) for change in changes],
        "manifest": NORMALIZED,
        "context": CONTEXT,
        "snapshot": snapshot,
    }


def asked(
    snapshot,
    previous=None,
    *,
    upgrade=False,
    keep=False,
    deploy=False,
    manifest=MANIFEST,
):
    """A check's payload; `deploy` is sent only when true."""
    payload = {
        "manifest": manifest,
        "snapshot": snapshot,
        "previous": previous,
        "upgrade": upgrade,
        "keep": keep,
    }
    if deploy:
        payload["deploy"] = True
    return payload


@pytest.fixture
def churn(tmp_path):
    root = tmp_path / "churn"
    root.mkdir()
    rows = ["id,x,y"] + [f"{i},{i / 10},{i % 2}" for i in range(6)]
    (root / "rows.csv").write_text("\n".join(rows) + "\n")
    (root / "gf.yaml").write_text(MANIFEST)
    return root


@pytest.fixture
def held(churn):
    """The churn directory with a context that holds a third of its rows out."""
    (churn / "gf.yaml").write_text(MANIFEST + HELD)
    return churn


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
    answer = checked()
    wire.expect("POST", "/snapshots")
    wire.expect("POST", "/predictors/churn/check", answer, payload=asked(name))
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
    wire.expect("POST", "/predictors/churn/check", answer, payload=asked(name, LOCK))
    code, _, stderr = run(
        "check", "--locked", str(churn), monkeypatch=monkeypatch, env=hosted
    )
    assert code == 0 and "Verified gf.lock" in stderr


def test_check_upgrade_asks_the_service_to_accept_a_new_model_digest(
    wire, hosted, churn, monkeypatch
):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    name, _ = uploaded(churn)
    upgraded = {
        **LOCK,
        "digest": "sha256:" + "f" * 64,
        "predictor": {**LOCK["predictor"], "model_digest": "sha256:" + "9" * 64},
    }
    change = ["~", "model_digest", "sha256:" + "a" * 64, "sha256:" + "9" * 64]
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        checked(upgraded, changes=[change]),
        payload=asked(name, LOCK, upgrade=True),
    )
    code, stdout, stderr = run(
        "--json", "check", "--upgrade", str(churn), monkeypatch=monkeypatch, env=hosted
    )
    assert code == 0, stderr
    result = json.loads(stdout)
    assert len(result["changes"]) == 1 and result["context"] == CONTEXT
    assert json.loads((churn / "gf.lock").read_text()) == upgraded


def test_a_check_the_service_refuses_shows_gfs_message_and_field(
    wire, hosted, churn, monkeypatch
):
    name, _ = uploaded(churn)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        {
            "code": "ValidationError",
            "message": "gf.yaml names predictor churn2, not churn",
            "param": "name",
        },
        payload=asked(name),
        status=422,
    )
    code, _, stderr = run("check", str(churn), monkeypatch=monkeypatch, env=hosted)
    assert code == 1
    assert "gf.yaml names predictor churn2, not churn" in stderr
    assert not (churn / "gf.lock").exists()


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
        checked(moved, changes=[("~", "target", "y", "z")], inferred={}),
        payload=asked(name, LOCK),
    )
    code, _, stderr = run(
        "--json", "check", "--locked", str(churn), monkeypatch=monkeypatch, env=hosted
    )
    assert code == 1
    assert json.loads(stderr)["error"]["message"] == "gf.lock would change"
    assert json.loads((churn / "gf.lock").read_text()) == LOCK


def evaluation(state, scored, metrics=None, *, number=1):
    return {
        "evaluation": number,
        "state": state,
        "held": 2,
        "repeats": 1,
        "context": 4,
        "task": "classification",
        "metric": "auroc",
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
        "billed": {"predictions": 1 if scored else 0, "rows": 6, "seconds": 0.4},
        "created": 1795035000.0,
        "started": 1795035001.0,
        "finished": None,
    }


def test_eval_starts_a_hosted_evaluation_and_polls_it_to_its_scores(
    wire, hosted, held, monkeypatch
):
    name, _ = uploaded(held)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        evaluation("queued", 0, number=3),
        payload={"manifest": MANIFEST + HELD, "snapshot": name},
    )
    wire.expect("GET", "/predictors/churn/evaluations/3", evaluation("running", 1))
    done = evaluation("finished", 2, {"auroc": 0.75, "log loss": 0.69}, number=3)
    wire.expect("GET", "/predictors/churn/evaluations/3", done)
    code, stdout, stderr = run("eval", str(held), monkeypatch=monkeypatch, env=hosted)
    assert code == 0, stderr
    assert "Evaluated churn  2 held-out rows" in stderr
    assert "auroc" in stdout and "0.750" in stdout and "0.690" in stdout
    assert [path for path, _ in wire.requests][-1] == (
        "/v1/projects/test/predictors/churn/evaluations/3"
    )


def test_eval_with_another_model_scores_the_manifests_model_beside_it(
    wire, hosted, held, monkeypatch
):
    wire.expect("POST", "/snapshots")
    name, _ = uploaded(held)
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        evaluation("queued", 0),
        payload={
            "manifest": MANIFEST + HELD,
            "snapshot": name,
            "models": [MODEL_ID, "general-forecast/x"],
        },
    )
    failed = evaluation("failed", 0)
    failed["models"][0]["state"] = "failed"
    failed["models"][0]["error"] = {
        "code": "ValidationError",
        "message": "the target has one class",
    }
    wire.expect("GET", "/predictors/churn/evaluations/1", failed)
    code, _, stderr = run(
        "--json",
        "eval",
        str(held),
        "--model",
        "general-forecast/x",
        monkeypatch=monkeypatch,
        env=hosted,
    )
    assert code == 1
    error = json.loads(stderr)["error"]
    assert error["message"] == "evaluation 1 failed"
    assert error["because"] == f"{MODEL_ID}: the target has one class"


def test_eval_refuses_a_manifest_that_holds_no_rows_out_before_uploading(
    wire, hosted, churn, monkeypatch
):
    code, stdout, stderr = run("eval", str(churn), monkeypatch=monkeypatch, env=hosted)
    assert code == 1 and stdout == ""
    assert stderr.startswith(
        "error: gf eval scores held-out rows, and gf.yaml holds none out\n"
    )
    assert "add `holdout` or `until` to the context in gf.yaml" in stderr
    assert wire.requests == []


@pytest.mark.parametrize(
    ("status", "error", "hint"),
    [
        (
            429,
            {
                "code": "CapacityExceeded",
                "message": "project test has 1 run going, its limit",
                "reason": "run_limit_reached",
                "limit": 1,
                "actual": 1,
            },
            "run `gf eval` again once the other ends",
        ),
        (
            422,
            {
                "code": "ValidationError",
                "message": "general-forecast/x is not a hosted classification model",
                "reason": "model_not_hostable",
                "param": "models",
            },
            "run `gf model list` for the models the service hosts",
        ),
        (
            422,
            {
                "code": "ValidationError",
                "message": "no held-out row has a label",
                "param": "holdout",
            },
            "give the context in gf.yaml a `holdout` or `until`",
        ),
    ],
)
def test_an_evaluation_the_service_refuses_says_what_to_do(
    wire, hosted, held, monkeypatch, status, error, hint
):
    name, _ = uploaded(held)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        error,
        payload={"manifest": MANIFEST + HELD, "snapshot": name},
        status=status,
    )
    code, _, stderr = run("eval", str(held), monkeypatch=monkeypatch, env=hosted)
    assert code == 1
    assert "the service refused the evaluation of predictor churn" in stderr
    assert error["message"] in stderr and hint in stderr
    # A standing limit is not retried: one request, one refusal.
    assert len(wire.requests) == 2


def test_the_sdk_evaluates_and_waits(wire, held):
    name, _ = uploaded(held)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/evaluations",
        evaluation("queued", 0),
        payload={"manifest": MANIFEST + HELD, "snapshot": name},
    )
    done = evaluation("finished", 2, {"auroc": 1.0})
    wire.expect("GET", "/predictors/churn/evaluations/1", done)
    with client(wire) as hosted:
        result = hosted.predictors.evaluate(held, poll_interval=0)
        assert result["state"] == "finished"
        assert result["models"][0]["metrics"] == {"auroc": 1.0}
        with pytest.raises(gf.InvalidInput, match="number from 1"):
            hosted.predictors.evaluation("churn", "ev-1")


def push_answers(wire, *, read=True, ready=False):
    """What a first push to staging meets once its check stored every labeled row."""
    if read:
        wire.expect("GET", "/predictors/churn", "NotFound", status=404)
    wire.expect("POST", "/predictors", {"name": "churn"}, payload={"name": "churn"})
    revision = {
        "predictor": "churn",
        "number": 1,
        "predictor_digest": LOCK["digest"],
        "model": MODEL_ID,
        "created_at": 1795035000.0,
        "status": "pending",
    }
    wire.expect(
        "POST",
        "/predictors/churn/revisions",
        revision,
        payload={
            "lock": DEPLOYED_LOCK,
            "snapshot": DEPLOYED["snapshot"]["digest"],
            "manifest": NORMALIZED,
        },
    )
    environment = {
        "predictor": "churn",
        "name": "staging",
        "revision": None,
        "generation": 0,
        "status": "pending",
    }
    if read:
        wire.expect(
            "GET", "/predictors/churn/environments/staging", "NotFound", status=404
        )
    wire.expect(
        "POST",
        "/predictors/churn/environments",
        environment,
        payload={"name": "staging"},
    )
    wire.expect(
        "POST",
        "/predictors/churn/environments/staging/promote",
        {
            **environment,
            "revision": 1,
            "generation": 1,
            "status": "ready" if ready else "pending",
        },
        payload={"revision": 1, "expected_generation": 0},
    )


def test_push_deploys_every_labeled_row_the_check_stored(wire, churn):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    name, _ = uploaded(churn)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        checked(deployed=True),
        payload=asked(name, LOCK, deploy=True),
    )
    push_answers(wire)
    with client(wire) as hosted:
        pushed = hosted.predictors.push(churn, env="staging")
    assert pushed.number == 1 and not wire.expected and not wire.errors


def test_the_cli_pushes_the_same_way(wire, hosted, churn, monkeypatch):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    name, _ = uploaded(churn)
    wire.expect("GET", "/predictors/churn", "NotFound", status=404)
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        checked(deployed=True),
        payload=asked(name, LOCK, deploy=True),
    )
    push_answers(wire, read=False, ready=True)
    code, _, stderr = run(
        "push",
        "--yes",
        "--env",
        "staging",
        str(churn),
        monkeypatch=monkeypatch,
        env=hosted,
    )
    assert code == 0, (stderr, wire.errors)
    assert "Uploading churn revision 1  rows.csv, 6 rows" in stderr
    assert "Context 6 rows, including the 2 gf eval held out" in stderr
    assert "Deployed churn revision 1 to staging" in stderr
    assert not wire.expected


def test_push_refuses_a_lock_the_service_no_longer_resolves(wire, churn):
    (churn / "gf.lock").write_text(json.dumps(LOCK))
    name, _ = uploaded(churn)
    moved = {**LOCK, "digest": "sha256:" + "e" * 64}
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/predictors/churn/check",
        checked(moved, deployed=True),
        payload=asked(name, LOCK, deploy=True),
    )
    with client(wire) as hosted, pytest.raises(gf.Error) as raised:
        hosted.predictors.push(churn, env="staging")
    assert raised.value.code == "LockRequired"
    assert raised.value.hint == "run `gf check` again"
    assert [path.rsplit("/", 1)[-1] for path, _ in wire.requests] == [
        "snapshots",
        "check",
    ]


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


def test_a_wide_table_carries_the_fewer_rows_its_models_cells_leave(wire):
    wide = {
        **MODEL,
        "limits": {"context_rows": 100, "queries": 100, "inputs": 512, "cells": 400},
    }
    limits = ModelInfo.from_reply(wide).limits
    assert (limits.cells, limits.rows(2), limits.rows(8)) == (400, 100, 50)
    assert ModelInfo.from_reply(MODEL).limits.cells is None
    wire.expect("GET", "/models/tabicl-v2-classifier", wide)
    rows = np.arange(60 * 8, dtype=float).reshape(60, 8)
    with client(wire) as hosted, pytest.raises(gf.CapacityExceeded) as raised:
        gf.Classifier(client=hosted).fit(rows, [0, 1] * 30)
    assert raised.value.message == "60 training rows exceed one request"
    assert "50 context rows per call at 8 inputs" in raised.value.because


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
    assert catalog.default("forecast") == "general-forecast/chronos-2"


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
    forecaster = {**MODEL, "name": "general-forecast/chronos-2", "task": "forecast"}
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
    assert sent_model == "general-forecast/chronos-2" and target == "units"
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
