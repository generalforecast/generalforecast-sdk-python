# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import asyncio

import numpy as np
import pandas as pd
import pytest

import generalforecast as gf
from generalforecast import _ops


def credit(rows=40):
    return pd.DataFrame(
        {
            "income": np.arange(rows, dtype=float),
            "defaulted": [i % 2 for i in range(rows)],
        }
    )


def test_a_small_snapshot_goes_as_one_arrow_stream(fake):
    snap = gf.fine_tunes.snapshots.create(
        "credit-2026-01",
        data=credit(),
        tags=["credit"],
        target="defaulted",
        task="classification",
    )
    assert snap.state == "ready" and snap.rows == 40 and snap.task == "classification"
    assert [c.type for c in snap.columns] == ["number", "number"]
    assert snap.tags == ("credit", "uploaded")
    sent = fake.calls_to("POST", r".*/snapshots")[0]
    assert sent.table is not None and sent.query == {
        "name": "credit-2026-01",
        "target": "defaulted",
        "task": "classification",
    }
    assert repr(snap) == (
        "Snapshot credit-2026-01  ready  40 rows  2 columns  "
        "classification, target defaulted  tags credit, uploaded"
    )
    again = gf.fine_tunes.snapshots.create("credit-2026-01", data=credit())
    assert again.digest == snap.digest
    with pytest.raises(gf.Conflict):
        gf.fine_tunes.snapshots.create("credit-2026-01", data=credit(41))
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.snapshots.create("x", data=credit(), target="defaulted")


@pytest.mark.parametrize("signed", [True, False], ids=["signed-url", "bytes-route"])
def test_a_large_snapshot_uploads_in_chunks(fake, monkeypatch, signed):
    monkeypatch.setattr(_ops, "STREAM_ROWS", 10)
    fake.chunk_bytes, fake.signed_uploads = 256, signed
    snap = gf.fine_tunes.snapshots.create(
        "big", data=credit(), target="defaulted", task="classification", tags=["x"]
    )
    assert snap.state == "ready" and snap.rows == 40
    opened = fake.calls_to("POST", r".*/snapshots")[0].payload
    assert opened["format"] == "parquet" and opened["metadata"]["target"] == "defaulted"
    if signed:
        # The bytes go to the upload URL the service gave, without the credential.
        chunks = fake.calls_to("PUT", r"/uploads/snapshots/big")
        assert all("authorization" not in c.headers for c in chunks)
    else:
        # A plane without object storage answers `url: null` and takes them itself.
        chunks = fake.calls_to("POST", r".*/snapshots/big/bytes")
        assert chunks[1].headers["content-range"].startswith("bytes 256-511/")
    assert len(chunks) > 1
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.snapshots.create("bigger", data=credit())
    assert gf.fine_tunes.snapshots.get(snap).name == "big"
    assert [s.name for s in gf.fine_tunes.snapshots.list(tag="x")] == ["big"]
    assert gf.fine_tunes.snapshots.tag("big", ["a", "b"]).tags == ("a", "b")
    gf.fine_tunes.snapshots.delete("big")


def test_synthesize_checks_settings_before_sending(fake):
    batch = gf.fine_tunes.snapshots.synthesize(
        "credit-warmup",
        count=4,
        seed=0,
        task="classification",
        rows=gf.Range(1024, 8192),
        missing=gf.Missing(0.05, ["random", "censored"]),
        reject=["trivial", "unlearnable"],
        graph_nodes=16,
    )
    assert batch.state == "pending" and batch.tag == "synthetic-credit-warmup"
    assert len(batch.tables) == 4
    sent = fake.calls_to("POST", r".*/snapshots/synthetic")[0].payload
    assert sent["rows"] == {"min": 1024, "max": 8192} and sent["graph_nodes"] == 16
    assert sent["missing"] == {"rate": 0.05, "patterns": ["random", "censored"]}
    gf.fine_tunes.snapshots.batch("credit-warmup")
    done = gf.fine_tunes.snapshots.batch(batch)
    assert done.state == "done" and done.counts["ready"] == 4
    assert repr(done).startswith(
        "SyntheticBatch credit-warmup  done  4 tables  4 ready"
    )
    calls = len(fake.calls)
    with pytest.raises(gf.InvalidInput) as error:
        gf.fine_tunes.snapshots.synthesize("x", rows=10)
    assert error.value.limit == 50_000
    with pytest.raises(gf.InvalidInput) as error:
        gf.fine_tunes.snapshots.synthesize("x", graph_node=3)
    assert error.value.hint == "a similar name exists: `graph_nodes`"
    assert len(fake.calls) == calls


def test_a_run_takes_snapshots_batches_contexts_and_tags(fake):
    snap = gf.fine_tunes.snapshots.create(
        "credit-2026-01", data=credit(), target="defaulted", task="classification"
    )
    gf.fine_tunes.snapshots.synthesize("warmup", count=2, task="classification")
    gf.contexts.create("adult", data=credit())
    run = gf.fine_tunes.runs.create(
        "credit-default",
        base="tabicl-v2",
        data=[snap, "warmup", "adult", "adult@1", gf.Tag("credit")],
        target="defaulted",
        plan=gf.Plan(steps=1000, learning_rate=5e-5, episode_rows=1024),
        evaluation=gf.Evaluation(holdout=0.1),
        budget=gf.Budget(credits=500),
        tags=["nightly"],
    )
    assert run.model == "acme/credit-default@ft1" and run.version == 1
    assert run.state == "queued"
    sent = fake.calls_to("POST", r".*/runs")[0].payload
    assert sent["data"] == [
        "credit-2026-01",
        {"tag": "synthetic-warmup"},
        {"context": "adult"},
        {"context": "adult", "version": 1},
        {"tag": "credit"},
    ]
    assert sent["method"] == {"type": "lora", "rank": 16}
    assert sent["plan"] == {
        "steps": 1000,
        "learning_rate": 5e-5,
        "episode_sampling": {"rows": 1024},
    }
    assert sent["evaluation"]["early_stopping_threshold"] == {"standard_errors": 2.0}
    assert run.data[2].kind == "context" and run.data[2].name == "adult"
    assert run.data[2].version == 1 and run.data[0].kind == "snapshot"
    assert sent["target"] == "defaulted" and run.target == "defaulted"


def test_a_run_on_a_context_names_its_target(fake):
    gf.contexts.create("adult", data=credit())
    with pytest.raises(gf.InvalidInput) as error:
        gf.fine_tunes.runs.create(
            "credit-default", base="tabicl-v2", data=["adult"], plan=gf.Plan(steps=10)
        )
    assert error.value.param == "target"
    assert not fake.calls_to("POST", r".*/runs")
    with pytest.raises(gf.InvalidInput) as error:
        gf.fine_tunes.runs.create(
            "credit-default",
            base="tabicl-v2",
            data=["adult"],
            target="incme",
            plan=gf.Plan(steps=10),
        )
    assert error.value.param == "target" and "incme" in error.value.message


def test_wait_reports_progress_and_the_published_version(fake):
    gf.fine_tunes.snapshots.create(
        "t", data=credit(), target="defaulted", task="classification"
    )
    seen = []
    run = gf.fine_tunes.runs.create(
        "credit-default",
        base="tabicl-v2",
        data=["t"],
        plan=gf.Plan(steps=1000),
        wait=True,
        progress=seen.append,
    )
    assert run.state == "finished" and run.published.kept_step == 1000
    assert run.published.report.delta == pytest.approx(0.023)
    assert [r.state for r in seen] == ["running", "finished"]
    assert seen[0].step == 500 and seen[0].loss == pytest.approx(0.45)
    # While the attempt runs its seconds are pending; once it ends, charged.
    assert seen[0].attempt == 1 and seen[0].seconds == 60.0
    assert (seen[0].pending_seconds, seen[0].charged_seconds) == (60.0, 0.0)
    assert (run.charged_seconds, run.pending_seconds, run.voided_seconds) == (
        130.0,
        0.0,
        0.0,
    )
    assert run.curve == ((500, 0.45, 0.43), (1000, 0.41, 0.398))
    assert repr(run) == (
        "credit-default@ft1  finished  step 1,000/1,000  loss 0.410  "
        "val 0.398 (base 0.421)  2m 10s  260.0 credits"
    )
    assert "<polyline" in run._repr_html_()
    second = gf.fine_tunes.runs.create(
        "credit-default", base="tabicl-v2", data=["t"], plan=gf.Plan(steps=10)
    )
    assert second.model.endswith("@ft2")
    assert gf.fine_tunes.runs.get("credit-default").version == 2
    assert gf.fine_tunes.runs.get("acme/credit-default@ft1").state == "finished"
    assert [r.version for r in gf.fine_tunes.runs.list()] == [1, 2]
    assert fake.calls_to("GET", r".*/runs")[-1].query == {}
    with pytest.raises(gf.Conflict):
        gf.fine_tunes.runs.create(
            "credit-default", base="mitra-2", data=["t"], plan=gf.Plan(steps=1)
        )


def test_watch_yields_each_event_once_and_a_failed_run_raises(fake):
    gf.fine_tunes.snapshots.create(
        "t", data=credit(), target="defaulted", task="classification"
    )
    gf.fine_tunes.runs.create("a", base="tabicl-v2", data=["t"], plan=gf.Plan(steps=10))
    kinds = [e.kind for e in gf.fine_tunes.runs.watch("a@ft1", poll=0.5)]
    assert kinds == [
        "created", "state", "attempt", "state", "calibration",
        "metrics", "metrics", "published", "state",
    ]  # fmt: skip
    assert [e.kind for e in gf.fine_tunes.runs.events("a")] == kinds
    fake.run_ends = "failed"
    gf.fine_tunes.runs.create("b", base="tabicl-v2", data=["t"], plan=gf.Plan(steps=10))
    with pytest.raises(gf.Error) as error:
        gf.fine_tunes.runs.wait("b")
    assert error.value.code == "RunFailed" and not error.value.recoverable
    assert error.value.message == "run b@ft1 failed"
    assert error.value.note == "the trainer ran out of memory"


def test_cancel_with_save_publishes(fake):
    gf.fine_tunes.snapshots.create(
        "t", data=credit(), target="defaulted", task="classification"
    )
    gf.fine_tunes.runs.create("a", base="tabicl-v2", data=["t"], plan=gf.Plan(steps=10))
    run = gf.fine_tunes.runs.cancel("a", save=True)
    assert run.state == "cancelled" and run.published.kept_step == 10
    assert fake.calls_to("POST", r".*/runs/a/cancel")[0].payload == {"save": True}


def test_run_arguments_fail_locally(fake):
    calls = len(fake.calls)
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.runs.create("a", base="tabicl-v2")
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.runs.create("a", base="tabicl-v2", data=["t"])
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.runs.create(
            "a", base="tabicl-v2", data=["t"], method={"type": "lora"}
        )
    with pytest.raises(TypeError):
        gf.Plan(stpes=10)
    with pytest.raises(gf.InvalidInput):
        gf.Full(components=["encoder"])
    with pytest.raises(gf.InvalidInput):
        gf.fine_tunes.runs.get("Not A Run")
    assert len(fake.calls) == calls
    with pytest.raises(gf.NotFound):
        gf.fine_tunes.runs.create(
            "a", base="tabicl-v2", data=["nothing"], plan=gf.Plan(steps=1)
        )


def test_a_curriculum_and_architecture_changes(fake):
    gf.fine_tunes.snapshots.create(
        "s1", data=credit(), target="defaulted", task="classification"
    )
    gf.fine_tunes.runs.create(
        "c",
        base="tabicl-v2",
        method=gf.LoRA(rank=8, input_adapter=True),
        phases=[
            gf.Phase("signal", ["s1"], 400, episode_rows=1024),
            gf.Phase("mixed", ["s1"], 600),
        ],
        plan=gf.Plan(learning_rate=1e-4),
        architecture=[
            gf.ArchitectureChange(
                "attention.query_scale", {"type": "gated_log_scale"}, stages=["icl"]
            )
        ],
    )
    sent = fake.calls_to("POST", r".*/runs")[0].payload
    assert sent["phases"][0] == {
        "name": "signal",
        "data": ["s1"],
        "plan": {"steps": 400, "episode_sampling": {"rows": 1024}},
    }
    assert sent["method"] == {"type": "lora", "rank": 8, "input_adapter": True}
    assert sent["architecture"][0]["stages"] == ["icl"]
    assert "data" not in sent


def test_the_async_client_fine_tunes_too(fake):
    async def go():
        async with gf.AsyncClient() as client:
            await client.fine_tunes.snapshots.create(
                "t", data=credit(), target="defaulted", task="classification"
            )
            run = await client.fine_tunes.runs.create(
                "a", base="tabicl-v2", data=["t"], plan=gf.Plan(steps=10), wait=True
            )
            events = [e async for e in client.fine_tunes.runs.watch(run)]
            return run, events

    run, events = asyncio.run(go())
    assert run.state == "finished" and events[-2].kind == "published"
