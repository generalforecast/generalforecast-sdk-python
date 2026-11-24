# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import asyncio

import numpy as np
import pandas as pd
import pytest

import generalforecast as gf

LORA = {"type": "lora", "rank": 16}


def credit(rows=40):
    return pd.DataFrame(
        {
            "income": np.arange(rows, dtype=float),
            "defaulted": [i % 2 for i in range(rows)],
        }
    )


def snapshot(name="t", rows=40):
    return gf.snapshots.create(
        name, credit(rows), target="defaulted", task="classification"
    )


def test_stream_sends_a_small_snapshot_as_one_arrow_stream(fake):
    snap = gf.snapshots.stream(
        "credit-2026-01", credit(), target="defaulted", task="classification"
    )
    assert snap.state == "ready" and snap.rows == 40 and snap.task == "classification"
    assert [c.type for c in snap.columns] == ["number", "number"]
    assert snap.tags == ("uploaded",)
    sent = fake.calls_to("POST", r".*/snapshots")[0]
    assert sent.table is not None and sent.query == {
        "name": "credit-2026-01",
        "target": "defaulted",
        "task": "classification",
    }
    assert repr(snap) == (
        "Snapshot credit-2026-01  ready  40 rows  2 columns  "
        "classification, target defaulted  tags uploaded"
    )
    again = gf.snapshots.stream("credit-2026-01", credit())
    assert again.digest == snap.digest
    with pytest.raises(gf.Conflict):
        gf.snapshots.stream("credit-2026-01", credit(41))
    with pytest.raises(gf.InvalidInput):
        gf.snapshots.stream("x", credit(), target="defaulted")
    fake.limits["snapshot_rows"] = 10
    with gf.Client() as other, pytest.raises(gf.CapacityExceeded) as error:
        other.snapshots.stream("big", credit())
    assert error.value.hint == "gf.snapshots.create uploads it in chunks"


@pytest.mark.parametrize("signed", [True, False], ids=["signed-url", "bytes-route"])
def test_create_uploads_in_chunks(fake, signed):
    fake.chunk_bytes, fake.signed_uploads = 256, signed
    snap = gf.snapshots.create(
        "big",
        credit(),
        target="defaulted",
        task="classification",
        column_types={"income": "numerical"},
        time="income",
        tags=["x"],
    )
    assert snap.state == "ready" and snap.rows == 40
    opened = fake.calls_to("POST", r".*/snapshots")[0].payload
    assert opened["format"] == "parquet" and opened["tags"] == ["x"]
    assert opened["metadata"] == {
        "target": "defaulted",
        "task": "classification",
        "column_types": {"income": "numerical"},
        "time": "income",
    }
    if signed:
        # The bytes go to the upload URL the service gave, without the credential.
        chunks = fake.calls_to("PUT", r"/uploads/snapshots/big")
        assert all("authorization" not in c.headers for c in chunks)
    else:
        # A plane without object storage answers `url: null` and takes them itself.
        chunks = fake.calls_to("POST", r".*/snapshots/big/bytes")
        assert chunks[1].headers["content-range"].startswith("bytes 256-511/")
    assert len(chunks) > 1
    assert gf.snapshots.get(snap).name == "big"
    assert [s.name for s in gf.snapshots.list(tag="x")] == ["big"]
    assert gf.snapshots.update("big", tags=["a", "b"]).tags == ("a", "b")
    assert fake.calls_to("POST", r".*/snapshots/big/update")[0].payload == {
        "tags": ["a", "b"]
    }
    gf.snapshots.delete("big")
    assert gf.snapshots.get("big").state == "deleted"


def test_create_without_wait_returns_the_open_upload(fake):
    fake.snapshot_chunks = False
    snap = gf.snapshots.create(
        "t", credit(), target="defaulted", task="classification", wait=False
    )
    assert snap.state != "ready"
    assert not fake.calls_to("GET", r".*/snapshots/t")


def test_a_snapshot_upload_keeps_to_the_services_limits_and_plan(fake):
    fake.limits.update(chunk_bytes=300)
    fake.snapshot_chunks = False
    snapshot("big")
    chunks = fake.calls_to("PUT", r"/uploads/snapshots/big")
    assert [len(c.body) for c in chunks[:-1]] == [300] * (len(chunks) - 1)
    assert len(fake.calls_to("GET", r"/v1/auth/status")) == 1
    fake.limits["upload_rows"] = 39
    with gf.Client() as other, pytest.raises(gf.CapacityExceeded) as error:
        other.snapshots.create(
            "bigger", credit(), target="defaulted", task="classification"
        )
    assert (error.value.limit, error.value.actual) == (39, 40)
    assert "bigger" not in fake.training
    fake.plan = "free"
    with pytest.raises(gf.PaymentRequired) as error:
        snapshot("biggest")
    assert error.value.reason == "plan_required"
    with pytest.raises(gf.PaymentRequired):
        gf.snapshots.create_synthetic("warmup", rows=100)
    # A small table still streams on the Free plan.
    assert gf.snapshots.stream("small", credit(10)).rows == 10


def test_create_synthetic_checks_options_before_sending(fake):
    batch = gf.snapshots.create_synthetic(
        "credit-warmup",
        count=4,
        seed=0,
        task="classification",
        rows={"min": 1024, "max": 8192},
        missing={"rate": 0.05, "patterns": ["random", "censored"]},
        reject=["trivial", "unlearnable"],
        graph_nodes=16,
        columns=None,
    )
    assert batch.state == "pending" and batch.tag == "synthetic-credit-warmup"
    assert len(batch.tables) == 4
    assert batch.created and batch.origin.surface == "notebook"
    assert batch.url == f"{fake.url}/?project=acme#/fine-tunes"
    sent = fake.calls_to("POST", r".*/snapshots/synthetic")[0].payload
    assert sent["rows"] == {"min": 1024, "max": 8192} and sent["graph_nodes"] == 16
    assert sent["missing"] == {"rate": 0.05, "patterns": ["random", "censored"]}
    assert "columns" not in sent
    gf.snapshots.synthetic("credit-warmup")
    done = gf.snapshots.synthetic(batch)
    assert done.state == "done" and done.counts["ready"] == 4
    assert repr(done).startswith(
        "SyntheticBatch credit-warmup  done  4 tables  4 ready"
    )
    calls = len(fake.calls)
    with pytest.raises(gf.InvalidInput) as error:
        gf.snapshots.create_synthetic("x", rows=10)
    assert error.value.limit == 50_000
    with pytest.raises(gf.InvalidInput):
        gf.snapshots.create_synthetic("x", rows={"min": 9000, "max": 1024})
    with pytest.raises(gf.InvalidInput) as error:
        gf.snapshots.create_synthetic("x", graph_node=3)
    assert error.value.hint == "a similar name exists: `graph_nodes`"
    with pytest.raises(gf.InvalidInput):
        gf.snapshots.create_synthetic("x", categorical={"shares": 0.2})
    assert len(fake.calls) == calls


def test_a_run_takes_snapshots_tags_and_contexts(fake):
    snapshot("credit-2026-01")
    batch = gf.snapshots.create_synthetic("warmup", count=2, task="classification")
    gf.contexts.create("adult", data=credit())
    run = gf.runs.create(
        "credit-default",
        base="tabicl-v2",
        method=LORA,
        data=[
            "credit-2026-01",
            {"tag": batch.tag},
            {"context": "adult"},
            {"context": "adult", "version": 1},
        ],
        target="defaulted",
        plan={"steps": 1000, "learning_rate": 5e-5, "episode_sampling": {"rows": 1024}},
        evaluation={"holdout": 0.1},
        budget={"credits": 500},
        tags=["nightly"],
    )
    assert run.model == "acme/credit-default@ft1" and not hasattr(run, "version")
    assert run.state == "queued"
    sent = fake.calls_to("POST", r".*/runs")[0].payload
    assert sent == {
        "name": "credit-default",
        "base": "tabicl-v2",
        "method": LORA,
        "data": [
            "credit-2026-01",
            {"tag": "synthetic-warmup"},
            {"context": "adult"},
            {"context": "adult", "version": 1},
        ],
        "plan": {
            "steps": 1000,
            "learning_rate": 5e-5,
            "episode_sampling": {"rows": 1024},
        },
        "evaluation": {"holdout": 0.1},
        "budget": {"credits": 500},
        "tags": ["nightly"],
        "target": "defaulted",
    }
    assert run.data[2].kind == "context" and run.data[2].name == "adult"
    assert run.data[2].version == 1 and run.data[0].kind == "snapshot"
    assert run.target == "defaulted"
    # The context lists the fine-tuned model made from it.
    assert gf.contexts.get("adult").fine_tunes == ("acme/credit-default@ft1",)


def test_a_run_on_a_context_names_its_target(fake):
    gf.contexts.create("adult", data=credit())
    with pytest.raises(gf.InvalidInput) as error:
        gf.runs.create(
            "credit-default",
            base="tabicl-v2",
            method=LORA,
            data=[{"context": "adult"}],
            plan={"steps": 10},
        )
    assert error.value.param == "target"
    assert not fake.calls_to("POST", r".*/runs")
    with pytest.raises(gf.InvalidInput) as error:
        gf.runs.create(
            "credit-default",
            base="tabicl-v2",
            method=LORA,
            data=[{"context": "adult"}],
            target="incme",
            plan={"steps": 10},
        )
    assert error.value.param == "target" and "incme" in error.value.message


def test_wait_passes_each_event_once_and_returns_the_published_version(fake):
    snapshot()
    run = gf.runs.create(
        "credit-default",
        base="tabicl-v2",
        method=LORA,
        data=["t"],
        plan={"steps": 1000},
    )
    seen = []
    run = gf.runs.wait(run, on_event=seen.append)
    assert run.state == "finished" and run.published.kept_step == 1000
    assert run.published.report.delta == pytest.approx(0.023)
    assert [e.kind for e in seen] == [e.kind for e in gf.runs.lineage(run).events]
    assert fake.slept[-1] == 5.0
    # Once the attempt ends its seconds are charged.
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
    second = gf.runs.create(
        "credit-default", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10}
    )
    assert second.model.endswith("@ft2")
    assert gf.runs.get("credit-default").model == "acme/credit-default@ft2"
    assert gf.runs.get("acme/credit-default@ft1").state == "finished"
    assert [r.model for r in gf.runs.list()] == [
        "acme/credit-default@ft1",
        "acme/credit-default@ft2",
    ]
    assert fake.calls_to("GET", r".*/runs")[-1].query == {}
    found = gf.runs.list(state="queued", data_digest=second.data_digest, tag="x")
    assert fake.calls_to("GET", r".*/runs")[-1].query == {
        "state": "queued",
        "data_digest": second.data_digest,
        "tag": "x",
    }
    assert found == []
    with pytest.raises(gf.Conflict):
        gf.runs.create(
            "credit-default", base="mitra-2", method=LORA, data=["t"], plan={"steps": 1}
        )


def test_watch_yields_each_event_once_and_a_failed_run_raises(fake):
    snapshot()
    gf.runs.create("a", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10})
    kinds = [e.kind for e in gf.runs.watch("a@ft1", poll=0.5)]
    assert kinds == [
        "created", "state", "attempt", "state", "calibration",
        "metrics", "metrics", "published", "state",
    ]  # fmt: skip
    lineage = gf.runs.lineage("a")
    assert lineage.model == "acme/a@ft1" and [e.kind for e in lineage.events] == kinds
    fake.run_ends = "failed"
    gf.runs.create("b", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10})
    with pytest.raises(gf.Error) as error:
        gf.runs.wait("b")
    assert error.value.code == "RunFailed" and not error.value.recoverable
    assert error.value.message == "run b@ft1 failed"
    assert error.value.note == "the trainer ran out of memory"


def test_a_wait_past_its_timeout_raises_and_the_run_carries_on(fake):
    snapshot()
    gf.runs.create("a", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10})
    with pytest.raises(gf.Timeout) as error:
        gf.runs.wait("a", timeout=0)
    assert error.value.hint == "gf.runs.wait('a@ft1') picks it up again"


def test_cancel_with_save_publishes(fake):
    snapshot()
    gf.runs.create("a", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10})
    run = gf.runs.cancel("a", save=True)
    assert run.state == "cancelled" and run.published.kept_step == 10
    assert fake.calls_to("POST", r".*/runs/a/cancel")[0].payload == {"save": True}


def test_run_arguments_fail_locally(fake):
    calls = len(fake.calls)
    with pytest.raises(gf.InvalidInput):
        gf.runs.create("a", base="tabicl-v2", method=LORA)
    with pytest.raises(gf.InvalidInput):
        gf.runs.create("a", base="tabicl-v2", method=LORA, data=["t"])
    with pytest.raises(gf.InvalidInput):
        gf.runs.create(
            "a",
            base="tabicl-v2",
            method=LORA,
            data=["t"],
            phases=[{"name": "p", "data": ["t"], "plan": {"steps": 1}}],
            plan={"steps": 1},
        )
    with pytest.raises(gf.InvalidInput):
        gf.runs.create(
            "a", base="tabicl-v2", method="lora", data=["t"], plan={"steps": 1}
        )
    with pytest.raises(gf.InvalidInput):
        gf.runs.create(
            "a", base="tabicl-v2", method=LORA, data=[{"tags": "x"}], plan={"steps": 1}
        )
    with pytest.raises(TypeError):
        gf.runs.create("a", base="tabicl-v2", data=["t"], plan={"steps": 1})
    with pytest.raises(gf.InvalidInput):
        gf.runs.get("Not A Run")
    assert len(fake.calls) == calls
    with pytest.raises(gf.NotFound):
        gf.runs.create(
            "a", base="tabicl-v2", method=LORA, data=["nothing"], plan={"steps": 1}
        )


def test_a_curriculum_and_architecture_changes(fake):
    snapshot("s1")
    phases = [
        {
            "name": "signal",
            "data": ["s1"],
            "plan": {"steps": 400, "episode_sampling": {"rows": 1024}},
        },
        {"name": "mixed", "data": ["s1"], "plan": {"steps": 600}},
    ]
    architecture = [
        {
            "component": "attention.query_scale",
            "stages": ["icl"],
            "to": {"type": "gated_log_scale"},
        }
    ]
    gf.runs.create(
        "c",
        base="tabicl-v2",
        method={"type": "lora", "rank": 8, "input_adapter": True},
        phases=phases,
        plan={"learning_rate": 1e-4},
        architecture=architecture,
    )
    sent = fake.calls_to("POST", r".*/runs")[0].payload
    assert sent["phases"] == phases
    assert sent["method"] == {"type": "lora", "rank": 8, "input_adapter": True}
    assert sent["architecture"] == architecture
    assert sent["plan"] == {"learning_rate": 1e-4} and "data" not in sent


def test_the_async_client_fine_tunes_too(fake):
    async def go():
        async with gf.AsyncClient() as client:
            await client.snapshots.create(
                "t", credit(), target="defaulted", task="classification"
            )
            run = await client.runs.create(
                "a", base="tabicl-v2", method=LORA, data=["t"], plan={"steps": 10}
            )
            seen = []

            async def on_event(event):
                seen.append(event.kind)

            run = await client.runs.wait(run, on_event=on_event)
            events = [e async for e in client.runs.watch(run)]
            lineage = await client.runs.lineage(run)
            return run, events, seen, lineage

    run, events, seen, lineage = asyncio.run(go())
    assert run.state == "finished" and events[-2].kind == "published"
    assert seen == [e.kind for e in events] == [e.kind for e in lineage.events]


def test_a_run_names_the_model_to_pass_and_its_short_name_resolves(fake):
    """A run's `version`, 1, read like a model; `credit-default@ft1` was refused."""
    snapshot("credit-2026-01")
    run = gf.runs.create(
        "credit-default",
        base="tabicl-v2",
        method=LORA,
        data=["credit-2026-01"],
        plan={"steps": 10},
    )
    assert run.model == "acme/credit-default@ft1"
    queries = pd.DataFrame({"income": [1.0]})
    context = pd.DataFrame({"income": [1.0, 2.0], "defaulted": [0, 1]})
    gf.models.predict(
        "credit-default@ft1", queries, context=context, target="defaulted"
    )
    assert fake.last_prediction["model"] == run.model
