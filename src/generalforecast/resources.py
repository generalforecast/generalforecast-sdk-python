# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The resources of a client: contexts, experiments, predictors, fine-tunes, sources, keys, models.

Each method takes a name or the object for every reference and returns
typed dataclasses. The work is in `generalforecast._ops`; the `Async`
classes await the same operations.
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from generalforecast import _ops
from generalforecast._ops import UNSET
from generalforecast.types import (
    STANDARD_PERMISSIONS,
    ArchitectureChange,
    Budget,
    ColumnType,
    Context,
    Deploy,
    Environment,
    Estimate,
    Evaluation,
    Experiment,
    FineTuneRun,
    Holdout,
    Key,
    Logs,
    LoRA,
    Method,
    Missing,
    Model,
    NewKey,
    Phase,
    Plan,
    Predictions,
    Predictor,
    PredictorInfo,
    Range,
    Revision,
    RunEvent,
    Slice,
    Snapshot,
    Source,
    SourceKind,
    SourceTable,
    SyntheticBatch,
    Tag,
    Task,
    WriteReceipt,
)

if TYPE_CHECKING:
    import os

    import pandas
    import polars  # type: ignore
    import pyarrow

    from generalforecast.client import AsyncClient, Client

    DataLike = (
        pandas.DataFrame | polars.DataFrame | pyarrow.Table | str | os.PathLike[str]
    )

__all__ = [
    "AsyncContexts",
    "AsyncExperiments",
    "AsyncFineTunes",
    "AsyncKeys",
    "AsyncModels",
    "AsyncPredictors",
    "AsyncRuns",
    "AsyncSnapshots",
    "AsyncSources",
    "Contexts",
    "Experiments",
    "FineTunes",
    "Keys",
    "Models",
    "Predictors",
    "Runs",
    "Snapshots",
    "Sources",
]


DEFAULT_METHOD = LoRA(rank=16)
"""A run's method when none is given: LoRA of rank 16."""


def _setup(**values: Any) -> dict[str, Any]:
    return values


class _Sync:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _run(self, op: _ops.Op) -> Any:
        return self._client._run(op)


class _Async:
    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def _run(self, op: _ops.Op) -> Any:
        return await self._client._run(op)


class Contexts(_Sync):
    """Contexts: the data models learn from, uploaded, imported from a source, or live."""

    def create(
        self,
        name: str,
        *,
        data: DataLike | None = None,
        source: str | Source | None = None,
        table: str | None = None,
        sql: str | None = None,
        live: bool = False,
        columns: Mapping[str, ColumnType] | None = None,
        targets: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        freshness: str | timedelta = "15m",
        wait: bool = True,
    ) -> Context:
        """Creates a context from exactly one origin.

        `data=` uploads a snapshot, version 1. `source=` with `table=` or
        `sql=` imports one; with `live=True` it is read at call time and
        cached for `freshness` (1m to 24h). `live=True` with `columns=`
        (or `data=` to infer them and seed rows) makes a context fed
        through `append`, with `label` filling `targets` by `id`.

        Raises:
            InvalidInput: For no origin, two, or arguments the origin does not take.
        """
        return self._run(
            _ops.context_create(
                name,
                data=data,
                source=source,
                table=table,
                sql=sql,
                live=live,
                columns=columns,
                targets=targets,
                id=id,
                time=time,
                freshness=freshness,
                wait=wait,
            )
        )

    def upload(
        self, context: str | Context, data: DataLike, *, wait: bool = True
    ) -> Context:
        """Uploads a new version, creating the context when it has none yet.

        Rows whose digest matches the latest version add none (`unchanged`),
        so CI and scripts can call it on every run.
        """
        return self._run(_ops.context_upload(context, data, wait=wait))

    def refresh(self, context: str | Context, *, wait: bool = True) -> Context:
        """Imports a new version from the context's source, when its rows changed."""
        return self._run(_ops.context_refresh(context, wait=wait))

    def append(
        self,
        context: str | Context,
        data: DataLike,
        *,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """Appends rows to a live context fed through the API."""
        return self._run(
            _ops.context_append(context, data, idempotency_key=idempotency_key)
        )

    def label(
        self,
        context: str | Context,
        data: DataLike,
        *,
        target: str,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """Fills `target` for rows already in a live context, matched by its `id` column."""
        return self._run(
            _ops.context_label(
                context, data, target=target, idempotency_key=idempotency_key
            )
        )

    def get(self, context: str | Context, *, version: int | None = None) -> Context:
        """A context, its latest version or the one pinned (`"adult@2"` or `version=2`)."""
        return self._run(_ops.context_get(context, version=version))

    def list(self) -> builtins.list[Context]:
        """The project's contexts."""
        return self._run(_ops.context_list())

    def rows(
        self, context: str | Context, *, limit: int = 20, version: int | None = None
    ) -> pandas.DataFrame:
        """The first `limit` rows of a context."""
        return self._run(_ops.context_rows(context, limit=limit, version=version))

    def delete(self, context: str | Context) -> None:
        """Deletes a context; raises `Conflict` while a predictor serves it."""
        self._run(_ops.context_delete(context))


class Experiments(_Sync):
    """Experiments: models evaluated on one context for one target, run by run."""

    def create(
        self,
        *,
        context: str | Context,
        target: str,
        name: str | None = None,
        title: str | None = None,
        task: Task | None = None,
        id: str | None = None,
        time: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        types: Mapping[str, ColumnType] | None = None,
        holdout: str | Holdout | None = None,
        sql: str | None = None,
        models: Sequence[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """Creates an experiment and starts run 1; with `wait`, returns once it ends.

        `name` defaults to `<context>-<target>`; `task` is inferred from the
        target; `models=None` runs the recommended set. `holdout` left out is
        the last 20% by `time` when one is declared, else a random 20%.

        Raises:
            Error: Not recoverable, when the run fails.
            Timeout: When the run is still going after `timeout` seconds.
        """
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return self._run(_ops.experiment_create(values, models, wait, timeout, poll))

    def estimate(
        self,
        *,
        context: str | Context,
        target: str,
        name: str | None = None,
        title: str | None = None,
        task: Task | None = None,
        id: str | None = None,
        time: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        types: Mapping[str, ColumnType] | None = None,
        holdout: str | Holdout | None = None,
        sql: str | None = None,
        models: Sequence[str] | None = None,
    ) -> Estimate:
        """What `create` with the same arguments would cost, without starting it."""
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return self._run(_ops.experiment_estimate(values, models))

    def run(
        self,
        experiment: str | Experiment,
        *,
        models: Sequence[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """Starts another run with the same setup."""
        return self._run(_ops.experiment_run(experiment, models, wait, timeout, poll))

    def update(
        self,
        experiment: str | Experiment,
        *,
        title: Any = UNSET,
        target: Any = UNSET,
        task: Any = UNSET,
        id: Any = UNSET,
        time: Any = UNSET,
        inputs: Any = UNSET,
        exclude: Any = UNSET,
        types: Any = UNSET,
        holdout: Any = UNSET,
        sql: Any = UNSET,
        context: Any = UNSET,
    ) -> Experiment:
        """Changes only the fields given; earlier runs become stale."""
        values = _setup(
            title=title, target=target, task=task, id=id, time=time, inputs=inputs,
            exclude=exclude, types=types, holdout=holdout, sql=sql, context=context,
        )  # fmt: skip
        return self._run(_ops.experiment_update(experiment, values))

    def get(
        self, experiment: str | Experiment, *, run: int | None = None
    ) -> Experiment:
        """An experiment with its latest run, or run `run`."""
        return self._run(_ops.experiment_get(experiment, run=run))

    def list(
        self, *, context: str | Context | None = None
    ) -> builtins.list[Experiment]:
        """The project's experiments, or those on one context."""
        return self._run(_ops.experiment_list(context))

    def wait(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """Waits for a run to end and returns the experiment with it.

        `poll` is the seconds between checks, as in every method that
        waits; left out, checks start 1s apart and slow to one every 5s.
        """
        return self._run(
            _ops.experiment_wait(experiment, run=run, timeout=timeout, poll=poll)
        )

    def cancel(
        self, experiment: str | Experiment, *, run: int | None = None
    ) -> Experiment:
        """Cancels a run, the latest unless `run` is given."""
        return self._run(_ops.experiment_cancel(experiment, run=run))

    def predictions(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        model: str | None = None,
    ) -> pandas.DataFrame:
        """A run's predictions on the held-out rows."""
        return self._run(_ops.experiment_predictions(experiment, run=run, model=model))

    def slices(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        model: str | None = None,
    ) -> builtins.list[Slice]:
        """A run's error by slice."""
        return self._run(_ops.experiment_slices(experiment, run=run, model=model))

    def delete(self, experiment: str | Experiment) -> None:
        """Deletes an experiment and its runs."""
        self._run(_ops.experiment_delete(experiment))


class Predictors(_Sync):
    """Predictors: named, versioned endpoints serving a context."""

    def create(
        self,
        name: str,
        *,
        env: str = "production",
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
        **spec_fields: Any,
    ) -> PredictorInfo:
        """Creates a predictor from `gf.Predictor`'s fields and deploys it to `env`.

        Raises:
            Conflict: When a predictor of that name exists.
            Error: Not recoverable, when the deploy fails to start.
        """
        return self._run(
            _ops.predictor_create(name, env, wait, timeout, spec_fields, poll)
        )

    def update(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
        **changed_fields: Any,
    ) -> PredictorInfo:
        """Makes a revision of the given fields over the serving spec (`None` clears one) and deploys it."""
        return self._run(
            _ops.predictor_update(predictor, env, wait, timeout, changed_fields, poll)
        )

    def deploy(
        self,
        *specs: Predictor,
        env: str = "production",
        dry_run: bool = False,
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
    ) -> builtins.list[Deploy]:
        """Makes `env` serve each spec: creates, revises, or leaves it as it is.

        It compares resolved values, so a short model name equals its full
        id. Predictors absent from `specs` are never deleted. `dry_run`
        returns the plan only.
        """
        return self._run(
            _ops.predictor_deploy(specs, env, dry_run, wait, timeout, poll)
        )

    def get(self, predictor: str | PredictorInfo) -> PredictorInfo:
        """A predictor."""
        return self._run(_ops.predictor_get(predictor))

    def list(self) -> builtins.list[PredictorInfo]:
        """The project's predictors."""
        return self._run(_ops.predictor_list())

    def revisions(self, predictor: str | PredictorInfo) -> builtins.list[Revision]:
        """A predictor's revisions."""
        return self._run(_ops.predictor_revisions(predictor))

    def predict(
        self,
        predictor: str | PredictorInfo,
        queries: DataLike,
        *,
        env: str = "production",
        as_of: datetime | str | None = None,
    ) -> Predictions:
        """Scores queries; they go as Arrow, in parts by bytes."""
        return self._run(
            _ops.predictor_predict(predictor, queries, env=env, as_of=as_of)
        )

    def promote(
        self,
        predictor: str | PredictorInfo,
        *,
        from_env: str = "staging",
        env: str = "production",
        revision: int | None = None,
        wait: bool = True,
        poll: float | None = None,
    ) -> Environment:
        """Points `env` at what `from_env` serves, or at `revision`."""
        payload = (
            {"revision": revision} if revision is not None else {"from_env": from_env}
        )
        return self._run(
            _ops.predictor_move(predictor, env, "promote", payload, wait, poll)
        )

    def rollback(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        revision: int | None = None,
        wait: bool = True,
        poll: float | None = None,
    ) -> Environment:
        """Points `env` back at what it served before, or at `revision`."""
        payload = {"revision": revision} if revision is not None else {}
        return self._run(
            _ops.predictor_move(predictor, env, "rollback", payload, wait, poll)
        )

    def logs(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        limit: int = 100,
        level: Literal["info", "warn", "error"] | None = None,
        q: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        after: str | None = None,
    ) -> Logs:
        """A page of an environment's logs; `after=logs.next` reads the next."""
        params = {
            "limit": limit,
            "level": level,
            "q": q,
            "since": since,
            "until": until,
            "after": after,
        }
        return self._run(_ops.predictor_logs(predictor, env, params))

    def delete(self, predictor: str | PredictorInfo) -> None:
        """Deletes a predictor and its revisions."""
        self._run(_ops.predictor_delete(predictor))


class Sources(_Sync):
    """Sources: workspace connections to warehouses and buckets."""

    def list(self) -> builtins.list[Source]:
        """The workspace's sources."""
        return self._run(_ops.source_list())

    def get(self, source: str | Source) -> Source:
        """A source."""
        return self._run(_ops.source_get(source))

    def create(
        self, name: str, *, kind: SourceKind, secret: str | None = None, **config: str
    ) -> Source:
        """Connects a source; `config` takes the keys its kind reads (section 3.8)."""
        return self._run(_ops.source_create(name, kind, secret, config))

    def test(self, source: str | Source) -> Source:
        """Tests a source's connection now."""
        return self._run(_ops.source_test(source))

    def tables(
        self, source: str | Source, *, prefix: str | None = None
    ) -> builtins.list[SourceTable]:
        """A source's tables."""
        return self._run(_ops.source_tables(source, prefix))

    def preview(
        self, source: str | Source, *, table: str | None = None, sql: str | None = None
    ) -> pandas.DataFrame:
        """Twenty rows of a table or a query."""
        return self._run(_ops.source_preview(source, table, sql))

    def delete(self, source: str | Source) -> None:
        """Deletes a source."""
        self._run(_ops.source_delete(source))


class Keys(_Sync):
    """API keys: by id, by name, or the `Key` itself."""

    def list(self) -> builtins.list[Key]:
        """The project's keys."""
        return self._run(_ops.key_list())

    def create(
        self,
        name: str,
        *,
        permissions: Sequence[str] = STANDARD_PERMISSIONS,
        expires: str | timedelta | None = "90d",
    ) -> NewKey:
        """Creates a key; its token is shown once. `expires=None` never expires."""
        return self._run(_ops.key_create(name, permissions, expires))

    def rotate(self, key: str | Key) -> NewKey:
        """Replaces a key's token; the old one stops working now."""
        return self._run(_ops.key_action(key, "rotate"))

    def revoke(self, key: str | Key) -> Key:
        """Revokes a key."""
        return self._run(_ops.key_action(key, "revoke"))

    def delete(self, key: str | Key) -> None:
        """Deletes an expired or revoked key."""
        self._run(_ops.key_action(key, "delete"))


class Models(_Sync):
    """Models: the catalog and the project's published fine-tune versions."""

    def list(
        self, *, task: Task | None = None, all: bool = False
    ) -> builtins.list[Model]:
        """Hostable models, or every model with `all`."""
        return self._run(_ops.model_list(task, all))

    def get(self, model: str, *, task: Task | None = None) -> Model:
        """A model by short name or full id; `task` picks among a family's."""
        return self._run(_ops.model_get(model, task))

    def predict(
        self,
        model: str,
        *,
        context: DataLike | Context,
        queries: DataLike,
        target: str,
        task: Task | None = None,
        **predictor_fields: Any,
    ) -> Predictions:
        """Scores queries with a model and a context sent once by digest.

        The context goes up once as an Arrow snapshot named by its digest
        (nothing is sent again for the same rows); the queries go as Arrow
        in parts by bytes.

        Raises:
            CapacityExceeded: At once, before any request, for a context past the snapshot ceiling.
        """
        return self._run(
            _ops.model_predict(
                self._client._state,
                model,
                context=context,
                queries=queries,
                target=target,
                task=task,
                fields=predictor_fields,
            )
        )


class Snapshots(_Sync):
    """Training snapshots and synthetic batches: named, immutable tables."""

    def create(
        self,
        name: str,
        *,
        data: DataLike,
        tags: Sequence[str] = (),
        target: str | None = None,
        task: Task | None = None,
        time: str | None = None,
        group: str | None = None,
    ) -> Snapshot:
        """Uploads a training table as `name`, waiting until the service has verified it.

        `target` and `task` are the training metadata a run reads; `time`
        and `group` say how its rows are held out. A name is never reused:
        the same rows again return the snapshot, other rows raise `Conflict`.
        """
        metadata = {"target": target, "task": task, "time": time, "group": group}
        return self._run(_ops.snapshot_create(name, data, tags, metadata))

    def synthesize(
        self,
        name: str,
        *,
        count: int = 1,
        seed: int | None = None,
        like: str | Snapshot | None = None,
        task: Task | None = None,
        rows: int | Range | None = None,
        columns: int | Range | None = None,
        classes: int | Range | None = None,
        missing: Missing | None = None,
        reject: Sequence[Literal["trivial", "unlearnable"]] = (),
        tags: Sequence[str] = (),
        **generator: Any,
    ) -> SyntheticBatch:
        """Generates `count` tables from random causal graphs, table `i` seeded `seed + i`.

        `like` copies a snapshot's columns, types, task and classes;
        `generator` takes the other generator settings (`categorical`,
        `functions`, `graph_nodes`, `irrelevant_columns`, ...), each
        checked against the service's bounds before anything is sent. It
        returns at once, every table pending; a run can name the batch now.
        """
        options = _ops.synthetic_options(
            task=task, rows=rows, columns=columns, classes=classes, missing=missing,
            reject=reject, generator=generator,
        )  # fmt: skip
        return self._run(
            _ops.synthetic_create(
                name, count=count, seed=seed, like=like, tags=tags, options=options
            )
        )

    def get(self, snapshot: str | Snapshot) -> Snapshot:
        """A snapshot."""
        return self._run(_ops.snapshot_get(snapshot))

    def batch(self, batch: str | SyntheticBatch) -> SyntheticBatch:
        """A synthetic batch and its tables' states."""
        return self._run(_ops.synthetic_get(batch))

    def list(self, *, tag: str | None = None) -> builtins.list[Snapshot]:
        """The project's snapshots, or those carrying `tag`."""
        return self._run(_ops.snapshot_list(tag))

    def tag(self, snapshot: str | Snapshot, tags: Sequence[str]) -> Snapshot:
        """Replaces a snapshot's tags; runs pinned to it are unchanged."""
        return self._run(_ops.snapshot_tag(snapshot, tags))

    def delete(self, snapshot: str | Snapshot) -> None:
        """Deletes a snapshot; its name stays taken. `Conflict` while a run reads it."""
        self._run(_ops.snapshot_delete(snapshot))


class Runs(_Sync):
    """Fine-tuning runs: each run of a name publishes its next version, `NAME@ftN`."""

    def create(
        self,
        name: str,
        *,
        base: str,
        data: Sequence[str | Snapshot | SyntheticBatch | Context | Tag] | None = None,
        method: Method = DEFAULT_METHOD,
        plan: Plan | None = None,
        phases: Sequence[Phase] | None = None,
        architecture: Sequence[ArchitectureChange] = (),
        evaluation: Evaluation | None = None,
        budget: Budget | None = None,
        seed: int | None = None,
        tags: Sequence[str] = (),
        wait: bool = False,
        progress: Callable[[FineTuneRun], None] | None = None,
        poll: float | None = None,
    ) -> FineTuneRun:
        """Starts the run that publishes `name`'s next version, trained afresh from `base`.

        `data` takes snapshots, synthetic batches, contexts (`"adult"`,
        `"adult@3"`, a `Context`; the run pins that version's snapshot) and
        `gf.Tag`s; a bare name is looked up as each. `phases` in its place
        trains a curriculum. A batch still generating makes the run wait.

        Raises:
            Error: Not recoverable, with `wait`, when the run fails.
        """
        return self._run(
            _ops.run_create(
                name, base=base, data=data, method=method, plan=plan, phases=phases,
                architecture=architecture, evaluation=evaluation, budget=budget,
                seed=seed, tags=tags, wait=wait, progress=progress, poll=poll,
            )
        )  # fmt: skip

    def get(self, run: str | FineTuneRun) -> FineTuneRun:
        """A run, `NAME@ftN`, or a name's latest run."""
        return self._run(_ops.run_get(run))

    def list(
        self,
        *,
        name: str | None = None,
        state: str | None = None,
        data_digest: str | None = None,
        tag: str | None = None,
    ) -> builtins.list[FineTuneRun]:
        """The project's runs, only those matching what is given."""
        params = {"name": name, "state": state, "data_digest": data_digest, "tag": tag}
        return self._run(_ops.run_list(params))

    def wait(
        self,
        run: str | FineTuneRun,
        *,
        timeout: float | None = None,
        progress: Callable[[FineTuneRun], None] | None = None,
        poll: float | None = None,
    ) -> FineTuneRun:
        """Waits for a run to end; `progress` is called with the run at each poll."""
        return self._run(_ops.run_wait(run, timeout, progress, poll))

    def watch(
        self, run: str | FineTuneRun, *, poll: float | None = None
    ) -> Iterator[RunEvent]:
        """Yields a run's events once each, oldest first, until it ends."""
        return self._client._iterate(_ops.run_watch(run, poll))

    def events(self, run: str | FineTuneRun) -> builtins.list[RunEvent]:
        """Every event of a run so far, oldest first."""
        return self._run(_ops.run_events(run))

    def cancel(self, run: str | FineTuneRun, *, save: bool = False) -> FineTuneRun:
        """Cancels a run; with `save`, one that is training publishes what it has."""
        return self._run(_ops.run_cancel(run, save))


class FineTunes:
    """Fine-tuning: training snapshots and synthetic batches, and the runs that train on them."""

    def __init__(self, client: Client) -> None:
        """Binds the snapshots and runs to `client`."""
        self.snapshots = Snapshots(client)
        self.runs = Runs(client)


class AsyncContexts(_Async):
    """`Contexts`, awaited."""

    async def create(
        self,
        name: str,
        *,
        data: DataLike | None = None,
        source: str | Source | None = None,
        table: str | None = None,
        sql: str | None = None,
        live: bool = False,
        columns: Mapping[str, ColumnType] | None = None,
        targets: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        freshness: str | timedelta = "15m",
        wait: bool = True,
    ) -> Context:
        """See `Contexts.create`."""
        return await self._run(
            _ops.context_create(
                name, data=data, source=source, table=table, sql=sql, live=live,
                columns=columns, targets=targets, id=id, time=time, freshness=freshness,
                wait=wait,
            )
        )  # fmt: skip

    async def upload(
        self, context: str | Context, data: DataLike, *, wait: bool = True
    ) -> Context:
        """See `Contexts.upload`."""
        return await self._run(_ops.context_upload(context, data, wait=wait))

    async def refresh(self, context: str | Context, *, wait: bool = True) -> Context:
        """See `Contexts.refresh`."""
        return await self._run(_ops.context_refresh(context, wait=wait))

    async def append(
        self,
        context: str | Context,
        data: DataLike,
        *,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """See `Contexts.append`."""
        return await self._run(
            _ops.context_append(context, data, idempotency_key=idempotency_key)
        )

    async def label(
        self,
        context: str | Context,
        data: DataLike,
        *,
        target: str,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """See `Contexts.label`."""
        return await self._run(
            _ops.context_label(
                context, data, target=target, idempotency_key=idempotency_key
            )
        )

    async def get(
        self, context: str | Context, *, version: int | None = None
    ) -> Context:
        """See `Contexts.get`."""
        return await self._run(_ops.context_get(context, version=version))

    async def list(self) -> builtins.list[Context]:
        """See `Contexts.list`."""
        return await self._run(_ops.context_list())

    async def rows(
        self, context: str | Context, *, limit: int = 20, version: int | None = None
    ) -> pandas.DataFrame:
        """See `Contexts.rows`."""
        return await self._run(_ops.context_rows(context, limit=limit, version=version))

    async def delete(self, context: str | Context) -> None:
        """See `Contexts.delete`."""
        await self._run(_ops.context_delete(context))


class AsyncExperiments(_Async):
    """`Experiments`, awaited."""

    async def create(
        self,
        *,
        context: str | Context,
        target: str,
        name: str | None = None,
        title: str | None = None,
        task: Task | None = None,
        id: str | None = None,
        time: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        types: Mapping[str, ColumnType] | None = None,
        holdout: str | Holdout | None = None,
        sql: str | None = None,
        models: Sequence[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """See `Experiments.create`."""
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return await self._run(
            _ops.experiment_create(values, models, wait, timeout, poll)
        )

    async def estimate(
        self,
        *,
        context: str | Context,
        target: str,
        name: str | None = None,
        title: str | None = None,
        task: Task | None = None,
        id: str | None = None,
        time: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        types: Mapping[str, ColumnType] | None = None,
        holdout: str | Holdout | None = None,
        sql: str | None = None,
        models: Sequence[str] | None = None,
    ) -> Estimate:
        """See `Experiments.estimate`."""
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return await self._run(_ops.experiment_estimate(values, models))

    async def run(
        self,
        experiment: str | Experiment,
        *,
        models: Sequence[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """See `Experiments.run`."""
        return await self._run(
            _ops.experiment_run(experiment, models, wait, timeout, poll)
        )

    async def update(
        self,
        experiment: str | Experiment,
        *,
        title: Any = UNSET,
        target: Any = UNSET,
        task: Any = UNSET,
        id: Any = UNSET,
        time: Any = UNSET,
        inputs: Any = UNSET,
        exclude: Any = UNSET,
        types: Any = UNSET,
        holdout: Any = UNSET,
        sql: Any = UNSET,
        context: Any = UNSET,
    ) -> Experiment:
        """See `Experiments.update`."""
        values = _setup(
            title=title, target=target, task=task, id=id, time=time, inputs=inputs,
            exclude=exclude, types=types, holdout=holdout, sql=sql, context=context,
        )  # fmt: skip
        return await self._run(_ops.experiment_update(experiment, values))

    async def get(
        self, experiment: str | Experiment, *, run: int | None = None
    ) -> Experiment:
        """See `Experiments.get`."""
        return await self._run(_ops.experiment_get(experiment, run=run))

    async def list(
        self, *, context: str | Context | None = None
    ) -> builtins.list[Experiment]:
        """See `Experiments.list`."""
        return await self._run(_ops.experiment_list(context))

    async def wait(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """See `Experiments.wait`."""
        return await self._run(
            _ops.experiment_wait(experiment, run=run, timeout=timeout, poll=poll)
        )

    async def cancel(
        self, experiment: str | Experiment, *, run: int | None = None
    ) -> Experiment:
        """See `Experiments.cancel`."""
        return await self._run(_ops.experiment_cancel(experiment, run=run))

    async def predictions(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        model: str | None = None,
    ) -> pandas.DataFrame:
        """See `Experiments.predictions`."""
        return await self._run(
            _ops.experiment_predictions(experiment, run=run, model=model)
        )

    async def slices(
        self,
        experiment: str | Experiment,
        *,
        run: int | None = None,
        model: str | None = None,
    ) -> builtins.list[Slice]:
        """See `Experiments.slices`."""
        return await self._run(_ops.experiment_slices(experiment, run=run, model=model))

    async def delete(self, experiment: str | Experiment) -> None:
        """See `Experiments.delete`."""
        await self._run(_ops.experiment_delete(experiment))


class AsyncPredictors(_Async):
    """`Predictors`, awaited."""

    async def create(
        self,
        name: str,
        *,
        env: str = "production",
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
        **spec_fields: Any,
    ) -> PredictorInfo:
        """See `Predictors.create`."""
        return await self._run(
            _ops.predictor_create(name, env, wait, timeout, spec_fields, poll)
        )

    async def update(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
        **changed_fields: Any,
    ) -> PredictorInfo:
        """See `Predictors.update`."""
        return await self._run(
            _ops.predictor_update(predictor, env, wait, timeout, changed_fields, poll)
        )

    async def deploy(
        self,
        *specs: Predictor,
        env: str = "production",
        dry_run: bool = False,
        wait: bool = True,
        timeout: float = 600,
        poll: float | None = None,
    ) -> builtins.list[Deploy]:
        """See `Predictors.deploy`."""
        return await self._run(
            _ops.predictor_deploy(specs, env, dry_run, wait, timeout, poll)
        )

    async def get(self, predictor: str | PredictorInfo) -> PredictorInfo:
        """See `Predictors.get`."""
        return await self._run(_ops.predictor_get(predictor))

    async def list(self) -> builtins.list[PredictorInfo]:
        """See `Predictors.list`."""
        return await self._run(_ops.predictor_list())

    async def revisions(
        self, predictor: str | PredictorInfo
    ) -> builtins.list[Revision]:
        """See `Predictors.revisions`."""
        return await self._run(_ops.predictor_revisions(predictor))

    async def predict(
        self,
        predictor: str | PredictorInfo,
        queries: DataLike,
        *,
        env: str = "production",
        as_of: datetime | str | None = None,
    ) -> Predictions:
        """See `Predictors.predict`."""
        return await self._run(
            _ops.predictor_predict(predictor, queries, env=env, as_of=as_of)
        )

    async def promote(
        self,
        predictor: str | PredictorInfo,
        *,
        from_env: str = "staging",
        env: str = "production",
        revision: int | None = None,
        wait: bool = True,
        poll: float | None = None,
    ) -> Environment:
        """See `Predictors.promote`."""
        payload = (
            {"revision": revision} if revision is not None else {"from_env": from_env}
        )
        return await self._run(
            _ops.predictor_move(predictor, env, "promote", payload, wait, poll)
        )

    async def rollback(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        revision: int | None = None,
        wait: bool = True,
        poll: float | None = None,
    ) -> Environment:
        """See `Predictors.rollback`."""
        payload = {"revision": revision} if revision is not None else {}
        return await self._run(
            _ops.predictor_move(predictor, env, "rollback", payload, wait, poll)
        )

    async def logs(
        self,
        predictor: str | PredictorInfo,
        *,
        env: str = "production",
        limit: int = 100,
        level: Literal["info", "warn", "error"] | None = None,
        q: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        after: str | None = None,
    ) -> Logs:
        """See `Predictors.logs`."""
        params = {
            "limit": limit,
            "level": level,
            "q": q,
            "since": since,
            "until": until,
            "after": after,
        }
        return await self._run(_ops.predictor_logs(predictor, env, params))

    async def delete(self, predictor: str | PredictorInfo) -> None:
        """See `Predictors.delete`."""
        await self._run(_ops.predictor_delete(predictor))


class AsyncSources(_Async):
    """`Sources`, awaited."""

    async def list(self) -> builtins.list[Source]:
        """See `Sources.list`."""
        return await self._run(_ops.source_list())

    async def get(self, source: str | Source) -> Source:
        """See `Sources.get`."""
        return await self._run(_ops.source_get(source))

    async def create(
        self, name: str, *, kind: SourceKind, secret: str | None = None, **config: str
    ) -> Source:
        """See `Sources.create`."""
        return await self._run(_ops.source_create(name, kind, secret, config))

    async def test(self, source: str | Source) -> Source:
        """See `Sources.test`."""
        return await self._run(_ops.source_test(source))

    async def tables(
        self, source: str | Source, *, prefix: str | None = None
    ) -> builtins.list[SourceTable]:
        """See `Sources.tables`."""
        return await self._run(_ops.source_tables(source, prefix))

    async def preview(
        self, source: str | Source, *, table: str | None = None, sql: str | None = None
    ) -> pandas.DataFrame:
        """See `Sources.preview`."""
        return await self._run(_ops.source_preview(source, table, sql))

    async def delete(self, source: str | Source) -> None:
        """See `Sources.delete`."""
        await self._run(_ops.source_delete(source))


class AsyncKeys(_Async):
    """`Keys`, awaited."""

    async def list(self) -> builtins.list[Key]:
        """See `Keys.list`."""
        return await self._run(_ops.key_list())

    async def create(
        self,
        name: str,
        *,
        permissions: Sequence[str] = STANDARD_PERMISSIONS,
        expires: str | timedelta | None = "90d",
    ) -> NewKey:
        """See `Keys.create`."""
        return await self._run(_ops.key_create(name, permissions, expires))

    async def rotate(self, key: str | Key) -> NewKey:
        """See `Keys.rotate`."""
        return await self._run(_ops.key_action(key, "rotate"))

    async def revoke(self, key: str | Key) -> Key:
        """See `Keys.revoke`."""
        return await self._run(_ops.key_action(key, "revoke"))

    async def delete(self, key: str | Key) -> None:
        """See `Keys.delete`."""
        await self._run(_ops.key_action(key, "delete"))


class AsyncModels(_Async):
    """`Models`, awaited."""

    async def list(
        self, *, task: Task | None = None, all: bool = False
    ) -> builtins.list[Model]:
        """See `Models.list`."""
        return await self._run(_ops.model_list(task, all))

    async def get(self, model: str, *, task: Task | None = None) -> Model:
        """See `Models.get`."""
        return await self._run(_ops.model_get(model, task))

    async def predict(
        self,
        model: str,
        *,
        context: DataLike | Context,
        queries: DataLike,
        target: str,
        task: Task | None = None,
        **predictor_fields: Any,
    ) -> Predictions:
        """See `Models.predict`."""
        return await self._run(
            _ops.model_predict(
                self._client._state,
                model,
                context=context,
                queries=queries,
                target=target,
                task=task,
                fields=predictor_fields,
            )
        )


class AsyncSnapshots(_Async):
    """`Snapshots`, awaited."""

    async def create(
        self,
        name: str,
        *,
        data: DataLike,
        tags: Sequence[str] = (),
        target: str | None = None,
        task: Task | None = None,
        time: str | None = None,
        group: str | None = None,
    ) -> Snapshot:
        """See `Snapshots.create`."""
        metadata = {"target": target, "task": task, "time": time, "group": group}
        return await self._run(_ops.snapshot_create(name, data, tags, metadata))

    async def synthesize(
        self,
        name: str,
        *,
        count: int = 1,
        seed: int | None = None,
        like: str | Snapshot | None = None,
        task: Task | None = None,
        rows: int | Range | None = None,
        columns: int | Range | None = None,
        classes: int | Range | None = None,
        missing: Missing | None = None,
        reject: Sequence[Literal["trivial", "unlearnable"]] = (),
        tags: Sequence[str] = (),
        **generator: Any,
    ) -> SyntheticBatch:
        """See `Snapshots.synthesize`."""
        options = _ops.synthetic_options(
            task=task, rows=rows, columns=columns, classes=classes, missing=missing,
            reject=reject, generator=generator,
        )  # fmt: skip
        return await self._run(
            _ops.synthetic_create(
                name, count=count, seed=seed, like=like, tags=tags, options=options
            )
        )

    async def get(self, snapshot: str | Snapshot) -> Snapshot:
        """See `Snapshots.get`."""
        return await self._run(_ops.snapshot_get(snapshot))

    async def batch(self, batch: str | SyntheticBatch) -> SyntheticBatch:
        """See `Snapshots.batch`."""
        return await self._run(_ops.synthetic_get(batch))

    async def list(self, *, tag: str | None = None) -> builtins.list[Snapshot]:
        """See `Snapshots.list`."""
        return await self._run(_ops.snapshot_list(tag))

    async def tag(self, snapshot: str | Snapshot, tags: Sequence[str]) -> Snapshot:
        """See `Snapshots.tag`."""
        return await self._run(_ops.snapshot_tag(snapshot, tags))

    async def delete(self, snapshot: str | Snapshot) -> None:
        """See `Snapshots.delete`."""
        await self._run(_ops.snapshot_delete(snapshot))


class AsyncRuns(_Async):
    """`Runs`, awaited."""

    async def create(
        self,
        name: str,
        *,
        base: str,
        data: Sequence[str | Snapshot | SyntheticBatch | Context | Tag] | None = None,
        method: Method = DEFAULT_METHOD,
        plan: Plan | None = None,
        phases: Sequence[Phase] | None = None,
        architecture: Sequence[ArchitectureChange] = (),
        evaluation: Evaluation | None = None,
        budget: Budget | None = None,
        seed: int | None = None,
        tags: Sequence[str] = (),
        wait: bool = False,
        progress: Callable[[FineTuneRun], None] | None = None,
        poll: float | None = None,
    ) -> FineTuneRun:
        """See `Runs.create`."""
        return await self._run(
            _ops.run_create(
                name, base=base, data=data, method=method, plan=plan, phases=phases,
                architecture=architecture, evaluation=evaluation, budget=budget,
                seed=seed, tags=tags, wait=wait, progress=progress, poll=poll,
            )
        )  # fmt: skip

    async def get(self, run: str | FineTuneRun) -> FineTuneRun:
        """See `Runs.get`."""
        return await self._run(_ops.run_get(run))

    async def list(
        self,
        *,
        name: str | None = None,
        state: str | None = None,
        data_digest: str | None = None,
        tag: str | None = None,
    ) -> builtins.list[FineTuneRun]:
        """See `Runs.list`."""
        params = {"name": name, "state": state, "data_digest": data_digest, "tag": tag}
        return await self._run(_ops.run_list(params))

    async def wait(
        self,
        run: str | FineTuneRun,
        *,
        timeout: float | None = None,
        progress: Callable[[FineTuneRun], None] | None = None,
        poll: float | None = None,
    ) -> FineTuneRun:
        """See `Runs.wait`."""
        return await self._run(_ops.run_wait(run, timeout, progress, poll))

    def watch(
        self, run: str | FineTuneRun, *, poll: float | None = None
    ) -> AsyncIterator[RunEvent]:
        """See `Runs.watch`; iterate it with `async for`."""
        return self._client._iterate(_ops.run_watch(run, poll))

    async def events(self, run: str | FineTuneRun) -> builtins.list[RunEvent]:
        """See `Runs.events`."""
        return await self._run(_ops.run_events(run))

    async def cancel(
        self, run: str | FineTuneRun, *, save: bool = False
    ) -> FineTuneRun:
        """See `Runs.cancel`."""
        return await self._run(_ops.run_cancel(run, save))


class AsyncFineTunes:
    """`FineTunes`, awaited."""

    def __init__(self, client: AsyncClient) -> None:
        """Binds the snapshots and runs to `client`."""
        self.snapshots = AsyncSnapshots(client)
        self.runs = AsyncRuns(client)
