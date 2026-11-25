# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The resources of a client: contexts, experiments, predictors, snapshots, runs, sources, keys, models.

Each method takes a name or the object for every reference and returns
typed dataclasses; fine-tuning settings are dicts, as the service takes
them. The work is in `generalforecast._ops`; the `Async` classes await the
same operations.
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
    ColumnType,
    Context,
    Deploy,
    Environment,
    Estimate,
    Experiment,
    FineTuneRun,
    Holdout,
    Key,
    Lineage,
    Logs,
    Model,
    NewKey,
    Precision,
    Predictions,
    Predictor,
    PredictorInfo,
    Revision,
    RunEvent,
    Slice,
    Snapshot,
    Source,
    SourceKind,
    SourceList,
    SourceTables,
    SyntheticBatch,
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
    "AsyncKeys",
    "AsyncModels",
    "AsyncPredictors",
    "AsyncRuns",
    "AsyncSnapshots",
    "AsyncSources",
    "Contexts",
    "Experiments",
    "Keys",
    "Models",
    "Predictors",
    "Runs",
    "Snapshots",
    "Sources",
]


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
        poll: float | None = None,
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
                self._client._state,
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
                poll=poll,
            )
        )

    def upload(
        self,
        context: str | Context,
        data: DataLike,
        *,
        wait: bool = True,
        poll: float | None = None,
    ) -> Context:
        """Uploads a new version, creating the context when it has none yet.

        Rows whose digest matches the latest version add none (`unchanged`),
        so CI and scripts can call it on every run.
        """
        return self._run(_ops.context_upload(context, data, wait=wait, poll=poll))

    def refresh(
        self, context: str | Context, *, wait: bool = True, poll: float | None = None
    ) -> Context:
        """Imports a new version from the context's source, when its rows changed."""
        return self._run(_ops.context_refresh(context, wait=wait, poll=poll))

    def append(
        self,
        context: str | Context,
        data: DataLike,
        *,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """Appends rows to a live context fed through the API."""
        return self._run(
            _ops.context_append(
                self._client._state, context, data, idempotency_key=idempotency_key
            )
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
                self._client._state,
                context,
                data,
                target=target,
                idempotency_key=idempotency_key,
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
        precision: Precision = "fast",
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """Creates an experiment and starts run 1; with `wait`, returns once it ends.

        `name` defaults to `<context>-<target>`; `task` is inferred from the
        target; `models=None` runs the recommended set. `holdout` left out is
        the last 20% by `time` when one is declared, else a random 20%.
        `precision` is what the models' matrix multiplies run at: `fast`
        (TF32) or `exact` (float32), which costs more.

        Raises:
            Error: Not recoverable, when the run fails.
            Timeout: When the run is still going after `timeout` seconds.
        """
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return self._run(
            _ops.experiment_create(values, models, wait, timeout, poll, precision)
        )

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
        precision: Precision = "fast",
    ) -> Estimate:
        """What `create` with the same arguments would cost, without starting it."""
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return self._run(_ops.experiment_estimate(values, models, precision))

    def run(
        self,
        experiment: str | Experiment,
        *,
        models: Sequence[str] | None = None,
        precision: Precision = "fast",
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """Starts another run with the same setup, at `precision` as `create` takes it."""
        return self._run(
            _ops.experiment_run(experiment, models, wait, timeout, poll, precision)
        )

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
        precision: Precision = "fast",
    ) -> Predictions:
        """Scores queries at `precision`, `fast` (TF32) or `exact` (float32, costs more); they go as Arrow, in parts by bytes."""
        return self._run(
            _ops.predictor_predict(
                predictor, queries, env=env, as_of=as_of, precision=precision
            )
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

    def list(self) -> SourceList:
        """The workspace's sources; `available` is False where the service connects none."""
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
    ) -> SourceTables:
        """A source's tables, only those whose names start with `prefix` when given.

        `more` says the warehouse holds more than were listed.
        """
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
        precision: Precision = "fast",
        **predictor_fields: Any,
    ) -> Predictions:
        """Scores queries with a model and a context sent once by digest, at `precision` as `predictors.predict` takes it.

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
                precision=precision,
            )
        )


class Snapshots(_Sync):
    """Named, immutable training tables: uploaded in chunks, streamed as Arrow, or generated in synthetic batches.

    A name is unique in the project and never reused, even once its
    snapshot is deleted. Runs read snapshots by name; the same name with
    the same table again returns the snapshot, and with another table is
    refused with `Conflict`.
    """

    def create(
        self,
        name: str,
        data: DataLike,
        *,
        target: str,
        task: Task,
        column_types: Mapping[str, str] | None = None,
        time: str | None = None,
        group: str | None = None,
        origin: Mapping[str, Any] | None = None,
        tags: builtins.list[str] | None = None,
        wait: bool = True,
    ) -> Snapshot:
        """Uploads a training table as `name`, in chunks, straight to object storage where the service says.

        `target` and `task` are the training metadata a run reads;
        `column_types` maps a column to `numerical` or `categorical`, and
        `time` and `group` say how a run holds its rows out. `tags` label
        it for `list(tag=...)`; `update` replaces them later. With `wait`,
        returns once the service has verified it and it is `ready`.

        Raises:
            CapacityExceeded: Before anything is sent, for a table past what one upload holds.
            InvalidInput: For an upload the service rejected.
        """
        metadata = _setup(
            target=target, task=task, column_types=column_types, time=time,
            group=group, origin=origin,
        )  # fmt: skip
        return self._run(
            _ops.snapshot_create(self._client._state, name, data, metadata, tags, wait)
        )

    def stream(
        self,
        name: str,
        data: DataLike,
        *,
        target: str | None = None,
        task: Task | None = None,
        time: str | None = None,
        group: str | None = None,
    ) -> Snapshot:
        """Sends a small table as one Arrow stream, `ready` at once.

        Its training metadata is optional: a run reads it once it names a
        `target` and `task`.

        Raises:
            CapacityExceeded: Before anything is sent, for a table past the service's stream limits.
        """
        metadata = _setup(target=target, task=task, time=time, group=group)
        return self._run(
            _ops.snapshot_stream(self._client._state, name, data, metadata)
        )

    def create_synthetic(
        self,
        name: str,
        *,
        count: int = 1,
        seed: int | None = None,
        schema_from: str | Snapshot | None = None,
        tags: builtins.list[str] | None = None,
        **options: Any,
    ) -> SyntheticBatch:
        """Generates a batch of `count` synthetic training tables, each its own snapshot.

        Table `i` is the snapshot `<name>-<i>`, drawn from a random causal
        graph seeded `seed + i`, and carries the tags `synthetic` and
        `synthetic-<name>`. `schema_from` names a ready snapshot whose
        columns, types, task and classes every table copies. `options` are
        the generator's settings, as `task="classification"`, `rows={"min":
        1024, "max": 8192}` (a range is drawn once per table), `missing=
        {"rate": 0.05, "patterns": ["random"]}` or `reject=["trivial"]`;
        each is checked against the service's bounds before anything is
        sent. It returns at once, every table pending; a run can name the
        batch now, as `{"tag": batch.tag}`.
        """
        return self._run(
            _ops.synthetic_create(
                name, count=count, seed=seed, schema_from=schema_from, tags=tags,
                options=options,
            )
        )  # fmt: skip

    def synthetic(self, name: str | SyntheticBatch) -> SyntheticBatch:
        """A synthetic batch by its name: its seed, tag, options and its tables' states."""
        return self._run(_ops.synthetic_get(name))

    def list(self, *, tag: str | None = None) -> builtins.list[Snapshot]:
        """The project's snapshots that are not deleted, oldest first; with `tag`, those carrying it."""
        return self._run(_ops.snapshot_list(tag))

    def get(self, name: str | Snapshot) -> Snapshot:
        """A snapshot by its name, a deleted one included."""
        return self._run(_ops.snapshot_get(name))

    def update(self, name: str | Snapshot, *, tags: builtins.list[str]) -> Snapshot:
        """Replaces a snapshot's tags and returns it; `[]` clears them. Runs pinned to it are unchanged."""
        return self._run(_ops.snapshot_update(name, tags))

    def delete(self, name: str | Snapshot) -> None:
        """Deletes a snapshot; its name stays taken. `Conflict` while a run reads it."""
        self._run(_ops.snapshot_delete(name))


class Runs(_Sync):
    """Fine-tuning runs: each run of a name publishes its next version, `NAME@ftN`.

    A run is named by the version it publishes, `NAME@ftN` (or
    `PROJECT/NAME@ftN`), or by `NAME` for the name's latest run.
    """

    def create(
        self,
        name: str,
        *,
        base: str,
        method: Mapping[str, Any],
        data: builtins.list[str | Mapping[str, Any]] | None = None,
        target: str | None = None,
        plan: Mapping[str, Any] | None = None,
        phases: builtins.list[Mapping[str, Any]] | None = None,
        architecture: builtins.list[Mapping[str, Any]] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        seed: int | None = None,
        tags: builtins.list[str] | None = None,
    ) -> FineTuneRun:
        """Starts the run that publishes `name`'s next version, trained afresh from `base`.

        `method` is how the network changes, `{"type": "lora", "rank": 16}`
        (with `"input_adapter": True` beside it), `{"type": "input_adapter"}`
        or `{"type": "full", "components": [...]}`. `data` names ready
        snapshots, `{"tag": "nightly"}` for every snapshot carrying a tag
        (a synthetic batch's, `{"tag": batch.tag}`), and `{"context":
        "adult", "version": 3}` for a context's version, the latest when
        `version` is left out. `plan` holds the steps and the trainer's
        settings, `{"steps": 1000, "learning_rate": 5e-5}`.

        `phases` in place of `data` trains a curriculum: 1 to 16 phases,
        each `{"name", "data", "plan"}` with a plan of its `steps`,
        `episode_sampling` and `episodes_per_step`; `plan` then holds the
        run-wide settings. `architecture` lists changes to the base, each
        `{"component", "to", "stages", "init"}`; `evaluation` how rows are
        held out and a step kept, `{"holdout": 0.2}`; `budget` where it
        stops and publishes its best kept step, `{"credits": 500}` or
        `{"gpu_hours": 2}`. `target` is the column a context's rows train
        to predict; a snapshot keeps the target its metadata names.

        Raises:
            InvalidInput: Before anything is sent, for `data` and `phases`
                both or neither, `data` without `plan`, or a context in
                the data and no `target`.
        """
        payload = _ops.run_spec(
            name, base=base, method=method, data=data, target=target, plan=plan,
            phases=phases, architecture=architecture, evaluation=evaluation,
            budget=budget, seed=seed, tags=tags,
        )  # fmt: skip
        return self._run(_ops.run_create(payload))

    def list(
        self,
        *,
        state: str | None = None,
        data_digest: str | None = None,
        tag: str | None = None,
    ) -> builtins.list[FineTuneRun]:
        """The project's runs by name and version, only those matching what is given."""
        params = {"state": state, "data_digest": data_digest, "tag": tag}
        return self._run(_ops.run_list(params))

    def get(self, run: str | FineTuneRun) -> FineTuneRun:
        """A run's state, pinned data, progress, usage and published version, with its loss curve."""
        return self._run(_ops.run_get(run))

    def cancel(self, run: str | FineTuneRun, *, save: bool = False) -> FineTuneRun:
        """Cancels a run; with `save`, one that is training publishes the weights it has kept."""
        return self._run(_ops.run_cancel(run, save))

    def lineage(self, run: str | FineTuneRun) -> Lineage:
        """Every change to a run, oldest first: its model and its events."""
        return self._run(_ops.run_lineage(run))

    def wait(
        self,
        run: str | FineTuneRun,
        *,
        timeout: float | None = None,
        poll: float = 5.0,
        on_event: Callable[[RunEvent], Any] | None = None,
    ) -> FineTuneRun:
        """Polls a run every `poll` seconds until it finishes or is cancelled, and returns it.

        With `on_event`, each lineage event is passed to it once, as
        `watch` yields them. A name alone waits for the run that was its
        latest when the wait began.

        Raises:
            Timeout: Past `timeout` seconds; the run carries on.
            Error: Not recoverable, when the run fails.
        """
        return self._run(_ops.run_wait(run, timeout, poll, on_event))

    def watch(
        self, run: str | FineTuneRun, *, timeout: float | None = None, poll: float = 5.0
    ) -> Iterator[RunEvent]:
        """Yields a run's lineage events once each, oldest first, until it ends.

        Raises:
            Timeout: Past `timeout` seconds; the run carries on.
        """
        return self._client._iterate(_ops.run_watch(run, timeout, poll))


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
        poll: float | None = None,
    ) -> Context:
        """See `Contexts.create`."""
        return await self._run(
            _ops.context_create(
                self._client._state, name, data=data, source=source, table=table, sql=sql, live=live,
                columns=columns, targets=targets, id=id, time=time, freshness=freshness,
                wait=wait, poll=poll,
            )
        )  # fmt: skip

    async def upload(
        self,
        context: str | Context,
        data: DataLike,
        *,
        wait: bool = True,
        poll: float | None = None,
    ) -> Context:
        """See `Contexts.upload`."""
        return await self._run(_ops.context_upload(context, data, wait=wait, poll=poll))

    async def refresh(
        self, context: str | Context, *, wait: bool = True, poll: float | None = None
    ) -> Context:
        """See `Contexts.refresh`."""
        return await self._run(_ops.context_refresh(context, wait=wait, poll=poll))

    async def append(
        self,
        context: str | Context,
        data: DataLike,
        *,
        idempotency_key: str | None = None,
    ) -> WriteReceipt:
        """See `Contexts.append`."""
        return await self._run(
            _ops.context_append(
                self._client._state, context, data, idempotency_key=idempotency_key
            )
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
                self._client._state,
                context,
                data,
                target=target,
                idempotency_key=idempotency_key,
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
        precision: Precision = "fast",
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
            _ops.experiment_create(values, models, wait, timeout, poll, precision)
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
        precision: Precision = "fast",
    ) -> Estimate:
        """See `Experiments.estimate`."""
        values = _setup(
            name=name, title=title, context=context, target=target, task=task, id=id,
            time=time, inputs=inputs, exclude=exclude, types=types, holdout=holdout, sql=sql,
        )  # fmt: skip
        return await self._run(_ops.experiment_estimate(values, models, precision))

    async def run(
        self,
        experiment: str | Experiment,
        *,
        models: Sequence[str] | None = None,
        precision: Precision = "fast",
        wait: bool = True,
        timeout: float | None = None,
        poll: float | None = None,
    ) -> Experiment:
        """See `Experiments.run`."""
        return await self._run(
            _ops.experiment_run(experiment, models, wait, timeout, poll, precision)
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
        precision: Precision = "fast",
    ) -> Predictions:
        """See `Predictors.predict`."""
        return await self._run(
            _ops.predictor_predict(
                predictor, queries, env=env, as_of=as_of, precision=precision
            )
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

    async def list(self) -> SourceList:
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
    ) -> SourceTables:
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
        precision: Precision = "fast",
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
                precision=precision,
            )
        )


class AsyncSnapshots(_Async):
    """`Snapshots`, awaited."""

    async def create(
        self,
        name: str,
        data: DataLike,
        *,
        target: str,
        task: Task,
        column_types: Mapping[str, str] | None = None,
        time: str | None = None,
        group: str | None = None,
        origin: Mapping[str, Any] | None = None,
        tags: builtins.list[str] | None = None,
        wait: bool = True,
    ) -> Snapshot:
        """See `Snapshots.create`."""
        metadata = _setup(
            target=target, task=task, column_types=column_types, time=time,
            group=group, origin=origin,
        )  # fmt: skip
        return await self._run(
            _ops.snapshot_create(self._client._state, name, data, metadata, tags, wait)
        )

    async def stream(
        self,
        name: str,
        data: DataLike,
        *,
        target: str | None = None,
        task: Task | None = None,
        time: str | None = None,
        group: str | None = None,
    ) -> Snapshot:
        """See `Snapshots.stream`."""
        metadata = _setup(target=target, task=task, time=time, group=group)
        return await self._run(
            _ops.snapshot_stream(self._client._state, name, data, metadata)
        )

    async def create_synthetic(
        self,
        name: str,
        *,
        count: int = 1,
        seed: int | None = None,
        schema_from: str | Snapshot | None = None,
        tags: builtins.list[str] | None = None,
        **options: Any,
    ) -> SyntheticBatch:
        """See `Snapshots.create_synthetic`."""
        return await self._run(
            _ops.synthetic_create(
                name, count=count, seed=seed, schema_from=schema_from, tags=tags,
                options=options,
            )
        )  # fmt: skip

    async def synthetic(self, name: str | SyntheticBatch) -> SyntheticBatch:
        """See `Snapshots.synthetic`."""
        return await self._run(_ops.synthetic_get(name))

    async def list(self, *, tag: str | None = None) -> builtins.list[Snapshot]:
        """See `Snapshots.list`."""
        return await self._run(_ops.snapshot_list(tag))

    async def get(self, name: str | Snapshot) -> Snapshot:
        """See `Snapshots.get`."""
        return await self._run(_ops.snapshot_get(name))

    async def update(
        self, name: str | Snapshot, *, tags: builtins.list[str]
    ) -> Snapshot:
        """See `Snapshots.update`."""
        return await self._run(_ops.snapshot_update(name, tags))

    async def delete(self, name: str | Snapshot) -> None:
        """See `Snapshots.delete`."""
        await self._run(_ops.snapshot_delete(name))


class AsyncRuns(_Async):
    """`Runs`, awaited."""

    async def create(
        self,
        name: str,
        *,
        base: str,
        method: Mapping[str, Any],
        data: builtins.list[str | Mapping[str, Any]] | None = None,
        target: str | None = None,
        plan: Mapping[str, Any] | None = None,
        phases: builtins.list[Mapping[str, Any]] | None = None,
        architecture: builtins.list[Mapping[str, Any]] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        seed: int | None = None,
        tags: builtins.list[str] | None = None,
    ) -> FineTuneRun:
        """See `Runs.create`."""
        payload = _ops.run_spec(
            name, base=base, method=method, data=data, target=target, plan=plan,
            phases=phases, architecture=architecture, evaluation=evaluation,
            budget=budget, seed=seed, tags=tags,
        )  # fmt: skip
        return await self._run(_ops.run_create(payload))

    async def list(
        self,
        *,
        state: str | None = None,
        data_digest: str | None = None,
        tag: str | None = None,
    ) -> builtins.list[FineTuneRun]:
        """See `Runs.list`."""
        params = {"state": state, "data_digest": data_digest, "tag": tag}
        return await self._run(_ops.run_list(params))

    async def get(self, run: str | FineTuneRun) -> FineTuneRun:
        """See `Runs.get`."""
        return await self._run(_ops.run_get(run))

    async def cancel(
        self, run: str | FineTuneRun, *, save: bool = False
    ) -> FineTuneRun:
        """See `Runs.cancel`."""
        return await self._run(_ops.run_cancel(run, save))

    async def lineage(self, run: str | FineTuneRun) -> Lineage:
        """See `Runs.lineage`."""
        return await self._run(_ops.run_lineage(run))

    async def wait(
        self,
        run: str | FineTuneRun,
        *,
        timeout: float | None = None,
        poll: float = 5.0,
        on_event: Callable[[RunEvent], Any] | None = None,
    ) -> FineTuneRun:
        """See `Runs.wait`; `on_event` may be a coroutine function."""
        return await self._run(_ops.run_wait(run, timeout, poll, on_event))

    def watch(
        self, run: str | FineTuneRun, *, timeout: float | None = None, poll: float = 5.0
    ) -> AsyncIterator[RunEvent]:
        """See `Runs.watch`; iterate it with `async for`."""
        return self._client._iterate(_ops.run_watch(run, timeout, poll))
