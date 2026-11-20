# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`generalforecast.Predictor`: one object for hosted and ad hoc predictors.

A hosted predictor is one `gf push` deployed, called by its name in an
environment. An ad hoc predictor is a catalog model, a target and context
rows, sent with each call to the hosted ad hoc route. Every prediction runs
on the service; a predictor directory runs once it is pushed.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast import tabular
from generalforecast.errors import Error, InvalidInput, boundary
from generalforecast.predict_request import (
    PredictRequest,
    Progress,
    _frame_of_result,
    _with_forecast_keys,
)
from generalforecast.validation import NAME

ENVIRONMENTS = ("staging", "production")
# A catalog model, `general-forecast/<name>`, or a project's fine-tune, `<project>/<name>`.
_MODEL = re.compile(r"[a-z][a-z0-9-]{0,62}/[A-Za-z0-9._-]+(@\S+)?")


def local_refused(path: str | os.PathLike[str]) -> InvalidInput:
    """The error for a predictor directory where a hosted predictor is wanted."""
    return InvalidInput(
        f"`{os.fspath(path)}` is a predictor directory; predictions run on the "
        "hosted service",
        hint="push it with `gf push DIR`, then use the predictor's name",
        param="ref",
    )


def resolve_reference(ref: Any) -> tuple[str, str]:
    """What a reference names: `("catalog", model id)` or `("hosted", name)`.

    A path (`./churn`, `~/churn`, `/srv/churn`, or `a/churn` holding
    `gf.yaml`) is refused: a predictor directory runs once it is pushed. A
    bare name such as `churn` is the hosted predictor, even where a
    directory of that name is here, as `gf predict` reads it.

    Raises:
        InvalidInput: For a path, a URL or a malformed name.
    """
    if isinstance(ref, os.PathLike):
        raise local_refused(ref)
    if not isinstance(ref, str) or not ref:
        raise InvalidInput("a predictor reference is a name or a model id", param="ref")
    if re.match(r"^https?://", ref):
        raise InvalidInput(
            "a predictor is named, not addressed by URL",
            hint="pass the hosted predictor's name; set the server with GF_URL",
            param="ref",
        )
    if ref.startswith((".", "~", "/")) or (
        "/" in ref and (Path(ref).expanduser() / "gf.yaml").is_file()
    ):
        raise local_refused(ref)
    if _MODEL.fullmatch(ref):
        return "catalog", ref
    if NAME.fullmatch(ref):
        return "hosted", ref
    raise InvalidInput(
        f"`{ref}` is not a predictor name or a model id",
        hint="a name is lowercase letters, digits and dashes; a model id is "
        "general-forecast/<name>",
        param="ref",
    )


def _query_batches(
    table: pa.Table, series: Sequence[str], cap: int
) -> list[np.ndarray]:
    """The row indices of each call, at most `cap` rows apiece, in row order.

    With series columns, each series' rows stay in one call, as a
    forecaster needs a series' whole horizon at once; series are packed in
    order of first appearance. A series of more than `cap` rows is a call
    of its own, which the service bounds.
    """
    if not series:
        return [
            np.arange(start, min(start + cap, table.num_rows))
            for start in range(0, table.num_rows, cap)
        ]
    keys = list(zip(*(table[name].to_pylist() for name in series), strict=True))
    members: dict[Any, list[int]] = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)
    batches: list[list[int]] = [[]]
    for rows in members.values():
        if batches[-1] and len(batches[-1]) + len(rows) > cap:
            batches.append([])
        batches[-1].extend(rows)
    return [np.asarray(sorted(rows), dtype=np.int64) for rows in batches if rows]


@dataclass(frozen=True)
class Context:
    """What a predictor learns from: its kind and how many rows it holds."""

    kind: str
    rows: int | None = None
    name: str | None = None
    at: Any = None
    _owner: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:  # noqa: D105
        rows = "" if self.rows is None else f" {self.rows:,} rows"
        return f"Context({self.kind}{rows})"

    def _live(self) -> Predictor:
        if self.kind != "live" or self._owner is None:
            raise InvalidInput(
                f"a {self.kind} context cannot change",
                hint="only a live context takes append, label and as_of",
            )
        return self._owner

    def append(self, rows: Any, *, idempotency_key: str | None = None) -> Any:
        """Appends rows to a live context, each carrying the predictor's id column.

        Args:
            rows: The rows to append.
            idempotency_key: The write's durable id, so a retried append is
                applied once; a fresh one if None.
        """
        owner = self._live()
        identity = owner._id_name()
        if identity is None:
            raise InvalidInput(
                "a live context's rows need an id column",
                hint="declare `id:` in gf.yaml so appended rows can be labeled later",
            )
        with boundary():
            return owner._client().contexts.append(
                self.name,
                tabular.frame(rows),
                id=identity,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def label(
        self,
        ids: Iterable[Any],
        values: Iterable[Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        """Sets the target of rows in a live context, by id."""
        owner = self._live()
        ids, values = list(ids), list(values)
        if len(ids) != len(values):
            raise InvalidInput("ids and values must have the same length")
        identity = owner._id_name() or "id"
        target = owner.target or owner._hosted_facts().get("target")
        if target is None:
            raise InvalidInput("the predictor names no target to label")
        rows = [
            {identity: key, target: value}
            for key, value in zip(ids, values, strict=True)
        ]
        with boundary():
            return owner._client().contexts.label(
                self.name,
                rows,
                id=identity,
                column=target,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def as_of(self, ts: Any) -> Predictor:
        """The same predictor reading the live context as of `ts`."""
        return self._live()._with_as_of(ts)


class Predictor:
    """A hosted predictor by name, or an ad hoc predictor on a catalog model.

    Args:
        ref: A hosted predictor's name, such as `churn`, or a catalog id,
            such as `general-forecast/tabicl-v2-classifier`, for an ad hoc
            predictor: give `target` and the context. A predictor directory
            is refused; push it with `gf push` and use its name.
        env: A hosted predictor's environment, `production` by default.
        target: An ad hoc predictor's target column.
        inputs: An ad hoc predictor's input columns; inferred if None.
        exclude: Columns an ad hoc predictor leaves out of the inferred inputs.
        id: The id column.
        time: The time column.
        series: A forecaster's series columns.
        frequency: A forecaster's step.
        horizon: A forecaster's steps.
        quantiles: A regressor's or forecaster's levels.
        context: An ad hoc predictor's context rows, so `predict` can omit them.
        selection: How each query's context is drawn: core, neighbors, kind.
        future: A forecaster's inputs whose values each query brings ahead;
            its other inputs are past covariates.
        client: The hosted client to use; one from the profile if None.
    """

    def __init__(
        self,
        ref: str,
        *,
        env: str | None = None,
        target: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        series: Sequence[str] | str | None = None,
        frequency: Any = None,
        horizon: int | None = None,
        quantiles: Sequence[float] | None = None,
        context: Any = None,
        selection: Mapping[str, Any] | None = None,
        future: Sequence[str] = (),
        client: Any = None,
    ) -> None:
        """Resolves the reference; nothing is sent until `predict`."""
        self._kind, where = resolve_reference(ref)
        self._client_given = client
        self._as_of: Any = None
        self._context_frame: pa.Table | None = None
        self._facts: dict[str, Any] | None = None
        self._entry: Any = None
        self.future = tuple(future)
        self.target = target
        self.inputs: tuple[str, ...] | None = (
            tuple(inputs) if inputs is not None else None
        )
        self.exclude = tuple(exclude)
        self.id = id
        self.time = time
        self.series: tuple[str, ...] = (
            (series,) if isinstance(series, str) else tuple(series or ())
        )
        self.frequency = frequency
        self.horizon = horizon
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.selection = dict(selection) if selection else None
        self._env: str | None = None
        if self._kind == "catalog":
            if env is not None:
                raise InvalidInput("an ad hoc predictor has no environment")
            if not target:
                raise InvalidInput(
                    f"an ad hoc predictor on {where} needs a target",
                    param="target",
                    hint='pass target="<column>"',
                )
            self.model = where
            self.name = "ad-hoc"
            self.task = "?"
            if context is not None:
                self._context_frame = tabular.frame(context)
        else:
            if env is not None and env not in ENVIRONMENTS:
                raise InvalidInput(
                    f"environment `{env}` is not one of staging, production",
                    param="env",
                )
            self.name = where
            self._env = env or "production"
            self.model = "?"
            self.task = "?"

    @property
    def env(self) -> str | None:
        """None for ad hoc, else `staging` or `production`."""
        return self._env

    @property
    def context(self) -> Context:
        """The predictor's context: its kind and rows, and a live context's handles."""
        if self._kind == "catalog":
            rows = None if self._context_frame is None else len(self._context_frame)
            return Context("request", rows)
        info = self._hosted_facts().get("context") or {}
        return Context(
            str(info.get("kind", "live")),
            info.get("rows"),
            info.get("name", self.name),
            self._as_of,
            self,
        )

    def _id_name(self) -> str | None:
        """The id column, asked of a hosted predictor once."""
        if self.id is None and self._kind == "hosted":
            self.id = self._hosted_facts().get("id") or None
        return self.id

    def _hosted_facts(self) -> dict[str, Any]:
        if self._kind != "hosted":
            return {}
        if self._facts is None:
            with boundary():
                found = self._client().predictors.get(self.name, env=self._env)
            self._facts = dict(found) if isinstance(found, Mapping) else {}
            self.model = str(self._facts.get("model", self.model))
            self.task = str(self._facts.get("task", self.task))
        return self._facts

    def _client(self) -> Any:
        if self._client_given is None:
            from generalforecast.client import Client

            with boundary():
                self._client_given = Client()
        return self._client_given

    def _model_entry(self) -> Any:
        """An ad hoc predictor's model as the catalog lists it, read once."""
        if self._entry is None:
            from generalforecast.catalog import ModelInfo

            client = self._client()
            with boundary():
                self._entry = ModelInfo.from_reply(client.models.get(self.model))
            self.task = self._entry.task
            # The client's predictions read the same entry, not the route again.
            seen = getattr(client, "_models_seen", None)
            if isinstance(seen, dict):
                seen[self.model] = self._entry
        return self._entry

    def __repr__(self) -> str:  # noqa: D105
        inputs = "?" if self.inputs is None else str(len(self.inputs))
        context = repr(self.context) if self._kind == "catalog" else "Context(?)"
        return (
            f"Predictor({self.name}, model={self.model}, inputs={inputs}, "
            f"context={context[len('Context(') : -1]}, env={self._env!r})"
        )

    def predict(
        self,
        queries: Any,
        context: Any = None,
        *,
        as_of: Any = None,
        progress: Callable[[Progress], None] | None = None,
    ) -> Any:
        """Scores rows on the service.

        Any number of query rows score: past one call's bound
        (`contracts.MAX_QUERIES`, or the model's own) they go as several
        calls, a forecaster's series each whole within one, and come back
        in the order given.

        Args:
            queries: Rows to score: a frame, list of dicts, Arrow table, or a
                path or URL to a .csv, .parquet or .json file.
            context: Rows the model learns from, for an ad hoc predictor or a
                hosted predictor whose context kind is `request`.
            as_of: For a live context, the time to read it as of.
            progress: Called with a `Progress` as each call's rows are
                scored; nothing is reported without it.

        Returns:
            A pandas DataFrame: `id` (if any), `label` and `p_<class>`
            columns for a classifier, or `value` and `q_<level>` columns for
            a regressor or forecaster.
        """
        import pandas as pd

        table = tabular.frame(queries)
        if self._kind == "catalog" and (as_of if as_of is not None else self._as_of):
            raise InvalidInput(
                "as_of reads a live context; an ad hoc predictor's is a request context",
                param="as_of",
            )
        cap = self._query_cap()
        named = self.model if self._kind == "catalog" else self.name
        scored = 0

        def tell() -> None:
            if progress is not None:
                progress(Progress("predicting", named, scored, table.num_rows))

        tell()
        series = self._series_columns()
        parts, order = [], []
        pending = list(reversed(_query_batches(table, series, cap)))
        while pending:
            rows = pending.pop()
            try:
                parts.append(self._predict_once(table.take(rows), context, as_of))
            except Error as error:
                # A reply past the SDK's response bound comes back as a
                # protocol error; half as many rows answer in half the bytes.
                halves = _query_batches(
                    table.take(rows), series, max(1, len(rows) // 2)
                )
                if error.code != "ProtocolError" or len(halves) < 2:
                    raise
                pending.extend(reversed([rows[half] for half in halves]))
                continue
            order.append(rows)
            scored += len(rows)
            tell()
        if not parts:
            return pd.DataFrame()
        frame = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        placed = np.empty(table.num_rows, dtype=np.int64)
        placed[np.concatenate(order)] = np.arange(table.num_rows)
        return frame.iloc[placed].reset_index(drop=True)

    def _query_cap(self) -> int:
        """Most query rows one call carries: the wire bound, or a model's lower one."""
        from generalforecast.contracts import MAX_QUERIES

        if self._kind == "catalog":
            entry = self._model_entry()
            if entry.task != "forecast":
                return max(1, min(MAX_QUERIES, entry.limits.queries))
        return MAX_QUERIES

    def _series_columns(self) -> tuple[str, ...]:
        """A forecaster's series columns, which a batch keeps whole; none otherwise."""
        if self._kind == "hosted" and not self.series:
            facts = self._hosted_facts()
            if facts.get("task") == "forecast":
                self.series = tuple(facts.get("series") or ())
        return tuple(self.series)

    def _predict_once(self, queries: pa.Table, context: Any, as_of: Any) -> Any:
        """One prediction call: at most one call's query rows."""
        when = as_of if as_of is not None else self._as_of
        if self._kind == "catalog":
            rows = context if context is not None else self._context_frame
            if rows is None:
                raise InvalidInput(
                    "context is required",
                    param="context",
                    hint="pass context= to predict, or give the predictor a context",
                )
            with boundary():
                result = self._client().predict(
                    self.model,
                    self.target,
                    rows,
                    queries,
                    inputs=self.inputs,
                    id=self.id,
                    time=self.time,
                    series=self.series or None,
                    frequency=self.frequency,
                    horizon=self.horizon,
                    quantiles=self.quantiles,
                    selection=self.selection,
                    exclude=self.exclude,
                    future=self.future,
                )
            frame = _frame_of_result(result, self.id)
            if self._model_entry().task == "forecast" and self.time is not None:
                frame = _with_forecast_keys(
                    frame, queries, self.series, self.time, self.id
                )
            return frame
        with boundary():
            result = self._client().predictors.predict(
                self.name, queries, context=context, as_of=when, env=self._env
            )
        return self._keyed(_frame_of_result(result, self._id_name()), queries)

    def _keyed(self, frame: Any, queries: pa.Table) -> Any:
        """A hosted forecast's frame with the series and time of its queries.

        The wire reply carries each prediction in query order, so the keys
        come from the queries themselves.
        """
        facts = self._hosted_facts()
        if facts.get("task") != "forecast" or not facts.get("time"):
            return frame
        series = tuple(facts.get("series") or ())
        return _with_forecast_keys(
            frame, queries, series, facts["time"], self._id_name()
        )

    def _with_as_of(self, ts: Any) -> Predictor:
        clone = Predictor.__new__(Predictor)
        clone.__dict__.update(self.__dict__)
        clone._as_of = ts
        return clone

    def request(self, queries: Any, context: Any = None) -> PredictRequest:
        """The validated ad hoc request `predict` sends for these rows."""
        if self._kind != "catalog":
            raise InvalidInput("only an ad hoc predictor builds a request")
        rows = context if context is not None else self._context_frame
        if rows is None:
            raise InvalidInput("context is required", param="context")
        assert self.target is not None
        return PredictRequest.build(
            model=self.model,
            target=self.target,
            context=rows,
            queries=queries,
            inputs=self.inputs,
            exclude=self.exclude,
            id=self.id,
            time=self.time,
            series=self.series,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=self.quantiles,
            selection=self.selection,
            future=self.future,
            entry=self._model_entry(),
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Writes an ad hoc predictor as a predictor directory: `gf.yaml` and its snapshot.

        The settings become a manifest whose snapshot is the context rows,
        `context.parquet`. `gf check` and `gf push` take the directory from
        there, as `push` does.
        """
        if self._kind != "catalog":
            raise InvalidInput(
                "only an ad hoc predictor can be saved",
                hint="a hosted predictor's manifest is its revision",
            )
        from pyarrow import parquet

        from generalforecast import manifest as manifests

        root = Path(path)
        name = root.resolve().name
        if not NAME.fullmatch(name):
            raise InvalidInput(
                f"`{name}` is not a predictor name",
                hint="use lowercase letters, digits and dashes, starting with a letter",
            )
        rows = self._context_frame
        if rows is None:
            raise InvalidInput(
                "an ad hoc predictor saves with its context rows",
                hint="give the predictor a context",
            )
        with boundary():
            document = self.request(rows.slice(0, 1), rows).manifest()
            document["name"] = name
            document["context"] = {"snapshot": "context.parquet"}
            root.mkdir(parents=True, exist_ok=True)
            parquet.write_table(rows, root / "context.parquet")
            manifests.write(document, root, force=True)
        return root

    def push(self, path: str | os.PathLike[str], *, env: str = "staging") -> Predictor:
        """Saves an ad hoc predictor at `path`, checks and pushes it, and returns the hosted predictor.

        Args:
            path: The predictor directory to write; its name is the predictor's.
            env: The environment, `staging` or `production`.
        """
        if env not in ENVIRONMENTS:
            raise InvalidInput(
                f"environment `{env}` is not one of staging, production", param="env"
            )
        from generalforecast import manifest as manifests

        root = self.save(path)
        client = self._client()
        with boundary():
            checked = client.predictors.check(root)
            manifests.write_lock(root, checked["lock"])
            client.predictors.push(root, env=env)
        return Predictor(root.resolve().name, env=env, client=client)


__all__ = [
    "Context",
    "PredictRequest",
    "Predictor",
    "Progress",
    "resolve_reference",
]
