# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""scikit-learn style estimators over `models.predict`.

Hosted, in-context: fit uploads your rows and trains nothing; predict reads
them as context. `fit` uploads the rows once as an Arrow snapshot named by
its digest, so rows the service already holds are never sent again, and
refuses at once, with `CapacityExceeded`, a context too big to send;
`predict` sends the queries only. With `verbose`, each call prints one
line to stderr; `usage_` adds up every call since `fit` and `last_usage_`
holds the last one's. `estimate` prices a call without making it.

They follow scikit-learn's conventions (`get_params`, `set_params`,
`clone`, fitted attributes ending in `_`, `__sklearn_tags__`) without
depending on it, so `cross_val_score` and `GridSearchCV` take them.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from time import monotonic
from typing import Any, Self

import numpy as np
import pyarrow as pa

from generalforecast import _data, _ops, _render
from generalforecast.errors import InvalidInput
from generalforecast.term import duration, plural
from generalforecast.types import (
    Context,
    Estimate,
    LargeContext,
    Precision,
    Predictions,
    money,
)

__all__ = ["Classifier", "Forecaster", "NotFittedError", "Regressor"]


class NotFittedError(InvalidInput, AttributeError):
    """An estimator was asked to predict before `fit`."""


def predicted_line(result: Predictions, context_rows: int, seconds: float) -> str:
    """`Predicted 6,513 rows with tabicl-v2-classifier from 26,048 context rows: 2 calls, 13.0 credits ($0.013) in 9.4s`."""
    line = (
        f"Predicted {plural(result.rows, 'row')} with {_render.model_name(result.model)} "
        f"from {plural(context_rows, 'context row')}: {plural(result.calls, 'call')}, "
        f"{_render.credits(result.credits)} ({money(result.usd)}) in {duration(seconds)}"
    )
    if result.gpu_seconds:
        line += f"; {result.gpu_seconds:g} GPU seconds past the large-context line"
    if result.context is not None and result.context.used == "retrieved":
        line += "; the context was too large for the device, so each query read its nearest rows"
    return line


class _Base:
    """Parameters, the client, the uploaded context and the usage so far."""

    model: str
    precision: Precision
    verbose: bool
    client: Any
    _task: str
    _estimator_type: str | None = None
    context_: str
    context_rows_: int
    context_inputs_: int
    usage_: Estimate
    last_usage_: Estimate

    @classmethod
    def _param_names(cls) -> list[str]:
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """The constructor's arguments as set."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        """Changes constructor arguments; a fitted estimator must be fit again."""
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise InvalidInput(
                    f"`{key}` is not a parameter of {type(self).__name__}",
                    hint="parameters: " + ", ".join(sorted(valid)),
                )
            setattr(self, key, value)
        if params:
            self._forget()
        return self

    def __sklearn_tags__(self) -> Any:
        """What scikit-learn's checks and splitters need to know: the kind, and that it takes text."""
        from sklearn.utils import (
            ClassifierTags,
            InputTags,
            RegressorTags,
            Tags,
            TargetTags,
        )

        kind = self._estimator_type
        return Tags(
            estimator_type=kind,
            target_tags=TargetTags(required=kind is not None),
            classifier_tags=ClassifierTags() if kind == "classifier" else None,
            regressor_tags=RegressorTags() if kind == "regressor" else None,
            input_tags=InputTags(allow_nan=True, categorical=True, string=True),
            non_deterministic=True,
        )

    def _forget(self) -> None:
        for name in [
            n for n in vars(self) if n.endswith("_") and not n.startswith("_")
        ]:
            delattr(self, name)

    def __repr__(self) -> str:
        """Says what fit does: upload rows, train nothing."""
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.get_params().items()
            if k in ("model", "target")
        )
        note = (
            f"{self.context_rows_:,} context rows uploaded; nothing trained"
            if hasattr(self, "context_")
            else "fit uploads rows; nothing is trained"
        )
        return f"{type(self).__name__}({params})  # {note}"

    def _hosted(self) -> Any:
        if self.client is None:
            from generalforecast.client import default

            return default()
        return self.client

    def _require_fitted(self) -> None:
        if not hasattr(self, "context_"):
            raise NotFittedError(
                f"this {type(self).__name__} is not fitted",
                hint="call fit with the labeled rows first",
            )

    def _upload(self, rows: pa.Table) -> None:
        client = self._hosted()
        digest, count = client._run(_ops.snapshot(client._state, rows))
        self.context_ = digest
        self.context_rows_ = count
        # Every column but the target is an input, as the service counts a context's cells.
        self.context_inputs_ = rows.num_columns - 1
        self.usage_ = Estimate()

    def _predict(self, queries: pa.Table, target: str, **fields: Any) -> Predictions:
        self._require_fitted()
        client = self._hosted()
        reference = Context(name="fit", digest=self.context_, rows=self.context_rows_)
        started = monotonic()
        result = client.models.predict(
            self.model,
            context=reference,
            queries=queries,
            target=target,
            precision=self.precision,
            **fields,
        )
        if result.table.num_rows != queries.num_rows:
            raise InvalidInput("the service answered a different number of rows")
        large = (
            LargeContext(gpu_seconds=result.gpu_seconds) if result.gpu_seconds else None
        )
        self.last_usage_ = Estimate(
            result.calls, result.rows, result.credits, result.usd, large
        )
        self.usage_ = self.usage_ + self.last_usage_
        if self.verbose:
            line = predicted_line(result, self.context_rows_, monotonic() - started)
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        return result

    def _estimate(self, queries: int) -> Estimate:
        """What predicting `queries` rows would cost, the large-context line included."""
        self._require_fitted()
        return self._hosted()._run(
            _ops.predict_estimate(
                self.model,
                self._task,
                self.context_rows_,
                queries,
                self.context_inputs_,
                self.precision,
            )
        )


def _features(X: Any, names: Sequence[str] | None = None) -> pa.Table:
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        columns = list(names) if names else [f"f{i}" for i in range(X.shape[1])]
        if len(columns) != X.shape[1]:
            raise InvalidInput(f"X has {X.shape[1]} columns; fit saw {len(columns)}")
        return pa.table({name: X[:, i] for i, name in enumerate(columns)})
    return _data.table(X, what="X")


def _labels(y: Any) -> list[Any]:
    values = np.asarray(y).ravel().tolist()
    return [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]


def _target_name(columns: Sequence[str]) -> str:
    name = "target"
    while name in columns:
        name = f"_{name}"
    return name


class _Tabular(_Base):
    """Rows of inputs and a target: what a classifier and a regressor share."""

    inputs: Sequence[str] | None

    def _fit(self, X: Any, y: Any) -> pa.Table:
        self._forget()
        features = _features(X)
        if self.inputs is not None:
            features = features.select(list(self.inputs))
        labels = _labels(y)
        if len(labels) != features.num_rows:
            raise InvalidInput(
                f"X has {features.num_rows} rows and y has {len(labels)}", param="y"
            )
        self.feature_names_in_ = np.asarray(features.column_names, dtype=object)
        self.n_features_in_ = features.num_columns
        self.target_ = _target_name(features.column_names)
        rows = features.append_column(self.target_, pa.array(labels))
        self._upload(rows)
        return rows

    def estimate(self, X: Any) -> Estimate:
        """What predicting `X` would cost, priced without calling it, the large-context line included."""
        return self._estimate(self._queries(X).num_rows)

    def _queries(self, X: Any) -> pa.Table:
        self._require_fitted()
        names = list(self.feature_names_in_)
        queries = _features(X, names)
        missing = [n for n in names if n not in queries.column_names]
        if missing:
            raise InvalidInput(f"X has no column `{missing[0]}`", param="X")
        return queries.select(names)


class Classifier(_Tabular):
    """Hosted, in-context: fit uploads your rows and trains nothing; predict reads them as context."""

    _task = "classification"
    _estimator_type = "classifier"

    def __init__(
        self,
        model: str = "tabicl-v2",
        *,
        inputs: Sequence[str] | None = None,
        precision: Precision = "fast",
        verbose: bool = True,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the rows.

        Args:
            model: A short name such as `tabicl-v2`, or a full id.
            inputs: The columns to read; None reads every column of `X`.
            precision: What its matrix multiplies run at: `fast` (TF32) or
                `exact` (float32), which costs more.
            verbose: Print one line to stderr for each call to the service.
            client: The `gf.Client` to call; None uses the module's.
        """
        self.model = model
        self.inputs = inputs
        self.precision = precision
        self.verbose = verbose
        self.client = client

    def fit(self, X: Any, y: Any) -> Self:
        """Uploads the labeled rows as the context; `classes_` in sorted order."""
        rows = self._fit(X, y)
        labels = [v for v in rows[self.target_].to_pylist() if v is not None]
        self.classes_ = np.asarray(
            sorted(set(labels), key=lambda v: (str(type(v)), v)), dtype=object
        )
        if len(self.classes_) < 2:
            self._forget()
            raise InvalidInput("y holds one class; a classifier needs two", param="y")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Probabilities over `classes_`, one row per query."""
        table = self._predict(
            self._queries(X), self.target_, task="classification"
        ).table
        keys = [
            f"p_{str(c).lower() if isinstance(c, (bool, np.bool_)) else c}"
            for c in self.classes_
        ]
        missing = [k for k in keys if k not in table.column_names]
        if missing:
            raise InvalidInput("the service returned different classes")
        return np.column_stack(
            [np.asarray(table[k].to_pylist(), dtype=float) for k in keys]
        )

    def predict(self, X: Any) -> np.ndarray:
        """The most probable label for each query."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def score(self, X: Any, y: Any) -> float:
        """Mean accuracy on the labeled rows."""
        predicted = self.predict(X)
        return float(
            np.mean([a == b for a, b in zip(predicted, _labels(y), strict=True)])
        )


class Regressor(_Tabular):
    """Hosted, in-context: fit uploads your rows and trains nothing; predict reads them as context."""

    _task = "regression"
    _estimator_type = "regressor"

    def __init__(
        self,
        model: str = "tabicl-v2",
        *,
        quantiles: Sequence[float] | None = None,
        inputs: Sequence[str] | None = None,
        precision: Precision = "fast",
        verbose: bool = True,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the rows.

        Args:
            model: A short name such as `tabicl-v2`, or a full id.
            quantiles: The levels `predict_quantiles` gives by default.
            inputs: The columns to read; None reads every column of `X`.
            precision: What its matrix multiplies run at: `fast` (TF32) or
                `exact` (float32), which costs more.
            verbose: Print one line to stderr for each call to the service.
            client: The `gf.Client` to call; None uses the module's.
        """
        self.model = model
        self.quantiles = quantiles
        self.inputs = inputs
        self.precision = precision
        self.verbose = verbose
        self.client = client

    def fit(self, X: Any, y: Any) -> Self:
        """Uploads the rows and their numeric targets as the context."""
        try:
            values = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise InvalidInput(
                "y holds text, but a Regressor predicts numbers",
                hint="use a Classifier to predict labels",
                param="y",
            ) from None
        self._fit(X, values)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """The predicted value of each query."""
        table = self._predict(self._queries(X), self.target_, task="regression").table
        return np.asarray(table["value"].to_pylist(), dtype=float)

    def predict_quantiles(
        self, X: Any, quantiles: Sequence[float] | None = None
    ) -> np.ndarray:
        """The predicted quantiles of each query: one column per level."""
        levels = [float(q) for q in (quantiles or self.quantiles or (0.1, 0.5, 0.9))]
        table = self._predict(
            self._queries(X), self.target_, task="regression", quantiles=levels
        ).table
        return np.column_stack(
            [
                np.asarray(table[f"q_{level:g}"].to_pylist(), dtype=float)
                for level in levels
            ]
        )

    def score(self, X: Any, y: Any) -> float:
        """The coefficient of determination, R^2."""
        truth = np.asarray(y, dtype=float).ravel()
        predicted = self.predict(X)
        total = float(((truth - truth.mean()) ** 2).sum())
        residual = float(((truth - predicted) ** 2).sum())
        return 1.0 - residual / total if total else 0.0


class Forecaster(_Base):
    """Hosted, in-context: fit uploads your rows and trains nothing; predict reads them as context.

    The history is the context: each series' past values of `target` at
    `time`. `forecast` takes the future rows, each series' times and any
    covariates known then, and predicts `horizon` steps.
    """

    _task = "forecast"

    def __init__(
        self,
        model: str = "chronos-2",
        *,
        target: str,
        time: str,
        series: Sequence[str] = (),
        horizon: int,
        frequency: str | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        precision: Precision = "fast",
        verbose: bool = True,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the history.

        Args:
            model: A short name such as `chronos-2`, or a full id.
            target: The column to forecast.
            time: The time column.
            series: The columns that name a series; none for one series.
            horizon: How many steps to forecast.
            frequency: The step, such as `1d`; None infers it from the history.
            quantiles: The levels each forecast gives.
            precision: What its matrix multiplies run at: `fast` (TF32) or
                `exact` (float32), which costs more.
            verbose: Print one line to stderr for each call to the service.
            client: The `gf.Client` to call; None uses the module's.
        """
        self.model = model
        self.target = target
        self.time = time
        self.series = series
        self.horizon = horizon
        self.frequency = frequency
        self.quantiles = quantiles
        self.precision = precision
        self.verbose = verbose
        self.client = client

    def fit(self, X: Any, y: Any = None) -> Self:
        """Uploads the history as the context."""
        self._forget()
        history = _data.table(X, what="X")
        if y is not None:
            values = np.asarray(y, dtype=float).ravel()
            if len(values) != history.num_rows:
                raise InvalidInput(
                    f"X has {history.num_rows} rows and y has {len(values)}", param="y"
                )
            if self.target in history.column_names:
                history = history.drop_columns([self.target])
            history = history.append_column(self.target, pa.array(values))
        for name in (self.time, *self.series, self.target):
            if name not in history.column_names:
                raise InvalidInput(f"the history has no column `{name}`", param="X")
        self._upload(history)
        return self

    def predict_quantiles(self, X: Any) -> np.ndarray:
        """The forecast quantiles of each future row: one column per level."""
        future = _data.table(X, what="X")
        table = self._predict(
            future,
            self.target,
            task="forecast",
            time=self.time,
            series=list(self.series),
            horizon=self.horizon,
            frequency=self.frequency,
            quantiles=list(self.quantiles),
        ).table
        return np.column_stack(
            [
                np.asarray(table[f"q_{float(q):g}"].to_pylist(), dtype=float)
                for q in self.quantiles
            ]
        )

    def predict(self, X: Any) -> np.ndarray:
        """The forecast median of each future row."""
        levels = [float(q) for q in self.quantiles]
        if 0.5 not in levels:
            raise InvalidInput("predict reports the median; add 0.5 to quantiles")
        return self.predict_quantiles(X)[:, levels.index(0.5)]

    def forecast(self, future: Any) -> np.ndarray:
        """The forecast quantiles of the future rows: series, times and known covariates."""
        return self.predict_quantiles(future)

    def estimate(self, future: Any) -> Estimate:
        """What forecasting `future` would cost, priced without calling it, the large-context line included."""
        return self._estimate(_data.table(future, what="X").num_rows)
