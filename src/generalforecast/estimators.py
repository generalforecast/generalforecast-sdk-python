# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""scikit-learn style estimators over `models.predict`.

Hosted, in-context: fit uploads your rows and trains nothing; predict reads
them as context. `fit` uploads the rows once as an Arrow snapshot named by
its digest and refuses at once, with `CapacityExceeded`, a context too big
to send; `predict` sends the queries only. They follow scikit-learn's
conventions (`get_params`, `set_params`, fitted attributes ending in `_`)
without depending on it.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pyarrow as pa

from generalforecast import _data, _ops
from generalforecast.errors import InvalidInput
from generalforecast.types import Estimate, Predictions

__all__ = ["Classifier", "Forecaster", "NotFittedError", "Regressor"]


class NotFittedError(InvalidInput, AttributeError):
    """An estimator was asked to predict before `fit`."""


class _Base:
    """Parameters, the client, the uploaded context and the usage so far."""

    model: str
    client: Any
    context_: str
    context_rows_: int
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
        self.usage_ = Estimate()

    def _predict(self, queries: pa.Table, target: str, **fields: Any) -> Predictions:
        from generalforecast.types import Context

        self._require_fitted()
        client = self._hosted()
        reference = Context(name="fit", digest=self.context_, rows=self.context_rows_)
        result = client.models.predict(
            self.model, context=reference, queries=queries, target=target, **fields
        )
        if result.table.num_rows != queries.num_rows:
            raise InvalidInput("the service answered a different number of rows")
        self.last_usage_ = Estimate(1, result.rows, result.credits, result.usd)
        self.usage_ = self.usage_ + self.last_usage_
        return result


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
    _task: str

    def __init__(
        self, model: str, *, inputs: Sequence[str] | None = None, client: Any = None
    ) -> None:
        self.model = model
        self.inputs = inputs
        self.client = client

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
        """What predicting `X` would cost, priced without calling it."""
        queries = self._queries(X)
        client = self._hosted()
        return client._run(
            _ops.predict_estimate(
                self.model, self._task, self.context_rows_, queries.num_rows
            )
        )

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

    def __init__(
        self,
        model: str = "tabicl-v2",
        *,
        inputs: Sequence[str] | None = None,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the rows."""
        super().__init__(model, inputs=inputs, client=client)

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

    def __init__(
        self,
        model: str = "tabicl-v2",
        *,
        quantiles: Sequence[float] | None = None,
        inputs: Sequence[str] | None = None,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the rows."""
        super().__init__(model, inputs=inputs, client=client)
        self.quantiles = quantiles

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
    """Hosted, in-context: fit uploads your rows and trains nothing; predict reads them as context."""

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
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` uploads the history."""
        self.model = model
        self.target = target
        self.time = time
        self.series = series
        self.horizon = horizon
        self.frequency = frequency
        self.quantiles = quantiles
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
