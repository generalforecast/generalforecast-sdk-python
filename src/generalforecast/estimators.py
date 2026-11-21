# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""scikit-learn style estimators that predict on the hosted service.

`fit` keeps the labeled rows; `predict` sends them as the context of an ad
hoc prediction (`Client.predict`) with the rows to score. Nothing is
trained or run in this process. A model reads at most so many context rows
in one request; past that, push the rows as a context and a predictor that
reads it (`gf init`, `gf check`, `gf push`), or pass a `selection` that
draws each query's context from them.

The estimators follow scikit-learn's conventions (`get_params`,
`set_params`, fitted attributes ending in `_`) without depending on it.
"""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import pyarrow as pa

from generalforecast import tabular
from generalforecast.errors import CapacityExceeded, Error, InvalidInput

__all__ = ["Classifier", "Forecaster", "NotFittedError", "Regressor"]


class NotFittedError(InvalidInput, AttributeError):
    """An estimator was asked to predict before `fit`."""


def ProtocolError(message: str) -> Error:
    """A reply the estimator cannot read."""
    return Error("ProtocolError", message, status=502)


class _Base:
    """`get_params`, `set_params` and `repr` from the constructor's keyword arguments."""

    @classmethod
    def _param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return [p for p in signature.parameters if p != "self"]

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
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def _hosted(self) -> Any:
        if getattr(self, "client", None) is None:
            from generalforecast.client import Client

            self.client = Client()
        return self.client

    def _require_fitted(self) -> None:
        if not hasattr(self, "context_"):
            raise NotFittedError(
                f"this {type(self).__name__} is not fitted",
                hint="call fit with the labeled rows first",
            )

    def _check_rows(self, rows: int, inputs: int = 0) -> None:
        """Refuses more training rows than one request carries at `inputs` columns, unless a selection draws them."""
        from generalforecast.catalog import ModelInfo

        client = self._hosted()
        entry = ModelInfo.from_reply(client.models.get(self.model))
        # The client's predictions read the same entry, not the route again.
        seen = getattr(client, "_models_seen", None)
        if isinstance(seen, dict):
            seen[self.model] = entry
        limit = entry.limits.rows(inputs)
        at = f" at {inputs:,} inputs" if limit < entry.limits.context_rows else ""
        if rows > limit and not getattr(self, "selection", None):
            raise CapacityExceeded(
                f"{rows:,} training rows exceed one request",
                because=f"{entry.id} reads at most {limit:,} context rows per call{at}",
                hint="push the rows as a context and a predictor that reads it "
                "(`gf init`, `gf check`, `gf push`), or pass selection= to draw "
                "each query's context from them",
                limit=limit,
                actual=rows,
                param="context",
            )


def _features(X: Any, names: Sequence[str] | None = None) -> pa.Table:
    """Rows as a table, numpy arrays named `f0`, `f1`, ... unless `names` are given."""
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(-1, 1)
    if isinstance(X, np.ndarray):
        columns = list(names) if names else [f"f{i}" for i in range(X.shape[1])]
        if len(columns) != X.shape[1]:
            raise InvalidInput(
                f"X has {X.shape[1]} columns; the estimator was fit on {len(columns)}"
            )
        return pa.table({name: X[:, i] for i, name in enumerate(columns)})
    return tabular.frame(X)


def _target_name(columns: Sequence[str]) -> str:
    """A column name for y that no feature already takes."""
    name = "target"
    while name in columns:
        name = f"_{name}"
    return name


def _labels(y: Any) -> list[Any]:
    values = np.asarray(y).ravel().tolist()
    return [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]


class _Tabular(_Base):
    """The shared fit and prediction of `Classifier` and `Regressor`."""

    def __init__(
        self,
        model: str,
        *,
        selection: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        """Keeps the settings, as scikit-learn's estimators do; `fit` keeps the rows."""
        self.model = model
        self.selection = selection
        self.client = client

    def _fit(self, X: Any, y: Any) -> pa.Table:
        self._forget()
        features = _features(X)
        labels = _labels(y)
        if len(labels) != features.num_rows:
            raise InvalidInput(
                f"X has {features.num_rows} rows and y has {len(labels)}",
                param="y",
            )
        self._check_rows(features.num_rows, features.num_columns)
        self.feature_names_in_ = np.asarray(features.column_names, dtype=object)
        self.n_features_in_ = features.num_columns
        self.target_ = _target_name(features.column_names)
        self.context_ = features.append_column(self.target_, pa.array(labels))
        return self.context_

    def _queries(self, X: Any) -> pa.Table:
        self._require_fitted()
        names = list(self.feature_names_in_)
        queries = _features(X, names)
        missing = [n for n in names if n not in queries.column_names]
        if missing:
            raise InvalidInput(f"X has no column `{missing[0]}`", param="X")
        return queries.select(names)

    def _result(self, X: Any, **options: Any) -> pa.Table:
        queries = self._queries(X)
        result = self._hosted().predict(
            self.model,
            self.target_,
            self.context_,
            queries,
            inputs=list(self.feature_names_in_),
            selection=dict(self.selection) if self.selection else None,
            **options,
        )
        table = result.table
        if table.num_rows != queries.num_rows:
            raise ProtocolError("the service answered a different number of rows")
        return table


class Classifier(_Tabular):
    """Classifies rows in context of the labeled rows it was fit on.

    Args:
        model: The catalog classifier.
        selection: How each query's context is drawn: core, neighbors, kind;
            the whole fit set by default.
        client: The hosted client; one from the profile if None.
    """

    def __init__(
        self,
        model: str = "general-forecast/tabicl-v2-classifier",
        *,
        selection: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` keeps the rows."""
        super().__init__(model, selection=selection, client=client)

    def fit(self, X: Any, y: Any) -> Self:
        """Keeps the labeled rows as the context, with `classes_` in sorted order."""
        context = self._fit(X, y)
        labels = [v for v in context[self.target_].to_pylist() if v is not None]
        self.classes_ = np.asarray(
            sorted(set(labels), key=lambda v: (str(type(v)), v)), dtype=object
        )
        if len(self.classes_) < 2:
            self._forget()
            raise InvalidInput("y holds one class; a classifier needs two", param="y")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Probabilities over `classes_`, one row per query."""
        table = self._result(X, outputs=["label", "probabilities"])
        if "probabilities" not in table.column_names:
            raise ProtocolError("the service returned no probabilities")
        keys = [
            str(label).lower() if isinstance(label, (bool, np.bool_)) else str(label)
            for label in self.classes_
        ]
        rows = table["probabilities"].to_pylist()
        rows = [dict(row) if isinstance(row, list) else row for row in rows]
        if not all(isinstance(row, dict) and set(row) == set(keys) for row in rows):
            raise ProtocolError("the service returned different classes")
        return np.asarray([[row[key] for key in keys] for row in rows], dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        """The most probable label for each query."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X: Any, y: Any) -> float:
        """Mean accuracy on the labeled rows."""
        predicted = self.predict(X)
        return float(
            np.mean([a == b for a, b in zip(predicted, _labels(y), strict=True)])
        )


class Regressor(_Tabular):
    """Predicts values in context of the rows it was fit on.

    Args:
        model: The catalog regressor.
        selection: How each query's context is drawn: core, neighbors, kind;
            the whole fit set by default.
        client: The hosted client; one from the profile if None.
    """

    def __init__(
        self,
        model: str = "general-forecast/tabicl-v2-regressor",
        *,
        selection: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` keeps the rows."""
        super().__init__(model, selection=selection, client=client)

    def fit(self, X: Any, y: Any) -> Self:
        """Keeps the rows and their numeric targets as the context."""
        try:
            np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise InvalidInput(
                "y holds text, but a Regressor predicts numbers",
                hint="use a Classifier to predict labels",
                param="y",
            ) from None
        self._fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """The predicted value of each query."""
        table = self._result(X, outputs=["value"])
        if "value" not in table.column_names:
            raise ProtocolError("the service returned no values")
        return np.asarray(table["value"].to_pylist(), dtype=float)

    def predict_quantiles(self, X: Any, quantiles: Sequence[float]) -> np.ndarray:
        """The predicted quantiles of each query: one row per query, one column per level.

        Args:
            X: Rows to predict.
            quantiles: 1 to 99 levels strictly between 0 and 1, such as
                `[0.05, 0.95]` for a 90% interval.
        """
        levels = [float(level) for level in quantiles]
        if not 0 < len(levels) <= 99 or any(not 0 < level < 1 for level in levels):
            raise InvalidInput(
                "quantiles are 1 to 99 levels strictly between 0 and 1",
                param="quantiles",
            )
        table = self._result(X, quantiles=levels, outputs=["value", "quantiles"])
        if "quantiles" not in table.column_names:
            raise ProtocolError("the service returned no quantiles")
        rows = table["quantiles"].to_pylist()
        rows = [dict(row) if isinstance(row, list) else row for row in rows]
        out = []
        for row in rows:
            by_level = {float(key): value for key, value in row.items()}
            out.append([by_level[level] for level in levels])
        return np.asarray(out, dtype=float)

    def score(self, X: Any, y: Any) -> float:
        """The coefficient of determination, R^2, on the labeled rows."""
        truth = np.asarray(y, dtype=float).ravel()
        predicted = self.predict(X)
        total = float(((truth - truth.mean()) ** 2).sum())
        residual = float(((truth - predicted) ** 2).sum())
        return 1.0 - residual / total if total else 0.0


class Forecaster(_Base):
    """Forecasts series ahead of the history it was fit on.

    `fit` takes the history: time, series keys, covariates and the target
    (or a separate `y`). `predict` takes the future rows: `horizon` rows per
    series with their times and future covariates, and returns their values
    and quantiles.

    Args:
        model: The catalog forecaster.
        target: The target column.
        time: The time column.
        series: The series columns; none for one series.
        horizon: Steps ahead each series is forecast.
        frequency: The series' step, such as `1d`; inferred if None.
        past_inputs: Covariates known only up to the present.
        future_inputs: Covariates whose values each future row brings.
        quantiles: The levels to report.
        client: The hosted client; one from the profile if None.
    """

    def __init__(
        self,
        model: str = "general-forecast/chronos-2",
        *,
        target: str = "target",
        time: str = "time",
        series: str | Sequence[str] = (),
        horizon: int = 1,
        frequency: str | float | None = None,
        past_inputs: Sequence[str] = (),
        future_inputs: Sequence[str] = (),
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        client: Any = None,
    ) -> None:
        """Keeps the settings; `fit` keeps the history."""
        self.model = model
        self.target = target
        self.time = time
        self.series = series
        self.horizon = horizon
        self.frequency = frequency
        self.past_inputs = past_inputs
        self.future_inputs = future_inputs
        self.quantiles = quantiles
        self.client = client

    def _series(self) -> list[str]:
        return [self.series] if isinstance(self.series, str) else list(self.series)

    def fit(self, X: Any, y: Any = None) -> Self:
        """Keeps the history as the context."""
        self._forget()
        history = tabular.frame(X)
        if y is not None:
            values = np.asarray(y, dtype=float).ravel()
            if len(values) != history.num_rows:
                raise InvalidInput(
                    f"X has {history.num_rows} rows and y has {len(values)}", param="y"
                )
            if self.target in history.column_names:
                history = history.drop_columns([self.target])
            history = history.append_column(self.target, pa.array(values))
        wanted = [
            self.time,
            *self._series(),
            *self.past_inputs,
            *self.future_inputs,
            self.target,
        ]
        missing = [name for name in wanted if name not in history.column_names]
        if missing:
            raise InvalidInput(f"the history has no column `{missing[0]}`", param="X")
        self._check_rows(history.num_rows)
        self.context_ = history.select(list(dict.fromkeys(wanted)))
        return self

    def predict_quantiles(self, X: Any) -> np.ndarray:
        """The forecast quantiles of each future row, in its order: one column per level."""
        self._require_fitted()
        future = tabular.frame(X)
        series = self._series()
        result = self._hosted().predict(
            self.model,
            self.target,
            self.context_,
            future,
            inputs=[*self.past_inputs, *self.future_inputs],
            time=self.time,
            series=series or None,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=list(self.quantiles),
            future=list(self.future_inputs),
        )
        table = result.table
        if "quantiles" not in table.column_names or table.num_rows != future.num_rows:
            raise ProtocolError("the service returned no forecast quantiles")
        rows = table["quantiles"].to_pylist()
        rows = [dict(row) if isinstance(row, list) else row for row in rows]
        out = []
        for row in rows:
            by_level = {float(key): value for key, value in row.items()}
            if any(float(level) not in by_level for level in self.quantiles):
                raise ProtocolError("the service returned different forecast quantiles")
            out.append([by_level[float(level)] for level in self.quantiles])
        return np.asarray(out, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        """The forecast median of each future row."""
        levels = [float(level) for level in self.quantiles]
        if 0.5 not in levels:
            raise InvalidInput("predict reports the median; add 0.5 to quantiles")
        return self.predict_quantiles(X)[:, levels.index(0.5)]

    def forecast(self, horizon: int | None = None, *, future: Any = None) -> np.ndarray:
        """Forecasts every fitted series past its last observation.

        Args:
            horizon: Steps to forecast; the fitted horizon if None.
            future: The future rows, one per series and step, with the series,
                time and every `future_inputs` column; `future_grid()` lists
                the times. Required when the forecaster has `future_inputs`.

        Returns:
            Quantiles `(series * horizon, levels)`, one row per series and
            step, series in the order they were first seen (the order of
            `future` when given).
        """
        self._require_fitted()
        if horizon is not None and horizon != self.horizon:
            context = self.context_
            self.set_params(horizon=horizon)
            self.fit(context)
        if future is None:
            if self.future_inputs:
                raise InvalidInput(
                    "this forecaster has known-future inputs; pass forecast(future=...)",
                    param="future",
                    hint="future rows need "
                    + ", ".join(self.future_inputs)
                    + f" at each of the {self.horizon} times future_grid() lists",
                )
            future = self.future_grid()
        return self.predict_quantiles(future)

    def future_grid(self) -> Any:
        """The time grid `forecast` scores: `horizon` steps after each series.

        Returns:
            A pandas DataFrame with the series and time columns, one row per
            series and step, series in the order they were first seen.
        """
        from generalforecast import frequency

        self._require_fitted()
        series = self._series()
        times: dict[tuple, list[Any]] = {}
        for row in self.context_.select([*series, self.time]).to_pylist():
            key = tuple(row[name] for name in series)
            times.setdefault(key, []).append(row[self.time])
        try:
            step = frequency.infer(times) if self.frequency is None else self.frequency
        except frequency.FrequencyError as error:
            raise InvalidInput(
                f"cannot place the forecast times: {error}".rstrip("."),
                hint="pass frequency=, such as '1d'",
                param="frequency",
            ) from None
        rows = []
        for key, values in times.items():
            last = max(values, key=frequency.instant)
            for when in frequency.grid(last, self.horizon, step):
                rows.append({**dict(zip(series, key, strict=True)), self.time: when})
        import pandas as pd

        return pd.DataFrame(rows, columns=[*series, self.time])
