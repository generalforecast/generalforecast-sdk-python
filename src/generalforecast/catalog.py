# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The hosted model catalog as the SDK reads it: each model's task and per-call limits.

The service's `GET /models` lists the catalog and `GET /models/{name}`
shows one model. A reply's `limits` bound one call: context rows, query
rows and input columns. `DEFAULTS` is the model `gf init` proposes for
each task.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from generalforecast.errors import InvalidInput, NotFound

ORGANIZATION = "general-forecast"
TASKS = ("classification", "regression", "forecast")
DEFAULTS: Mapping[str, str] = {
    "classification": f"{ORGANIZATION}/kumo-tabular-medium-classifier",
    "regression": f"{ORGANIZATION}/kumo-tabular-medium-regressor",
    "forecast": f"{ORGANIZATION}/chronos-2",
}


@dataclass(frozen=True)
class Limits:
    """What one call to a model may carry."""

    context_rows: int
    queries: int
    inputs: int


@dataclass(frozen=True)
class ModelInfo:
    """One catalog model: its id, task and per-call limits."""

    id: str
    task: str
    limits: Limits

    @classmethod
    def from_reply(cls, value: Mapping[str, Any]) -> "ModelInfo":
        """A `Model` reply of the API, read for its name, task and limits."""
        limits = value.get("limits")
        limits = limits if isinstance(limits, Mapping) else {}

        def bound(key: str) -> int:
            number = limits.get(key)
            return number if type(number) is int and number > 0 else 1 << 62

        return cls(
            id=str(value.get("name") or value.get("model") or "").partition("@sha256:")[
                0
            ],
            task=str(value.get("task") or ""),
            limits=Limits(bound("context_rows"), bound("queries"), bound("inputs")),
        )


class Catalog:
    """The models one listing holds, read by id."""

    def __init__(self, models: Iterable[ModelInfo]) -> None:
        """Keeps the listing in the order the service gave it."""
        self.models = tuple(models)

    @classmethod
    def from_reply(cls, models: Iterable[Mapping[str, Any]]) -> "Catalog":
        """The catalog of a `GET /models` reply's `models`."""
        return cls(ModelInfo.from_reply(model) for model in models)

    def catalog(self, task: str | None = None) -> list[ModelInfo]:
        """The models, optionally for one task."""
        return [m for m in self.models if task is None or m.task == task]

    def default(self, task: str) -> str:
        """The model `gf init` proposes for a task: `DEFAULTS`, else the task's first."""
        listed = self.catalog(task)
        if any(m.id == DEFAULTS.get(task) for m in listed) or not listed:
            return DEFAULTS[task]
        return listed[0].id

    def canonical(self, ref: str) -> str:
        """`general-forecast/<name>[@variant]`, or an error naming the right id."""
        base, _, variant = ref.partition("@")
        suffix = f"@{variant}" if variant else ""
        for model in self.models:
            if base == model.id:
                return ref
            if base == model.id.rpartition("/")[2]:
                raise InvalidInput(
                    f"model `{ref}` is not a canonical id",
                    hint=f"use {model.id}{suffix}",
                    param="model",
                )
        raise NotFound(
            f"model `{ref}` is not in the catalog",
            hint="run `gf model list` to see the catalog",
            param="model",
        )

    def entry(self, ref: str) -> ModelInfo:
        """The listed model a reference names."""
        base = self.canonical(ref).partition("@")[0]
        return next(m for m in self.models if m.id == base)

    def task_of(self, ref: str) -> str:
        """classification, regression or forecast."""
        return self.entry(ref).task

    def limits(self, ref: str) -> Limits:
        """The model's per-call limits."""
        return self.entry(ref).limits
