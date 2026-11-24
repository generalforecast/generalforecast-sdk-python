# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The objects the SDK returns and the specs it takes: frozen dataclasses.

They hold data only; no method makes a network call. Times are
timezone-aware `datetime`s and money is `credits` (1 credit = $0.001) beside
`usd`. Every reference a method takes is a name or the object itself, whose
`name` is used.
"""

from __future__ import annotations

import builtins
import re
import types as _types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields, is_dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, Union

from generalforecast.errors import InvalidInput

if TYPE_CHECKING:
    import pandas
    import pyarrow

__all__ = [
    "STANDARD_PERMISSIONS",
    "BatchTable",
    "Change",
    "Column",
    "ColumnType",
    "Context",
    "ContextSource",
    "ContextUse",
    "DataPin",
    "Deploy",
    "Entry",
    "Environment",
    "EnvironmentFailure",
    "Estimate",
    "Experiment",
    "FineTuneRun",
    "Holdout",
    "Key",
    "KeyOwner",
    "LargeContext",
    "Limits",
    "Lineage",
    "LogEntry",
    "Logs",
    "Me",
    "Model",
    "NewKey",
    "Origin",
    "Predictions",
    "Predictor",
    "PredictorInfo",
    "Published",
    "Report",
    "Resolved",
    "Revision",
    "Run",
    "RunEvent",
    "RunEventKind",
    "RunSummary",
    "Selection",
    "Slice",
    "Snapshot",
    "Source",
    "SourceKind",
    "SourceList",
    "SourceState",
    "SourceTable",
    "SourceTables",
    "SyntheticBatch",
    "Task",
    "Timing",
    "TrainedOn",
    "UploadLimits",
    "WriteReceipt",
]

Task = Literal["classification", "regression", "forecast"]
ColumnType = Literal["number", "category", "text", "boolean", "date", "timestamp"]
SourceKind = Literal["bigquery", "gcs", "snowflake", "databricks", "s3", "postgres"]
TASKS: tuple[str, ...] = typing.get_args(Task)
COLUMN_TYPES: tuple[str, ...] = typing.get_args(ColumnType)
SOURCE_KINDS: tuple[str, ...] = typing.get_args(SourceKind)

STANDARD_PERMISSIONS: tuple[str, ...] = (
    "predictors:read",
    "predictors:write",
    "predictors:predict",
    "contexts:read",
    "contexts:write",
    "snapshots:read",
    "snapshots:write",
    "models:read",
)
"""The permissions the console pre-checks for a new key."""

NAME = re.compile(r"^[a-z][a-z0-9-]{0,62}$")


def instant(value: Any) -> datetime | None:
    """An ISO 8601 string or Unix seconds as an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


@cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _empty(kind: Any) -> Any:
    origin = typing.get_origin(kind)
    if origin in (Union, _types.UnionType) and type(None) in typing.get_args(kind):
        return None
    if origin is tuple:
        return ()
    if origin in (Mapping, dict) or kind is Mapping:
        return {}
    return {str: "", int: 0, float: 0.0, bool: False}.get(kind)


def _convert(kind: Any, value: Any) -> Any:
    if kind is Any:
        return value
    origin = typing.get_origin(kind)
    args = typing.get_args(kind)
    if origin in (Union, _types.UnionType):
        if value is None:
            return None
        options = [a for a in args if a is not type(None)]
        for option in options:
            if is_dataclass(option) and isinstance(value, Mapping):
                return load(option, value)
        for option in options:
            if option in (str, int, float, bool) and isinstance(value, option):
                return value
        return _convert(options[0], value)
    if value is None:
        return _empty(kind)
    if origin is Literal:
        return value
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value or ())
        return tuple(_convert(a, v) for a, v in zip(args, value, strict=False))
    if origin in (Mapping, dict) or kind is Mapping:
        item = args[1] if len(args) == 2 else Any
        return {str(k): _convert(item, v) for k, v in dict(value).items()}
    if is_dataclass(kind):
        return load(kind, value) if isinstance(value, Mapping) else value
    if kind is datetime:
        return instant(value)
    if kind is date:
        return date.fromisoformat(str(value)[:10]) if value else None
    if kind is timedelta:
        return timedelta(seconds=float(value))
    if kind is str:
        if isinstance(value, Mapping):
            return str(value.get("message") or value.get("name") or value)
        return str(value)
    if kind is bool:
        return bool(value)
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    return value


def load[T](cls: type[T], data: Mapping[str, Any]) -> T:
    """A dataclass from a service result, leniently.

    A field missing from the result takes its default, or an empty value;
    keys the class does not know are ignored, so a newer service still reads.

    Raises:
        InvalidInput: For a result that is not an object.
    """
    if not isinstance(data, Mapping):
        raise InvalidInput(f"the service sent a {cls.__name__} that is not an object")
    custom = getattr(cls, "_load", None)
    if custom is not None:
        return custom(data)
    hints = _hints(cls)
    values: dict[str, Any] = {}
    for item in fields(typing.cast(Any, cls)):
        if not item.init:
            continue
        key = item.metadata.get("wire", item.name)
        if key in data:
            values[item.name] = _convert(hints[item.name], data[key])
        elif item.name in data:
            values[item.name] = _convert(hints[item.name], data[item.name])
        elif item.default is MISSING and item.default_factory is MISSING:
            values[item.name] = _empty(hints[item.name])
    return cls(**values)


def loads[T](cls: type[T], items: Any) -> builtins.list[T]:
    """A list of dataclasses from a result's list."""
    return [load(cls, item) for item in items or ()]


def money(usd: float | None) -> str:
    """Dollars to two significant figures: `$0.013`, `$0.04`, `$12.50`."""
    if usd is None:
        return ""
    if usd >= 1:
        return f"${usd:,.2f}"
    if usd == 0:
        return "$0"
    text = f"{usd:.2g}"
    if "e" in text:
        text = f"{usd:.6f}".rstrip("0")
    return "$" + text


@dataclass(frozen=True)
class Origin:
    """Where a resource was made: the surface, the notebook tool, the client and who."""

    surface: Literal["console", "cli", "notebook", "script", "api"] = "api"
    notebook: str | None = None
    client: str | None = None
    by: str | None = None


@dataclass(frozen=True)
class Column:
    """A context column's profile."""

    name: str
    type: ColumnType
    missing: int = 0
    distinct: int | None = None


@dataclass(frozen=True)
class ContextSource:
    """The source a context imports from or reads live."""

    source: str
    kind: SourceKind | None = None
    table: str | None = None
    sql: str | None = None


@dataclass(frozen=True, repr=False)
class Context:
    """A context: data a model learns from, uploaded, imported or live.

    `predictors` and `experiments` name what reads it, and `fine_tunes`
    the fine-tuned models made from it, as `acme/credit-default@ft3`.
    `file` names the
    file an uploaded version came from, when it came from one. `unchanged`
    is true on an upload whose rows matched the latest version, which then
    stays the latest.
    """

    name: str
    kind: Literal["snapshot", "live"] = "snapshot"
    state: Literal["uploading", "importing", "ready", "failed"] = "ready"
    version: int | None = None
    versions: int = 0
    digest: str | None = None
    rows: int | None = None
    columns: tuple[Column, ...] = ()
    id: str | None = None
    time: str | None = None
    source: ContextSource | None = None
    freshness: timedelta | None = field(
        default=None, metadata={"wire": "freshness_seconds"}
    )
    read_at: datetime | None = None
    error: str | None = None
    predictors: tuple[str, ...] = ()
    experiments: tuple[str, ...] = ()
    fine_tunes: tuple[str, ...] = ()
    origin: Origin | None = None
    created: datetime | None = None
    updated: datetime | None = None
    url: str | None = None
    file: str | None = None
    unchanged: bool = False

    @property
    def ref(self) -> str:
        """The name pinned to this version, `adult@2`, or the name alone."""
        return f"{self.name}@{self.version}" if self.version else self.name

    def __repr__(self) -> str:
        """`Context adult  snapshot v1  32,561 rows  15 columns`."""
        from generalforecast import _render

        return _render.context_line(self)

    def _repr_html_(self) -> str:
        from generalforecast import _render

        return _render.context_html(self)

    def _mime_(self) -> tuple[str, str]:
        return ("text/html", self._repr_html_())


@dataclass(frozen=True)
class WriteReceipt:
    """Rows accepted into an API-fed live context."""

    context: str
    rows: int = 0
    write_id: str = ""
    sequence: int = 0


_HOLDOUT = re.compile(
    r"^\s*(?:(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?P<rest>.*)|(?P<rows>\d[\d,_]*)\s+rows?"
    r"|until\s+(?P<until>\d{4}-\d{2}-\d{2})|(?P<none>none))\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Holdout:
    """How an experiment holds out rows to score on.

    `Holdout.parse` reads the short forms: `"20%"`, `"20% stratified"`,
    `"20% by time"`, `"20% by <column>"`, `"1000 rows"`,
    `"until 2025-06-01"` and `"none"`.
    """

    kind: Literal["random", "group", "time", "none"] = "random"
    fraction: float | None = None
    rows: int | None = None
    column: str | None = None
    until: date | None = None
    stratify: bool = False
    seed: int = 0
    repeats: int = 1

    def __post_init__(self) -> None:
        """Checks the fields agree with the kind."""
        if self.kind not in ("random", "group", "time", "none"):
            raise InvalidInput(f"`{self.kind}` is not a holdout kind")
        if self.fraction is not None and not 0 < self.fraction < 1:
            raise InvalidInput("a holdout fraction is between 0 and 1, exclusive")
        if self.kind == "group" and not self.column:
            raise InvalidInput("a group holdout names its column")

    @classmethod
    def parse(cls, text: str) -> Holdout:
        """A holdout from its short form.

        Raises:
            InvalidInput: For text that is none of the forms.
        """
        found = _HOLDOUT.match(text or "")
        if found is None:
            raise InvalidInput(
                f"`{text}` is not a holdout",
                hint='write "20%", "20% stratified", "20% by time", '
                '"20% by <column>", "1000 rows", "until 2025-06-01" or "none"',
                param="holdout",
            )
        if found["none"]:
            return cls("none")
        if found["until"]:
            return cls("time", until=date.fromisoformat(found["until"]))
        if found["rows"]:
            return cls(
                "random", rows=int(found["rows"].replace(",", "").replace("_", ""))
            )
        fraction = float(found["pct"]) / 100
        rest = (found["rest"] or "").strip()
        if not rest:
            return cls("random", fraction)
        if rest.lower() == "stratified":
            return cls("random", fraction, stratify=True)
        by = re.match(r"^by\s+(\S+)$", rest, re.IGNORECASE)
        if by is None:
            raise InvalidInput(f"`{text}` is not a holdout", param="holdout")
        if by[1].lower() == "time":
            return cls("time", fraction)
        return cls("group", fraction, column=by[1])

    def to_json(self) -> dict[str, Any]:
        """The holdout as an experiment's setup sends it."""
        out: dict[str, Any] = {"kind": self.kind}
        for key in ("fraction", "rows", "column"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.until is not None:
            out["until"] = self.until.isoformat()
        if self.stratify:
            out["stratify"] = True
        if self.seed:
            out["seed"] = self.seed
        if self.repeats != 1:
            out["repeats"] = self.repeats
        return out

    def __str__(self) -> str:
        """The short form."""
        if self.kind == "none":
            return "none"
        if self.until is not None:
            return f"until {self.until.isoformat()}"
        if self.rows is not None:
            return f"{self.rows} rows"
        pct = f"{(self.fraction or 0.2) * 100:g}%"
        if self.kind == "time":
            return f"{pct} by time"
        if self.kind == "group":
            return f"{pct} by {self.column}"
        return f"{pct} stratified" if self.stratify else pct


@dataclass(frozen=True)
class Entry:
    """One model's row on a run's leaderboard."""

    model: str
    short: str = ""
    state: str = "queued"
    score: float | None = None
    interval: tuple[float, float] | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    rows: int = 0
    credits: float = 0.0
    usd: float = 0.0
    error: str | None = None


@dataclass(frozen=True, repr=False)
class Run:
    """One run of an experiment: its models scored on the held-out rows."""

    number: int
    state: Literal["queued", "running", "finished", "failed", "cancelled"] = "queued"
    stale: bool = False
    models: tuple[str, ...] = ()
    metric: str = ""
    lower_is_better: bool = False
    held_rows: int = 0
    context_rows: int = 0
    repeats: int = 1
    leaderboard: tuple[Entry, ...] = ()
    rows: int = 0
    credits: float = 0.0
    usd: float = 0.0
    created: datetime | None = None
    started: datetime | None = None
    finished: datetime | None = None
    origin: Origin | None = None
    error: str | None = None
    experiment: str | None = None

    @property
    def done(self) -> bool:
        """Whether the run has ended."""
        return self.state in ("finished", "failed", "cancelled")

    def __repr__(self) -> str:
        """The leaderboard as aligned text."""
        from generalforecast import _render

        return "\n".join(_render.leaderboard(self))

    def _repr_html_(self) -> str:
        from generalforecast import _render

        return _render.leaderboard_html(self)

    def _mime_(self) -> tuple[str, str]:
        return ("text/html", self._repr_html_())


@dataclass(frozen=True)
class RunSummary:
    """A run as an experiment lists it."""

    number: int
    state: str = "queued"
    models: tuple[str, ...] = ()
    best: str | None = None
    score: float | None = None
    origin: Origin | None = None
    created: datetime | None = None


@dataclass(frozen=True, repr=False)
class Experiment:
    """Models evaluated on one context for one target, run by run."""

    name: str
    title: str | None = None
    context: str = ""
    context_version: int | None = None
    target: str = ""
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] = ()
    types: Mapping[str, ColumnType] = field(default_factory=dict)
    holdout: Holdout | None = None
    sql: str | None = None
    metric: str = ""
    runs: tuple[RunSummary, ...] = ()
    run: Run | None = None
    origin: Origin | None = None
    created: datetime | None = None
    updated: datetime | None = None
    url: str | None = None

    @property
    def best(self) -> Entry | None:
        """The run's best finished entry by its metric, or None."""
        if self.run is None:
            return None
        scored = [
            e
            for e in self.run.leaderboard
            if e.score is not None and e.state == "finished"
        ]
        if not scored:
            return None
        pick = min if self.run.lower_is_better else max
        return pick(scored, key=lambda e: e.score)  # type: ignore[arg-type, return-value]

    def __repr__(self) -> str:
        """The header line, then the leaderboard, then the console link."""
        from generalforecast import _render

        return _render.experiment_text(self)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        from generalforecast import _render

        return _render.experiment_html(self)

    def _mime_(self) -> tuple[str, str]:
        return ("text/html", self._repr_html_())


@dataclass(frozen=True)
class Slice:
    """How a model scores on the rows where one column takes one value."""

    column: str
    value: str
    rows: int = 0
    score: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class LargeContext:
    """The large-context line: GPU seconds past `large_context_cells`, and their cost."""

    gpu_seconds: float = 0.0
    credits: float = 0.0
    usd: float = 0.0


@dataclass(frozen=True, repr=False)
class Estimate:
    """What calls would cost, priced without making them."""

    calls: int = 0
    rows: int = 0
    credits: float = 0.0
    usd: float = 0.0
    large_context: LargeContext | None = None
    warnings: tuple[str, ...] = ()

    def __add__(self, other: Estimate) -> Estimate:
        """Two estimates together."""
        large = None
        if self.large_context or other.large_context:
            a = self.large_context or LargeContext()
            b = other.large_context or LargeContext()
            large = LargeContext(
                a.gpu_seconds + b.gpu_seconds, a.credits + b.credits, a.usd + b.usd
            )
        return Estimate(
            self.calls + other.calls,
            self.rows + other.rows,
            round(self.credits + other.credits, 6),
            round(self.usd + other.usd, 9),
            large,
            self.warnings + other.warnings,
        )

    def __repr__(self) -> str:
        """`Estimate(calls=6, rows=19539, credits=39.1, usd=0.0391)`."""
        text = (
            f"Estimate(calls={self.calls}, rows={self.rows}, "
            f"credits={self.credits:g}, usd={self.usd:g}"
        )
        if self.large_context is not None:
            text += f", large_context={self.large_context!r}"
        return text + ")"


@dataclass(frozen=True)
class Selection:
    """How a predictor draws each query's context from a large one: a core plus neighbors."""

    core: float | int | None = None
    neighbors: float | int | None = None
    kind: Literal["stratified", "random"] | None = None
    recent: int | None = None
    seed: int = 0

    def to_json(self) -> dict[str, Any]:
        """The selection as a spec sends it."""
        out = {
            k: getattr(self, k)
            for k in ("core", "neighbors", "kind", "recent")
            if getattr(self, k) is not None
        }
        if self.seed:
            out["seed"] = self.seed
        return out


PREDICTOR_FIELDS = (
    "context",
    "model",
    "target",
    "task",
    "id",
    "time",
    "inputs",
    "exclude",
    "types",
    "series",
    "frequency",
    "horizon",
    "quantiles",
    "past",
    "future",
    "selection",
)


def _names(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise InvalidInput(
            f"`{what}` is a list of column names, not one string",
            hint=f'write {what}=["{value}"]',
            param=what,
        )
    items = tuple(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise InvalidInput(f"`{what}` holds column names", param=what)
    return items


def ref_of(value: Any, what: str = "context") -> str:
    """The name, or pinned name, a reference stands for."""
    if isinstance(value, str):
        if not value:
            raise InvalidInput(f"{what} needs a name", param=what)
        return value
    if isinstance(value, Context):
        return value.ref
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    raise InvalidInput(f"{what} must be a name or a {what} object", param=what)


@dataclass(frozen=True)
class Predictor:
    """A predictor's spec; constructing one is local and makes no call.

    `context` is a stored context, `"adult"` (the latest version at each
    deploy) or `"adult@3"` (pinned); `model` a short name or a full id.
    `inputs=None` reads every other column.
    """

    name: str
    _: KW_ONLY
    context: str | Context
    model: str
    target: str
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: Sequence[str] | None = None
    exclude: Sequence[str] = ()
    types: Mapping[str, ColumnType] | None = None
    series: Sequence[str] = ()
    frequency: str | None = None
    horizon: int | None = None
    quantiles: Sequence[float] = ()
    past: Sequence[str] = ()
    future: Sequence[str] = ()
    selection: Selection | None = None

    def __post_init__(self) -> None:
        """Checks every field and freezes the sequences as tuples.

        Raises:
            InvalidInput: For a name, task, type or field the service would refuse.
        """
        if not isinstance(self.name, str) or not NAME.match(self.name):
            raise InvalidInput(
                f"`{self.name}` is not a predictor name",
                hint="start with a letter, then lowercase letters, digits and hyphens, at most 64",
                param="name",
            )
        ref_of(self.context)
        for key in ("model", "target"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise InvalidInput(f"a predictor needs `{key}`", param=key)
        if self.task is not None and self.task not in TASKS:
            raise InvalidInput(f"`{self.task}` is not a task", param="task")
        set_ = object.__setattr__
        if self.inputs is not None:
            set_(self, "inputs", _names(self.inputs, "inputs"))
        for key in ("exclude", "series", "past", "future"):
            set_(self, key, _names(getattr(self, key), key))
        quantiles = tuple(float(q) for q in self.quantiles)
        if any(not 0 < q < 1 for q in quantiles):
            raise InvalidInput("quantiles are between 0 and 1", param="quantiles")
        set_(self, "quantiles", quantiles)
        if self.types is not None:
            for column, kind in self.types.items():
                if kind not in COLUMN_TYPES:
                    raise InvalidInput(
                        f"`{kind}` is not a column type for `{column}`",
                        hint="types are " + ", ".join(COLUMN_TYPES),
                        param="types",
                    )
            set_(self, "types", dict(self.types))
        if self.horizon is not None and (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon < 1
        ):
            raise InvalidInput(
                "horizon is a whole number of steps, at least 1", param="horizon"
            )
        if self.selection is not None and not isinstance(self.selection, Selection):
            if isinstance(self.selection, Mapping):
                set_(self, "selection", Selection(**self.selection))
            else:
                raise InvalidInput("selection is a gf.Selection", param="selection")

    def __hash__(self) -> int:
        """Hashes by name and spec."""
        return hash((self.name, repr(self.to_json())))

    def to_json(self) -> dict[str, Any]:
        """The spec as the service takes it, defaults left out."""
        return spec_json(
            {key: getattr(self, key) for key in PREDICTOR_FIELDS}, partial=False
        )

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> Predictor:
        values: dict[str, Any] = {}
        for key in PREDICTOR_FIELDS:
            value = data.get(key)
            if value is None:
                continue
            if key == "selection":
                value = load(Selection, value)
            values[key] = value
        values.setdefault("context", "")
        values.setdefault("model", "")
        values.setdefault("target", "")
        predictor = object.__new__(cls)
        object.__setattr__(predictor, "name", data.get("name") or "")
        for item in fields(cls):
            if item.name in ("name", "_"):
                continue
            default = (
                item.default
                if item.default is not MISSING
                else item.default_factory()
                if item.default_factory is not MISSING
                else None
            )
            value = values.get(item.name, default)
            if isinstance(value, list):
                value = tuple(value)
            object.__setattr__(predictor, item.name, value)
        return predictor


def spec_json(values: Mapping[str, Any], *, partial: bool) -> dict[str, Any]:
    """Predictor fields as JSON; a partial spec keeps `None` to clear a field."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in PREDICTOR_FIELDS:
            raise InvalidInput(f"gf.Predictor has no field `{key}`", param=key)
        if key == "context" and value is not None:
            value = ref_of(value)
        elif isinstance(value, Selection):
            value = value.to_json()
        elif isinstance(value, tuple):
            value = list(value)
        if not partial and value in (None, [], (), {}):
            continue
        out[key] = value
    return out


@dataclass(frozen=True)
class EnvironmentFailure:
    """Why an environment's revision failed to start.

    `request_id` names the deploy or promote that started it, and
    `serving_revision` the revision the environment still serves (None
    when it served none).
    """

    message: str = ""
    request_id: str | None = None
    serving_revision: int | None = None


@dataclass(frozen=True)
class Environment:
    """Where a predictor serves a revision: `production`, `staging` and others.

    `resolved` is what the revision it serves resolved to, None while it
    serves none.
    """

    name: str
    revision: int = 0
    state: Literal["starting", "ready", "failed", "stopped"] = "starting"
    generation: int = 0
    resolved: Resolved | None = None
    calls_today: int = 0
    p50_seconds: float | None = None
    updated: datetime | None = None
    url: str | None = None
    error: EnvironmentFailure | None = None


@dataclass(frozen=True)
class Resolved:
    """A spec as the service resolved it: full model id and digest, context version, task, inputs.

    `base_digest` is, for a derived model (a project's fine-tune), the
    digest of the base it was trained from; None for a catalog model.
    """

    model: str = ""
    short: str = ""
    model_digest: str | None = None
    base_digest: str | None = None
    task: Task | None = None
    context: str = ""
    context_version: int | None = None
    context_digest: str | None = None
    inputs: tuple[str, ...] = ()
    types: Mapping[str, ColumnType] = field(default_factory=dict)
    context_rows: int | None = None


@dataclass(frozen=True)
class Revision:
    """One revision of a predictor: its spec and what it resolved to.

    `status` is where its build stands (`ready` once it can serve; whether
    an environment serves it is the environment's `state`), and
    `lock_digest` the digest of the lock the service keeps beside it.
    """

    number: int
    status: Literal["pending", "building", "ready", "failed", "stopped"] | None = None
    lock_digest: str | None = None
    spec: Predictor | None = None
    resolved: Resolved | None = None
    origin: Origin | None = None
    created: datetime | None = None


@dataclass(frozen=True, repr=False)
class PredictorInfo:
    """A stored predictor: its serving spec, revision and environments."""

    name: str
    spec: Predictor | None = None
    revision: int = 0
    environments: Mapping[str, Environment] = field(default_factory=dict)
    created: datetime | None = None
    url: str | None = None
    resolved: Resolved | None = None

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> PredictorInfo:
        environments = data.get("environments") or {}
        if isinstance(environments, list):
            environments = {e.get("name"): e for e in environments}
        spec = data.get("spec")
        if isinstance(spec, Mapping):
            spec = load(Predictor, {"name": data.get("name"), **spec})
        return cls(
            name=data.get("name") or "",
            spec=spec,
            revision=int(data.get("revision") or 0),
            environments={
                str(k): load(Environment, {"name": k, **v})
                for k, v in environments.items()
            },
            created=instant(data.get("created")),
            url=data.get("url"),
            resolved=load(Resolved, data["resolved"]) if data.get("resolved") else None,
        )

    def __repr__(self) -> str:
        """A short aligned summary."""
        from generalforecast import _render

        return _render.predictor_text(self)

    def _repr_html_(self) -> str:
        from generalforecast import _render

        return _render.predictor_html(self)

    def _mime_(self) -> tuple[str, str]:
        return ("text/html", self._repr_html_())


@dataclass(frozen=True)
class Change:
    """One field of a predictor's spec, before and after a deploy."""

    field: str
    before: str | None = None
    after: str | None = None


@dataclass(frozen=True)
class Deploy:
    """What deploying one spec did, or with `dry_run` would do."""

    predictor: str
    env: str = "production"
    action: Literal["created", "updated", "unchanged"] = "unchanged"
    changes: tuple[Change, ...] = ()
    revision: Revision | None = None
    environment: Environment | None = None


@dataclass(frozen=True)
class Timing:
    """Where a call's time went: waiting in the queue, a cold start, and compute.

    `calls` is the model calls the engine made to answer it, as many as
    the device decoded its queries in. `compile_seconds` is the time the
    call's steps spent compiling or loading programs they waited on, which
    `compute_seconds` leaves out.
    """

    calls: int = 0
    queued_seconds: float = 0.0
    cold_start_seconds: float = 0.0
    compute_seconds: float = 0.0
    compile_seconds: float = 0.0


@dataclass(frozen=True)
class ContextUse:
    """How much of the context a call read: whole, or retrieved past the device's memory.

    `cells` is rows times input columns, what the large-context line is
    drawn on; the service leaves `used`, `core` and `cells` out when it has
    nothing to say, and then the whole context was read.
    """

    rows: int = 0
    used: Literal["exact", "retrieved"] = "exact"
    core: int | None = None
    cells: int | None = None


@dataclass(frozen=True, repr=False)
class Predictions:
    """A prediction's answer and its usage.

    `table` has `id` (when the predictor declares one), then `label` and a
    `p_<class>` column per class, or `value` and a `q_<level>` per quantile.
    `gpu_seconds` is the large-context meter, set only past its line;
    `timing` keeps queueing and cold start apart from compute. `calls` is
    the model calls the engine made, its `timing.calls`, which the usage
    copies.
    """

    predictor: str | None
    revision: int | None
    environment: str | None
    model: str
    task: str | None
    rows: int
    credits: float
    usd: float
    request_id: str | None
    table: pyarrow.Table
    gpu_seconds: float | None = None
    timing: Timing | None = None
    context: ContextUse | None = None
    notes: tuple[str, ...] = ()
    calls: int = 0

    def to_pandas(self) -> pandas.DataFrame:
        """The predictions as a DataFrame."""
        return self.table.to_pandas()

    def __len__(self) -> int:
        """How many rows were predicted."""
        return self.table.num_rows

    def __repr__(self) -> str:
        """`Predictions 5 rows with tabicl-v2-classifier  0.01 credits ($0.00001)`."""
        from generalforecast._render import model_name

        model = model_name(self.model)
        text = f"Predictions {self.rows:,} rows with {model}  {self.credits:g} credits ({money(self.usd)})"
        if self.gpu_seconds:
            text += f"  {self.gpu_seconds:g} GPU seconds"
        return text


@dataclass(frozen=True)
class LogEntry:
    """One line of a predictor environment's log."""

    at: datetime | None = None
    level: str = "info"
    source: str | None = None
    code: str | None = None
    message: str = ""
    request_id: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Logs:
    """A page of log entries, with cursors to the next and previous pages."""

    entries: tuple[LogEntry, ...] = ()
    next: str | None = None
    previous: str | None = None


SourceState = Literal[
    "untested", "connected", "unauthorized", "not_found", "unavailable", "expired"
]
SOURCE_STATES: tuple[str, ...] = typing.get_args(SourceState)


@dataclass(frozen=True)
class Source:
    """A workspace connection to a warehouse or bucket.

    `access` says how the service reads it: as the workspace's
    `service_account`, or with the source's `secret`. `state` is what the
    last test found: `connected`, or why not (`unauthorized`, `not_found`,
    `unavailable`, `expired`), and `untested` before any test. `tables` is
    how many tables that test found, `contexts` how many of the project's
    contexts were made from it.
    """

    name: str
    kind: SourceKind | None = None
    access: Literal["service_account", "secret"] | None = None
    state: SourceState = "untested"
    config: Mapping[str, str] = field(default_factory=dict)
    secret_expires: datetime | None = None
    tested_at: datetime | None = None
    tables: int | None = None
    contexts: int = 0
    error: str | None = None
    origin: Origin | None = None
    created: datetime | None = None
    url: str | None = None


class SourceList(builtins.list[Source]):
    """The project's sources; `available` is False on a service that connects none."""

    available: bool = True

    def __init__(self, sources: Sequence[Source] = (), *, available: bool = True):
        """The sources listed, and whether the service connects any."""
        super().__init__(sources)
        self.available = available


@dataclass(frozen=True)
class SourceTable:
    """A table of a source's location, as the warehouse lists it.

    `type` is the warehouse's kind of table, such as `TABLE`, `VIEW` or
    `MODEL`; `readable` whether a context can be made of it.
    """

    name: str
    type: str = "TABLE"
    readable: bool = True


@dataclass(frozen=True)
class SourceTables:
    """The tables of a source's location: its BigQuery `project` and `dataset`.

    `more` says the warehouse holds more than were listed; a `prefix`
    narrows them. It iterates, indexes and counts as its tables do.
    """

    project: str = ""
    dataset: str = ""
    tables: tuple[SourceTable, ...] = ()
    more: bool = False

    def __iter__(self) -> typing.Iterator[SourceTable]:
        """The tables listed."""
        return iter(self.tables)

    def __len__(self) -> int:
        """How many tables were listed."""
        return len(self.tables)

    def __getitem__(self, index: int) -> SourceTable:
        """A listed table by its position."""
        return self.tables[index]


@dataclass(frozen=True)
class KeyOwner:
    """The signed-in account behind a key: `issuer` and `subject` name it, and stay when its email changes."""

    issuer: str = ""
    subject: str = ""
    email: str | None = None


@dataclass(frozen=True)
class Key:
    """An API key; its token is shown once, in `NewKey`.

    `owner` is the account that made it, None for a `service` key (one the
    service made for its own use) and a `legacy` one (made before keys
    recorded their owners). `machine` names the machine a `gf login` made
    it for.
    """

    id: str
    name: str = ""
    permissions: tuple[str, ...] = ()
    prefix: str | None = None
    state: Literal["active", "expired", "revoked"] = "active"
    created: datetime | None = None
    expires: datetime | None = None
    last_used: datetime | None = None
    owner: KeyOwner | None = None
    service: bool = False
    legacy: bool = False
    machine: str | None = None

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> Key:
        created = instant(data.get("created_at", data.get("created")))
        expires = instant(data.get("expires_at", data.get("expires")))
        state = data.get("state")
        if state not in ("active", "expired", "revoked"):
            if data.get("revoked_at"):
                state = "revoked"
            elif expires is not None and expires <= datetime.now(UTC):
                state = "expired"
            else:
                state = "active"
        token = data.get("token")
        prefix = data.get("prefix") or (token[:12] if isinstance(token, str) else None)
        owner = data.get("owner")
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            permissions=tuple(data.get("permissions") or ()),
            prefix=prefix,
            state=state,
            created=created,
            expires=expires,
            last_used=instant(data.get("last_used_at", data.get("last_used"))),
            owner=load(KeyOwner, owner) if isinstance(owner, Mapping) else None,
            service=bool(data.get("service")),
            legacy=bool(data.get("legacy")),
            machine=data.get("machine"),
        )


@dataclass(frozen=True)
class NewKey:
    """A key just created or rotated, with its token; the token is not shown again."""

    key: Key
    token: str = field(repr=False)


@dataclass(frozen=True)
class TrainedOn:
    """What a model saw in training; a note on results, never enforced."""

    rows: int | None = None
    inputs: int | None = None
    classes: int | None = None


@dataclass(frozen=True)
class Limits:
    """The context rows and inputs a model was trained and measured to; a call past them carries a note.

    No count of queries is a limit: a call's queries decode in as many
    model calls as the device needs, which `Timing.calls` counts.
    """

    context_rows: int | None = None
    inputs: int | None = None


@dataclass(frozen=True)
class Model:
    """A model `list` and `get` both return: a catalog entry or a published fine-tune version."""

    id: str
    short: str = ""
    task: Task | None = None
    family: str | None = None
    kind: Literal["catalog", "fine-tune"] = "catalog"
    base: str | None = None
    trained_on: TrainedOn | None = None
    hostable: bool = True
    digest: str | None = None
    architecture: Mapping[str, Any] = field(default_factory=dict)
    limits: Limits | None = None


@dataclass(frozen=True)
class UploadLimits:
    """The sizes the service takes data in, which the SDK sizes its requests by.

    One streamed snapshot holds at most `snapshot_rows` rows and
    `snapshot_bytes` bytes; a resumable upload at most `upload_rows` rows
    and `upload_bytes` bytes, or `csv_bytes` for a CSV, in chunks of
    `chunk_bytes` unless the upload names its own; a JSON request body at
    most `request_bytes`.
    """

    snapshot_bytes: int
    snapshot_rows: int
    upload_bytes: int
    upload_rows: int
    csv_bytes: int
    chunk_bytes: int
    request_bytes: int


@dataclass(frozen=True)
class Me:
    """Who the credential acts for.

    `plan` is the plan of the project's workspace (`free`, `pro` or
    `enterprise`), `trains` whether the project may fine-tune here, its
    plan included, and `limits` the sizes the service takes data in; a
    device login's reply says none of them, so they are None there.
    """

    email: str | None
    workspace: str | None
    project: str | None
    key: Key | None
    token_expires: datetime | None
    url: str
    plan: str | None = None
    trains: bool | None = None
    limits: UploadLimits | None = None


FINE_TUNE_STATES = ("queued", "running", "finished", "failed", "cancelled")
_RUN_STATES = {
    "setting_up": "running",
    "training": "running",
    "preempted": "running",
    "cancelling": "running",
}
_ARROW_TYPES = {
    "double": "number",
    "float": "number",
    "int64": "number",
    "int32": "number",
    "bool": "boolean",
    "string": "category",
    "timestamp": "timestamp",
    "date32": "date",
}


@dataclass(frozen=True, repr=False)
class Snapshot:
    """A named, immutable training table.

    `target` and `task` are the training metadata a run reads. A snapshot
    has no console page: the `url` the service sends with one is where an
    open upload sends its bytes, which `snapshots.create` uses.
    """

    name: str
    digest: str | None = None
    rows: int | None = None
    columns: tuple[Column, ...] = ()
    task: Task | None = None
    target: str | None = None
    bytes: int | None = None
    tags: tuple[str, ...] = ()
    state: str = "ready"
    origin: Mapping[str, Any] = field(default_factory=dict)
    created: datetime | None = None
    error: str | None = None

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> Snapshot:
        metadata = data.get("metadata") or {}
        columns = tuple(
            Column(
                str(c.get("name")),
                _ARROW_TYPES.get(str(c.get("type")), c.get("type") or "category"),
                int(c.get("missing") or 0),
                c.get("distinct"),
            )
            for c in data.get("columns") or ()
        )
        error = data.get("error")
        return cls(
            name=str(data.get("name") or ""),
            digest=data.get("digest"),
            rows=data.get("rows"),
            columns=columns,
            task=data.get("task") or metadata.get("task"),
            target=data.get("target") or metadata.get("target"),
            bytes=data.get("bytes"),
            tags=tuple(data.get("tags") or ()),
            state=str(data.get("state") or "ready"),
            origin=dict(data.get("origin") or metadata.get("origin") or {}),
            created=instant(data.get("created")),
            error=error.get("message") if isinstance(error, Mapping) else error,
        )

    def __repr__(self) -> str:
        """`Snapshot credit-2026-01  ready  500,000 rows  12 columns  classification, target defaulted`."""
        from generalforecast import _render

        return _render.snapshot_line(self)


@dataclass(frozen=True)
class BatchTable:
    """One table of a synthetic batch."""

    name: str
    state: str = "pending"
    error: str | None = None


@dataclass(frozen=True, repr=False)
class SyntheticBatch:
    """Synthetic training tables drawn from random causal graphs, each its own snapshot."""

    name: str
    tag: str = ""
    count: int = 1
    seed: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    state: Literal["pending", "done"] = "pending"
    counts: Mapping[str, int] = field(default_factory=dict)
    tables: tuple[BatchTable, ...] = ()
    created: datetime | None = None
    origin: Origin | None = None
    url: str | None = None

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> SyntheticBatch:
        tables = data.get("tables")
        if tables is None:
            tables = [
                {"name": n, "state": "pending"} for n in data.get("snapshots") or ()
            ]
        known = {"name", "tag", "count", "seed", "state", "counts", "tables", "snapshots",
                 "created", "origin", "url", "options", "tags", "schema_from"}  # fmt: skip
        options = data.get("options")
        if options is None:
            options = {k: v for k, v in data.items() if k not in known}
        return cls(
            name=str(data.get("name") or ""),
            tag=str(data.get("tag") or f"synthetic-{data.get('name')}"),
            count=int(data.get("count") or len(tables) or 1),
            seed=data.get("seed"),
            options=dict(options),
            state="done" if data.get("state") == "done" else "pending",
            counts=dict(data.get("counts") or {}),
            tables=tuple(load(BatchTable, t) for t in tables),
            created=instant(data.get("created")),
            origin=load(Origin, data["origin"]) if data.get("origin") else None,
            url=data.get("url"),
        )

    def __repr__(self) -> str:
        """`SyntheticBatch credit-warmup  pending  64 tables  12 ready, 52 pending`."""
        from generalforecast import _render

        return _render.batch_line(self)


@dataclass(frozen=True)
class Report:
    """A published version scored against its base on rows that chose nothing."""

    metric: str = "loss"
    base: float = 0.0
    tuned: float = 0.0
    delta: float = 0.0
    interval: tuple[float, float] | None = None
    rows: int = 0

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> Report:
        nested = data.get("data")
        table: Mapping[str, Any] = nested if isinstance(nested, Mapping) else data
        interval = table.get("ci95", table.get("interval"))
        return cls(
            metric=str(table.get("metric") or "loss"),
            base=float(table.get("base") or 0.0),
            tuned=float(table.get("kept", table.get("tuned")) or 0.0),
            delta=float(table.get("gain", table.get("delta")) or 0.0),
            interval=tuple(interval) if interval else None,  # type: ignore[arg-type]
            rows=int(table.get("rows") or 0),
        )


@dataclass(frozen=True)
class Published:
    """The version a run published: its model, weights' digest, kept step and report."""

    model: str
    digest: str | None = None
    kept_step: int | None = None
    validation_loss: float | None = None
    base_validation_loss: float | None = None
    report: Report | None = None


@dataclass(frozen=True)
class DataPin:
    """One table a run reads, pinned to its digest: a snapshot, or a context's version."""

    name: str
    kind: Literal["snapshot", "context"] = "snapshot"
    version: int | None = None
    digest: str | None = None
    phase: str | None = None

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> DataPin:
        context = data.get("context")
        return cls(
            name=str(context or data.get("name") or ""),
            kind="context" if context else "snapshot",
            version=data.get("version"),
            digest=data.get("digest"),
            phase=data.get("phase"),
        )


RunEventKind = Literal[
    "created",
    "data",
    "attempt",
    "state",
    "metrics",
    "resumed",
    "published",
    "failure",
    "calibration",
    "kept_nothing",
    "orphaned",
]


@dataclass(frozen=True)
class RunEvent:
    """One event of a run's lineage, as the service records it.

    `kind` says what happened: `created`, the run as created; `data`, the
    data a waiting run pinned; `attempt`, an attempt started; `state`, a
    change of state; `metrics`, a chunk of training; `resumed`, an attempt
    picked up again; `published`, the version it published; `failure`,
    why it failed; `calibration`, the losses it started from;
    `kept_nothing`, a run that ended publishing nothing, with what the
    trainer kept; `orphaned`, an attempt its trainer stopped answering for.
    """

    at: datetime | None = None
    kind: RunEventKind = "state"
    body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Lineage:
    """Every change to a run, oldest first: the version it is for and its events."""

    model: str
    events: tuple[RunEvent, ...] = ()


@dataclass(frozen=True, repr=False)
class FineTuneRun:
    """A fine-tuning run: the one that publishes version `<name>@ft<version>`.

    `model` is that version's model id, `<project>/<name>@ft<version>`: pass
    it, or `<name>@ft<version>` in the same project, wherever a model goes,
    as `gf.predictors.create(..., model=run.model)`.

    `curve` holds `(step, loss, validation_loss)` points for the loss chart,
    read from the run's metrics events. In the API, CLI and console it is a run.

    `attempt` counts the run's attempts (a preempted one starts another).
    `seconds` are the GPU seconds its budget counts, charged or still
    pending; `charged_seconds` those of attempts that ended charged,
    `pending_seconds` those not yet settled, and `voided_seconds` those
    lost on the service's side, never charged. `credits` and `usd` are
    what `seconds` cost.
    """

    model: str
    name: str = ""
    base: str = ""
    target: str | None = None
    state: Literal["queued", "running", "finished", "failed", "cancelled"] = "queued"
    waiting: int | None = None
    phase: str | None = None
    step: int = 0
    steps: int | None = None
    loss: float | None = None
    validation_loss: float | None = None
    base_validation_loss: float | None = None
    kept_step: int | None = None
    published: Published | None = None
    data: tuple[DataPin, ...] = ()
    data_digest: str | None = None
    method: Mapping[str, Any] = field(default_factory=dict)
    plan: Mapping[str, Any] = field(default_factory=dict)
    phases: tuple[Mapping[str, Any], ...] = ()
    architecture: tuple[Mapping[str, Any], ...] = ()
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    tags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    attempt: int = 0
    seconds: float = 0.0
    charged_seconds: float = 0.0
    voided_seconds: float = 0.0
    pending_seconds: float = 0.0
    credits: float = 0.0
    usd: float = 0.0
    error: str | None = None
    origin: Origin | None = None
    created: datetime | None = None
    started: datetime | None = None
    finished: datetime | None = None
    url: str | None = None
    curve: tuple[tuple[int, float, float | None], ...] = ()

    @property
    def done(self) -> bool:
        """Whether the run has ended."""
        return self.state in ("finished", "failed", "cancelled")

    @classmethod
    def _load(cls, data: Mapping[str, Any]) -> FineTuneRun:
        progress = data.get("progress") or {}
        published = data.get("published")
        found = load(Published, published) if isinstance(published, Mapping) else None
        usage = data.get("usage") or {}
        plan = dict(data.get("plan") or {})
        steps = plan.get("steps")
        if steps is None and data.get("phases"):
            steps = (
                sum(
                    int((p.get("plan") or {}).get("steps") or 0) for p in data["phases"]
                )
                or None
            )
        waiting = data.get("waiting")
        if isinstance(waiting, Mapping):
            waiting = waiting.get("pending")
        state = str(data.get("state") or "queued")
        error = data.get("error") or (data.get("reason") if state == "failed" else None)
        credits = float(usage.get("credits", data.get("credits")) or 0.0)
        seconds = float(usage.get("gpu_seconds", data.get("seconds")) or 0.0)
        charged = float(usage.get("charged_gpu_seconds") or 0.0)
        warnings = tuple(
            str(w.get("code") or w.get("message") or w)
            if isinstance(w, Mapping)
            else str(w)
            for w in data.get("warnings") or ()
        )
        model = str(data.get("model") or "")
        return cls(
            model=model,
            name=str(data.get("name") or model.split("/")[-1].split("@ft")[0]),
            base=str(data.get("base") or ""),
            target=data.get("target"),
            state=typing.cast(Any, _RUN_STATES.get(state, state)),
            waiting=waiting,
            phase=progress.get("phase") or data.get("phase"),
            step=int(progress.get("step", data.get("step")) or 0),
            steps=steps,
            loss=progress.get("loss", data.get("loss")),
            validation_loss=(found.validation_loss if found else None)
            if data.get("validation_loss") is None
            else data["validation_loss"],
            base_validation_loss=(found.base_validation_loss if found else None)
            if data.get("base_validation_loss") is None
            else data["base_validation_loss"],
            kept_step=found.kept_step if found else data.get("kept_step"),
            published=found,
            data=tuple(load(DataPin, d) for d in data.get("data") or ()),
            data_digest=data.get("data_digest"),
            method=dict(data.get("method") or {}),
            plan=plan,
            phases=tuple(dict(p) for p in data.get("phases") or ()),
            architecture=tuple(dict(a) for a in data.get("architecture") or ()),
            evaluation=dict(data.get("evaluation") or {}),
            budget=dict(data.get("budget") or {}),
            seed=data.get("seed"),
            tags=tuple(data.get("tags") or ()),
            warnings=warnings,
            attempt=int(data.get("attempt") or 0),
            seconds=seconds,
            charged_seconds=charged,
            voided_seconds=float(usage.get("voided_gpu_seconds") or 0.0),
            pending_seconds=max(seconds - charged, 0.0),
            credits=credits,
            usd=float(usage.get("usd", data.get("usd", credits / 1000)) or 0.0),
            error=error.get("message") if isinstance(error, Mapping) else error,
            origin=load(Origin, data["origin"])
            if isinstance(data.get("origin"), Mapping)
            else None,
            created=instant(data.get("created")),
            started=instant(data.get("started")),
            finished=instant(data.get("finished")),
            url=data.get("url"),
        )

    def __repr__(self) -> str:
        """One aligned status line."""
        from generalforecast import _render

        return _render.run_line(self)

    def _repr_html_(self) -> str:
        from generalforecast import _render

        return _render.run_html(self)

    def _mime_(self) -> tuple[str, str]:
        return ("text/html", self._repr_html_())
