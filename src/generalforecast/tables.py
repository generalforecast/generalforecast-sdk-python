# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Reading a context table for `gf init`: column kinds, role candidates and the query split.

A column has one kind, the vocabulary `gf.yaml`'s `inputs.types` uses:
`int`, `float`, `0/1`, `category`, `text`, `timestamp` or `bool`. `kind_of`
reads it from a column's values. The service makes the same reading when
`gf check` resolves the manifest; this module only proposes the roles
`gf init` offers.
"""

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import InvalidInput
from generalforecast.validation import PredictorError

pc: Any = _compute  # ty cannot see Arrow's generated compute functions
FORMATS = (".csv", ".tsv", ".parquet", ".arrow", ".feather", ".json")
CATEGORY_DISTINCT = 20
CATEGORY_SHARE = 0.05
ID_NAMES = ("id", "key", "uuid", "guid", "pk", "code", "number")
TIME_NAMES = ("time", "date", "timestamp", "ts", "at", "when", "day", "month")
TARGET_NAMES = ("target", "label", "outcome", "class", "churn", "fraud", "y")


@dataclass(frozen=True)
class ColumnInfo:
    """One column of an inspected table.

    Attributes:
        name: The column name.
        kind: One of `int`, `float`, `0/1`, `category`, `text`, `timestamp`, `bool`.
        unique: Whether every row has a distinct, present value.
        empty: How many rows have no value.
        distinct: How many distinct present values there are.
    """

    name: str
    kind: str
    unique: bool
    empty: int
    distinct: int


@dataclass(frozen=True)
class TableInfo:
    """What `inspect_table` learns about a context file.

    Attributes:
        path: The file inspected, as given.
        rows: The row count.
        columns: Each column with its inferred kind.
        targets: Target candidates, best first.
        ids: Id candidates, best first.
        times: Time candidates, best first.
    """

    path: str
    rows: int
    columns: tuple[ColumnInfo, ...]
    targets: tuple[str, ...]
    ids: tuple[str, ...]
    times: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """The column names in file order."""
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnInfo:
        """The column with a name."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def kinds(self) -> dict[str, str]:
        """Each column's kind by name."""
        return {column.name: column.kind for column in self.columns}


# The largest table `gf init` reads.
MAX_DATA_BYTES = 1024 * 1024 * 1024


def read_frame(root: str | Path, path: str) -> pa.Table:
    """Reads a .csv, .parquet, .arrow or .json file inside a predictor directory.

    Raises:
        InvalidInput: If the format is unknown, or the file is missing, too
            large or malformed.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in FORMATS:
        raise InvalidInput(
            message=f"`{path}` is not a .csv, .parquet, .arrow or .json file",
            param="context.snapshot",
            hint="save the context as one of those formats",
        )
    file = Path(root) / path
    if not file.is_file():
        raise InvalidInput(
            message=f"`{path}` does not exist",
            param="context.snapshot",
            hint="check the path",
        )
    if file.stat().st_size > MAX_DATA_BYTES:
        raise InvalidInput(
            message=f"`{path}` is too large to read",
            param="context.snapshot",
            hint=f"snapshots are at most {MAX_DATA_BYTES // (1024 * 1024)} MiB",
        )
    try:
        return tabular.read_table(file.read_bytes(), suffix)
    except PredictorError as error:
        raise InvalidInput(
            message=f"cannot read `{path}`: {error}".rstrip("."),
            param="context.snapshot",
        ) from None


def inspect_table(path: str | Path) -> TableInfo:
    """Reads a table file and infers each column's kind and role candidates.

    Args:
        path: A .csv, .parquet, .arrow or .json file.

    Returns:
        The columns with kinds, uniqueness and empty counts, plus ranked target,
        id and time candidates.

    Raises:
        InvalidInput: If the file is missing, unreadable or empty.
    """
    file = Path(path)
    frame = read_frame(file.parent if str(file.parent) else ".", file.name)
    return inspect_frame(frame, path=str(path))


def inspect_frame(frame: Any, *, path: str = "") -> TableInfo:
    """Infers kinds and role candidates for rows already in memory."""
    table = to_table(frame)
    if table.num_rows == 0:
        raise InvalidInput(
            message=f"`{Path(path).name if path else 'context'}` has no rows",
            param="context.snapshot",
            hint="the context needs rows to learn from",
        )
    columns = tuple(
        _column(name, table[name], table.num_rows) for name in table.column_names
    )
    return TableInfo(
        path=path,
        rows=table.num_rows,
        columns=columns,
        targets=_targets(columns),
        ids=_ids(columns),
        times=_times(columns),
    )


def to_table(frame: Any) -> pa.Table:
    """Any supported rows as an Arrow table."""
    try:
        return tabular.to_arrow(frame)
    except PredictorError as error:
        raise InvalidInput(
            message=f"cannot read the rows: {error}".rstrip("."),
            param="context",
        ) from None


def _column(name: str, values: pa.ChunkedArray, rows: int) -> ColumnInfo:
    empty = _empty_count(values)
    present = pc.drop_null(_blank_to_null(values))
    distinct = pc.count_distinct(present).as_py() if len(present) else 0
    kind = kind_of(values, present, distinct)
    return ColumnInfo(name, kind, empty == 0 and distinct == rows, empty, distinct)


def _blank_to_null(values: pa.ChunkedArray) -> pa.ChunkedArray:
    kind = values.type
    if pa.types.is_dictionary(kind):
        values = pc.cast(values, kind.value_type)
        kind = values.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        stripped = pc.utf8_trim_whitespace(values)
        # A length of zero casts to false, so no Python scalar is compared
        # (`tabular.null` says why that matters).
        filled = pc.cast(pc.binary_length(stripped), pa.bool_())
        return pc.if_else(filled, stripped, tabular.null(kind))
    if pa.types.is_floating(kind):
        return pc.if_else(pc.is_nan(values), tabular.null(kind), values)
    return values


def _empty_count(values: pa.ChunkedArray) -> int:
    return _blank_to_null(values).null_count


def kind_of(values: pa.ChunkedArray, present: pa.ChunkedArray, distinct: int) -> str:
    """The inferred kind of a column from its present values."""
    kind = present.type
    if pa.types.is_boolean(kind):
        return "bool"
    if pa.types.is_date(kind) or pa.types.is_timestamp(kind):
        return "timestamp"
    if pa.types.is_integer(kind):
        return "0/1" if _binary(present, distinct) else "int"
    if pa.types.is_floating(kind):
        return "0/1" if _binary(present, distinct) else "float"
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        if not len(present):
            # Nothing says what an empty column holds, so it is not judged a time.
            return "text"
        if distinct and distinct <= 2:
            lowered = set(pc.utf8_lower(pc.unique(present)).to_pylist())
            if lowered <= {"0", "1"}:
                return "0/1"
            if lowered <= {"true", "false"} or lowered <= {"yes", "no"}:
                return "bool"
        if _timestamps(present):
            return "timestamp"
        if distinct <= CATEGORY_DISTINCT or distinct <= CATEGORY_SHARE * len(present):
            return "category"
        return "text"
    raise InvalidInput(
        message=f"column type {kind} is not supported",
        param="context",
        hint="use numbers, text, booleans or timestamps",
    )


# The Arrow storage a kind takes in a schema compiled without rows.


def _binary(present: pa.ChunkedArray, distinct: int) -> bool:
    if not 1 <= distinct <= 2:
        return False
    return set(pc.unique(present).to_pylist()) <= {0, 1}


def _timestamps(present: pa.ChunkedArray) -> bool:
    """Whether text reads as times, by `iso.timestamps`, judged by its first rows."""
    from generalforecast.iso import timestamps

    try:
        timestamps(present.slice(0, 1000))
        return True
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False


def _named_like(name: str, words: Sequence[str]) -> bool:
    lowered = name.lower()
    parts = [part for part in lowered.replace("-", "_").split("_") if part]
    return any(
        part in words or any(part.endswith(word) for word in words if len(word) > 2)
        for part in parts
    )


def _ids(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    unique = [
        column
        for column in columns
        if column.unique and column.kind in ("int", "text", "category")
    ]
    named = [column.name for column in unique if _named_like(column.name, ID_NAMES)]
    rest = [column.name for column in unique if column.name not in named]
    return (*named, *rest)


def _times(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    stamps = [column for column in columns if column.kind == "timestamp"]
    named = [column.name for column in stamps if _named_like(column.name, TIME_NAMES)]
    rest = [column.name for column in stamps if column.name not in named]
    return (*named, *rest)


def _targets(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    tiers = {"0/1": 0, "bool": 1, "category": 2, "int": 3, "float": 3}
    ranked = []
    for index, column in enumerate(columns):
        tier = tiers.get(column.kind)
        if tier is None or column.unique or column.distinct < 2:
            continue
        if column.kind == "category" and column.distinct > CATEGORY_DISTINCT:
            continue
        bonus = 0 if _named_like(column.name, TARGET_NAMES) else 1
        ranked.append((tier, bonus, index, column.name))
    return tuple(name for _, _, _, name in sorted(ranked))


# Tasks by the names the CLI and the lock use.
_TASKS = {
    "classify": "classify",
    "classification": "classify",
    "regress": "regress",
    "regression": "regress",
    "forecast": "forecast",
}
# Share of distinct values past which a text-like column reads as an id.
_NEAR_UNIQUE = 0.9


def target_candidates(info: TableInfo, task: str | None = None) -> tuple[str, ...]:
    """The columns that can be `task`'s target, best first; empty if none.

    Never an id candidate, a timestamp, a free-text column or a near-unique
    text or integer column (an id in all but name). A column named like a
    target (`label`, `outcome`, `y`) ranks first. For `regress` and
    `forecast` only numeric columns qualify: floats before ints, then later
    columns before earlier ones, as datasets usually put the target last.
    For `classify` only low-cardinality columns do: 0/1 flags, booleans,
    categories, then ints with few values (`task_for`), in file order.

    With no task, the candidates `gf init` offers when nothing declares
    one, each column once: a table with a time column lists its forecast
    candidates and then its classification ones, any other table its
    classification candidates and then its regression ones. The first is
    the target `gf init --yes` takes, and `task_for` then names its task.

    Args:
        info: What `inspect_table` or `inspect_frame` learned.
        task: `classify`, `regress` or `forecast` (or `classification`,
            `regression`); None for the tasks `gf init` considers.

    Raises:
        ValueError: For another task.
    """
    if task is None:
        tasks = (
            ("forecast", "classification")
            if info.times
            else ("classification", "regression")
        )
        return tuple(
            dict.fromkeys(
                name for each in tasks for name in target_candidates(info, each)
            )
        )
    if (wanted := _TASKS.get(task)) is None:
        raise ValueError(f"unknown task `{task}`")
    excluded = {*info.ids, *info.times}
    ranked: list[tuple[int, int, int, str]] = []
    for index, column in enumerate(info.columns):
        if column.name in excluded or column.distinct < 2:
            continue
        present = info.rows - column.empty
        near_unique = present > 0 and column.distinct >= _NEAR_UNIQUE * present
        named = 0 if _named_like(column.name, TARGET_NAMES) else 1
        if wanted == "classify":
            tiers = {"0/1": 0, "bool": 1, "category": 2, "int": 3}
            tier = tiers.get(column.kind)
            if tier is None or (column.kind == "int" and near_unique):
                continue
            if tier == 3 and task_for(column, info.rows) != "classification":
                continue
            ranked.append((named, tier, index, column.name))
        else:
            tier = {"float": 0, "int": 1}.get(column.kind)
            if tier is None or (column.kind == "int" and near_unique):
                continue
            if _named_like(column.name, ID_NAMES):
                continue
            ranked.append((named, tier, -index, column.name))
    return tuple(name for *_, name in sorted(ranked))


# The share of the rows a holdout takes when nothing says otherwise.
HOLDOUT_FRACTION = 0.2
# The ways `gf init` splits queries off a snapshot, in the order it offers them.
QUERY_RULES = ("empty", "date", "fraction", "none")


def default_holdout(rows: int, queries: int) -> float:
    """The holdout fraction `gf init` proposes for a table of `rows` rows.

    A fifth of the rows (`HOLDOUT_FRACTION`), or, when that is more rows
    than the model scores in one call (`queries`), the fraction that holds
    out one call's worth, to four significant digits.
    """
    if round(HOLDOUT_FRACTION * rows) <= queries:
        return HOLDOUT_FRACTION
    return float(f"{queries / rows:.4g}")


def id_candidates(info: TableInfo, taken: Iterable[str | None] = ()) -> tuple[str, ...]:
    """The id candidates that hold no other role, best first; `taken` are the columns that do."""
    held = set(taken)
    return tuple(name for name in info.ids if name not in held)


def query_rules(
    info: TableInfo, target: str, time: str | None = None
) -> tuple[str, ...]:
    """The ways to split queries off a table, of `QUERY_RULES`, the one `gf init --yes` takes first.

    `empty` (the rows with no target are the queries) when the target has
    blank rows; `date` (`context.until`) when `time`, or else the table's
    first time candidate, gives dates to compare; then `fraction`
    (`context.holdout`) and `none`, which every table can take.
    """
    rules = []
    if info.column(target).empty:
        rules.append("empty")
    if time is not None or info.times:
        rules.append("date")
    return (*rules, "fraction", "none")


def default_until(
    table: pa.Table,
    *,
    time: str,
    series: Sequence[str],
    horizon: int,
    frequency: str | None = None,
) -> str | None:
    """The `context.until` that holds out exactly `horizon` steps of every series.

    `table` is the labeled history (rows with a target); each series' last
    `horizon` times, one frequency step apart, become the queries `gf check`
    locks and `gf eval` scores, so the two agree. A series that ends before
    the cutoff holds out nothing, which `gf check` admits; any other count
    it refuses.

    Args:
        table: The labeled rows.
        time: The time column.
        series: The series columns; none for a single series.
        horizon: The steps each series holds out.
        frequency: The series' frequency in any spelling; inferred through
            `frequency.infer` when None.

    Returns:
        The cutoff as an ISO-8601 date (for dates) or UTC timestamp, or None
        when no one cutoff holds out `horizon` steps of every series: a
        series has no more than `horizon` times, the times follow no single
        frequency, or the series end at times that leave some of them a
        partial horizon.
    """
    from generalforecast import frequency as _frequency

    if horizon < 1 or time not in table.column_names:
        return None
    column = table[time]
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        try:
            column = pc.cast(column, pa.timestamp("us"))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return None
    keys = (
        list(zip(*(table[name].to_pylist() for name in series), strict=True))
        if series
        else [()] * table.num_rows
    )
    times: dict[tuple, list[Any]] = {}
    for key, value in zip(keys, column.to_pylist(), strict=True):
        times.setdefault(tuple(key), []).append(value)
    try:
        if frequency is None:
            _frequency.infer(times)
        else:
            _frequency.check(times, frequency)
    except (_frequency.FrequencyError, TypeError, ValueError):
        return None
    ordered = {
        key: sorted(values, key=_frequency.instant) for key, values in times.items()
    }
    if any(len(values) <= horizon for values in ordered.values()):
        return None
    cutoff = max(
        (values[-horizon] for values in ordered.values()), key=_frequency.instant
    )
    at = _frequency.instant(cutoff)
    for values in ordered.values():
        held = sum(_frequency.instant(value) >= at for value in values)
        if held not in (0, horizon):
            return None
    if isinstance(cutoff, dt.datetime):
        if cutoff.tzinfo is not None:
            cutoff = cutoff.astimezone(dt.UTC).replace(tzinfo=None)
        return cutoff.isoformat()
    if isinstance(cutoff, dt.date):
        return cutoff.isoformat()
    return None


def whole_numbers(values: pa.ChunkedArray) -> pa.ChunkedArray | None:
    """A float column cast to int64 when every present value is a whole number.

    A CSV with blanks in an integer column often spells its values `1.0`; the
    classes a lock records and a server checks labels against stay integers.
    """
    if not pa.types.is_floating(values.type):
        return None
    present = pc.drop_null(values)
    present = present.filter(pc.invert(pc.is_nan(present)))
    if present.length() == 0:
        return None
    if not pc.all(pc.equal(present, pc.floor(present))).as_py():
        return None
    return pc.cast(values, pa.int64(), safe=False)


def labels(values: pa.ChunkedArray) -> pa.ChunkedArray:
    """A target column's class labels: its present values, whole floats as ints."""
    values = whole_numbers(values) or values
    return pc.drop_null(_blank_to_null(values))


def task_for(column: ColumnInfo, rows: int) -> str:
    """The task a target column calls for.

    Booleans, 0/1 flags, categories and text are classes. So is an int with few
    distinct values, by the rule that makes a text column a category: at most
    `CATEGORY_DISTINCT` of them, or at most `CATEGORY_SHARE` of the rows. Every
    other number is regressed.
    """
    if column.kind == "float":
        return "regression"
    if column.kind == "int":
        few = (
            column.distinct <= CATEGORY_DISTINCT
            or column.distinct <= CATEGORY_SHARE * rows
        )
        return "classification" if few else "regression"
    return "classification"
