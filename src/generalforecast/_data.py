# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Tables on their way to the service: read, normalized, hashed, split.

Whatever a caller holds (pandas, Polars, Arrow, or a file) becomes one
normalized Arrow table. A context goes up once as an Arrow snapshot named
by its digest; queries go as Arrow streams split by bytes, never by a
fixed row count (section 3.9 of the surface spec).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from generalforecast.errors import CapacityExceeded, InvalidInput

if TYPE_CHECKING:
    import pandas

QUERY_PART_BYTES = 8 * 1024 * 1024
"""The most Arrow bytes of queries one prediction request carries."""

FILE_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "xlsx",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}

ARROW_STREAM = "application/vnd.apache.arrow.stream"


def _from_file(path: Path) -> pa.Table:
    kind = FILE_FORMATS.get(path.suffix.lower())
    if kind is None:
        raise InvalidInput(
            f"`{path}` is not a file this reads",
            hint="pass a .csv, .tsv, .parquet, .xlsx, .arrow, .feather or .jsonl file",
        )
    if not path.is_file():
        raise InvalidInput(f"no file at `{path}`")
    try:
        if kind in ("csv", "tsv"):
            from pyarrow import csv

            options = csv.ParseOptions(delimiter="\t" if kind == "tsv" else ",")
            return csv.read_csv(path, parse_options=options)
        if kind == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path)
        if kind == "xlsx":
            return _workbook(path)
        if kind == "jsonl":
            from pyarrow import json

            return json.read_json(path)
        from pyarrow import feather

        return feather.read_table(path)
    except (pa.ArrowException, OSError) as error:
        raise InvalidInput(f"cannot read `{path}`: {error}") from None


def _cell_text(value: Any) -> str:
    """A cell as text, a whole number without Excel's `.0`."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sheet_column(values: list[Any]) -> pa.Array:
    """One column of a sheet, typed by what its cells hold, as the service reads one.

    Excel keeps every number as a float: a column of whole numbers is int64.
    Dates with times are read as UTC, which a workbook does not name. A
    column whose cells differ in type is text.
    """
    held = [v for v in values if v is not None]
    if held and all(type(v) is bool for v in held):
        return pa.array(values, pa.bool_())
    if held and all(type(v) in (int, float) for v in held):
        if all(float(v).is_integer() and abs(v) < 2**53 for v in held):
            return pa.array([None if v is None else int(v) for v in values], pa.int64())
        return pa.array(values, pa.float64())
    if held and all(isinstance(v, dt.date) for v in held):
        if all(type(v) is dt.date for v in held):
            return pa.array(values, pa.date32())
        return pa.array(
            [
                None
                if v is None
                else dt.datetime(*v.timetuple()[:6], v.microsecond, tzinfo=dt.UTC)
                if isinstance(v, dt.datetime)
                else dt.datetime(v.year, v.month, v.day, tzinfo=dt.UTC)
                for v in values
            ],
            pa.timestamp("us", tz="UTC"),
        )
    return pa.array([None if v is None else _cell_text(v) for v in values], pa.string())


def _workbook(path: Path) -> pa.Table:
    """An XLSX file's first sheet: its first row names the columns.

    Rows with no value are skipped, as a CSV's blank lines are, and so is a
    column with neither a name nor a value.
    """
    from python_calamine import CalamineWorkbook

    try:
        book = CalamineWorkbook.from_path(str(path))
        rows = book.get_sheet_by_index(0).iter_rows() if book.sheet_names else iter(())
    except Exception as error:  # noqa: BLE001 - any unreadable workbook is refused
        raise InvalidInput(f"cannot read `{path}`: {error}") from None
    header = next(rows, None)
    if header is None:
        raise InvalidInput(f"`{path}` has no sheet with a header row")
    names = [None if cell == "" else _cell_text(cell) for cell in header]
    columns: list[list[Any]] = [[] for _ in names]
    for row in rows:
        cells = [None if cell == "" else cell for cell in row]
        if any(cell is not None for cell in cells):
            for column, cell in zip(columns, cells, strict=True):
                column.append(cell)
    kept = [
        (name, column)
        for name, column in zip(names, columns, strict=True)
        if name is not None or any(cell is not None for cell in column)
    ]
    if any(name is None for name, _ in kept):
        raise InvalidInput(
            f"`{path}` has a column with values and no name in its first row"
        )
    return pa.table(
        [_sheet_column(column) for _, column in kept],
        names=[name for name, _ in kept],
    )


def table(data: Any, *, what: str = "data") -> pa.Table:
    """Rows as one normalized Arrow table.

    Takes a pandas or Polars frame, an Arrow table or record batch, a path
    to a .csv, .tsv, .parquet, .xlsx, .arrow, .feather or .jsonl file, a mapping
    of columns, or a list of row mappings. Schema metadata is dropped and
    dictionary columns decoded and large strings made plain; `normalized`
    goes on to the table the service holds.

    Raises:
        InvalidInput: For anything that is not rows, or rows Arrow cannot hold.
    """
    module = type(data).__module__.split(".")[0]
    try:
        if isinstance(data, pa.Table):
            result = data
        elif isinstance(data, pa.RecordBatch):
            result = pa.Table.from_batches([data])
        elif isinstance(data, (str, os.PathLike)):
            result = _from_file(Path(os.fspath(data)).expanduser())
        elif module == "pandas":
            if getattr(data, "ndim", None) == 1:
                data = data.to_frame()
            result = pa.Table.from_pandas(data, preserve_index=False)
        elif module == "polars":
            if type(data).__name__ == "LazyFrame":
                data = data.collect()
            result = data.to_arrow()
        elif isinstance(data, Mapping):
            result = pa.table(dict(data))
        elif isinstance(data, list) and all(isinstance(row, Mapping) for row in data):
            result = pa.Table.from_pylist([dict(row) for row in data])
        else:
            raise InvalidInput(
                f"{what} must be a DataFrame, an Arrow table or a file path",
                hint="pass a pandas or Polars DataFrame, a pyarrow.Table, "
                "or a .csv, .tsv, .parquet, .xlsx, .arrow or .jsonl path",
                param=what,
            )
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise InvalidInput(
            f"{what} cannot be read as a table: {error}",
            hint="give each column one type; convert mixed object columns first",
            param=what,
        ) from None
    names = result.column_names
    if len(set(names)) != len(names) or not all(
        isinstance(n, str) and n for n in names
    ):
        raise InvalidInput(f"{what} needs unique, non-empty column names", param=what)
    columns = []
    for column in result.columns:
        if pa.types.is_dictionary(column.type):
            column = column.cast(column.type.value_type)
        if pa.types.is_large_string(column.type):
            column = column.cast(pa.string())
        elif pa.types.is_large_binary(column.type):
            column = column.cast(pa.binary())
        columns.append(column)
    return pa.table(columns, names=names).combine_chunks()


def stream(data: pa.Table, metadata: Mapping[str, str] | None = None) -> bytes:
    """A table's Arrow IPC stream: deterministic, with `metadata` on its schema."""
    schema = data.schema.remove_metadata()
    if metadata:
        schema = schema.with_metadata(dict(metadata))
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(data.replace_schema_metadata(schema.metadata))
    return sink.getvalue()


def read_stream(body: bytes) -> pa.Table:
    """The table an Arrow IPC stream holds."""
    return pa.ipc.open_stream(body).read_all()


# The Arrow type each kind of column is held as, as the engine normalizes a
# table before it hashes it (`gf_engine.tabular.to_arrow`).
HELD_TYPES = (
    (pa.types.is_boolean, pa.bool_()),
    (pa.types.is_integer, pa.int64()),
    (pa.types.is_floating, pa.float64()),
    (
        lambda kind: pa.types.is_string(kind) or pa.types.is_large_string(kind),
        pa.string(),
    ),
    (
        lambda kind: pa.types.is_binary(kind) or pa.types.is_large_binary(kind),
        pa.binary(),
    ),
    (pa.types.is_date, pa.date32()),
    (pa.types.is_timestamp, pa.timestamp("us", tz="UTC")),
)
DIGEST_BATCH_ROWS = 1_000_000
"""The rows of each record batch in the stream a table's digest hashes, as the engine writes it."""


def _held_column(name: str, column: pa.ChunkedArray) -> pa.ChunkedArray:
    """A column as the service holds it (`normalized`)."""
    import pyarrow.compute as pc

    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    if pa.types.is_null(column.type):
        # A column with no value at all is text, every cell empty, as the
        # engine reads a file's.
        column = column.cast(pa.string())
    kind = next((held for test, held in HELD_TYPES if test(column.type)), None)
    if kind is None:
        raise InvalidInput(
            f"the column `{name}` is {column.type}, which the service does not hold",
            hint="cast it to a number, text, boolean, date or timestamp column",
            param=name,
        )
    try:
        column = pc.cast(column, kind, safe=True)
    except pa.ArrowInvalid as error:
        raise InvalidInput(
            f"the column `{name}` cannot be held as {kind}: {error}",
            hint="timestamps are held to the microsecond, whole numbers in 64 bits",
            param=name,
        ) from None
    if kind == pa.float64():
        # Compute functions by name: pyarrow generates them, so its stubs name none.
        if pc.call_function("any", [pc.call_function("is_inf", [column])]).as_py():
            raise InvalidInput(
                f"the column `{name}` holds an infinity, which the service does not take",
                param=name,
            )
        nan = pc.call_function("is_nan", [column])
        column = pc.call_function("if_else", [nan, pa.nulls(1, kind)[0], column])
    return column


def _zero(kind: pa.DataType) -> pa.Scalar:
    """The zero of a held column's type: all-zero bytes, or an empty string."""
    if pa.types.is_string(kind) or pa.types.is_binary(kind):
        buffers = [None, pa.py_buffer(bytes(8)), pa.py_buffer(b"")]
    else:
        buffers = [None, pa.py_buffer(bytes(max(1, kind.bit_width // 8)))]
    return pa.Array.from_buffers(kind, 1, buffers)[0]


def _zero_under_nulls(column: pa.ChunkedArray) -> pa.ChunkedArray:
    """A column whose cells under its nulls hold its type's zero, the engine's `_logical`.

    Arrow keeps whatever bytes lie under a null (a NaN made null stays NaN
    in a stream, Parquet reads back a zero), so with the zero there the
    same values hash alike whichever format carried them.
    """
    import pyarrow.compute as pc

    if column.null_count == 0:
        return column
    chunks = []
    for chunk in column.chunks:
        valid = pc.call_function("is_valid", [chunk])
        filled = pc.fill_null(chunk, _zero(chunk.type))
        buffers = [valid.buffers()[1], *filled.buffers()[1:]]
        chunks.append(
            pa.Array.from_buffers(
                chunk.type, len(chunk), buffers, null_count=chunk.null_count
            )
        )
    return pa.chunked_array(chunks, type=column.type)


def normalized(data: pa.Table) -> pa.Table:
    """A table as the service holds and hashes it, as the engine normalizes one.

    Each column becomes a 64-bit integer, a double with NaN as null, a
    boolean, text, bytes, a date, or a UTC timestamp in microseconds;
    dictionaries are decoded and a column with no value is text. Every
    field is nullable and carries no metadata, and every cell under a null
    holds its type's zero.

    Raises:
        InvalidInput: For a column of another type, an infinity, or a value
            its type cannot hold, such as a nanosecond.
    """
    columns = [_held_column(name, data.column(name)) for name in data.column_names]
    table = pa.table(columns, names=data.column_names).combine_chunks()
    return pa.table(
        [_zero_under_nulls(column) for column in table.columns],
        names=table.column_names,
    )


class _Hashing:
    """A write-only file that hashes what it is given and keeps none of it."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256()
        self.closed = False

    def write(self, data: Any) -> int:
        view = memoryview(data)
        self.hash.update(view)
        return view.nbytes

    def flush(self) -> None:
        """Nothing is buffered."""

    def close(self) -> None:
        self.closed = True


def table_digest(data: pa.Table) -> str:
    """The digest the service names a table by, the engine's `table_digest`.

    It hashes the normalized table's Arrow IPC stream. Every cell under a
    null holds its type's zero, so a table has one digest whether it is
    sent as Parquet or as an Arrow stream.
    """
    table = normalized(data)
    sink = _Hashing()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=DIGEST_BATCH_ROWS)
    return "sha256:" + sink.hash.hexdigest()


def parquet(data: pa.Table) -> bytes:
    """A table as Parquet bytes, how a context upload sends it."""
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    pq.write_table(data, sink, compression="zstd")
    return sink.getvalue()


def read_parquet(body: bytes) -> pa.Table:
    """The table Parquet bytes hold."""
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(body))


def file_name(data: Any) -> str | None:
    """The file name a context's source shows, for data given as a path."""
    if isinstance(data, (str, os.PathLike)):
        return Path(os.fspath(data)).name
    return None


def column_types(data: pa.Table) -> dict[str, str]:
    """Each column's `ColumnType`, read from its Arrow type."""
    types: dict[str, str] = {}
    for field in data.schema:
        kind = field.type
        if pa.types.is_boolean(kind):
            types[field.name] = "boolean"
        elif (
            pa.types.is_integer(kind)
            or pa.types.is_floating(kind)
            or pa.types.is_decimal(kind)
        ):
            types[field.name] = "number"
        elif pa.types.is_timestamp(kind):
            types[field.name] = "timestamp"
        elif pa.types.is_date(kind):
            types[field.name] = "date"
        else:
            types[field.name] = "category"
    return types


def check_table(
    what: str,
    rows: int,
    nbytes: int,
    most: tuple[int, int],
    *,
    holds: str,
    hint: str,
    param: str,
) -> None:
    """Refuses, before anything is sent, a table past `most` rows and bytes, all one `holds` holds.

    Raises:
        CapacityExceeded: Naming the size, the limit and the fix.
    """
    most_rows, most_bytes = most
    if rows <= most_rows and nbytes <= most_bytes:
        return
    limit, actual = (most_rows, rows) if rows > most_rows else (most_bytes, nbytes)
    raise CapacityExceeded(
        f"the {what} is {rows:,} rows and {size(nbytes)}, past the {most_rows:,} "
        f"rows and {size(most_bytes)} one {holds} holds",
        hint=hint,
        param=param,
        limit=limit,
        actual=actual,
    )


def size(count: int) -> str:
    """A byte count as `1.1 MiB`."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def split(
    data: pa.Table, *, max_bytes: int | None = None, series: Sequence[str] = ()
) -> list[pa.Table]:
    """Queries in parts of at most `max_bytes` Arrow bytes each, in order.

    A forecast's series stay whole: rows of one series never straddle two
    parts. A single row, or series, past `max_bytes` is a part on its own.
    `max_bytes` defaults to `QUERY_PART_BYTES`.
    """
    max_bytes = max_bytes or QUERY_PART_BYTES
    if data.num_rows == 0:
        return [data]
    if series:
        return _split_series(data, max_bytes, series)
    parts: list[pa.Table] = []
    start = 0
    per_row = max(1, data.nbytes // max(1, data.num_rows))
    while start < data.num_rows:
        count = max(1, min(data.num_rows - start, max_bytes // per_row))
        while True:
            part = data.slice(start, count)
            if count == 1 or len(stream(part)) <= max_bytes:
                break
            count //= 2
        parts.append(part)
        start += count
    return parts


def _split_series(
    data: pa.Table, max_bytes: int, series: Sequence[str]
) -> list[pa.Table]:
    missing = [column for column in series if column not in data.column_names]
    if missing:
        raise InvalidInput(f"the queries have no series column `{missing[0]}`")
    keys = list(zip(*(data.column(c).to_pylist() for c in series), strict=True))
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    per_row = max(1, data.nbytes // data.num_rows)
    parts: list[pa.Table] = []
    current: list[int] = []
    for indices in groups.values():
        if current and (len(current) + len(indices)) * per_row > max_bytes:
            parts.append(data.take(current))
            current = []
        current.extend(indices)
    if current:
        parts.append(data.take(current))
    return parts


def frame(columns: Any, rows: Any) -> pandas.DataFrame:
    """A reply's `{columns, rows}` as a DataFrame.

    `columns` are names or `{name, type}` objects; `rows` are lists in
    column order or mappings by name.
    """
    import pandas

    names = [c["name"] if isinstance(c, Mapping) else str(c) for c in columns or ()]
    rows = list(rows or ())
    if rows and isinstance(rows[0], Mapping):
        found = pandas.DataFrame.from_records(rows)
        return found[[n for n in names if n in found.columns]] if names else found
    return pandas.DataFrame.from_records(rows, columns=names or None)
