"""Query rows in calls the service takes, and their results as one.

A call carries at most `MAX_QUERIES` query rows in a body of at most
`MAX_BODY_BYTES`, so more go as several calls under one request id,
`<id>-1`, `<id>-2` and so on, and their results come back as if from one
call, in the order the rows were given. A forecaster's series stay whole
within one call, since a series' horizon is forecast at once.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast.contracts import PredictionResult
from generalforecast.service.contracts import ServiceError


def query_batches(table: pa.Table, series: Sequence[str], cap: int) -> list[np.ndarray]:
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


# The most a prediction's request body may be, as the service bounds it; a
# part's payload keeps this much under it for the envelope around it.
MAX_BODY_BYTES = 1_048_576
ENVELOPE_BYTES = 4096


def body_bytes(payload: Any) -> int:
    """About how many bytes a payload sends: its JSON, as loosely as it may be written."""
    return len(json.dumps(payload, default=str).encode())


def fitted(
    batches: list[np.ndarray],
    table: pa.Table,
    series: Sequence[str],
    payload: Callable[[pa.Table], Any],
) -> list[np.ndarray]:
    """The batches split in halves until each one's payload fits a request body.

    A batch that cannot halve, one row or one series, goes as it is, for
    the service to refuse if it must.
    """
    budget = MAX_BODY_BYTES - ENVELOPE_BYTES
    fitting: list[np.ndarray] = []
    pending = list(reversed(batches))
    while pending:
        rows = pending.pop()
        if body_bytes(payload(table.take(pa.array(rows)))) <= budget:
            fitting.append(rows)
            continue
        halves = query_batches(
            table.take(pa.array(rows)), series, max(1, len(rows) // 2)
        )
        if len(halves) < 2:
            fitting.append(rows)
            continue
        pending.extend(reversed([rows[half] for half in halves]))
    return fitting


def part_id(request_id: str, part: int) -> str:
    """The request id of a call's `part`, counted from 1."""
    return f"{request_id}-{part}"


def described(rows: np.ndarray, part: int, parts: int, total: int) -> str:
    """What a part of the queries is, as an error it answers names it."""
    first, last = int(rows[0]) + 1, int(rows[-1]) + 1
    if last - first + 1 == len(rows):
        return f"query rows {first:,} to {last:,} of {total:,}"
    return (
        f"part {part} of {parts} of the queries, {len(rows):,} rows from row {first:,}"
    )


def stitched(
    results: Sequence[PredictionResult], order: Sequence[np.ndarray]
) -> PredictionResult:
    """The parts' results as one, a row per query in the order the queries came.

    Every part must come from the same model, predictor revision and
    environment; an environment that moved between parts is a conflict.
    """
    first = results[0]
    if len(results) == 1:
        return first
    for result in results[1:]:
        if (result.model, result.predictor, result.revision, result.environment) != (
            first.model,
            first.predictor,
            first.revision,
            first.environment,
        ):
            raise ServiceError(
                "Conflict",
                "the predictor moved to another revision while its queries were "
                "scored in parts",
                409,
                hint="call again",
            )
    table = pa.concat_tables(
        [result.table for result in results], promote_options="default"
    )
    rows = np.concatenate(order)
    placed = np.empty(len(rows), dtype=np.int64)
    placed[rows] = np.arange(len(rows))
    return PredictionResult(
        table=table.take(pa.array(placed)),
        model=first.model,
        task=first.task,
        context=first.context,
        predictor=first.predictor,
        revision=first.revision,
        environment=first.environment,
    )
