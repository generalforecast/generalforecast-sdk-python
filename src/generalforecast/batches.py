"""Query rows in calls of at most `contracts.MAX_QUERIES`, and their results as one.

A call carries at most `MAX_QUERIES` query rows, so more go as several calls
under one request id, `<id>-1`, `<id>-2` and so on, and their results come
back as if from one call, in the order the rows were given. A forecaster's
series stay whole within one call, since a series' horizon is forecast at once.
"""

from __future__ import annotations

from collections.abc import Sequence
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
