# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""How results print: plain aligned text for terminals, HTML for notebooks.

Text tables fit `WIDTH` columns by dropping columns in a named order
(`term.columns`, which the CLI's tables use too); HTML escapes every value.
Nothing here makes a network call.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

from generalforecast.term import columns, duration
from generalforecast.types import money

if TYPE_CHECKING:
    from generalforecast.types import (
        Context,
        Entry,
        Experiment,
        FineTuneRun,
        PredictorInfo,
        Run,
        Snapshot,
        SyntheticBatch,
    )

WIDTH = 100

METRICS = {
    "auroc": "AUROC",
    "roc auc": "AUROC",
    "log loss": "Log loss",
    "logloss": "Log loss",
    "accuracy": "Accuracy",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "r2": "R²",
    "crps": "CRPS",
    "wql": "WQL",
    "mase": "MASE",
}


def metric_name(metric: str) -> str:
    """A metric as a column header: `auroc` is `AUROC`."""
    key = (metric or "").lower()
    return METRICS.get(key, metric.capitalize() if metric else "Score")


def spaced(value: float) -> str:
    """A long duration with spaces, as a run's line shows it: `2m 10s`."""
    total = round(value)
    if total < 60:
        return f"{total}s"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {rest}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def number(value: float | None, digits: int = 3) -> str:
    """A score to `digits` decimals, or empty."""
    return "" if value is None else f"{value:.{digits}f}"


def credits(value: float) -> str:
    """Credits as results show them: `0.4 credits`, `52.1 credits`, `1,204 credits`, `0.004 credits`."""
    if value >= 1000:
        text = f"{value:,.0f}"
    elif value >= 0.05 or value == 0:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2g}"
    return f"{text} credits"


def _ranked(run: Run) -> list[Entry]:
    scored = [e for e in run.leaderboard if e.score is not None]
    others = [e for e in run.leaderboard if e.score is None]
    scored.sort(key=lambda e: e.score or 0.0, reverse=not run.lower_is_better)
    return scored + others


def model_name(model: str) -> str:
    """A full model id without its owner and digest: `tabicl-v2-classifier`."""
    return model.split("/")[-1].split("@")[0]


def _model(entry: Entry) -> str:
    return entry.short or model_name(entry.model)


def leaderboard_table(
    run: Run,
) -> tuple[list[str], list[list[str]], str, tuple[int, ...]]:
    """A run's leaderboard as a table: header, rows best first, alignment, and drop order Interval, Time, Cost."""
    header = ["", "Model", metric_name(run.metric), "Interval", "Time", "Cost"]
    rows = []
    for rank, entry in enumerate(_ranked(run), 1):
        scored = entry.score is not None
        rows.append(
            [
                str(rank),
                _model(entry),
                number(entry.score) if scored else entry.state,
                f"{number(entry.interval[0])} to {number(entry.interval[1])}"
                if scored and entry.interval
                else "",
                duration(entry.seconds) if entry.seconds else "",
                money(entry.usd) if entry.usd else "",
            ]
        )
    return header, rows, "><>", (3, 4, 5)


def leaderboard(run: Run, *, width: int = WIDTH) -> list[str]:
    """A run's leaderboard as aligned lines `width` wide."""
    header, rows, align, drop = leaderboard_table(run)
    return columns(header, rows, align=align, drop=drop, width=width).lines


def failures(run: Run) -> list[str]:
    """Why each of a run's models failed, as `mitra-2 failed: inference timed out`."""
    return [
        f"{_model(entry)} failed: {entry.error}"
        for entry in _ranked(run)
        if entry.state == "failed" and entry.error
    ]


def _header(experiment: Experiment) -> str:
    text = f"Experiment {experiment.name}"
    run = experiment.run
    if run is None:
        return text + "  no runs yet"
    text += f"  run {run.number}"
    if run.state != "finished":
        text += f" {run.state}"
    if run.stale:
        text += ", stale"
    task = experiment.task or "task to infer"
    held = f", {run.held_rows:,} held-out rows" if run.held_rows else ""
    return f"{text}  {task}, target {experiment.target}{held}"


def experiment_text(experiment: Experiment, *, width: int = WIDTH) -> str:
    """The header line, the leaderboard and the console link."""
    lines = [_header(experiment)]
    if experiment.run is not None and experiment.run.leaderboard:
        lines += leaderboard(experiment.run, width=width)
        lines += [f"  {line}" for line in failures(experiment.run)]
    if experiment.run is not None and experiment.run.error:
        lines.append(f"  failed: {experiment.run.error}")
    if experiment.url:
        lines.append(f"  {experiment.url}")
    return "\n".join(lines)


_STYLE = (
    "border-collapse:collapse;font-variant-numeric:tabular-nums;"
    "font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px"
)
_CELL = "padding:2px 10px;text-align:left;border-bottom:1px solid #e5e5e5"
_NUM = "padding:2px 10px;text-align:right;border-bottom:1px solid #e5e5e5"


def _html_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    bold: int | None = None,
    numeric: Sequence[int] = (),
) -> str:
    head = "".join(
        f'<th style="{_NUM if i in numeric else _CELL}">{html.escape(h)}</th>'
        for i, h in enumerate(header)
    )
    body = []
    for index, row in enumerate(rows):
        weight = ' style="font-weight:600"' if index == bold else ""
        cells = "".join(
            f'<td style="{_NUM if i in numeric else _CELL}">{html.escape(c)}</td>'
            for i, c in enumerate(row)
        )
        body.append(f"<tr{weight}>{cells}</tr>")
    return (
        f'<table style="{_STYLE}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _link(url: str | None) -> str:
    if not url:
        return ""
    safe = html.escape(url, quote=True)
    return (
        f'<div style="margin-top:6px;font-size:13px"><a href="{safe}" '
        f'target="_blank" rel="noopener">View in the console</a></div>'
    )


def _title(text: str) -> str:
    return f'<div style="font-weight:600;margin-bottom:4px">{html.escape(text)}</div>'


def leaderboard_html(run: Run) -> str:
    """A run's leaderboard as an HTML table, the best row bold."""
    header, rows, _, _ = leaderboard_table(run)
    ranked = _ranked(run)
    best = 0 if ranked and ranked[0].score is not None else None
    return _html_table(header, rows, bold=best, numeric=(0, 2, 4, 5))


def experiment_html(experiment: Experiment) -> str:
    """The experiment's leaderboard as HTML with a console link."""
    parts = [f"<div>{_title(_header(experiment))}"]
    if experiment.run is not None and experiment.run.leaderboard:
        parts.append(leaderboard_html(experiment.run))
        parts += [
            f"<div>{html.escape(line)}</div>" for line in failures(experiment.run)
        ]
    if experiment.run is not None and experiment.run.error:
        parts.append(f"<div>failed: {html.escape(experiment.run.error)}</div>")
    parts.append(_link(experiment.url))
    parts.append("</div>")
    return "".join(parts)


def context_line(context: Context) -> str:
    """`Context adult  snapshot v1  32,561 rows  15 columns`."""
    kind = context.kind
    if context.kind == "snapshot" and context.version:
        kind += f" v{context.version}"
    parts = [f"Context {context.name}", kind]
    if context.state != "ready":
        parts.append(
            f"{context.state}: {context.error}" if context.error else context.state
        )
    if context.rows is not None:
        parts.append(f"{context.rows:,} rows")
    if context.columns:
        parts.append(f"{len(context.columns)} columns")
    if context.unchanged:
        parts.append("unchanged")
    return "  ".join(parts)


def context_html(context: Context) -> str:
    """A context's summary and its columns as HTML."""
    columns = [
        [
            c.name,
            c.type,
            f"{c.missing:,}",
            "" if c.distinct is None else f"{c.distinct:,}",
        ]
        for c in context.columns
    ]
    body = _title(context_line(context))
    if columns:
        body += _html_table(
            ["Column", "Type", "Missing", "Distinct"], columns, numeric=(2, 3)
        )
    return f"<div>{body}{_link(context.url)}</div>"


def predictor_text(info: PredictorInfo) -> str:
    """A predictor's summary and its environments, aligned."""
    spec = info.spec
    head = f"Predictor {info.name}  revision {info.revision}"
    if spec is not None:
        model = (
            info.resolved.short if info.resolved and info.resolved.short else spec.model
        )
        head += f"  {model}, target {spec.target}"
    lines = [head]
    rows = [
        [
            env.name,
            f"revision {env.revision}",
            env.state,
            f"{env.calls_today:,} calls today",
        ]
        for env in info.environments.values()
    ]
    if rows:
        lines += columns(None, rows).lines
    if info.url:
        lines.append(f"  {info.url}")
    return "\n".join(lines)


def predictor_html(info: PredictorInfo) -> str:
    """A predictor's environments as HTML."""
    rows = [
        [
            env.name,
            str(env.revision),
            env.state,
            f"{env.calls_today:,}",
            duration(env.p50_seconds) if env.p50_seconds else "",
        ]
        for env in info.environments.values()
    ]
    body = _title(predictor_text(info).splitlines()[0])
    if rows:
        body += _html_table(
            ["Environment", "Revision", "State", "Calls today", "p50"],
            rows,
            numeric=(1, 3, 4),
        )
    return f"<div>{body}{_link(info.url)}</div>"


def _run_parts(run: FineTuneRun) -> list[str]:
    parts = []
    if run.state == "queued" and run.waiting:
        parts.append(f"waiting for {run.waiting:,} tables")
    if run.phase:
        parts.append(f"phase {run.phase}")
    if run.state != "queued":
        if run.steps:
            parts.append(f"step {run.step:,}/{run.steps:,}")
        elif run.step:
            parts.append(f"step {run.step:,}")
    if run.loss is not None:
        parts.append(f"loss {run.loss:.3f}")
    if run.validation_loss is not None:
        parts.append(f"val {validation(run.validation_loss, run.base_validation_loss)}")
    if run.seconds:
        parts.append(spaced(run.seconds))
    if run.credits:
        parts.append(credits(run.credits))
    if run.error:
        parts.append(f"error: {run.error}")
    return parts


def validation(loss: float, base: float | None) -> str:
    """`0.398 (base 0.421)`: a validation loss beside the base's."""
    return f"{loss:.3f}" + (f" (base {base:.3f})" if base is not None else "")


def run_name(run: FineTuneRun) -> str:
    """`credit-default@ft2`, the run's model without its project."""
    return run.model.split("/")[-1] or run.name


def run_line(run: FineTuneRun) -> str:
    """`credit-default@ft2  running  step 340/1,000  loss 0.412  val 0.398 (base 0.421)  2m 10s  31.0 credits`."""
    return "  ".join([run_name(run), run.state, *_run_parts(run)])


def run_progress(run: FineTuneRun) -> tuple[str, str]:
    """A waited run's progress line, `Training credit-default@ft2  step 340/1,000 ...`, and its line in a log."""
    head = f"{'Queued' if run.state == 'queued' else 'Training'} {run_name(run)}"
    return "  ".join([head, *_run_parts(run)]), head


def loss_chart(run: FineTuneRun, *, width: int = 360, height: int = 120) -> str:
    """An inline SVG of a run's loss and validation loss, with the base's line."""
    points = [p for p in run.curve if p[1] is not None]
    if len(points) < 2:
        return ""
    values = [p[1] for p in points] + [p[2] for p in points if p[2] is not None]
    if run.base_validation_loss is not None:
        values.append(run.base_validation_loss)
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    first, last = points[0][0], points[-1][0]
    steps = (last - first) or 1

    def xy(step: float, value: float) -> str:
        x = 4 + (step - first) / steps * (width - 8)
        y = 4 + (high - value) / span * (height - 8)
        return f"{x:.1f},{y:.1f}"

    loss = " ".join(xy(s, v) for s, v, _ in points)
    val = " ".join(xy(s, v) for s, _, v in points if v is not None)
    lines = [
        f'<polyline fill="none" stroke="#2563eb" stroke-width="1.5" points="{loss}"/>'
    ]
    if val:
        lines.append(
            f'<polyline fill="none" stroke="#d97706" stroke-width="1.5" points="{val}"/>'
        )
    if run.base_validation_loss is not None:
        y = xy(first, run.base_validation_loss).split(",")[1]
        lines.append(
            f'<line x1="4" x2="{width - 4}" y1="{y}" y2="{y}" stroke="#737373" '
            'stroke-dasharray="4 3" stroke-width="1"/>'
        )
    legend = (
        '<div style="font-size:12px;color:#525252">'
        '<span style="color:#2563eb">loss</span>&nbsp;&nbsp;'
        '<span style="color:#d97706">validation loss</span>&nbsp;&nbsp;'
        "dashed: base</div>"
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="loss by step">{"".join(lines)}</svg>{legend}'
    )


def run_html(run: FineTuneRun) -> str:
    """A run's status line and loss chart as HTML."""
    return f"<div>{_title(run_line(run))}{loss_chart(run)}{_link(run.url)}</div>"


def snapshot_line(snapshot: Snapshot) -> str:
    """`Snapshot credit-2026-01  ready  500,000 rows  12 columns  classification, target defaulted`."""
    parts = [f"Snapshot {snapshot.name}", snapshot.state]
    if snapshot.rows is not None:
        parts.append(f"{snapshot.rows:,} rows")
    if snapshot.columns:
        parts.append(f"{len(snapshot.columns)} columns")
    if snapshot.task:
        parts.append(
            f"{snapshot.task}, target {snapshot.target}"
            if snapshot.target
            else snapshot.task
        )
    if snapshot.tags:
        parts.append("tags " + ", ".join(snapshot.tags))
    return "  ".join(parts)


def batch_line(batch: SyntheticBatch) -> str:
    """`SyntheticBatch credit-warmup  pending  64 tables  12 ready, 52 pending`."""
    parts = [f"SyntheticBatch {batch.name}", batch.state, f"{batch.count:,} tables"]
    counts = ", ".join(f"{n:,} {state}" for state, n in batch.counts.items() if n)
    if counts:
        parts.append(counts)
    parts.append(f"tag {batch.tag}")
    return "  ".join(parts)
