# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""How results print: plain aligned text for terminals, HTML for notebooks.

Text tables fit `WIDTH` columns by dropping columns in a named order; HTML
escapes every value. Nothing here makes a network call.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

from generalforecast.types import money

if TYPE_CHECKING:
    from generalforecast.types import (
        Context,
        Entry,
        Experiment,
        PredictorInfo,
        Run,
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


def seconds(value: float) -> str:
    """A duration as status lines show it: `0.21s`, `6.8s`, `41s`, `1m12s`."""
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{value:.0f}s"
    minutes, rest = divmod(round(value), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def number(value: float | None, digits: int = 3) -> str:
    """A score to `digits` decimals, or empty."""
    return "" if value is None else f"{value:.{digits}f}"


def table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    drop: Sequence[int] = (),
    right: Sequence[int] = (),
    indent: int = 2,
    width: int = WIDTH,
) -> list[str]:
    """Aligned columns two spaces apart, dropping columns in `drop` order until they fit."""
    keep = list(range(len(header)))

    def render(columns: list[int]) -> list[str]:
        widths = {
            c: max([len(header[c]), *(len(row[c]) for row in rows)]) for c in columns
        }
        lines = []
        for row in [header, *rows]:
            cells = []
            for position, c in enumerate(columns):
                cell = row[c]
                last = position == len(columns) - 1
                if c in right:
                    cells.append(cell.rjust(widths[c]))
                else:
                    cells.append(cell if last else cell.ljust(widths[c]))
            lines.append((" " * indent + "  ".join(cells)).rstrip())
        return lines

    lines = render(keep)
    for column in drop:
        if max(len(line) for line in lines) <= width:
            break
        keep.remove(column)
        lines = render(keep)
    return lines


def _ranked(run: Run) -> list[Entry]:
    scored = [e for e in run.leaderboard if e.score is not None]
    others = [e for e in run.leaderboard if e.score is None]
    scored.sort(key=lambda e: e.score or 0.0, reverse=not run.lower_is_better)
    return scored + others


def _model(entry: Entry) -> str:
    return entry.short or entry.model.split("/")[-1].split("@")[0]


def leaderboard(run: Run, *, width: int = WIDTH) -> list[str]:
    """A run's leaderboard as aligned lines; drop order Interval, Time, Cost."""
    header = ["", "Model", metric_name(run.metric), "Interval", "Time", "Cost"]
    rows = []
    for rank, entry in enumerate(_ranked(run), 1):
        if entry.score is None:
            score, interval = entry.state, ""
        else:
            score = number(entry.score)
            interval = (
                f"{number(entry.interval[0])} to {number(entry.interval[1])}"
                if entry.interval
                else ""
            )
        rows.append(
            [
                str(rank),
                _model(entry),
                score,
                interval,
                seconds(entry.seconds) if entry.seconds else "",
                money(entry.usd) if entry.usd else "",
            ]
        )
    return table(header, rows, drop=(3, 4, 5), right=(0,), width=width)


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
    header = ["", "Model", metric_name(run.metric), "Interval", "Time", "Cost"]
    rows = []
    ranked = _ranked(run)
    for rank, entry in enumerate(ranked, 1):
        rows.append(
            [
                str(rank),
                _model(entry),
                number(entry.score) if entry.score is not None else entry.state,
                f"{number(entry.interval[0])} to {number(entry.interval[1])}"
                if entry.interval
                else "",
                seconds(entry.seconds) if entry.seconds else "",
                money(entry.usd) if entry.usd else "",
            ]
        )
    best = 0 if ranked and ranked[0].score is not None else None
    return _html_table(header, rows, bold=best, numeric=(0, 2, 4, 5))


def experiment_html(experiment: Experiment) -> str:
    """The experiment's leaderboard as HTML with a console link."""
    parts = [f"<div>{_title(_header(experiment))}"]
    if experiment.run is not None and experiment.run.leaderboard:
        parts.append(leaderboard_html(experiment.run))
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
        lines += table(["", "", "", ""], rows)[1:]
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
            seconds(env.p50_seconds) if env.p50_seconds else "",
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
