# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""How each command's result reads: status lines on stderr, tables and fields on stdout.

A view takes the run of a command and the SDK's result and writes through
`term.Output`, which lays every line out for the width; the helpers at the
top word what several views share (times, counts, money, a context's
version), so each is worded once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from generalforecast import _render
from generalforecast.term import count, duration, plural
from generalforecast.types import STANDARD_PERMISSIONS, money

if TYPE_CHECKING:
    import pandas

    from generalforecast.cli._call import Call
    from generalforecast.types import (
        Change,
        Context,
        Deploy,
        Environment,
        Experiment,
        FineTuneRun,
        Key,
        LogEntry,
        Model,
        NewKey,
        Predictions,
        PredictorInfo,
        Resolved,
        Revision,
        RunEvent,
        Slice,
        Snapshot,
        Source,
        SourceTables,
        SyntheticBatch,
        WriteReceipt,
    )

KINDS = {
    "bigquery": "BigQuery",
    "gcs": "Cloud Storage",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "s3": "S3",
    "postgres": "Postgres",
}
# A context's columns `get` lists before `and N more`, unless `-v`.
COLUMNS_SHOWN = 8


def now() -> datetime:
    """The time relative times are told from; a test puts a clock of its own here."""
    return datetime.now(UTC)


def when(at: datetime | None) -> str:
    """`just now`, `2m ago`, `Nov 21 09:12`, or `Nov 21 2025` in another year."""
    if at is None:
        return ""
    ago = (now() - at).total_seconds()
    if 0 <= ago < 60:
        return "just now"
    if 0 <= ago < 3600:
        return f"{int(ago // 60)}m ago"
    local = at.astimezone()
    if local.year != now().astimezone().year:
        return day(at)
    return f"{local:%b} {local.day} {local:%H:%M}"


def day(at: datetime | None) -> str:
    """`Feb 19 2027`."""
    if at is None:
        return ""
    local = at.astimezone()
    return f"{local:%b} {local.day} {local.year}"


def span(value: timedelta | float | None) -> str:
    """A length of time as it was asked for: `5m`, `15m`, `1h`, `90s`, `2d`."""
    if value is None:
        return ""
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"


def listed(items: Sequence[Any]) -> str:
    """`1`, `1 and 2`, `1, 2 and 3`."""
    words = [str(item) for item in items]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def usage(credits: float, usd: float, calls: int = 0) -> str:
    """`2 calls, 0.4 credits ($0.0004)`."""
    head = f"{plural(calls, 'call')}, " if calls else ""
    return f"{head}{_render.credits(credits)} ({money(usd)})"


def full_model(resolved: Resolved) -> str:
    """`general-forecast/tabicl-v2-classifier@sha256:b14107`."""
    pin = f"@{digest(resolved.model_digest)}" if resolved.model_digest else ""
    return resolved.model + pin


def serves(resolved: Resolved | None) -> str:
    """`kumo-medium on adult v1`, what an environment serves; empty while it serves nothing."""
    return f"{short_model(resolved)} on {served(resolved)}" if resolved else ""


def short_model(resolved: Resolved) -> str:
    """`tabicl-v2`, or the model's name when the service gave no short name."""
    return resolved.short or _render.model_name(resolved.model)


def served(resolved: Resolved) -> str:
    """`customers v8`."""
    return f"{resolved.context} v{resolved.context_version}"


def stamp(at: datetime | None) -> str:
    """`Nov 21 10:04:12`, a log's time."""
    if at is None:
        return ""
    local = at.astimezone()
    return f"{local:%b} {local.day} {local:%H:%M:%S}"


def kind_name(kind: str | None) -> str:
    """A source kind as people write it: `BigQuery`."""
    return KINDS.get(kind or "", kind or "")


def fields(call: Call, rows: Sequence[tuple[str, str | Sequence[str]]]) -> None:
    """Rows of key and value on stdout, those with no value left out."""
    call.out.details([(k, v, "") for k, v in rows if v], stream=call.out.out)


def pinned(context: str, version: int | None, rows: int | None = None) -> str:
    """`adult version 1, 32,561 rows`."""
    text = context if version is None else f"{context} version {version}"
    return text if rows is None else f"{text}, {plural(rows, 'row')}"


def shape(context: Context | Snapshot) -> str:
    """`32,561 rows, 15 columns`."""
    return (
        f"{plural(context.rows or 0, 'row')}, {plural(len(context.columns), 'column')}"
    )


def digest(value: str | None) -> str:
    """A digest cut to twelve hex digits: `sha256:5fb6a4c6f24b`."""
    if not value:
        return ""
    algorithm, _, hexdigits = value.partition(":")
    return f"{algorithm}:{hexdigits[:12]}" if hexdigits else value[:12]


def cell(value: Any) -> str:
    """A table's value as text: empty for missing, `false` for False, `0.5` for 0.5."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def rows_table(call: Call, frame: pandas.DataFrame) -> None:
    """Rows of data: as many columns as fit, left to right, then how many did not."""
    from pandas.api.types import is_numeric_dtype

    header = [str(name) for name in frame.columns]
    align = "".join(
        ">" if is_numeric_dtype(frame[name]) and frame[name].dtype != bool else "<"
        for name in frame.columns
    )
    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False)]
    hidden = call.out.table(
        header, rows, align=align, drop=range(len(header) - 1, 0, -1)
    )
    if hidden:
        call.out.line(
            f"  and {plural(hidden, 'more column')}; widen the terminal or pass --json"
        )


def write(call: Call, frame: pandas.DataFrame) -> None:
    """Writes rows to `--output` (`.csv`, `.tsv`, `.parquet` or `.jsonl`), or shows them."""
    path = call.options.get("output")
    if path is None:
        rows_table(call, frame)
        return
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif suffix == ".jsonl":
        frame.to_json(target, orient="records", lines=True)
    else:
        frame.to_csv(target, index=False, sep="\t" if suffix == ".tsv" else ",")
    call.out.status("Wrote", f"{path}  {plural(len(frame), 'row')}")


def deleted(noun: str) -> Any:
    """`Deleted context signups`."""

    def show(call: Call, result: None) -> None:
        call.out.status("Deleted", f"{noun} {call.name}")

    return show


def contexts_created(call: Call, context: Context) -> None:
    """`Created context adult version 1  32,561 rows, 15 columns in 3.4s`, or a live context's line."""
    out = call.out
    if context.kind == "snapshot":
        versioned("Created")(call, context)
    elif context.source is not None:
        out.status(
            "Created",
            f"live context {context.name}  read from {context.source.source} at call "
            f"time, fresh for {span(context.freshness)}",
        )
        if context.read_at is not None:
            out.status("Read", shape(context), seconds=call.seconds())
    else:
        out.status(
            "Created",
            f"live context {context.name}  {plural(len(context.columns), 'column')}, "
            "rows arrive through the API",
        )


def versioned(verb: str) -> Any:
    """`Uploaded context adult version 2  32,561 rows, 15 columns in 3.4s`; `Unchanged` when the rows were."""

    def show(call: Call, context: Context) -> None:
        name = f"context {context.name} version {context.version}"
        if context.unchanged:
            call.out.status("Unchanged", f"{name}  same rows as before")
        else:
            call.out.status(verb, f"{name}  {shape(context)}", seconds=call.seconds())

    return show


def context_source(context: Context) -> str:
    """`upload adult.csv`, `bigquery-acme`, `snowflake-prod, SQL`, `API`."""
    if context.source is not None:
        return context.source.source + (", SQL" if context.source.sql else "")
    if context.kind == "live":
        return "API"
    return f"upload {context.file}" if context.file else "upload"


def contexts_list(call: Call, contexts: list[Context]) -> None:
    """Name, kind, source, rows, updated and what uses each; drops Used by, Updated, Source."""
    rows = [
        [
            c.name,
            c.kind,
            context_source(c),
            "" if c.rows is None else count(c.rows),
            when(c.updated),
            ", ".join(dict.fromkeys([*c.predictors, *c.experiments, *c.fine_tunes]))
            or "none",
        ]
        for c in contexts
    ]
    header = ["Name", "Kind", "Source", "Rows", "Updated", "Used by"]
    call.out.table(header, rows, align="<<<>", drop=(5, 4, 2))


def contexts_get(call: Call, context: Context) -> None:
    """The context's fields, then its columns, eight unless `-v`."""
    out = call.out
    if context.kind == "snapshot":
        kind = f"snapshot, version {context.version} of {context.versions}"
    elif context.source is not None:
        kind = f"live, read at call time, fresh for {span(context.freshness)}"
    else:
        kind = "live, rows arrive through the API"
    used = [f"experiment {name}" for name in context.experiments]
    used += [f"predictor {name}" for name in context.predictors]
    used += [f"fine-tune {model}" for model in context.fine_tunes]
    rows = [
        ("Name", context.name),
        ("Kind", kind),
        ("Source", context_source(context)),
        ("Rows", "" if context.rows is None else count(context.rows)),
        ("Digest", digest(context.digest)),
        ("Id", context.id or ""),
        ("Time", context.time or ""),
        ("Used by", ", ".join(used) or "none"),
    ]
    fields(call, rows)
    if not context.columns:
        return
    shown = context.columns if out.verbose else context.columns[:COLUMNS_SHOWN]
    out.line()
    out.table(
        ["Column", "Type", "Missing"],
        [[c.name, c.type, count(c.missing)] for c in shown],
        align="<<>",
    )
    if len(shown) < len(context.columns):
        out.line(
            f"  and {len(context.columns) - len(shown)} more; -v shows every column"
        )


def appended(call: Call, receipt: WriteReceipt) -> None:
    """`Appended 250 rows to signups  sequence 1,904`."""
    call.out.status(
        "Appended",
        f"{plural(receipt.rows, 'row')} to {receipt.context}  sequence {count(receipt.sequence)}",
    )


def labeled(call: Call, receipt: WriteReceipt) -> None:
    """`Labeled 41 rows in signups`."""
    call.out.status("Labeled", f"{plural(receipt.rows, 'row')} in {receipt.context}")


def experiment_created(call: Call, experiment: Experiment) -> None:
    """`Created experiment adult-income  classification, 6,513 held-out rows, about $0.039`."""
    parts = [experiment.task or "task to infer"]
    if experiment.run is not None and experiment.run.held_rows:
        parts.append(f"{count(experiment.run.held_rows)} held-out rows")
    estimate = call.state.get("estimate")
    if estimate is not None:
        parts.append(f"about {money(estimate.usd)}")
    call.out.status("Created", f"experiment {experiment.name}  {', '.join(parts)}")


def leaderboard(call: Call, experiment: Experiment) -> None:
    """A run's leaderboard on stdout; drops Interval, Time, Cost."""
    if experiment.run is not None and experiment.run.leaderboard:
        header, rows, align, drop = _render.leaderboard_table(experiment.run)
        call.out.table(header, rows, align=align, drop=drop)


def run_ended(call: Call, experiment: Experiment) -> None:
    """`Finished run 1 in 1m12s  19,539 rows, $0.039`, then the leaderboard; `Started run 2 of ...` without a wait."""
    run = experiment.run
    if run is None:
        return
    head = f"run {run.number} of {experiment.name}"
    if not run.done:
        call.out.status("Started", head)
        return
    taken = (
        (run.finished - run.started).total_seconds()
        if run.finished and run.started
        else call.seconds()
    )
    verb = {"finished": "Finished", "cancelled": "Cancelled"}.get(run.state, "Ended")
    rows = f"{plural(run.rows, 'row')}, {money(run.usd)}"
    call.out.status(verb, f"run {run.number} in {duration(taken)}  {rows}")
    leaderboard(call, experiment)


def experiment_estimate(call: Call, estimate: Any) -> None:
    """`Estimated 6 calls on 19,539 rows  39.1 credits ($0.039)`, then any warnings."""
    call.out.status(
        "Estimated",
        f"{plural(estimate.calls, 'call')} on {plural(estimate.rows, 'row')}  "
        + usage(estimate.credits, estimate.usd),
    )
    for warning in estimate.warnings:
        call.out.warning(warning)


def experiments_list(call: Call, experiments: list[Experiment]) -> None:
    """Name, context, target, runs, best and score, origin, updated; drops Updated, Origin, Context."""
    rows = []
    for e in experiments:
        latest = e.runs[-1] if e.runs else None
        score = (
            f"{_render.metric_name(e.metric)} {latest.score:.3f}"
            if latest is not None and latest.score is not None
            else ""
        )
        rows.append(
            [
                e.name,
                e.context,
                e.target,
                count(len(e.runs)),
                (latest.best or "") if latest else "",
                score,
                e.origin.surface if e.origin else "",
                when(e.updated),
            ]
        )
    header = ["Name", "Context", "Target", "Runs", "Best", "Score", "Origin", "Updated"]
    call.out.table(header, rows, align="<<<>", drop=(7, 6, 1))


def experiment_cancelled(call: Call, experiment: Experiment) -> None:
    """`Cancelled run 2 of adult-income  1 of 1 models stopped, $0.004 spent`."""
    run = experiment.run
    if run is None:
        return
    stopped = sum(1 for e in run.leaderboard if e.state == "cancelled")
    total = len(run.leaderboard) or len(run.models)
    call.out.status(
        "Cancelled",
        f"run {run.number} of {experiment.name}  {stopped} of {total} models "
        f"stopped, {money(run.usd)} spent",
    )


def experiment_updated(call: Call, experiment: Experiment) -> None:
    """`Updated experiment adult-income  runs 1 and 2 are now stale`."""
    numbers = [run.number for run in experiment.runs]
    text = f"experiment {experiment.name}"
    if len(numbers) == 1:
        text += f"  run {numbers[0]} is now stale"
    elif numbers:
        text += f"  runs {listed(numbers)} are now stale"
    call.out.status("Updated", text)


def experiment_slices(call: Call, slices: list[Slice]) -> None:
    """Column, value, rows, the metric and its change against the whole run."""
    metric = _render.metric_name(call.before.metric) if call.before else "Score"
    rows = [
        [s.column, s.value, count(s.rows), f"{s.score:.3f}", f"{s.delta:+.3f}"]
        for s in slices
    ]
    call.out.table(["Column", "Value", "Rows", metric, "Change"], rows, align="<<>>>")


def experiment_deleted(call: Call, result: None) -> None:
    """`Deleted experiment churn-old  3 runs`."""
    runs = len(call.before.runs) if call.before else 0
    call.out.status("Deleted", f"experiment {call.name}  {plural(runs, 'run')}")


def _ref(value: str) -> tuple[str, int | None]:
    name, _, version = value.partition("@")
    return name, int(version) if version.isdigit() else None


def change_value(change: Change, deploy: Deploy) -> tuple[str, str]:
    """A change's sign and value: `~`, `kumo-small → kumo-medium`; `+ context` reads `adult version 1, 32,561 rows`."""
    resolved = deploy.revision.resolved if deploy.revision else None
    before, after = change.before, change.after
    if change.field == "context":
        if before is not None and after is not None:
            (old, v1), (new, v2) = _ref(before), _ref(after)
            if old == new and v1 is not None and v2 is not None:
                return "~", f"{new} version {v1} → {v2}"
            return "~", f"{pinned(old, v1)} → {pinned(new, v2)}"
        if after is not None and resolved is not None:
            return "+", pinned(
                resolved.context, resolved.context_version, resolved.context_rows
            )
    if change.field == "model" and before is None and after is not None and resolved:
        return "+", f"{after}, {full_model(resolved)}"
    if before is None:
        return "+", after or ""
    if after is None:
        return "-", before
    return "~", f"{before} → {after}"


def change_items(
    changes: Sequence[Change], deploy: Deploy, indent: int
) -> list[tuple[int, str, str, str]]:
    """Changes as `term.change_lines` takes them, at `indent`."""
    items = []
    for change in changes:
        sign, value = change_value(change, deploy)
        items.append((indent, sign, change.field, value))
    return items


def plan_lines(
    deploys: Sequence[Deploy], indent: int = 2
) -> list[tuple[int, str, str, str]]:
    """Each predictor's line and its changes' lines under it, as `term.change_lines` takes them."""
    items = []
    for deploy in deploys:
        number = deploy.revision.number if deploy.revision else None
        was = deploy.environment.revision if deploy.environment else None
        sign, head = {
            "created": ("+", f"revision {number}"),
            "updated": (
                "~",
                f"revision {was} → {number}" if was else f"revision {number}",
            ),
        }.get(deploy.action, ("=", f"revision {number}, unchanged"))
        items.append((indent, sign, deploy.predictor, head))
        items += change_items(deploy.changes, deploy, indent + 4)
    return items


def predictor_reached(call: Call, deploy: Deploy) -> None:
    """`Created predictor adult-income revision 1  tabicl-v2-classifier, adult version 1`, or `Updated` and its changes."""
    number = deploy.revision.number if deploy.revision else None
    name = f"predictor {deploy.predictor} revision {number}"
    resolved = deploy.revision.resolved if deploy.revision else None
    if deploy.action == "created":
        detail = (
            f"  {_render.model_name(resolved.model)}, "
            f"{pinned(resolved.context, resolved.context_version)}"
            if resolved
            else ""
        )
        call.out.status("Created", name + detail)
    elif deploy.action == "updated":
        call.out.status("Updated", name)
        call.out.changes(change_items(deploy.changes, deploy, 14))
    else:
        call.out.status("Unchanged", f"{name}  nothing to change")
    call.state["deploy"] = deploy


def predictor_deployed(call: Call, info: PredictorInfo) -> None:
    """`Deployed adult-income revision 1 to production in 48s`, once the wait is over."""
    deploy = call.state.get("deploy")
    if (
        deploy is None
        or deploy.action == "unchanged"
        or not call.values.get("wait", True)
    ):
        return
    env = deploy.env
    number = deploy.revision.number if deploy.revision else info.revision
    call.out.status(
        "Deployed", f"{info.name} revision {number} to {env}", seconds=call.seconds()
    )


def predictors_get(call: Call, info: PredictorInfo) -> None:
    """The predictor's task, model and context, then its environments and what each serves."""
    out = call.out
    spec, resolved = info.spec, info.resolved
    task = resolved.task if resolved and resolved.task else spec.task if spec else None
    rows = [("Name", info.name)]
    if spec is not None:
        rows.append(("Task", f"{task or 'task to infer'}, target {spec.target}"))
    if resolved is not None:
        context = (resolved.context, resolved.context_version, resolved.context_rows)
        rows += [
            ("Model", f"{short_model(resolved)}, {full_model(resolved)}"),
            ("Context", pinned(*context)),
        ]
    fields(call, rows)
    if info.environments:
        out.line()
        rows = [
            [
                e.name,
                count(e.revision),
                e.state,
                serves(e.resolved),
                count(e.calls_today),
                duration(e.p50_seconds) if e.p50_seconds else "",
            ]
            for e in info.environments.values()
        ]
        out.table(
            ["Environment", "Revision", "State", "Serves", "Calls today", "p50"],
            rows,
            align="<><<>>",
            drop=(3,),
        )


def _envs(infos: Iterable[PredictorInfo]) -> list[str]:
    names = {name for info in infos for name in info.environments}
    order = ["production", "staging"]
    return [n for n in order if n in names] + sorted(names - set(order))


def predictors_list(call: Call, infos: list[PredictorInfo]) -> None:
    """Name, model, context, each environment's revision and state, calls today."""
    envs = _envs(infos)
    rows = []
    for info in infos:
        resolved = info.resolved
        model = short_model(resolved) if resolved else ""
        context = served(resolved) if resolved else ""
        serving = [
            f"{info.environments[e].revision} {info.environments[e].state}"
            if e in info.environments
            else ""
            for e in envs
        ]
        calls = sum(e.calls_today for e in info.environments.values())
        rows.append([info.name, model, context, *serving, count(calls)])
    header = [
        "Name",
        "Model",
        "Context",
        *(e.capitalize() for e in envs),
        "Calls today",
    ]
    last = len(header) - 1
    drop = (last, *range(last - 1, 3, -1), 2)
    call.out.table(header, rows, align="<<<" + "<" * len(envs) + ">", drop=drop)


def predictors_revisions(call: Call, revisions: list[Revision]) -> None:
    """Revision, model, context, origin, created, and where it serves; drops Origin, Created."""
    info = call.before
    rows = []
    for r in revisions:
        resolved = r.resolved
        serving = (
            [e.name for e in info.environments.values() if e.revision == r.number]
            if info
            else []
        )
        rows.append(
            [
                count(r.number),
                short_model(resolved) if resolved else "",
                served(resolved) if resolved else "",
                r.origin.surface if r.origin else "",
                when(r.created),
                ", ".join(serving),
            ]
        )
    header = ["Revision", "Model", "Context", "Origin", "Created", "Serving in"]
    call.out.table(header, rows, align=">", drop=(3, 4))


def predicted(call: Call, result: Predictions) -> None:
    """`Predicted 1,000 rows in 2.1s  10 calls, 2.0 credits ($0.002)`, then the rows or the file."""
    call.out.status(
        "Predicted",
        f"{plural(result.rows, 'row')} in {duration(call.seconds())}  "
        + usage(result.credits, result.usd, result.calls),
    )
    for note in result.notes:
        call.out.warning(note)
    write(call, result.to_pandas())


def _was(call: Call, env: Environment) -> str:
    """`  was revision 4`: what `env` served before, when that was another revision."""
    info = call.before
    found = info.environments.get(env.name) if info is not None else None
    if found is None or found.revision == env.revision:
        return ""
    return f"  was revision {found.revision}"


def promoted(call: Call, env: Environment) -> None:
    """`Promoted churn revision 5 to production in 7.2s  was revision 4`."""
    taken = duration(call.seconds())
    call.out.status(
        "Promoted",
        f"{call.name} revision {env.revision} to {env.name} in {taken}{_was(call, env)}",
    )


def rolled_back(call: Call, env: Environment) -> None:
    """`Rolled back churn in production to revision 4 in 1.1s  was revision 5`."""
    taken = duration(call.seconds())
    call.out.status(
        "Rolled back",
        f"{call.name} in {env.name} to revision {env.revision} in {taken}"
        + _was(call, env),
    )


def log_rows(entries: Iterable[LogEntry]) -> list[list[str]]:
    """Log entries as table rows: time, level, source, request and message."""
    return [
        [stamp(e.at), e.level, e.source or "", e.request_id or "", e.message]
        for e in entries
    ]


def predictor_question(call: Call) -> str:
    """`Delete predictor churn and its 5 revisions? Production serves 18,330 calls a day`."""
    info, revisions = call.before
    question = (
        f"Delete predictor {info.name} and its {plural(len(revisions), 'revision')}?"
    )
    production = info.environments.get("production")
    if production is not None and production.calls_today:
        question += f" Production serves {count(production.calls_today)} calls a day"
    return question


def sources_list(call: Call, sources: list[Source]) -> None:
    """Name, kind, state, tables and contexts, and when each was last tested."""
    if getattr(sources, "available", True) is False:
        call.out.say("This service connects no sources.")
    rows = [
        [s.name, kind_name(s.kind), words(s.state),
         "" if s.tables is None else count(s.tables), count(s.contexts), when(s.tested_at)]
        for s in sources
    ]  # fmt: skip
    call.out.table(["Name", "Kind", "State", "Tables", "Contexts", "Tested"], rows,
                   align="<<<>><", drop=(5, 4, 3))  # fmt: skip


def sources_get(call: Call, source: Source) -> None:
    """The source's kind and settings, how it is read, its state and what it holds."""
    kind = ", ".join(
        [
            kind_name(source.kind),
            *(f"{k} {v}" for k, v in source.config.items()),
        ]
    )
    state = words(source.state) + (
        f", tested {when(source.tested_at)}" if source.tested_at else ""
    )
    rows = [("Name", source.name), ("Kind", kind)]
    if source.access:
        rows.append(("Access", ACCESS.get(source.access, source.access)))
    rows.append(("State", state))
    holds = [] if source.tables is None else [plural(source.tables, "table")]
    rows.append(("Holds", ", ".join([*holds, plural(source.contexts, "context")])))
    if source.error:
        rows.append(("Error", source.error))
    fields(call, rows)


ACCESS = {"service_account": "the workspace's service account", "secret": "its secret"}


def words(state: str) -> str:
    """A state as words: `not_found` is `not found`."""
    return state.replace("_", " ")


def source_tested(call: Call, source: Source) -> None:
    """`Tested source bigquery-acme  ready in 0.4s`."""
    call.out.status(
        "Tested", f"source {source.name}  {words(source.state)}", seconds=call.seconds()
    )


def source_created(call: Call, source: Source) -> None:
    """`Created source pg-replica  Postgres, tested in 0.4s`."""
    kind = kind_name(source.kind)
    call.out.status(
        "Created", f"source {source.name}  {kind}, tested in {duration(call.seconds())}"
    )


def sources_tables(call: Call, tables: SourceTables) -> None:
    """Each table's name and type, and whether a context can be made of it."""
    rows = [[t.name, t.type, "yes" if t.readable else "no"] for t in tables]
    call.out.table(["Table", "Type", "Readable"], rows)
    if tables.more:
        call.out.say("The warehouse holds more tables; narrow them with --prefix.")


def permissions(key: Key) -> str:
    """`standard`, or how many permissions a key has."""
    if set(key.permissions) == set(STANDARD_PERMISSIONS):
        return "standard"
    return plural(len(key.permissions), "permission")


def owner(key: Key) -> str:
    """`maya@acme.com`, or `service` or `legacy` for a key no account owns."""
    if key.owner is not None and key.owner.email:
        return key.owner.email
    return "service" if key.service else "legacy" if key.legacy else ""


def keys_list(call: Call, keys: list[Key]) -> None:
    """Name, id, permissions, expiry, last use and owner; drops Owner, Last used, Permissions."""
    rows = [
        [
            k.name,
            k.id,
            permissions(k),
            k.state if k.state != "active" else day(k.expires) or "never",
            when(k.last_used) or "never",
            owner(k),
        ]
        for k in keys
    ]
    call.out.table(
        ["Name", "Id", "Permissions", "Expires", "Last used", "Owner"],
        rows,
        drop=(5, 4, 2),
    )


def key_created(verb: str, detail: str) -> Any:
    """`Created key ci-deploy  standard permissions, expires Feb 19 2027; shown once`, then the token on stdout."""

    def show(call: Call, new: NewKey) -> None:
        key = new.key
        expires = f"expires {day(key.expires)}" if key.expires else "never expires"
        text = detail.format(permissions=permissions(key), expires=expires)
        call.out.status(verb, f"key {key.name}  {text}; shown once")
        call.out.line(new.token)

    return show


def key_changed(verb: str) -> Any:
    """`Revoked key ci-deploy`, `Deleted key old-laptop`, named from the key `before` found."""

    def show(call: Call, result: Any) -> None:
        call.out.status(verb, f"key {call.before.name if call.before else call.name}")

    return show


def models_list(call: Call, models: list[Model]) -> None:
    """One row per model family: its tasks, and the context rows and inputs it was measured to."""
    families: dict[str, list[Model]] = {}
    for model in models:
        families.setdefault(model.short or model.id, []).append(model)
    rows = []
    for short, found in families.items():
        limits = next((m.limits for m in found if m.limits), None)
        rows.append(
            [
                short,
                ", ".join(dict.fromkeys(m.task or "" for m in found if m.task)),
                count(limits.context_rows) if limits and limits.context_rows else "",
                count(limits.inputs) if limits and limits.inputs else "",
            ]
        )
    header = ["Model", "Tasks", "Context rows", "Inputs"]
    call.out.table(header, rows, align="<<>>", drop=(3, 2))


def models_get(call: Call, model: Model) -> None:
    """The model's id, task, limits, and what it was trained on."""
    rows = [("Model", f"{model.short}, {model.id}"), ("Task", model.task or "")]
    if model.kind == "fine-tune":
        rows.append(("Base", model.base or ""))
    limits = model.limits
    if limits is not None:
        parts = [
            f"{count(value)} {noun}"
            for value, noun in (
                (limits.context_rows, "context rows"),
                (limits.inputs, "inputs"),
            )
            if value
        ]
        rows.append(("Limits", listed(parts)))
    trained = model.trained_on
    if trained is not None:
        parts = [
            f"{count(value)} {noun}"
            for value, noun in (
                (trained.rows, "rows"),
                (trained.inputs, "inputs"),
                (trained.classes, "classes"),
            )
            if value
        ]
        rows.append(
            ("Trained on", f"up to {listed(parts)}; past that quality is untested")
        )
    fields(call, rows)


def snapshot_created(call: Call, snapshot: Snapshot) -> None:
    """`Created snapshot credit-2026-01  500,000 rows, 12 columns in 8.1s`."""
    call.out.status(
        "Created",
        f"snapshot {snapshot.name}  {shape(snapshot)}",
        seconds=call.seconds(),
    )


def batch_started(call: Call, batch: SyntheticBatch) -> None:
    """`Started synthetic batch credit-warmup  64 tables, tag synthetic-credit-warmup`."""
    call.out.status(
        "Started",
        f"synthetic batch {batch.name}  {plural(batch.count, 'table')}, tag {batch.tag}",
    )


def snapshots_list(call: Call, snapshots: list[Snapshot]) -> None:
    """Name, state, rows, columns, task, tags, created; drops Tags, Created, Task."""
    rows = [
        [
            s.name,
            s.state,
            "" if s.rows is None else count(s.rows),
            count(len(s.columns)),
            s.task or "",
            ", ".join(s.tags),
            when(s.created),
        ]
        for s in snapshots
    ]
    header = ["Name", "State", "Rows", "Columns", "Task", "Tags", "Created"]
    call.out.table(header, rows, align="<<>>", drop=(5, 6, 4))


def snapshots_get(call: Call, snapshot: Snapshot) -> None:
    """The snapshot's state, shape, training metadata, digest and tags."""
    task = snapshot.task or ""
    if snapshot.target:
        task = f"{task}, target {snapshot.target}"
    rows = [
        ("Name", snapshot.name),
        ("State", snapshot.state + (f": {snapshot.error}" if snapshot.error else "")),
        ("Rows", shape(snapshot)),
        ("Task", task),
        ("Digest", digest(snapshot.digest)),
        ("Tags", ", ".join(snapshot.tags)),
        ("Created", when(snapshot.created)),
    ]
    fields(call, rows)


def batch_get(call: Call, batch: SyntheticBatch) -> None:
    """The batch's state, its tables by state, tag and seed."""
    counts = ", ".join(f"{n:,} {state}" for state, n in batch.counts.items() if n)
    rows = [
        ("Name", batch.name),
        ("State", batch.state),
        ("Tables", plural(batch.count, "table") + (f": {counts}" if counts else "")),
        ("Tag", batch.tag),
        ("Seed", "" if batch.seed is None else str(batch.seed)),
    ]
    fields(call, rows)


def snapshot_tagged(call: Call, snapshot: Snapshot) -> None:
    """`Tagged snapshot credit-2026-01  nightly, credit`."""
    call.out.status(
        "Tagged", f"snapshot {snapshot.name}  {', '.join(snapshot.tags) or 'no tags'}"
    )


def run_started(call: Call, run: FineTuneRun) -> None:
    """`Started run credit-default@ft2  base tabicl-v2, 2 tables pinned`."""
    call.out.status(
        "Started",
        f"run {_render.run_name(run)}  base {_render.model_name(run.base)}, "
        f"{plural(len(run.data), 'table')} pinned",
    )


def run_ended_line(call: Call, run: FineTuneRun) -> None:
    """`Finished run credit-default@ft2 in 2m10s  val 0.398 (base 0.421), published step 1,000`."""
    if not run.done:
        return
    verb = {"finished": "Finished", "cancelled": "Cancelled"}.get(run.state, "Ended")
    published = run.published
    if published is not None:
        detail = f"published step {count(published.kept_step or 0)}"
        if published.validation_loss is not None:
            loss = _render.validation(
                published.validation_loss, published.base_validation_loss
            )
            detail = f"val {loss}, {detail}"
    else:
        detail = "nothing beat the base, so nothing was published"
    call.out.status(
        verb, f"run {_render.run_name(run)} in {duration(run.seconds)}  {detail}"
    )


def steps(run: FineTuneRun) -> str:
    """`340/1,000`, or the step alone when the run has no fixed number of steps."""
    return f"{count(run.step)}/{count(run.steps)}" if run.steps else count(run.step)


def runs_list(call: Call, runs: list[FineTuneRun]) -> None:
    """Run, state, base, step, loss, validation loss, cost, created; drops Created, Cost, Base, Loss."""
    rows = [
        [
            _render.run_name(r),
            r.state,
            _render.model_name(r.base),
            steps(r),
            "" if r.loss is None else f"{r.loss:.3f}",
            "" if r.validation_loss is None else f"{r.validation_loss:.3f}",
            money(r.usd) if r.usd else "",
            when(r.created),
        ]
        for r in runs
    ]
    header = ["Run", "State", "Base", "Step", "Loss", "Val loss", "Cost", "Created"]
    call.out.table(header, rows, align="<<<>>>>", drop=(7, 6, 2, 4))


def runs_get(call: Call, run: FineTuneRun) -> None:
    """The run's state, base, data pins, progress, losses and what it published."""
    pins = [
        f"{pin.name}"
        + (f" version {pin.version}" if pin.version else "")
        + f" {digest(pin.digest)}"
        for pin in run.data
    ]
    method = ", ".join(f"{k} {v}" for k, v in run.method.items())
    validation = (
        ""
        if run.validation_loss is None
        else _render.validation(run.validation_loss, run.base_validation_loss)
    )
    published = run.published
    rows = [
        ("Run", _render.run_name(run)),
        ("State", run.state + (f": {run.error}" if run.error else "")),
        ("Base", run.base),
        ("Data", pins),
        ("Method", method),
        ("Step", steps(run)),
        ("Loss", "" if run.loss is None else f"{run.loss:.3f}"),
        ("Validation", validation),
        (
            "Published",
            f"{published.model}, step {count(published.kept_step or 0)}"
            if published
            else "",
        ),
        ("Cost", usage(run.credits, run.usd) if run.credits else ""),
        ("Created", when(run.created)),
    ]
    fields(call, rows)


def event_line(event: RunEvent) -> str:
    """`Nov 21 10:04:12  metrics    step 340  loss 0.412  val 0.398`."""
    body = event.body
    if event.kind == "metrics":
        parts = [f"step {count(int(body.get('step', body.get('steps', 0))))}"]
        if body.get("loss") is not None:
            parts.append(f"loss {float(body['loss']):.3f}")
        if body.get("validation_loss") is not None:
            parts.append(f"val {float(body['validation_loss']):.3f}")
        text = "  ".join(parts)
    elif event.kind == "published":
        text = str(body.get("model", ""))
    else:
        text = str(body.get("state", ""))
    return f"  {stamp(event.at)}  {event.kind:<9}  {text}".rstrip()


def run_events(call: Call, events: Iterable[RunEvent]) -> None:
    """Each event on its own line, as it arrives when watched."""
    for event in events:
        call.out.line(event_line(event))


def run_cancelled(call: Call, run: FineTuneRun) -> None:
    """`Cancelled run credit-default@ft2`, and the step it published with `--save`."""
    saved = (
        f"  published step {count(run.published.kept_step or 0)}"
        if run.published is not None
        else ""
    )
    call.out.status("Cancelled", f"run {_render.run_name(run)}{saved}")
