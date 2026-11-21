# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The command table: every `gf` command, the SDK method it calls and how its result shows.

Each row mirrors `gf.<resource>.<verb>(NAME, **flags)`; the few that do
more than call the SDK once say how in their `run`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from generalforecast.cli import _login, _views
from generalforecast.cli._call import Call
from generalforecast.cli._spec import (
    BUDGET,
    CLEAR,
    EXPIRES,
    FILE_TEXT,
    FRACTION,
    METHOD,
    MISSING,
    NAMES,
    NUMBER,
    NUMBERS,
    PAIRS,
    SECONDS,
    SELECTION,
    SETTINGS,
    SPAN,
    STDIN,
    TEXT,
    WHOLE,
    YES,
    Arg,
    Command,
    Flag,
    Kind,
    Usage,
    choice,
    switch,
)
from generalforecast.term import duration, plural
from generalforecast.types import TASKS, Evaluation, Plan

# Seconds between pages of `gf predictors logs --follow`.
FOLLOW_SECONDS = 2.0

NAME = Arg("NAME", "The name")
FILE = Arg("FILE", "A .csv, .tsv, .parquet, .arrow, .feather or .jsonl file")
RUN = Arg("RUN", "A run, NAME@ftN, or NAME for its latest")
KEY = Arg("ID", "The key's id, or its name")
TASK = choice(*TASKS)
CONFIG = Kind("KEY=VALUE", PAIRS.parse, "KEY=VALUE pairs", spread=True)

WAIT = (
    Flag("no-wait", "Return once it has started", switch(False), to="wait"),
    Flag(
        "wait",
        "Wait at most this many seconds (default: until it ends)",
        SECONDS,
        to="timeout",
    ),
)
YES_FLAG = Flag("yes", "Go ahead without asking", YES, to=None)
OUTPUT = Flag(
    "output",
    "Write the rows to this file: .csv, .tsv, .parquet or .jsonl",
    TEXT,
    to=None,
    metavar="F",
)
ENV = Flag("env", "The environment (default: production)", metavar="E")


def _flag(
    name: str, help: str, kind: Kind = TEXT, metavar: str = "C", **more: Any
) -> Flag:
    return Flag(name, help, kind, metavar=metavar if kind is TEXT else None, **more)


SQL = (
    _flag("sql", "Read the rows of this SQL query", metavar="Q"),
    Flag("sql-file", "Read the SQL query from this file", FILE_TEXT, to="sql"),
)
SETUP = (
    _flag("title", "A title to show in the console", metavar="T"),
    _flag("target", "The column to predict"),
    _flag("task", "classification, regression or forecast (default: from the target)", TASK),
    _flag("id", "The column that identifies a row"),
    _flag("time", "The time column; the default holdout is then the latest 20% by it"),
    _flag("inputs", "The columns the models read (default: every other column)", NAMES, metavar="C,..."),
    _flag("exclude", "Columns to leave out of the inputs", NAMES, metavar="C,..."),
    Flag("type", "A column's type, as age=number (repeatable)", PAIRS, to="types", metavar="C=TYPE", repeat=True),
    Flag("holdout", 'The rows held out to score on: "20%", "20% by time", "1000 rows", ...', metavar="H"),
    *SQL,
    _flag("context", "The context, NAME or NAME@VERSION"),
)  # fmt: skip
MODELS = _flag(
    "models",
    "The models to compare (default: the recommended set)",
    NAMES,
    metavar="M,...",
)
SPEC = (
    _flag("context", "The context it serves, NAME (the latest at each deploy) or NAME@VERSION"),
    _flag("model", "The model: a short name such as tabicl-v2, or a full id", metavar="M"),
    *(f for f in SETUP if f.name in ("target", "task", "id", "time", "inputs", "exclude", "type")),
    _flag("series", "The columns that name a forecast's series", NAMES, metavar="C,..."),
    _flag("frequency", "A forecast's step, such as 1d or 1h", metavar="F"),
    _flag("horizon", "How many steps a forecast predicts", WHOLE),
    _flag("quantiles", "The quantiles to predict, as 0.1,0.5,0.9", NUMBERS),
    _flag("past", "Covariates known only in the past", NAMES, metavar="C,..."),
    _flag("future", "Covariates known in the future too", NAMES, metavar="C,..."),
    Flag("selection", "How each query's context is drawn from a large one", SELECTION),
)  # fmt: skip


def _ask(noun: str, verb: str = "Delete") -> Any:
    return lambda call: f"{verb} {noun} {call.name}?"


def experiment_create(call: Call) -> Any:
    """Prices run 1, then creates the experiment, so `Created` can say what it should cost."""
    setup = {k: v for k, v in call.values.items() if k not in ("wait", "timeout")}
    call.state["estimate"] = call.method("experiments.estimate")(**setup)
    return call.invoke()


def experiment_question(call: Call) -> str:
    """`Delete experiment churn-old and its 3 runs?`."""
    return (
        f"Delete experiment {call.name} and its {plural(len(call.before.runs), 'run')}?"
    )


def logs(call: Call) -> Any:
    """A page of logs; with `--follow`, every page from there on as it is written."""
    page = call.invoke()
    if not call.options.get("follow"):
        return page

    def pages() -> Iterator[Any]:
        current = page
        while True:
            yield current
            time.sleep(FOLLOW_SECONDS)
            current = call.invoke(after=current.next or call.values.get("after"))

    return pages()


def logs_view(call: Call, result: Any) -> None:
    """Each page of entries as aligned lines: time, level, source, request and message."""
    for page in result if isinstance(result, Iterator) else [result]:
        call.out.table(None, _views.log_rows(page.entries))


def deploy(call: Call) -> list[Any]:
    """`gf deploy FILE`: loads the specs, prints the plan, then deploys what changed in file order."""
    from generalforecast.cli._load import predictors

    out, env = call.out, call.values.get("env", "production")
    path = call.args[0]
    specs = predictors(path)
    out.status("Loading", f"{path}  {plural(len(specs), 'predictor')}")
    out.status("Planning", env)
    sdk = call.method("predictors.deploy")
    plan = sdk(*specs, env=env, dry_run=True)
    # A plan with changes stands apart from the lines around it.
    apart = any(d.action != "unchanged" for d in plan)
    if apart:
        out.say()
    out.changes(_views.plan_lines(plan))
    if apart:
        out.say()
    if call.values.get("dry_run"):
        return plan
    waits = {k: call.values[k] for k in ("wait", "timeout") if k in call.values}
    deployed = []
    for spec, planned in zip(specs, plan, strict=True):
        if planned.action == "unchanged":
            deployed.append(planned)
            continue
        started = call.seconds()
        found = sdk(spec, env=env, **waits)[0]
        if waits.get("wait", True):
            number = found.revision.number if found.revision else None
            out.status(
                "Deployed",
                f"{found.predictor} revision {number} to {env}",
                seconds=call.seconds() - started,
            )
        deployed.append(found)
    return deployed


def deploy_view(call: Call, deploys: list[Any]) -> None:
    """`Finished 1 created, 1 updated, 3 unchanged in 55s`, or that it was a plan only."""
    if call.values.get("dry_run"):
        call.out.status("Finished", "plan only; nothing deployed")
        return
    counts = {
        action: sum(d.action == action for d in deploys)
        for action in ("created", "updated", "unchanged")
    }
    if counts["created"] + counts["updated"] == 0:
        call.out.status("Finished", f"nothing to deploy in {duration(call.seconds())}")
        return
    done = ", ".join(f"{n} {action}" for action, n in counts.items() if n)
    call.out.status("Finished", f"{done} in {duration(call.seconds())}")
    url = next(
        (d.environment.url for d in deploys if d.environment and d.environment.url),
        None,
    )
    call.out.view(url.split("#/predictors")[0] + "#/predictors" if url else None)


def run_create(call: Call) -> Any:
    """Starts a fine-tuning run from its flags, or from `--from FILE`; `--dry-run` only resolves it."""
    from generalforecast import _ops

    settings: dict[str, Any] = dict(call.values)
    wait = settings.pop("wait", False)
    source = call.options.get("from")
    if source is not None:
        if settings:
            given = ", ".join(f"--{k.replace('_', '-')}" for k in settings)
            raise Usage(
                f"--from takes every setting from its file, not {given}",
                hint="set them in the file's gf.FineTuneSpec",
            )
        from generalforecast.cli._load import fine_tune

        settings = fine_tune(source).arguments()
    if call.options.get("dry_run"):
        return call.client._run(_ops.run_request(call.name, **settings))
    return call.method()(call.name, wait=wait, **settings)


def run_created(call: Call, result: Any) -> None:
    """The run's end once waited for, or with `--dry-run` the request that would start it."""
    if call.options.get("dry_run"):
        import json

        call.out.line(json.dumps(result, indent=2))
        return
    _views.run_ended_line(call, result)


def predictor_and_revisions(call: Call) -> tuple[Any, Any]:
    """The predictor a delete is about and its revisions, for the question it asks."""
    return call.method("predictors.get")(call.name), call.method(
        "predictors.revisions"
    )(call.name)


def find_key(call: Call) -> Any:
    """The key an id or a name names, for the question and the line after."""
    from generalforecast.types import Key

    found = [k for k in call.method("keys.list")() if call.name in (k.id, k.name)]
    return found[0] if found else Key(call.name, call.name)


def _key_question(verb: str, after: str = "") -> Any:
    return lambda call: f"{verb} key {call.before.name}?{after}"


def _get(path: str, **keywords: Any) -> Any:
    """A `before` that reads the resource the command is about."""
    return lambda call: call.method(path)(
        call.name, **{k: call.values.get(v) for k, v in keywords.items()}
    )


def _command(words: str, summary: str, show: Any, *args: Arg, **more: Any) -> Command:
    return Command(tuple(words.split()), summary, show, args=args, **more)


COMMANDS_LIST = [
    _command("login", "Log this machine in through your browser, or with a key", _login.logged_in,
             run=_login.login, sdk="login", flags=(
                 Flag("key", "Read an API key from a prompt, or from stdin when piped", YES, to=None),
                 Flag("url", "The server (default: GF_URL, else app.generalforecast.com)", to=None, metavar="URL"),
             )),
    _command("logout", "Forget this machine's login, revoking a key `gf login` made", _login.logged_out,
             run=_login.logout, sdk="logout",
             flags=(Flag("url", "The server (default: GF_URL)", to=None, metavar="URL"),)),
    _command("whoami", "Show who the login acts for, and in which project", _login.whoami_view,
             run=_login.whoami, sdk="whoami"),
    _command("contexts create", "Create a context from a file, a source's table or query, or the API",
             _views.contexts_created, NAME, flags=(
                 Flag("file", "Upload this file as version 1", to="data", metavar="F"),
                 _flag("source", "Import from this source", metavar="S"),
                 _flag("table", "The source's table", metavar="T"),
                 *SQL,
                 Flag("live", "Read the source at call time, or take rows through the API", YES),
                 Flag("columns", "A live context's columns, as plan=category,seats=number", PAIRS, metavar="C=TYPE,..."),
                 _flag("targets", "The columns `gf contexts label` fills", NAMES, metavar="C,..."),
                 _flag("id", "The column that identifies a row"),
                 _flag("time", "The time column"),
                 _flag("freshness", "How long a live read is cached, 1m to 24h (default: 15m)", metavar="15m"),
                 WAIT[0],
             )),
    _command("contexts upload", "Upload a new version; rows equal to the latest add none",
             _views.versioned("Uploaded"), NAME, FILE, flags=(WAIT[0],)),
    _command("contexts refresh", "Import a new version from the context's source", _views.versioned("Refreshed"),
             NAME, flags=(WAIT[0],)),
    _command("contexts append", "Append rows to a live context fed through the API", _views.appended, NAME, FILE,
             flags=(_flag("idempotency-key", "Make the write safe to repeat under this key", metavar="K"),)),
    _command("contexts label", "Fill a target for rows already in a live context, by its id", _views.labeled,
             NAME, FILE, flags=(_flag("target", "The column to fill"),
                                _flag("idempotency-key", "Make the write safe to repeat", metavar="K"))),
    _command("contexts get", "Show a context and its columns", _views.contexts_get, NAME,
             flags=(_flag("version", "Show this version (default: the latest)", WHOLE),)),
    _command("contexts list", "List the project's contexts", _views.contexts_list),
    _command("contexts rows", "Show a context's first rows", _views.rows_table, NAME, flags=(
        _flag("limit", "How many rows (default: 20)", WHOLE), _flag("version", "Read this version", WHOLE))),
    _command("contexts delete", "Delete a context", _views.deleted("context"), NAME, flags=(YES_FLAG,),
             ask=_ask("context")),
    _command("experiments create", "Create an experiment and start run 1", _views.run_ended,
             run=experiment_create, reached=_views.experiment_created,
             flags=(*SETUP[-1:], *SETUP[1:2], _flag("name", "The name (default: CONTEXT-TARGET)", metavar="N"),
                    SETUP[0], *SETUP[2:-1], MODELS, *WAIT)),
    _command("experiments estimate", "Price run 1 of an experiment without starting it", _views.experiment_estimate,
             flags=(*SETUP[-1:], *SETUP[1:2], *SETUP[2:-1], MODELS)),
    _command("experiments run", "Start another run with the same setup", _views.run_ended, NAME,
             flags=(MODELS, *WAIT)),
    _command("experiments update", "Change an experiment's setup; earlier runs become stale",
             _views.experiment_updated, NAME, flags=SETUP),
    _command("experiments get", "Show an experiment's leaderboard", _views.leaderboard, NAME,
             flags=(_flag("run", "Show this run (default: the latest)", WHOLE),)),
    _command("experiments wait", "Wait for a run to end", _views.run_ended, NAME, flags=(
        _flag("run", "Wait for this run (default: the latest)", WHOLE),
        Flag("wait", "Wait at most this many seconds", SECONDS, to="timeout"))),
    _command("experiments list", "List experiments", _views.experiments_list,
             flags=(_flag("context", "Only those on this context"),)),
    _command("experiments cancel", "Cancel a run", _views.experiment_cancelled, NAME,
             flags=(_flag("run", "Cancel this run (default: the latest)", WHOLE),)),
    _command("experiments predictions", "Show or write a run's predictions on the held-out rows", _views.write,
             NAME, flags=(_flag("run", "This run (default: the latest)", WHOLE),
                          _flag("model", "Only this model's", metavar="M"), OUTPUT)),
    _command("experiments slices", "Show a run's error by slice, worst first", _views.experiment_slices, NAME,
             flags=(_flag("run", "This run (default: the latest)", WHOLE), _flag("model", "This model", metavar="M")),
             before=_get("experiments.get", run="run"), key="slices"),
    _command("experiments delete", "Delete an experiment and its runs", _views.experiment_deleted, NAME,
             flags=(YES_FLAG,), before=_get("experiments.get"), ask=experiment_question),
    _command("deploy", "Deploy the gf.Predictor specs a Python file defines; never prompts or prunes",
             deploy_view, Arg("FILE", "A Python file of gf.Predictor specs"), run=deploy, key="deploys",
             python="gf.predictors.deploy(*specs, **flags)",
             flags=(ENV, Flag("dry-run", "Print the plan only", YES), *WAIT)),
    _command("predictors create", "Create a predictor and deploy it", _views.predictor_deployed, NAME,
             reached=_views.predictor_reached, flags=(*SPEC, ENV, WAIT[0])),
    _command("predictors update", "Deploy a revision of the given fields over the serving spec",
             _views.predictor_deployed, NAME, reached=_views.predictor_reached,
             flags=(*SPEC, Flag("clear", "Clear a field (repeatable)", CLEAR, repeat=True), ENV, WAIT[0])),
    _command("predictors get", "Show a predictor and its environments", _views.predictors_get, NAME),
    _command("predictors list", "List predictors", _views.predictors_list),
    _command("predictors revisions", "List a predictor's revisions and where each serves",
             _views.predictors_revisions, NAME, before=_get("predictors.get"), key="revisions"),
    _command("predictors predict", "Score rows with a predictor", _views.predicted, NAME, flags=(
        Flag("queries", "The rows to score", metavar="F"), ENV,
        _flag("as-of", "Read a live context as it was then", metavar="T"), OUTPUT)),
    _command("predictors promote", "Point an environment at what another serves", _views.promoted, NAME,
             before=_get("predictors.get"), flags=(
                 Flag("from", "The environment to take from (default: staging)", to="from_env", metavar="E"),
                 _flag("env", "The environment to point (default: production)", metavar="E"),
                 _flag("revision", "Point at this revision instead", WHOLE))),
    _command("predictors rollback", "Point an environment back at what it served before", _views.rolled_back,
             NAME, before=_get("predictors.get"),
             flags=(ENV, _flag("revision", "Point at this revision instead", WHOLE))),
    _command("predictors logs", "Show an environment's logs", logs_view, NAME, run=logs, flags=(
        ENV, _flag("limit", "How many entries (default: 100)", WHOLE),
        _flag("level", "Only entries at this level", choice("info", "warn", "error")),
        _flag("q", "Only entries matching this text", metavar="TEXT"),
        _flag("since", "Entries from this time", metavar="T"), _flag("until", "Entries up to this time", metavar="T"),
        Flag("follow", "Keep printing entries as they are written", YES, to=None))),
    _command("predictors delete", "Delete a predictor and its revisions", _views.deleted("predictor"), NAME,
             flags=(YES_FLAG,), before=predictor_and_revisions, ask=_views.predictor_question),
    _command("sources list", "List the workspace's sources", _views.sources_list),
    _command("sources get", "Show a source", _views.sources_get, NAME),
    _command("sources test", "Test a source's connection now", _views.source_tested, NAME),
    _command("sources create", "Connect a warehouse or bucket", _views.source_created, NAME, flags=(
        _flag("kind", "bigquery, gcs, snowflake, databricks, s3 or postgres", metavar="K"),
        Flag("config", "A setting of the kind, as host=10.2.0.4 (repeatable)", CONFIG, repeat=True),
        Flag("secret-stdin", "Read the source's secret from stdin", STDIN, to="secret"))),
    _command("sources tables", "List a source's tables", _views.sources_tables, NAME,
             flags=(_flag("prefix", "Only tables starting with this", metavar="P"),), key="tables"),
    _command("sources preview", "Show twenty rows of a source's table or query", _views.rows_table, NAME,
             flags=(_flag("table", "The table", metavar="T"), _flag("sql", "A query", metavar="Q"))),
    _command("sources delete", "Delete a source", _views.deleted("source"), NAME, flags=(YES_FLAG,),
             ask=_ask("source")),
    _command("keys list", "List the project's keys", _views.keys_list),
    _command("keys create", "Create a key; its token is shown once",
             _views.key_created("Created", "{permissions} permissions, {expires}"), NAME, flags=(
                 _flag("permissions", "The permissions (default: the standard set)", NAMES, metavar="P,..."),
                 Flag("expires", "When it expires (default: 90d)", EXPIRES))),
    _command("keys rotate", "Replace a key's token; the old one stops working now",
             _views.key_created("Rotated", "the old token stops working now"), KEY, flags=(YES_FLAG,),
             before=find_key, ask=_key_question("Rotate", " Its token stops working now")),
    _command("keys revoke", "Revoke a key", _views.key_changed("Revoked"), KEY, flags=(YES_FLAG,),
             before=find_key, ask=_key_question("Revoke")),
    _command("keys delete", "Delete an expired or revoked key", _views.key_changed("Deleted"), KEY,
             flags=(YES_FLAG,), before=find_key, ask=_key_question("Delete")),
    _command("models list", "List the models you can host, fine-tune versions included", _views.models_list,
             flags=(_flag("task", "Only models for this task", TASK), Flag("all", "Every model", YES))),
    _command("models get", "Show a model's task and limits", _views.models_get, NAME,
             flags=(_flag("task", "The task, for a family with several", TASK),)),
    _command("models predict", "Score rows with a model and context rows sent once by digest", _views.predicted,
             NAME, flags=(Flag("context", "The context rows", metavar="F"), Flag("queries", "The rows to score",
                                                                              metavar="F"),
                          *SPEC[2:], OUTPUT)),
    _command("fine-tunes snapshots create", "Upload a training snapshot", _views.snapshot_created, NAME, flags=(
        Flag("file", "The table", to="data", metavar="F"),
        _flag("tag", "A tag (repeatable)", metavar="T", to="tags", repeat=True),
        _flag("target", "The column a run learns to predict"),
        _flag("task", "classification or regression", TASK))),
    _command("fine-tunes snapshots synthesize", "Generate synthetic training tables", _views.batch_started, NAME,
             flags=(
                 _flag("count", "How many tables, 1 to 64 (default: 1)", WHOLE),
                 _flag("seed", "Table i is seeded SEED + i", WHOLE),
                 _flag("like", "Copy a snapshot's columns, types, task and classes", metavar="S"),
                 _flag("task", "classification or regression", TASK),
                 Flag("rows", "Rows per table, or a range", SPAN), Flag("columns", "Columns per table", SPAN),
                 Flag("classes", "Classes per table", SPAN), Flag("missing", "The share of cells missing", MISSING),
                 _flag("reject", "Drop tables that are trivial or unlearnable", NAMES, metavar="R,..."),
                 Flag("set", "Another generator setting, as graph_nodes=16 (repeatable)", SETTINGS, repeat=True),
                 _flag("tag", "A tag (repeatable)", metavar="T", to="tags", repeat=True))),
    _command("fine-tunes snapshots list", "List training snapshots", _views.snapshots_list,
             flags=(_flag("tag", "Only those with this tag", metavar="T"),)),
    _command("fine-tunes snapshots get", "Show a snapshot", _views.snapshots_get, NAME),
    _command("fine-tunes snapshots batch", "Show a synthetic batch", _views.batch_get, NAME, sdk="fine_tunes.snapshots.batch"),
    _command("fine-tunes snapshots tag", "Replace a snapshot's tags", _views.snapshot_tagged, NAME,
             Arg("T,...", "The tags", NAMES)),
    _command("fine-tunes snapshots delete", "Delete a snapshot; its name stays taken",
             _views.deleted("snapshot"), NAME, flags=(YES_FLAG,), ask=_ask("snapshot")),
    _command("fine-tunes runs create", "Start a run that publishes NAME's next version, NAME@ftN", run_created,
             NAME, run=run_create, reached=_views.run_started, settings={"plan": Plan, "evaluation": Evaluation},
             flags=(
                 _flag("base", "The model it trains from", metavar="M"),
                 _flag("data", "Snapshots, batches, contexts and tag:T", NAMES, metavar="S,..."),
                 _flag("target", "The column a context's rows train to predict", metavar="COLUMN"),
                 Flag("method", "lora:16, lora:16+adapter, adapter or full:COMPONENTS", METHOD),
                 _flag("steps", "Training steps", WHOLE, to="plan.steps"),
                 _flag("learning-rate", "The learning rate", NUMBER, to="plan.learning_rate"),
                 _flag("episode-rows", "Rows per episode", WHOLE, to="plan.episode_rows"),
                 Flag("holdout", "The share of rows held out to keep a step", FRACTION, to="evaluation.holdout"),
                 Flag("budget", "Stop and publish the best step by then", BUDGET),
                 _flag("seed", "The seed", WHOLE),
                 _flag("tag", "A tag (repeatable)", metavar="T", to="tags", repeat=True),
                 Flag("wait", "Wait for the run to end", YES),
                 Flag("from", "Take every setting from a Python file defining `run = gf.FineTuneSpec(...)`",
                      to=None, metavar="FILE"),
                 Flag("dry-run", "Print the resolved run instead of starting it", YES, to=None))),
    _command("fine-tunes runs list", "List fine-tuning runs", _views.runs_list, flags=(
        _flag("state", "Only runs in this state", metavar="S"), _flag("tag", "Only runs with this tag", metavar="T"))),
    _command("fine-tunes runs get", "Show a run", _views.runs_get, RUN),
    _command("fine-tunes runs wait", "Wait for a run to end", _views.run_ended_line, RUN,
             flags=(Flag("wait", "Wait at most this many seconds", SECONDS, to="timeout"),)),
    _command("fine-tunes runs watch", "Print a run's events as they happen, until it ends", _views.run_events, RUN,
             key="events"),
    _command("fine-tunes runs events", "Print every event of a run", _views.run_events, RUN, key="events"),
    _command("fine-tunes runs cancel", "Cancel a run", _views.run_cancelled, RUN,
             flags=(Flag("save", "Publish the best step it has kept so far", YES),)),
]  # fmt: skip

COMMANDS: dict[tuple[str, ...], Command] = {c.words: c for c in COMMANDS_LIST}

GROUPS: dict[tuple[str, ...], str] = {
    (
        "contexts",
    ): "The data models learn from, uploaded, imported from a source, or live",
    ("experiments",): "Models evaluated on one context for one target, run by run",
    ("predictors",): "Named, versioned endpoints serving a context",
    (
        "fine-tunes",
    ): "Training snapshots, synthetic batches, and runs that publish NAME@ftN",
    (
        "fine-tunes",
        "snapshots",
    ): "Snapshots and synthetic batches: named, immutable tables",
    (
        "fine-tunes",
        "runs",
    ): "Runs: each run of a name publishes its next version, NAME@ftN",
    ("sources",): "Workspace connections to warehouses and buckets",
    ("keys",): "API keys",
    ("models",): "The catalog, and the project's published fine-tune versions",
}
