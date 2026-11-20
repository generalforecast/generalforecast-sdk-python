# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The predictor commands: init, check, eval, predict, push and the hosted pointers.

`gf init` asks for what it cannot infer and writes `gf.yaml`. `gf check`
uploads the manifest's snapshot, has the service resolve the manifest and
writes the `gf.lock` it answers with. `gf eval` starts a hosted evaluation
of the manifest and follows it to its scores. `gf predict` scores rows with
a pushed predictor or a catalog model on the service. The other commands
use the hosted predictor routes. Each command returns its result for
`generalforecast.render` to write.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from generalforecast import term
from generalforecast.command_specs import DEFAULT_ENV
from generalforecast.errors import Error, InvalidInput, NotFound, during

if TYPE_CHECKING:
    from generalforecast.retry import PredictionWait

MANIFEST = "gf.yaml"
LOCK = "gf.lock"
TABLE_SUFFIXES = (".csv", ".parquet", ".json")
# A page of log entries when `--limit` names none, as the service counts it,
# and how often `gf admin logs --follow` asks for what is new.
LOG_PAGE = 100
LOG_POLL_SECONDS = 2.0
LOG_LEVELS = ("info", "warn", "error")
# How long ago `--since 15m` means: a count of seconds, minutes, hours or days.
_AGO = re.compile(r"([0-9]{1,6})([smhd])")
_AGO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _ago(value: str | None) -> str | float | None:
    """`--since`'s time: `15m` as the instant that long ago, in epoch seconds, else as given."""
    found = None if value is None else _AGO.fullmatch(value)
    if found is None:
        return value
    return time.time() - int(found[1]) * _AGO_SECONDS[found[2]]


def _compute() -> Any:
    """`pyarrow.compute`, imported by the first command that reads a table.

    It builds its kernels at import, so its stubs name none of them, and the
    import is most of what `gf init` waits for: it happens under a status
    line, not before one.
    """
    import pyarrow.compute as pc

    return pc


def _table(path: Path) -> Any:
    from generalforecast.commands import read_data

    return read_data(str(path), Path.cwd())


def _previous_lock(root: Path) -> dict[str, Any] | None:
    from generalforecast import manifest

    return manifest.read_lock(root)


def _short(digest: str | None) -> str:
    if not digest:
        return "no digest"
    return digest.split(":")[-1][:7]


def _shown_value(value: Any) -> str:
    if value is None:
        return "whole table"
    if type(value) is int:
        return term.count(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _shown_changes(
    changes: Sequence[tuple[str, str, Any, Any]], model: str
) -> list[tuple[str, str, Any, Any]]:
    """Lock changes as the CLI lists them.

    The library diffs dotted keys with raw values. The listing joins the model
    and its digest into one `model` line with short digests, every changed
    selection field into one `selection` line, counts in thousands, and
    orders the lines model, inputs, selection, then the rest.
    """
    shown: list[tuple[str, str, Any, Any]] = []
    pinned = {
        key: (old, new)
        for _, key, old, new in changes
        if key in ("model", "model_digest")
    }
    selection = [
        (key.removeprefix("selection."), old, new)
        for _, key, old, new in changes
        if key.startswith("selection.")
    ]
    for op, key, old, new in changes:
        if key in ("model", "model_digest"):
            if not pinned:
                continue
            old_digest, new_digest = pinned.get("model_digest", (None, None))
            if "model" in pinned:
                before, after = pinned["model"]
                line = (
                    f"{before} {_short(old_digest)}",
                    f"{after} {_short(new_digest)}",
                )
            else:
                line = (f"{model} {_short(old_digest)}", _short(new_digest))
            shown.append(("~", "model", *line))
            pinned = {}
        elif key.startswith("selection."):
            if not selection:
                continue
            parts = [
                f"{field} {_shown_value(old)} → {_shown_value(new)}"
                for field, old, new in selection
            ]
            shown.append(("~", "selection", None, ", ".join(parts)))
            selection = []
        elif key == "input":
            shown.append((op, key, old, new))
        else:
            shown.append(
                (
                    op,
                    key,
                    _shown_value(old) if old is not None else None,
                    _shown_value(new) if new is not None else None,
                )
            )
    order = {"model": 0, "input": 1, "selection": 2}
    return sorted(shown, key=lambda change: order.get(change[1], 3))


def _manifests() -> Any:
    """The predictor directory reader, `generalforecast.manifest`."""
    from generalforecast import manifest

    return manifest


def _shown(root: Path) -> str:
    text = os.path.relpath(root)
    if text == ".":
        return "."
    if text.startswith(".."):
        return str(root)
    return text if text.startswith((".", "/")) else f"./{text}"


def is_local(reference: str) -> bool:
    """Whether a DIR|NAME argument names a directory here.

    One rule for every command: a path, such as `.`, `./churn` or `~/churn`,
    names a directory; a bare name such as `churn` is the hosted predictor,
    even where a directory of that name holds a `gf.yaml`.
    """
    if re.match(r"^https?://", reference):
        return False
    return reference.startswith((".", "~")) or "/" in reference or os.sep in reference


def _shadowed(reference: str, out: term.Output) -> None:
    """Warns when a bare name taken as hosted is also a predictor directory here."""
    if not is_local(reference) and (Path(reference) / MANIFEST).is_file():
        out.warning(
            f"`{reference}` is the hosted predictor; pass ./{reference} for the "
            "directory here"
        )


def _tables_in(root: Path) -> list[Path]:
    return (
        sorted(p for p in root.iterdir() if p.is_file() and p.suffix in TABLE_SUFFIXES)
        if root.is_dir()
        else []
    )


def _output_path(
    text: str | None, *inputs: str | Path | None, predictor: Path | None = None
) -> Path | None:
    """The `--output` file, checked before any scoring.

    Its directory must exist, and it must be none of the files the command
    reads, which writing it would destroy: the queries and context it was
    given, and a predictor directory's manifest, lock and snapshot.
    """
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.parent.is_dir():
        raise NotFound(
            f"cannot write `{text}`: directory `{path.parent}` does not exist",
            param="output",
        )
    if path.is_dir():
        raise InvalidInput(f"cannot write `{text}`: it is a directory", param="output")
    read = list(inputs)
    if predictor is not None:
        read += [predictor / MANIFEST, predictor / LOCK]
        snapshot = _manifests().load(predictor).snapshot
        if snapshot is not None:
            read.append(predictor / snapshot)
    for given in read:
        if given is not None and Path(given).expanduser().resolve() == path.resolve():
            raise InvalidInput(
                f"--output `{text}` is also an input; it would be overwritten",
                hint="write the scored rows to another file",
                param="output",
            )
    return path


def _mode(path: Path) -> int:
    """The mode a written file takes: the one it replaces, else what the umask gives."""
    if path.exists():
        return path.stat().st_mode & 0o777
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _write_frame(frame: Any, path: Path) -> None:
    """Writes rows as their suffix says, atomically: a reader sees all or nothing."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix == ".json":
            from generalforecast.render import dumps

            # The encoder --json writes stdout with, so a file and stdout
            # carry the same text: ISO times and every float digit.
            Path(temporary).write_text(dumps(frame) + "\n", encoding="utf-8")
        else:
            frame.to_csv(temporary, index=False)
        os.chmod(temporary, _mode(path))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def execute(args: argparse.Namespace, out: term.Output) -> dict | None:
    """Runs one predictor command."""
    return getattr(PredictorCommands(args, out), args.command)()


class PredictorCommands:
    """The predictor commands, bound to one invocation's arguments and renderer."""

    def __init__(self, args: argparse.Namespace, out: term.Output) -> None:
        """Binds the arguments and the renderer."""
        self.args, self.out = args, out

    @cached_property
    def client(self) -> Any:
        """The authenticated SDK client, built on first use."""
        from generalforecast.commands import client_for

        return client_for(self.args)

    @cached_property
    def catalog(self) -> Any:
        """The hosted model catalog, read on first use."""
        from generalforecast.commands import model_registry

        return model_registry(self.args)

    @property
    def root(self) -> Path:
        """The predictor directory a DIR argument names."""
        return Path(self.args.directory).expanduser()

    def reference(self) -> tuple[str, Path | None]:
        """A DIR|NAME argument as `(hosted name, local root or None)`."""
        target = self.args.target
        _shadowed(target, self.out)
        if is_local(target):
            root = Path(target).expanduser()
            return _manifests().load(root).name, root
        return target, None

    def call(
        self,
        method: str,
        predictor: str,
        suffix: str = "",
        payload: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """One request on the predictor's hosted routes; `params` are a GET's query."""
        from generalforecast.service.contracts import name

        return self.client._call(
            method,
            f"/predictors/{name(predictor)}" + suffix,
            "predictor",
            predictor,
            payload,
            **({} if params is None else {"params": params}),
        )

    def confirm(self, question: str, *, declined: str) -> None:
        """Asks `question [y/N]` unless `--yes`.

        Anything but yes ends the command with `declined`, the line that
        says what did not happen.
        """
        term.Prompter(self.out, yes=self.args.yes).confirm(question, declined=declined)

    def init(self) -> dict:
        """Writes a predictor manifest, asking for what the context does not say."""
        args, out = self.args, self.out
        root = self.root
        manifest = root / MANIFEST
        if manifest.exists() and not args.force:
            raise Error(
                "InvalidInput",
                f"{MANIFEST} already exists in {_shown(root)}",
                hint="pass --force to overwrite it, or run `gf check` to validate it",
            )
        if args.context:
            source = Path(args.context).expanduser()
            if not source.is_file():
                raise Error(
                    "NotFound",
                    f"context file `{args.context}` does not exist",
                    hint="pass --context with a .csv, .parquet or .json file",
                )
            try:
                relative = source.resolve().relative_to(root.resolve())
            except ValueError:
                raise Error(
                    "InvalidInput",
                    f"context file `{args.context}` is outside {_shown(root)}",
                    hint=f"move it into {_shown(root)}; a predictor's snapshot lives beside its {MANIFEST}",
                ) from None
        else:
            tables = _tables_in(root)
            if len(tables) != 1:
                if not tables:
                    raise Error(
                        "NotFound",
                        f"no context table in {_shown(root)}",
                        hint="pass --context FILE, or put a .csv, .parquet or .json file here",
                    )
                raise Error(
                    "InvalidInput",
                    f"{_shown(root)} has {len(tables)} tables",
                    because="the context must be one file: "
                    + ", ".join(t.name for t in tables),
                    hint="pass --context FILE to pick one",
                )
            source = tables[0]
            relative = source.relative_to(root)
        # Arrow loads here, so the line is up before the wait starts.
        with out.working(log=False) as work:
            work.show("Reading", str(relative))
            from generalforecast import tables

            info = tables.inspect_table(source)
        out.status(
            "Found",
            f"{relative}  {term.plural(info.rows, 'row')}, {term.plural(len(info.columns), 'column')}",
        )
        out.say()
        prompter = term.Prompter(out, yes=args.yes, no_input=args.no_input)
        registry = self.catalog
        options = {c.name: term.Option(c.name, c.name, c.kind) for c in info.columns}

        target = args.target
        if target is not None and target not in options:
            raise Error(
                "InvalidInput",
                f"target column `{target}` is not in {relative}",
                hint=f"{relative} has {', '.join(info.names)}",
                param="target",
            )
        # A task the flags declare picks the target candidates; otherwise the
        # chosen target decides the task.
        model = args.model
        declared: tuple[str, str] | None = None
        if model is not None:
            model = registry.canonical(model)
            declared = registry.task_of(model), "from --model"
        elif args.task is not None:
            declared = args.task, "from --task"
        elif args.time or args.frequency or args.horizon:
            flag = next(
                f
                for f, v in (
                    ("--time", args.time),
                    ("--frequency", args.frequency),
                    ("--horizon", args.horizon),
                )
                if v
            )
            declared = "forecast", f"from {flag}"
        if target is None:
            names = tables.target_candidates(
                info, None if declared is None else declared[0]
            )
            target = prompter.choose(
                "Target", [options[c] for c in names], flag="--target"
            )
        else:
            out.collapsed("Target", target)

        column = info.column(target)
        if declared is not None:
            task, why = declared
        else:
            task = tables.task_for(column, info.rows)
            why = f"{target} has {term.count(column.distinct)} distinct {column.kind} values"
        out.collapsed("Task", task, why)
        forecast = task == "forecast"
        time_column, series, frequency, horizon = (
            args.time,
            args.series,
            args.frequency,
            args.horizon,
        )
        if forecast:
            if time_column is None:
                times = [options[c] for c in info.times] or [
                    options[c] for c in info.names if c != target
                ]
                time_column = prompter.choose("Time", times, flag="--time")
            else:
                out.collapsed("Time", time_column)
            if series is None:
                rest = [
                    options[c] for c in info.names if c not in (target, time_column)
                ]
                series = prompter.choose_many(
                    "Series",
                    rest,
                    flag="--series",
                    selected=[
                        i
                        for i, o in enumerate(rest)
                        if info.column(o.value).kind in ("text", "category")
                    ],
                    noun="column",
                )
            else:
                out.collapsed("Series", list(series) or "none", noun="column")
            if frequency is None:
                frequency = prompter.ask("Frequency", "1d", flag="--frequency")
            else:
                out.collapsed("Frequency", str(frequency))
            if horizon is None:
                horizon = int(
                    prompter.ask(
                        "Horizon", "7", flag="--horizon", check=_horizon_problem
                    )
                )
            else:
                out.collapsed("Horizon", str(horizon))

        if model is None:
            default = registry.default(task)
            listed = sorted(registry.catalog(task=task), key=lambda m: m.id != default)
            catalog = [term.Option(m.id, m.id, m.task) for m in listed]
            model = prompter.choose("Model", catalog, flag="--model")
        else:
            out.collapsed("Model", model)

        taken = {target, time_column, *(series or ())} - {None}
        identifier = args.id
        if identifier is None:
            ids = [options[c] for c in tables.id_candidates(info, taken)]
            identifier = prompter.choose(
                "Id", [*ids, term.Option(None, "none", "")], flag="--id"
            )
        else:
            if identifier not in options:
                raise Error(
                    "InvalidInput",
                    f"id column `{identifier}` is not in {relative}",
                    param="id",
                )
            out.collapsed("Id", identifier)
        taken.add(identifier)

        remaining = [options[c] for c in info.names if c not in taken]
        if args.inputs is not None:
            missing = [c for c in args.inputs if c not in options]
            if missing:
                raise Error(
                    "InvalidInput",
                    f"input column `{missing[0]}` is not in {relative}",
                    param="inputs",
                )
            inputs: list[str] | None = list(args.inputs)
            out.collapsed("Inputs", inputs or "none", noun="column")
        else:
            preselected = [
                i
                for i, o in enumerate(remaining)
                if o.value not in (args.exclude or ())
            ]
            chosen = prompter.choose_many(
                "Inputs",
                remaining,
                flag="--inputs",
                selected=preselected,
                noun="column",
            )
            inputs = (
                None
                if chosen == [o.value for o in remaining] and not args.exclude
                else chosen
            )

        empty = column.empty
        time_name = time_column or (info.times[0] if info.times else None)
        context: dict[str, Any] = {"snapshot": str(relative)}
        if args.until is not None:
            if time_name is None:
                raise Error(
                    "InvalidInput",
                    "--until needs a time column",
                    because=f"{relative} has no timestamp column to compare dates against",
                    hint="pass --holdout FRACTION for a random split",
                    param="until",
                )
            context["until"] = args.until
            out.collapsed("Queries", f"rows at or after {args.until}")
        elif args.holdout is not None:
            context["holdout"] = {"fraction": args.holdout, "seed": 0}
            out.collapsed("Queries", f"a random {args.holdout:g} of the rows")
        elif args.queries == "none":
            out.collapsed("Queries", "none")
        else:
            offered = {
                "empty": term.Option(
                    "empty",
                    "rows with no target",
                    term.plural(empty, "row"),
                    f"{term.count(empty)} rows with no target",
                ),
                "date": term.Option("date", "rows after a date", ""),
                "fraction": term.Option("fraction", "a random fraction", ""),
                "none": term.Option("none", "none", ""),
            }
            rules = [
                offered[rule] for rule in tables.query_rules(info, target, time_name)
            ]
            rule = prompter.choose(
                "Queries", rules, flag="--until, --holdout or --queries none"
            )
            if rule == "date":
                assert time_name is not None
                table = _table(source)
                last, rows_after = _dates(table.column(time_name))
                if forecast:
                    # Each series holds out exactly its horizon, as eval scores it.
                    assert horizon is not None
                    pc = _compute()
                    last = (
                        tables.default_until(
                            table.filter(pc.is_valid(table.column(target))),
                            time=time_name,
                            series=list(series or ()),
                            horizon=horizon,
                            frequency=None if frequency is None else str(frequency),
                        )
                        or last
                    )
                else:
                    # About the share a random holdout takes, never the last day's rows alone.
                    last = (
                        _split_day(
                            table,
                            time_name,
                            target,
                            tables.default_holdout(
                                info.rows, registry.limits(model).queries
                            ),
                        )
                        or last
                    )
                until = prompter.ask(
                    "Until",
                    last,
                    flag="--until",
                    note=lambda v: term.plural(rows_after(v), "row"),
                )
                context["until"] = until
            elif rule == "fraction":
                cap = registry.limits(model).queries
                fraction = prompter.ask(
                    "Holdout",
                    f"{tables.default_holdout(info.rows, cap):g}",
                    flag="--holdout",
                    check=_fraction_problem,
                    note=lambda v: _holdout_note(float(v), info.rows, model, cap),
                )
                context["holdout"] = {"fraction": float(fraction), "seed": 0}

        name = (
            root.resolve().name
            if root.resolve().name
            and re.fullmatch(r"[a-z][a-z0-9-]{0,62}", root.resolve().name)
            else "predictor"
        )
        spec: dict[str, Any] = {"name": name, "model": model, "target": target}
        if identifier:
            spec["id"] = identifier
        if forecast:
            spec.update(
                time=time_column,
                series=list(series or ()),
                frequency=_number(frequency),
                horizon=horizon,
            )
        elif "until" in context:
            spec["time"] = time_name
        if inputs is not None:
            spec["inputs"] = inputs
        elif args.exclude:
            spec["inputs"] = {"exclude": list(args.exclude)}
        spec["context"] = context
        path = _manifests().write(spec, root, force=args.force)
        out.say()
        out.status("Initialized", f"predictor {name} in {_shown(root)}/{MANIFEST}")
        return {"name": name, "path": str(path), "manifest": spec}

    def check(self) -> dict:
        """Has the service resolve `gf.yaml`, prints what it inferred and writes `gf.lock`."""
        args, out = self.args, self.out
        started = time.perf_counter()
        root = self.root
        found = _manifests().load(root)
        previous = _previous_lock(root)
        if (args.frozen or args.locked) and previous is None:
            raise Error(
                "InvalidInput",
                f"no {LOCK} in {_shown(root)}",
                hint="run `gf check` to write it",
            )
        # The snapshot uploads, then the service resolves the manifest
        # against it; a terminal says so until `Resolved` replaces it.
        with out.working(log=False) as work:
            work.show("Checking", found.name)
            with during(f"the check of predictor {found.name}"):
                answer = self.client.predictors.check(
                    root, upgrade=bool(getattr(args, "upgrade", False))
                )
        lock = answer["lock"]
        current = previous is not None and previous.get("digest") == lock.get("digest")
        if args.frozen:
            if not current:
                raise Error(
                    "InvalidInput",
                    f"{LOCK} is not current",
                    hint="run `gf check` to write it again",
                )
            out.status(
                "Verified",
                f"{LOCK} in {_shown(root)}",
                seconds=time.perf_counter() - started,
            )
            return {"name": found.name, "frozen": True}
        predictor = lock.get("predictor") or {}
        model = str(answer.get("model") or predictor.get("model") or found.model)
        inferred = answer.get("inferred") or {}
        out.status(
            "Resolved",
            f"{found.name}  {predictor.get('task') or inferred.get('task') or 'predictor'} "
            f"on {model}",
        )
        out.details(_inferred_rows(inferred))
        changes = _shown_changes(_changes(answer.get("changes")), model)
        out.changes(changes)
        if args.locked and (changes or not current):
            listed = f"{' is' if len(changes) == 1 else 's are'} listed above"
            raise Error(
                "InvalidInput",
                f"{LOCK} would change",
                # Under -q and --json nothing was listed, so the error lists them.
                because="\n".join(line.strip() for line in term.change_lines(changes))
                if out.quiet or not changes
                else f"{len(changes)} change{listed}",
                hint="run `gf check` without --locked to write them",
            )
        if args.locked:
            out.status("Verified", LOCK, seconds=time.perf_counter() - started)
        else:
            _manifests().write_lock(root, lock)
            out.status("Locked", LOCK, seconds=time.perf_counter() - started)
        return {
            "name": found.name,
            "model": model,
            "inferred": inferred,
            "changes": [list(change) for change in changes],
            "context": answer.get("context"),
            "lock": lock,
        }

    def eval(self) -> dict:
        """Starts a hosted evaluation of the manifest and follows it to its scores."""
        from generalforecast._operations import EVALUATION_RUNNING
        from generalforecast.command_specs import EVAL_POLL_SECONDS

        args, out = self.args, self.out
        root = self.root
        found = _manifests().load(root)
        if args.wait is not None and args.wait <= 0:
            raise InvalidInput("--wait must be more than 0", param="wait")
        if found.context_kind != "snapshot" or not (
            "holdout" in found.context or "until" in found.context
        ):
            raise InvalidInput(
                f"gf eval scores held-out rows, and {MANIFEST} holds none out",
                because="an evaluation needs a snapshot context with `holdout` or `until`",
                hint=f"add `holdout` or `until` to the context in {MANIFEST}",
                param="context",
            )
        models = _evaluated(found, _previous_lock(root), list(args.model))
        started = time.perf_counter()
        with out.working(log=False) as work:
            work.show("Uploading", found.name)
            with during(f"the evaluation of predictor {found.name}"), _eval_refusal():
                state = self.client.predictors.evaluate(root, models=models, wait=False)
            evaluation = state["evaluation"]
            deadline = None if args.wait is None else time.monotonic() + args.wait
            while True:
                line = _evaluation_line(state)
                work.show("Evaluating", lambda seconds, line=line: line)
                if state["state"] not in EVALUATION_RUNNING:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise Error(
                        "Timeout",
                        f"evaluation {evaluation} is still {state['state']}",
                        hint="pass a longer --wait, or run `gf eval` again later",
                    )
                time.sleep(EVAL_POLL_SECONDS)
                with during(f"the request for evaluation {evaluation}"):
                    state = self.client.predictors.evaluation(found.name, evaluation)
        scored = [m for m in state.get("models") or [] if isinstance(m, dict)]
        failed = [m for m in scored if m.get("error") or m.get("state") == "failed"]
        if state["state"] != "finished":
            raise Error(
                "Internal",
                f"evaluation {evaluation} {state['state']}",
                because="; ".join(
                    f"{m.get('model')}: {_error_message(m.get('error'))}"
                    for m in failed
                )
                or None,
                hint="fix the manifest and run `gf eval` again",
            )
        for model in failed:
            out.warning(
                f"{model.get('model')} failed: {_error_message(model.get('error'))}"
            )
        held = state.get("held")
        out.status(
            "Evaluated",
            f"{found.name}  "
            + (
                term.plural(held, "held-out row")
                if type(held) is int
                else "held-out rows"
            ),
            seconds=time.perf_counter() - started,
        )
        return {
            "name": found.name,
            "evaluation": evaluation,
            "task": state.get("task"),
            "metric": state.get("metric"),
            "rows": held,
            "repeats": state.get("repeats"),
            "context": state.get("context"),
            "metrics": {str(m.get("model")): m.get("metrics") or {} for m in scored},
            "models": scored,
            "billed": state.get("billed"),
        }

    def predict(self) -> dict:
        """Scores rows with a hosted predictor or, with `--model`, an ad hoc predictor."""
        from generalforecast.commands import client_for
        from generalforecast.predictor import Predictor

        args, out = self.args, self.out
        reference = args.target
        if args.model is not None:
            for flag, value, why in (
                ("NAME", reference, "give a predictor NAME or --model, not both"),
                ("--env", args.env, "--env names a hosted predictor's environment"),
            ):
                if value is not None:
                    raise InvalidInput(why, param=flag.lstrip("-").lower())
            if not args.target_column:
                raise InvalidInput(
                    "--model needs a target",
                    hint="pass '--target C' and optionally '--inputs C,C'",
                    param="target",
                )
            if args.context is None:
                raise InvalidInput(
                    "--model needs context rows",
                    hint="pass '--context FILE' with rows whose target is known",
                    param="context",
                )
        else:
            if reference is None:
                raise InvalidInput(
                    "name a hosted predictor, or pass --model",
                    hint="run `gf predict NAME --queries FILE`, or `gf predict "
                    "--model ID --target C --context FILE --queries FILE`",
                    param="target",
                )
            _refuse_directory(reference)
            _shadowed(reference, out)
            given = [
                flag
                for flag, value in (
                    ("--target", args.target_column),
                    ("--inputs", args.inputs),
                )
                if value is not None
            ]
            if given:
                raise InvalidInput(
                    f"{given[0]} applies to --model; a hosted predictor has its own",
                    param=given[0].lstrip("-"),
                )
        output = _output_path(args.output, args.queries, args.context)
        if not 0 <= args.wait_seconds <= 86400:
            raise InvalidInput(
                "--wait-seconds must be from 0 to 86400", param="wait-seconds"
            )
        client = client_for(args, wait_seconds=args.wait_seconds, on_wait=self.waiting)
        predictor = Predictor(
            args.model or reference,
            env=None if args.model else (args.env or DEFAULT_ENV),
            target=args.target_column,
            inputs=args.inputs,
            client=client,
        )
        started = time.perf_counter()
        with out.working() as work:
            work.show(
                "Predicting", predictor.name if args.model is None else args.model
            )
            with during("the prediction"):
                frame = predictor.predict(args.queries, context=args.context)
        out.status(
            "Predicted",
            term.plural(len(frame), "row"),
            seconds=time.perf_counter() - started,
        )
        if output is None:
            return {"rows": len(frame), "predictions": frame}
        _write_frame(frame, output)
        out.status("Wrote", args.output)
        return {"rows": len(frame), "output": str(output)}

    def waiting(self, wait: PredictionWait) -> None:
        """Shows a hosted prediction the service accepted and is still working on, as one updating line."""
        what = "to start" if wait.state == "starting" else "to answer"
        self.out.progress(
            "Waiting",
            f"for hosted inference {what}  {term.elapsed(wait.elapsed)}",
        )

    def push(self) -> dict:
        """Uploads the locked predictor as a revision and points an environment at it."""
        from generalforecast._operations import registered
        from generalforecast.contracts import Environment, PromoteRequest, Revision
        from generalforecast.service.contracts import ServiceError
        from generalforecast.service.sdk import pushed

        args, out = self.args, self.out
        root = self.root
        spec = _manifests().load(root)
        if _previous_lock(root) is None:
            raise Error(
                "InvalidInput",
                f"no current {LOCK} in {_shown(root)}",
                hint="run `gf check` first",
            )
        checked = pushed(root)
        predictor = checked.lock.to_dict()["predictor"]
        name, env = spec.name, args.env
        try:
            with during(f"the request for predictor {name}"):
                self.call("GET", name)
            exists = True
        except ServiceError as error:
            if error.status != 404:
                raise
            exists = False
        current: Environment | None = None
        if exists:
            try:
                with during(f"the request for environment {env} of {name}"):
                    current = Environment.from_dict(
                        self.call("GET", name, f"/environments/{env}")
                    )
            except ServiceError as error:
                if error.status != 404:
                    raise
        number = (current.revision or 0) + 1 if current else 1
        context = predictor.get("context") or {}
        resources = spec.document.get("resources")
        resources = resources if isinstance(resources, dict) else {}
        replicas = resources.get("replicas")
        replicas = replicas if isinstance(replicas, dict) else {}
        accelerator = resources.get("accelerator") or "L4"
        hardware = (
            f"{accelerator}, "
            f"{replicas.get('min') or 0}-{replicas.get('max') or 1} replicas, "
            f"idle {resources.get('idle_seconds') or 300}s"
        )
        plan = [
            ("name", name),
            ("revision", str(number)),
            ("environment", env),
            (
                "model",
                f"{predictor.get('model')}@{_short(predictor.get('model_digest', ''))}",
            ),
            (
                "context",
                f"{context.get('kind')}, {term.count(context.get('rows', 0))} rows"
                + (
                    " and those gf eval holds out"
                    if context.get("holdout") or context.get("until")
                    else ""
                )
                + (f", {_short(context['digest'])}" if context.get("digest") else ""),
            ),
            ("hardware", hardware),
            ("first call", _cost_line(accelerator)),
        ]
        # The plan is what a confirmation answers, so it shows whenever one is asked.
        out.table(plan, indent=2, force=not args.yes)
        self.confirm("Push?", declined="Nothing was pushed.")
        rows = checked.checked.snapshot
        if rows is not None:
            source = _snapshot_source(spec.document)
            out.status(
                "Uploading",
                f"{name} revision {number}  {source}, "
                f"{term.count(rows.num_rows)} rows, {_bytes(rows.nbytes)}",
            )
        else:
            out.status("Uploading", f"{name} revision {number}  {context.get('kind')}")
        # The service checks the directory again and stores a snapshot's
        # every labeled row, those gf eval held out included, which the
        # revision then names by their digest.
        with during(f"the check of predictor {name}"):
            answer = self.client.predictors.check(root, deploy=True)
        payload = registered(answer, checked.lock)
        deployed = answer.get("deployed")
        if isinstance(deployed, dict):
            out.status("Context", _deployed_rows(deployed))
        started = time.perf_counter()
        if not exists:
            with during(f"the registration of predictor {name}"):
                self.client._call(
                    "POST", "/predictors", "predictor", name, {"name": name}
                )
        with during(f"revision {number} of {name}"):
            revision = Revision.from_dict(
                self.call("POST", name, "/revisions", payload)
            )
        if current is None:
            with during(f"the new environment {env} of {name}"):
                current = Environment.from_dict(
                    self.call(
                        "POST",
                        name,
                        "/environments",
                        {"name": env},
                    )
                )
        request = PromoteRequest(revision.number, current.generation)
        with during(f"the promotion of {name} revision {revision.number} to {env}"):
            promoted = Environment.from_dict(
                self.call(
                    "POST", name, f"/environments/{env}/promote", request.to_dict()
                )
            )
        ready = self.wait(promoted)
        out.status(
            "Deployed",
            f"{name} revision {revision.number} to {env}",
            seconds=time.perf_counter() - started,
        )
        return {
            "name": name,
            "revision": revision.to_dict(),
            "environment": ready.to_dict(),
        }

    def wait(self, environment: Any) -> Any:
        """Polls the environment until it is ready, for at most `--wait` seconds.

        `--timeout` bounds each request; `--wait` bounds the whole start, which
        a cold start of a GPU replica can take most of a minute or more.
        """
        from generalforecast.contracts import Environment

        deadline = time.monotonic() + self.args.wait
        current = environment
        while current.status != "ready":
            if current.status in ("failed", "stopped"):
                raise Error(
                    "Internal",
                    f"environment {environment.name} failed to start",
                    hint=f"run `gf admin logs {environment.predictor} --env {environment.name}`",
                )
            if time.monotonic() >= deadline:
                raise Error(
                    "Internal",
                    f"environment {environment.name} is not ready yet",
                    hint=f"run `gf status {environment.predictor}` to follow it, or pass a longer '--wait'",
                )
            time.sleep(0.25)
            with during(f"the request for the status of {environment.name}"):
                current = Environment.from_dict(
                    self.call(
                        "GET",
                        environment.predictor,
                        f"/environments/{environment.name}",
                    )
                )
        return current

    def status(self) -> dict:
        """The predictor's environments."""
        name, _ = self.reference()
        with during(f"the request for predictor {name}"):
            predictor = self.call("GET", name)
        with during(f"the request for the environments of {name}"):
            environments = self.call("GET", name, "/environments")
        return {
            "name": name,
            "predictor": predictor,
            "environments": environments.get("environments", environments),
        }

    def promote(self) -> dict:
        """Points `--to` at the revision `--from` serves."""
        from generalforecast.contracts import Environment

        name, _ = self.reference()
        with during(f"the request for environment {self.args.source} of {name}"):
            source = Environment.from_dict(
                self.call("GET", name, f"/environments/{self.args.source}")
            )
        if source.revision is None:
            raise Error(
                "InvalidInput",
                f"{self.args.source} serves no revision of {name}",
                hint=f"run `gf push --env {self.args.source}` first",
            )
        return self._point(name, self.args.env, source.revision, "Promoted")

    def rollback(self) -> dict:
        """Points `--env` at an earlier revision."""
        from generalforecast.contracts import Environment

        name, _ = self.reference()
        with during(f"the request for environment {self.args.env} of {name}"):
            current = Environment.from_dict(
                self.call("GET", name, f"/environments/{self.args.env}")
            )
        revision = (
            self.args.revision
            if self.args.revision is not None
            else (current.revision or 1) - 1
        )
        if revision < 1:
            raise Error(
                "InvalidInput",
                f"{self.args.env} serves revision {current.revision} of {name}; there is nothing earlier",
                hint="pass '--revision N' to pick one",
            )
        return self._point(name, self.args.env, revision, "Pointed")

    def _point(self, name: str, env: str, revision: int, verb: str) -> dict:
        from generalforecast.contracts import Environment, PromoteRequest
        from generalforecast.service.contracts import ServiceError

        started = time.perf_counter()
        try:
            with during(f"the request for environment {env} of {name}"):
                current = Environment.from_dict(
                    self.call("GET", name, f"/environments/{env}")
                )
        except ServiceError as error:
            if error.status != 404:
                raise
            with during(f"the new environment {env} of {name}"):
                current = Environment.from_dict(
                    self.call(
                        "POST",
                        name,
                        "/environments",
                        {"name": env},
                    )
                )
        request = PromoteRequest(revision, current.generation)
        moved = "promotion" if verb == "Promoted" else "rollback"
        with during(f"the {moved} of {env} to {name} revision {revision}"):
            promoted = Environment.from_dict(
                self.call(
                    "POST", name, f"/environments/{env}/promote", request.to_dict()
                )
            )
        ready = self.wait(promoted)
        self.out.status(
            verb,
            f"{env} at {name} revision {revision}"
            if verb == "Pointed"
            else f"{name} revision {revision} to {env}",
            seconds=time.perf_counter() - started,
        )
        return {"name": name, "environment": ready.to_dict()}

    def logs(self) -> dict | None:
        """An environment's log: its newest entries, or with `--follow` each as it is written.

        `--level`, `--since`, `--until` and `--grep` select entries on the
        service, and `--limit` how many of the newest are read. Following
        reads that page, then asks every `LOG_POLL_SECONDS` for what was
        written after it and prints each entry as it comes, a JSON object a
        line under `--json`, until Ctrl-C.
        """
        from generalforecast._operations import log_selection
        from generalforecast.render import log_page

        name, _ = self.reference()
        args = self.args
        if args.limit is not None and not 1 <= args.limit <= 500:
            raise InvalidInput("--limit is a number from 1 to 500", param="limit")
        if args.level not in (None, *LOG_LEVELS):
            raise InvalidInput(
                "--level is " + ", ".join(LOG_LEVELS[:-1]) + f" or {LOG_LEVELS[-1]}",
                param="level",
            )
        try:
            selected = log_selection(
                limit=args.limit,
                level=args.level,
                q=args.grep,
                since=_ago(args.since),
                until=_ago(args.until),
            )
        except InvalidInput as error:
            if error.param not in ("since", "until"):
                raise
            raise InvalidInput(
                f"--{error.param} is a time, such as 2026-10-27T08:00:00Z, "
                "or how long ago, such as 15m, 2h or 1d",
                param=error.param,
            ) from None
        suffix = f"/environments/{args.env}/logs"

        def read(after: str | None = None) -> dict:
            params = {k: v for k, v in selected.items() if v is not None}
            if after is not None:
                params["after"] = after
            with during(f"the request for the logs of {name}"):
                return self.call("GET", name, suffix, params=params or None)

        page = read()
        if not args.follow:
            return page
        log_page(args, page, self.out)
        while True:
            # A full page may have more behind it; an emptier one is the end for now.
            if len(page["logs"]) < (args.limit or LOG_PAGE):
                time.sleep(LOG_POLL_SECONDS)
            page = read(page["next"])
            log_page(args, page, self.out)

    def diff(self) -> dict:
        """The lock changes between this directory and what an environment serves."""
        from generalforecast.contracts import Environment

        name, root = self.reference()
        if root is None:
            raise Error(
                "InvalidInput",
                "diff needs a predictor directory",
                hint="run it inside the predictor, or pass its path",
            )
        local = _previous_lock(root)
        if local is None:
            raise Error(
                "InvalidInput",
                f"no {LOCK} in {_shown(root)}",
                hint="run `gf check` first",
            )
        with during(f"the request for environment {self.args.env} of {name}"):
            environment = Environment.from_dict(
                self.call("GET", name, f"/environments/{self.args.env}")
            )
        if environment.revision is None:
            raise Error("NotFound", f"{self.args.env} serves no revision of {name}")
        with during(f"the request for revision {environment.revision} of {name}"):
            remote = self.call("GET", name, f"/revisions/{environment.revision}")
        changes = _lock_changes(remote.get("lock") or {}, local)
        if changes:
            self.out.changes(changes)
        else:
            self.out.status(
                "Unchanged",
                f"{name} matches {self.args.env} revision {environment.revision}",
            )
        return {
            "name": name,
            "revision": environment.revision,
            "changes": [list(c) for c in changes],
        }

    def delete(self) -> dict:
        """Deletes a hosted predictor and every environment of it."""
        name, _ = self.reference()
        self.confirm(
            f"Delete {name} and its environments?", declined="Nothing was deleted."
        )
        with during(f"the deletion of {name}"):
            result = self.call("POST", name, "/delete", {})
        self.out.status("Deleted", f"predictor {name}")
        return result if isinstance(result, dict) else {"name": name, "deleted": True}


def _changes(value: Any) -> list[tuple[str, str, Any, Any]]:
    """The service's lock changes as `(op, key, old, new)`: from lists or objects."""
    changes: list[tuple[str, str, Any, Any]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            changes.append(
                (
                    str(item.get("op", "~")),
                    str(item.get("key", "")),
                    item.get("old"),
                    item.get("new"),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            changes.append((str(item[0]), str(item[1]), item[2], item[3]))
    return changes


def _lock_changes(old: dict, new: dict) -> list[tuple[str, str, Any, Any]]:
    """What differs between two locks' `predictor` sections, key by key."""
    before = old.get("predictor") if isinstance(old.get("predictor"), dict) else {}
    after = new.get("predictor") if isinstance(new.get("predictor"), dict) else {}
    changes: list[tuple[str, str, Any, Any]] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) == after.get(key):
            continue
        op = "+" if key not in before else "-" if key not in after else "~"
        changes.append((op, key, before.get(key), after.get(key)))
    return changes


def _inferred_rows(inferred: Any) -> list[tuple[str, str | list[str], str]]:
    """What the service inferred, as the role table lists it: key, value and a note."""
    rows: list[tuple[str, str | list[str], str]] = []
    for key, value in inferred.items() if isinstance(inferred, dict) else ():
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            rows.append((str(key), list(value) or "none", "inferred"))
        elif isinstance(value, dict):
            rows.append(
                (str(key), ", ".join(f"{k} {v}" for k, v in value.items()), "inferred")
            )
        else:
            rows.append((str(key), str(value), "inferred"))
    return rows


def _evaluation_line(state: dict) -> str:
    """Where an evaluation has got to: each model's rows scored and its score so far."""
    metric = state.get("metric")
    parts = []
    for model in state.get("models") or []:
        if not isinstance(model, dict):
            continue
        name = str(model.get("model", "")).split("/", 1)[-1]
        scored, total = model.get("scored"), model.get("total")
        part = name
        if type(scored) is int and type(total) is int:
            part += f" {term.count(scored)} of {term.count(total)}"
        metrics = model.get("metrics")
        if metric and isinstance(metrics, dict) and metric in metrics:
            value = metrics[metric]
            part += f" {metric} {value:.4g}" if isinstance(value, float) else ""
        parts.append(part)
    return f"{state.get('state', '')}  " + ", ".join(parts)


def _evaluated(
    found: Any, previous: dict[str, Any] | None, extra: list[str]
) -> list[str] | None:
    """The models `gf eval --model` scores: the manifest's own, then each named one.

    None without `--model`, when the service scores the manifest's model.
    The manifest's model is the one it names, or else the one its lock pins.
    """
    if not extra:
        return None
    own = found.model
    if own is None and isinstance(previous, dict):
        locked = previous.get("predictor")
        own = locked.get("model") if isinstance(locked, dict) else None
    return list(dict.fromkeys(([own] if isinstance(own, str) else []) + extra))


def _error_message(error: Any) -> str:
    """A model's failure as the evaluation reports it: an error object or text."""
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "failed")
    return str(error or "failed")


@contextlib.contextmanager
def _eval_refusal() -> Iterator[None]:
    """Says what to do about the refusals only an evaluation meets."""
    from generalforecast.account import wire_code

    try:
        yield
    except Error as error:
        if error.hint is None:
            code, reason = wire_code(error)
            if reason == "run_limit_reached":
                error.hint = (
                    "the project runs one evaluation or Playground run at a time; "
                    "run `gf eval` again once the other ends"
                )
            elif reason == "model_not_hostable":
                error.hint = "run `gf model list` for the models the service hosts"
            elif code == "ValidationError" and error.param in ("context", "holdout"):
                error.hint = (
                    "gf eval scores a snapshot context's labelled held-out rows: "
                    f"give the context in {MANIFEST} a `holdout` or `until`"
                )
            elif code == "ValidationError" and error.param == "model":
                error.hint = "gf eval scores classification and regression predictors"
        raise


def _refuse_directory(reference: str) -> None:
    """Refuses a predictor directory where a hosted name is wanted."""
    if is_local(reference):
        named = ""
        try:
            named = _manifests().load(Path(reference)).name
        except Error:
            pass
        if "/" in reference and not Path(reference).expanduser().exists():
            raise InvalidInput(
                f"`{reference}` is not a predictor name",
                hint=f"to predict with a catalog model, pass --model {reference}",
                param="target",
            )
        raise InvalidInput(
            f"`{reference}` is a predictor directory; predictions run on the "
            "hosted service",
            hint=f"push it with `gf push {reference}`, then run "
            f"`gf predict {named or 'NAME'} --queries FILE`",
            param="target",
        )


def _holdout_note(fraction: float, rows: int, model: str, cap: int) -> str:
    """How many rows a typed holdout fraction takes, and when that is one call's worth."""
    from generalforecast import tables

    held = round(fraction * rows)
    note = term.plural(held, "row")
    if held == cap and round(tables.HOLDOUT_FRACTION * rows) > cap:
        note += f", the most {model} scores in one call"
    return note


def _fraction_problem(text: str) -> str | None:
    """What is wrong with a typed holdout fraction, or None."""
    try:
        value = float(text)
    except ValueError:
        return "must be a number between 0 and 1, like 0.2"
    return None if 0 < value < 1 else "must be between 0 and 1, like 0.2"


def _horizon_problem(text: str) -> str | None:
    """What is wrong with a typed horizon, or None."""
    return (
        None
        if text.isascii() and text.isdigit() and int(text) >= 1
        else "must be a whole number of steps"
    )


def _dates(column: Any) -> tuple[str, Callable[[str], int]]:
    """A time column's last date, and a count of its rows at or after a date.

    The column is read once; the count runs on every keystroke of the prompt.
    """
    pc = _compute()
    last = pc.max(column).as_py()
    shown = (
        str(last)[:10]
        if last is not None
        else datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    )
    text = column.cast("string")

    def after(value: str) -> int:
        return int(pc.sum(pc.greater_equal(text, value)).as_py() or 0)

    return shown, after


def _split_day(table: Any, time: str, target: str, fraction: float) -> str | None:
    """The day the labelled rows' latest `fraction` starts on, as an ISO date; None without dates."""
    pc = _compute()
    import pyarrow as pa

    try:
        stamps = pc.cast(table.column(time), pa.timestamp("us"))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return None
    labelled = pc.filter(
        stamps, pc.and_(pc.is_valid(stamps), pc.is_valid(table.column(target)))
    )
    days = sorted(stamp.date() for stamp in labelled.to_pylist())
    if not days:
        return None
    return days[min(len(days) - 1, int(len(days) * (1 - fraction)))].isoformat()


def _number(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    return value


def _snapshot_source(document: Any) -> str:
    """The file a manifest's snapshot context names, as gf.yaml writes it."""
    context = document.get("context") if isinstance(document, dict) else None
    source = context.get("snapshot") if isinstance(context, dict) else None
    return str(source) if isinstance(source, str) else "snapshot"


def _deployed_rows(deployed: dict) -> str:
    """What a push deploys: every labeled row, and how many of them gf eval held out."""
    rows = term.count(deployed["context"]["rows"])
    held = deployed["held_out"]
    if held:
        return f"{rows} rows, including the {term.count(held)} gf eval held out"
    return f"{rows} rows"


def _bytes(size: float) -> str:
    """A size in binary units, as the service and its console write one: `27.2 KiB`."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GiB"


def _cost_line(accelerator: str) -> str:
    rates = {"cpu": 0.0002, "L4": 0.0004, "A10G": 0.0006, "L40S": 0.0010}
    starts = {"cpu": 15, "L4": 40, "A10G": 40, "L40S": 45}
    return f"about ${rates.get(accelerator, 0.0004):.4f} after a cold start of about {starts.get(accelerator, 40)}s"
