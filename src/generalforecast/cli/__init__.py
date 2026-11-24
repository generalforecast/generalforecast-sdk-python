# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The `gf` command: the Python API from a shell.

`gf <resource> <verb> [NAME] [flags]` calls `gf.<resource>.<verb>(NAME,
**flags)`; the commands are one table (`generalforecast.cli._commands`), and
this module runs any of them the same way: it reads the command line, makes
the client, asks before a delete, calls the SDK, and shows the result, or
under `--json` prints it as one JSON object (one per line for a command
that streams, such as `gf runs watch`). Status lines go to
stderr and results to stdout.

Every failure goes through `report`, written as cargo writes one: `error:`,
`note:`, `help:`. One the person can fix is bold; a real failure (the
service failed, or a run or a deploy did) is red. A hint the SDK words for
Python, such as `gf.experiments.wait('x')`, reads as the command,
`gf experiments wait x`. The exit code says how the command ended:

    0    it did what was asked, including when there was nothing to change
    1    it failed; the error says why
    2    the command line was refused, or a question was declined or could not be asked
    130  Ctrl-C interrupted it
    141  the reader of stdout closed the pipe
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shlex
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from generalforecast import term
from generalforecast.cli._call import Call
from generalforecast.cli._spec import (
    SECONDS,
    TEXT,
    YES,
    Command,
    Flag,
    Parser,
    Usage,
    switch,
)
from generalforecast.errors import Error

if TYPE_CHECKING:
    from generalforecast._ops import Say
    from generalforecast.client import Client

__all__ = ["main"]

DEFAULT_TIMEOUT = 40.0
USAGE_EXIT = 2
INTERRUPTED_EXIT = 130
BROKEN_PIPE_EXIT = 141

GLOBALS = (
    Flag("json", "Print the result, or the error, as one JSON object on stdout", YES, to=None),
    Flag("quiet", "Print only results and errors", YES, to=None, short="-q"),
    Flag("verbose", "Print every column and name, and request ids", YES, to=None, short="-v"),
    Flag("timeout", f"Seconds to wait for each request (default: {DEFAULT_TIMEOUT:g})", SECONDS, to=None),
    Flag("project", "The project to use (default: GF_PROJECT, else the login's)", TEXT, to=None, metavar="NAME"),
    Flag("help", "Print help", switch(), to=None, short="-h"),
)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Runs one `gf` command line; returns its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    early = {token for token in argv[: argv.index("--") if "--" in argv else None]}
    out = term.Output(
        quiet=bool(early & {"-q", "--quiet"}),
        verbose=bool(early & {"-v", "--verbose"}),
    )
    as_json = "--json" in early
    try:
        return _run(argv, out)
    except term.Declined as declined:
        report(out, declined, as_json=as_json)
        return USAGE_EXIT
    except (Usage, term.Aborted) as error:
        report(out, error, as_json=as_json)
        return USAGE_EXIT
    except Error as error:
        report(out, error, as_json=as_json)
        return 1
    except KeyboardInterrupt:
        out.end()
        return INTERRUPTED_EXIT
    except BrokenPipeError:
        # Python flushes stdout once more at exit; point it somewhere that takes it.
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return BROKEN_PIPE_EXIT


def _run(argv: list[str], out: term.Output) -> int:
    from generalforecast.cli import _help
    from generalforecast.cli._commands import COMMANDS, GROUPS

    words, rest = _split(argv, COMMANDS, GROUPS)
    if words[:1] == ("help",):
        words, rest = _split([*words[1:], *rest], COMMANDS, GROUPS)
        rest = [*rest, "--help"]
    command = COMMANDS.get(words)
    if command is None:
        if not words and {"-V", "--version"} & set(rest):
            from generalforecast import __version__

            out.line(f"gf {__version__}")
            return 0
        given = [token for token in rest if not token.startswith("-")]
        if given:
            raise _unknown(words, given[0], COMMANDS, GROUPS)
        if {"-h", "--help"} & set(rest) or not words:
            out.line(_help.group(words, COMMANDS, GROUPS, out.width(out.out)))
            return 0
        line = " ".join(["gf", *words])
        raise Usage(f"`{line}` needs a command", hint=f"`{line} --help` lists them")
    if {"-h", "--help"} & set(rest):
        out.line(_help.command(command, GLOBALS, out.width(out.out)))
        return 0
    parsed = Parser(command, [*command.flags, *GLOBALS]).parse_args(rest)
    call = _call(command, parsed, out)
    call.require(*command.needs)
    with out.working() as activity:
        call.state["activity"] = activity
        if command.before is not None:
            call.before = command.before(call)
        if command.ask is not None:
            out.confirm(command.ask(call), yes=bool(call.options.get("yes")))
        result = command.run(call) if command.run is not None else call.invoke()
        if call.json and isinstance(result, Iterator):
            for item in result:
                out.line(json.dumps(jsonable(item), ensure_ascii=False))
        elif call.json:
            out.line(json.dumps(as_json(call, result), ensure_ascii=False))
        else:
            command.show(call, result)
            out.view(getattr(result, "url", None))
    return 0


def _split(
    argv: list[str], commands: dict[tuple[str, ...], Command], groups: dict
) -> tuple[tuple[str, ...], list[str]]:
    """The command's words, and the rest of the line: its arguments and flags."""
    takes_value = {f.option for f in GLOBALS if not f.kind.switch}
    words: list[str] = []
    rest: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        known = (*words, token)
        if token in takes_value:
            rest += [token, next(tokens, "")]
        elif token.startswith("-") or tuple(words) in commands:
            rest.append(token)
        elif known in commands or known in groups or known == ("help",):
            words.append(token)
        else:
            rest.append(token)
    return tuple(words), rest


def _unknown(words: tuple[str, ...], word: str, commands: dict, groups: dict) -> Usage:
    options = {
        key[len(words)]
        for key in [*commands, *groups]
        if key[: len(words)] == words and len(key) > len(words)
    }
    near = term.suggest(word, options)
    line = " ".join(["gf", *words])
    return Usage(
        f"`{line} {word}` is not a command",
        tip=f"a similar command exists: `{line} {near}`" if near else None,
        hint=f"`{line} --help` lists the commands",
    )


def _call(command: Command, parsed: Any, out: term.Output) -> Call:
    """The run of a command: its arguments, SDK keywords and the CLI's own options."""
    texts = [getattr(parsed, f"arg{index}") for index in range(len(command.args))]
    args = [
        None if text is None else arg.kind.read(text, arg.metavar)
        for arg, text in zip(command.args, texts, strict=True)
    ]
    values: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for flag in [*command.flags, *GLOBALS]:
        raw = getattr(parsed, flag.attribute)
        if raw is None or raw is False:
            continue
        if flag.kind.switch:
            value = flag.kind.parse(None)
        else:
            pieces = [
                flag.kind.read(text, flag.option)
                for text in (raw if flag.repeat else [raw])
            ]
            value = _joined(pieces) if flag.repeat else pieces[0]
        if flag.keyword is None:
            options[flag.name.replace("-", "_")] = value
        elif flag.kind.spread:
            for key, item in value.items():
                _put(values, key, item)
        else:
            _put(values, flag.keyword, value)
    return Call(
        command, out, args=args, values=values, options=options, connect=_connect
    )


def _joined(pieces: list[Any]) -> Any:
    if all(isinstance(piece, dict) for piece in pieces):
        return {key: value for piece in pieces for key, value in piece.items()}
    return [
        item
        for piece in pieces
        for item in (piece if isinstance(piece, list) else [piece])
    ]


def _put(values: dict[str, Any], key: str, value: Any) -> None:
    *path, last = key.split(".")
    for head in path:
        values = values.setdefault(head, {})
    values[last] = value


class Watcher:
    """The CLI's view of an SDK call in progress, in place of a client's progress line.

    A `Say` step becomes a progress line, its first word the verb; a step
    the SDK reaches is shown by the command's `reached`.
    """

    def __init__(self, activity: term.Activity, reached: Callable[[Any], None]) -> None:
        """Draws on `activity` and hands steps to `reached`."""
        self.activity, self._reached = activity, reached

    def start(self) -> None:
        """Nothing to do: the command's lines are timed from its start."""

    def say(self, step: Say) -> None:
        """Shows the step's text as a progress line."""
        if step.text:
            verb, _, message = step.text.partition(" ")
            log = step.log.partition(" ")[2] if step.log else None
            self.activity.show(verb, message, log=log)

    def reached(self, value: Any) -> None:
        """Hands a step the SDK reached to the command."""
        self._reached(value)

    def clear(self) -> None:
        """Leaves the line for the next status line to replace."""


def _connect(call: Call) -> Client:
    """The client a command calls: the cli origin, its project and timeout, and a `Watcher`."""
    from generalforecast._origin import Detected
    from generalforecast.client import Client

    client = Client(
        project=call.options.get("project"),
        timeout=call.options.get("timeout") or DEFAULT_TIMEOUT,
        progress=False,
        origin=Detected("cli").header(),
    )
    reached = call.command.reached
    client._line = Watcher(
        call.state["activity"],
        (lambda value: reached(call, value)) if reached else (lambda value: None),
    )
    return client


def as_json(call: Call, result: Any) -> Any:
    """A command's result as `--json` prints it: the SDK's dataclass, a list under its noun."""
    from generalforecast.types import SourceList

    if result is None:
        return {"name": call.name, "deleted": True}
    if isinstance(result, list):
        key = call.command.key or call.command.words[-2].replace("-", "_")
        listed: dict[str, Any] = {key: [jsonable(item) for item in result]}
        if isinstance(result, SourceList):
            listed["available"] = result.available
        return listed
    return jsonable(result)


def jsonable(value: Any) -> Any:
    """Any SDK result as JSON values: dataclasses by field, times in ISO 8601, tables as rows."""
    import dataclasses
    import math
    from collections.abc import Mapping
    from datetime import date, datetime, timedelta

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_pylist"):
        return jsonable(value.to_pylist())
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return {"rows": jsonable(value.to_dict(orient="records"))}
    if hasattr(value, "item") and not isinstance(value, (str, int, bool)):
        return jsonable(value.item())
    return value


def report(out: term.Output, error: Error, *, as_json: bool) -> None:
    """Prints an error: cargo's shape for a person, `{"error": {...}}` under `--json`.

    A declined question prints only the line saying what did not happen.
    The request's id is a note for a real failure, or any error under `-v`.
    """
    out.end()
    if as_json:
        out.line(json.dumps({"error": error.to_dict()}, ensure_ascii=False))
        return
    if isinstance(error, term.Declined):
        out.say(error.said, force=True)
        return
    notes = [for_cli(error.note)] if error.note else []
    if error.request_id and (not error.recoverable or out.verbose):
        request = f"request {error.request_id}"
        notes = [f"{notes[0]} ({request})"] if notes else [request]
    out.error(
        error.message,
        notes=notes,
        hint=for_cli(error.hint) if error.hint else None,
        tip=getattr(error, "tip", None),
        pointer=getattr(error, "pointer", None),
        failure=not error.recoverable,
    )


_SDK_CALL = re.compile(r"`?gf\.([a-z_]+(?:\.[a-z_]+)*)\(([^()]*)\)`?")
_KEYWORD = re.compile(r"(?<![\w-])([a-z][a-z_]*)=(?=$|[\s,;.)])")


def for_cli(text: str) -> str:
    """A hint worded for Python, worded for the shell.

    `gf.experiments.wait('adult-income')` reads `gf experiments wait
    adult-income`, and a bare keyword such as `key=` reads `--key`: the
    command line mirrors the SDK word for word.
    """

    def shown(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return shlex.quote(str(value))

    def command(match: re.Match[str]) -> str:
        try:
            call = ast.parse(f"f({match[2]})", mode="eval").body
            assert isinstance(call, ast.Call)
            parts = ["gf", *(word.replace("_", "-") for word in match[1].split("."))]
            parts += [shown(ast.literal_eval(arg)) for arg in call.args]
            for keyword in call.keywords:
                value = ast.literal_eval(keyword.value)
                flag = "--" + str(keyword.arg).replace("_", "-")
                parts += [flag] if value is True else [flag, shown(value)]
        except (SyntaxError, ValueError, AssertionError):
            return match[0]
        return "`" + " ".join(parts) + "`"

    text = _SDK_CALL.sub(command, text)
    return _KEYWORD.sub(lambda match: "--" + match[1].replace("_", "-"), text)
