# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`gf --help`, `gf <resource> --help` and `gf help <command>`, written from the command table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from generalforecast.cli._spec import YES, Command, Flag
from generalforecast.term import layout

TAGLINE = "General Forecast from the command line: the Python API, word for word"
USAGE = "gf <resource> <verb> [NAME] [flags]"


def _rows(rows: Sequence[tuple[str, str]], width: int) -> list[str]:
    return layout([(key, value, "") for key, value in rows], width, limit=None)


def _flags(flags: Sequence[Flag], width: int) -> list[str]:
    return _rows([(flag.shown(), flag.help) for flag in flags], width)


def group(
    words: tuple[str, ...],
    commands: Mapping[tuple[str, ...], Command],
    groups: Mapping[tuple[str, ...], str],
    width: int,
) -> str:
    """The help of `gf` or of a group of commands: what is in it, one line each."""
    from generalforecast.cli import GLOBALS

    depth = len(words)
    inside = [
        key
        for key in [*groups, *commands]
        if key[:depth] == words and len(key) == depth + 1
    ]
    nested = [(key[-1], groups[key]) for key in inside if key in groups]
    leaves = [(key[-1], commands[key].summary) for key in inside if key in commands]
    line = " ".join(["gf", *words])
    lines = [groups.get(words, TAGLINE), ""]
    lines.append(f"Usage: {USAGE if not words else line + ' <verb> [NAME] [flags]'}")
    for title, rows in (("Resources", nested), ("Commands", leaves)):
        if rows:
            lines += ["", f"{title}:", *_rows(rows, width)]
    if not words:
        version = Flag("version", "Print the version", YES, short="-V")
        lines += ["", "Global flags:", *_flags([*GLOBALS, version], width)]
    lines += ["", f"`{line} <command> --help` shows a command's arguments and flags."]
    return "\n".join(lines)


def command(command: Command, globals_: Sequence[Flag], width: int) -> str:
    """A command's help: what it does, its arguments, its flags and the SDK method it calls."""
    lines = [command.summary, "", f"Usage: {command.line} [flags]"]
    if command.args:
        lines += [
            "",
            "Arguments:",
            *_rows([(a.metavar, a.help) for a in command.args], width),
        ]
    if command.flags:
        lines += ["", "Flags:", *_flags(command.flags, width)]
    lines += ["", "Global flags:", *_flags(globals_, width)]
    arguments = ", ".join([*(a.metavar for a in command.args), "**flags"])
    lines += ["", f"Python: {command.python or f'gf.{command.path}({arguments})'}"]
    return "\n".join(lines)
