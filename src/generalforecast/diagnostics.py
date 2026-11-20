# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Diagnostics a person reads, written the way cargo writes them.

One `error:` or `warning:` line, then optionally where it points,
`  --> path:line:col`, then `note:` lines for why and `help:` lines for what
to do instead:

    error: gf.yaml has no `target`
      --> gf.yaml:3:1
    note: a classifier learns the target column
    help: add `target: <column>`

That is the gf CLI's shape (`term.Output.error`): its first line says what
failed in the person's terms and carries no code. A code is information
only for an error a service sent back, and there it is a note beside the
service's own words, under a first line that says what gf was asking for:

    error: the service refused the snapshot upload
    note: ValidationError (422): application/json required
    note: request r-524a763d64e2

`head` can still put a code in brackets on the first line,
`error[ValidationError]: ...`. `Error.__str__` writes it that way, since
what an SDK caller reads in a traceback is the name to catch.

Messages start lowercase, unless with a name such as GF_API_KEY or
Parquet, and end without a period; tests/test_repository.py holds every
message literal in gf to that. Wire codes and reasons are not messages.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import Literal, TextIO

Severity = Literal["error", "warning"]


def head(severity: Severity, code: str | None = None) -> str:
    """The first line's label: `error`, `error[Code]`, `warning`, `warning[Code]`."""
    return severity if code is None else f"{severity}[{code}]"


def lines(
    severity: Severity,
    message: str,
    *,
    code: str | None = None,
    location: str | None = None,
    notes: Iterable[str] = (),
    help: Iterable[str] | str | None = None,
    style: Callable[[str, str], str] = lambda text, _kind: text,
) -> list[str]:
    """The diagnostic's lines, each label styled by `style(text, kind)`.

    `location` is `path:line:col`; a message, note or help of several lines
    keeps its later lines under its first.
    """
    out: list[str] = []

    def labelled(label: str, kind: str, text: str) -> None:
        first, *rest = text.split("\n") or [""]
        out.append(f"{style(label + ':', kind)} {first}")
        pad = " " * (len(label) + 2)
        out.extend(pad + line for line in rest)

    labelled(head(severity, code), severity, message)
    if location is not None:
        out.append(f"  {style('-->', 'location')} {location}")
    for note in notes:
        if note:
            labelled("note", "note", note)
    helps = [help] if isinstance(help, str) else list(help or ())
    for item in helps:
        if item:
            labelled("help", "help", item)
    return out


def report(
    severity: Severity,
    message: str,
    *,
    code: str | None = None,
    location: str | None = None,
    notes: Iterable[str] = (),
    help: Iterable[str] | str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Writes one diagnostic to `stream`, standard error by default."""
    out = sys.stderr if stream is None else stream
    for line in lines(
        severity, message, code=code, location=location, notes=notes, help=help
    ):
        out.write(line + "\n")
    out.flush()


__all__ = ["head", "lines", "report"]
