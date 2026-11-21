# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Diagnostics a person reads, written the way cargo writes them.

One `error:` or `warning:` line, then optionally where it points,
`  --> path:line:col`, then `note:` lines for why and `help:` lines for what
to do instead:

    error: predictors.py could not be loaded
      --> predictors.py:9:5
    help: a similar argument exists: `model`

That is the gf CLI's shape (`term.Output.error`): its first line says what
failed in the person's terms and carries no code. What a service said
about a failure is a note under it, with the request's id:

    error: churn revision 5 failed to start in production
    note: the replica exited loading the model (request r-8f2c41d0)
    help: production still serves revision 4

`head` can still put a code in brackets on the first line,
`error[ValidationError]: ...`.

Messages start lowercase, unless with a name such as GF_API_KEY or
Parquet, and end without a period. Wire codes and reasons are not messages.
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
