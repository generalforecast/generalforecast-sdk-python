# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""What a command is: its words, arguments and flags, and how text becomes SDK values.

A command mirrors `gf.<resource>.<verb>(NAME, **flags)`: its arguments are
the method's positional arguments and each flag one keyword argument, `-`
written for `_`. A `Kind` turns a flag's text into the value the keyword
takes (`--models a,b` is a list, `--type age=number` a mapping); a dotted
keyword (`plan.steps`) fills one field of a setting's dict. The parser and
the help both read these, so a flag is described in one place.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from generalforecast.errors import Error

if TYPE_CHECKING:
    from generalforecast.cli._call import Call


class Usage(Error):
    """A command line `gf` cannot read; `tip` names a near miss."""

    code = "UsageError"

    def __init__(
        self, message: str, *, tip: str | None = None, hint: str | None = None
    ) -> None:
        """Keeps the message, a near-miss tip and what to run for help."""
        super().__init__(message, hint=hint, reason="usage_error")
        self.tip = tip


@dataclass(frozen=True)
class Kind:
    """How the text of an argument or a flag becomes the value the SDK takes.

    A `switch` takes no text; `parse` gets None and says what it stands for.
    A `spread` value is a mapping whose keys are keyword arguments
    themselves, as `--config host=...` is. Repeated flags join their values.
    """

    metavar: str
    parse: Callable[[Any], Any] = str
    what: str = "a value"
    switch: bool = False
    spread: bool = False

    def read(self, text: str, flag: str) -> Any:
        """The value of one piece of text.

        Raises:
            Usage: For text this kind does not read.
        """
        try:
            return self.parse(text)
        except (ValueError, TypeError) as error:
            detail = f": {error}" if str(error) and isinstance(error, TypeError) else ""
            raise Usage(f"`{text}` is not {self.what} for {flag}{detail}") from None


def _names(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _scalar(text: str) -> Any:
    """`5` is an int, `0.5` a float, `true` a bool; anything else stays text."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _pairs(text: str, value: Callable[[str], Any] = str) -> dict[str, Any]:
    out = {}
    for part in _names(text):
        key, sep, item = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(part)
        out[key.strip()] = value(item.strip())
    return out


def _whole(text: str) -> int:
    value = int(text.replace(",", "").replace("_", ""))
    return value


def _seconds(text: str) -> float:
    value = float(text)
    if value < 0:
        raise ValueError(text)
    return value


def _fraction(text: str) -> float:
    """`20%` or `0.2`."""
    text = text.strip()
    value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    if not 0 <= value <= 1:
        raise ValueError(text)
    return value


def _span(text: str) -> Any:
    """`1024`, or `1024..8192` for a range, `{"min": 1024, "max": 8192}`."""
    low, sep, high = text.partition("..")
    if not sep:
        return _scalar(text)
    return {"min": _scalar(low), "max": _scalar(high)}


def _missing(text: str) -> dict[str, Any]:
    """`5%`, or `1%..10%` for a range of rates."""
    low, sep, high = text.partition("..")
    if sep:
        return {"rate": {"min": _fraction(low), "max": _fraction(high)}}
    return {"rate": _fraction(text)}


FULL_COMPONENTS = (
    "column_encoder",
    "row_encoder",
    "in_context_transformer",
    "prediction_head",
)


def _method(text: str) -> dict[str, Any]:
    """`lora:16`, `lora:16+adapter`, `adapter`, or `full` with every component or `full:COMPONENT,...`."""
    name, _, rest = text.partition(":")
    if name == "adapter" and not rest:
        return {"type": "input_adapter"}
    if name == "lora":
        rank, plus, extra = rest.partition("+")
        if plus and extra != "adapter":
            raise ValueError(text)
        method: dict[str, Any] = {"type": "lora", "rank": int(rank or 16)}
        return {**method, "input_adapter": True} if plus else method
    if name == "full":
        return {"type": "full", "components": _names(rest) or list(FULL_COMPONENTS)}
    raise ValueError(text)


def _budget(text: str) -> dict[str, float]:
    """`500credits` or `2h`."""
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(credits?|h|hours?)\s*", text)
    if found is None:
        raise ValueError(text)
    return {"credits" if found[2].startswith("c") else "gpu_hours": float(found[1])}


def _data(text: str) -> list[Any]:
    """Snapshot names, `tag:T` and `context:NAME[@VERSION]`, as a run's data."""
    items: list[Any] = []
    for part in _names(text):
        kind, sep, rest = part.partition(":")
        if not sep:
            items.append(part)
        elif kind == "tag" and rest:
            items.append({"tag": rest})
        elif kind == "context" and rest:
            name, at, version = rest.partition("@")
            items.append(
                {"context": name, **({"version": _whole(version)} if at else {})}
            )
        else:
            raise ValueError(part)
    return items


def _selection(text: str) -> Any:
    from generalforecast.types import Selection

    return Selection(**_pairs(text, _scalar))


def _expires(text: str) -> str | None:
    return None if text == "never" else text


def _file_text(path: str) -> str:
    from pathlib import Path

    try:
        return Path(path).expanduser().read_text()
    except OSError as error:
        raise ValueError(error.strerror) from None


def _stdin(_: Any) -> str:
    import sys

    return sys.stdin.read().rstrip("\n")


TEXT = Kind("TEXT")
WHOLE = Kind("N", _whole, "a whole number")
NUMBER = Kind("X", float, "a number")
SECONDS = Kind("SECONDS", _seconds, "a number of seconds")
NAMES = Kind("A,B", _names, "a list")
NUMBERS = Kind(
    "X,Y", lambda text: [float(v) for v in _names(text)], "a list of numbers"
)
PAIRS = Kind("KEY=VALUE", _pairs, "KEY=VALUE pairs")
SETTINGS = Kind(
    "KEY=VALUE", lambda text: _pairs(text, _scalar), "KEY=VALUE pairs", spread=True
)
CLEAR = Kind("FIELD", lambda text: dict.fromkeys(_names(text)), "a field", spread=True)
FILE_TEXT = Kind("F", _file_text, "a file to read")
FRACTION = Kind("20%", _fraction, "a share such as 20% or 0.2")
SPAN = Kind("N|MIN..MAX", _span, "a number or a range such as 1024..8192")
DATA = Kind("S,...", _data, "snapshots, tag:T and context:NAME[@VERSION]")
MISSING = Kind("5%", _missing, "a share of cells such as 5%")
METHOD = Kind("METHOD", _method, "lora:16, lora:16+adapter, adapter or full:COMPONENTS")
BUDGET = Kind("500credits|2h", _budget, "a budget such as 500credits or 2h")
SELECTION = Kind(
    "core=X,neighbors=Y", _selection, "a selection such as core=0.5,neighbors=0.5"
)
EXPIRES = Kind("90d|never", _expires)
STDIN = Kind("", _stdin, switch=True)


def switch(value: Any = True) -> Kind:
    """A flag that takes no value and stands for `value`."""
    return Kind("", lambda _: value, switch=True)


YES = switch()


def choice(*values: str) -> Kind:
    """One of `values`."""

    def parse(text: str) -> str:
        if text not in values:
            raise ValueError(text)
        return text

    return Kind("|".join(values), parse, " or ".join(values))


@dataclass(frozen=True)
class Arg:
    """A positional argument: NAME, FILE, RUN; the SDK method's next positional argument."""

    metavar: str
    help: str
    kind: Kind = TEXT


@dataclass(frozen=True)
class Flag:
    """A flag: one keyword argument of the SDK method, or `to=None` for one the CLI keeps.

    `to` defaults to the name with `-` as `_`; a dotted `to` (`plan.steps`)
    is one field of the dict the keyword takes, nested at each dot.
    """

    name: str
    help: str
    kind: Kind = TEXT
    to: str | None = ""
    metavar: str | None = None
    repeat: bool = False
    short: str | None = None

    @property
    def option(self) -> str:
        """`--file`."""
        return f"--{self.name}"

    @property
    def keyword(self) -> str | None:
        """The keyword it fills, or None for one the CLI keeps."""
        return None if self.to is None else self.to or self.name.replace("-", "_")

    @property
    def attribute(self) -> str:
        return "flag_" + self.name.replace("-", "_")

    def shown(self) -> str:
        """`--file F`, as help lists it."""
        metavar = self.kind.metavar if self.metavar is None else self.metavar
        text = self.option + ("" if self.kind.switch else f" {metavar}")
        return f"{self.short}, {text}" if self.short else f"    {text}"


View = Callable[["Call", Any], None]


@dataclass(frozen=True)
class Command:
    """One `gf` command: its words, the SDK method it calls, and how its result shows.

    `run` replaces the plain call of `sdk` with the arguments; `before`
    reads what the command is about first, for `ask`, the question asked on
    a terminal before anything changes, and for `show`. `reached` shows a
    step the SDK reports before the call returns. `needs` are the keywords
    the SDK method has no default for, refused when left out. A list
    result is printed under `key` in `--json`.
    """

    words: tuple[str, ...]
    summary: str
    show: View
    args: tuple[Arg, ...] = ()
    flags: tuple[Flag, ...] = ()
    sdk: str | None = None
    run: Callable[[Call], Any] | None = None
    reached: Callable[[Call, Any], None] | None = None
    before: Callable[[Call], Any] | None = None
    ask: Callable[[Call], str] | None = None
    needs: tuple[str, ...] = ()
    key: str | None = None
    python: str = ""

    @property
    def path(self) -> str:
        """The SDK method, `contexts.create`."""
        return self.sdk or ".".join(word.replace("-", "_") for word in self.words)

    @property
    def line(self) -> str:
        """`gf contexts create NAME`."""
        return " ".join(["gf", *self.words, *(arg.metavar for arg in self.args)])


class Parser(argparse.ArgumentParser):
    """argparse that raises `Usage` instead of exiting, and never abbreviates a flag."""

    def __init__(self, command: Command, flags: Sequence[Flag]) -> None:
        super().__init__(
            prog=" ".join(["gf", *command.words]), add_help=False, allow_abbrev=False
        )
        self.flags = flags
        for index, arg in enumerate(command.args):
            self.add_argument(f"arg{index}", metavar=arg.metavar)
        for item in flags:
            names = [item.option, *([item.short] if item.short else [])]
            if item.kind.switch:
                self.add_argument(*names, dest=item.attribute, action="store_true")
            else:
                self.add_argument(
                    *names,
                    dest=item.attribute,
                    action="append" if item.repeat else "store",
                )

    def error(self, message: str) -> NoReturn:
        """Turns argparse's complaint into a `Usage` error naming a near miss."""
        from generalforecast.term import suggest

        unknown = re.search(r"unrecognized arguments: (\S+)", message)
        if unknown:
            word = unknown[1]
            if word.startswith("-"):
                near = suggest(word, [f.option for f in self.flags])
                raise Usage(
                    f"`{self.prog}` has no flag `{word}`",
                    tip=f"a similar flag exists: `{near}`" if near else None,
                    hint=f"`{self.prog} --help` lists its flags",
                )
            raise Usage(
                f"`{self.prog}` takes no argument `{word}`",
                hint=f"`{self.prog} --help` shows what it takes",
            )
        missing = re.search(r"the following arguments are required: (.+)", message)
        if missing:
            raise Usage(
                f"`{self.prog}` needs {missing[1].upper()}",
                hint=f"`{self.prog} --help` shows what it takes",
            )
        expected = re.search(
            r"argument (\S+?)(?:/\S+)?: expected one argument", message
        )
        if expected:
            raise Usage(f"{expected[1]} needs a value", hint=f"`{self.prog} --help`")
        raise Usage(message, hint=f"`{self.prog} --help`")
