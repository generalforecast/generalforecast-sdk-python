# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Python files of specs: `gf deploy FILE` and `gf fine-tunes runs create NAME --from FILE`.

A file runs with `runpy` in a fresh namespace and its module-level names
are read; loading it makes no call to the service. A file that fails to
load is an error pointing into it, at the argument that is wrong where one
is: `--> predictors.py:9:5`.
"""

from __future__ import annotations

import ast
import re
import traceback
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from generalforecast import types
from generalforecast.errors import Error, InvalidInput
from generalforecast.term import Pointer, suggest


class LoadError(InvalidInput):
    """A spec file that could not be loaded; `pointer` is where in it."""

    def __init__(
        self, message: str, *, pointer: Pointer | None = None, **fields: Any
    ) -> None:
        """Keeps the pointer beside the error's fields."""
        super().__init__(message, **fields)
        self.pointer = pointer


def _run(path: str) -> dict[str, Any]:
    import runpy

    file = Path(path).expanduser()
    if not file.is_file():
        raise InvalidInput(
            f"{path} is not a file", hint="pass the path of a Python file"
        )
    try:
        return runpy.run_path(str(file), run_name="__gf__")
    except SyntaxError as error:
        source = error.text or ""
        raise LoadError(
            f"{path} could not be loaded",
            pointer=Pointer(
                path,
                error.lineno or 1,
                error.offset or 1,
                source.rstrip("\n"),
                error.msg,
            ),
        ) from None
    except Exception as error:  # noqa: BLE001
        raise _located(path, file, error) from None


_UNEXPECTED = re.compile(
    r"(\w+)\.__init__\(\) got an unexpected keyword argument '(\w+)'"
)


def _located(path: str, file: Path, error: Exception) -> LoadError:
    """The error that stopped a file, pointing at the line of the file it came from."""
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if Path(frame.filename).resolve() == file.resolve()
    ]
    message = error.message if isinstance(error, Error) else str(error)
    hint = error.hint if isinstance(error, Error) else None
    if not frames:
        return LoadError(f"{path} could not be loaded", note=message, hint=hint)
    frame = frames[-1]
    lines = file.read_text().splitlines()
    line = frame.lineno or 1
    column = (frame.colno or 0) + 1
    width = (frame.end_colno or column) - column + 1 if frame.end_lineno == line else 1
    unexpected = _UNEXPECTED.search(message)
    if unexpected:
        name, argument = unexpected.groups()
        found = _keyword(file, line, argument)
        if found is not None:
            line, column = found
            width = len(argument)
        message = f"gf.{name}() has no argument `{argument}`"
        near = suggest(argument, _arguments(name))
        hint = f"a similar argument exists: `{near}`" if near else None
    else:
        message = re.sub(r"(\w+)\.__init__\(\)", r"gf.\1()", message)
    source = lines[line - 1] if 0 < line <= len(lines) else ""
    return LoadError(
        f"{path} could not be loaded",
        pointer=Pointer(path, line, column, source, message, max(width, 1)),
        hint=hint,
    )


def _keyword(file: Path, line: int, name: str) -> tuple[int, int] | None:
    """Where the keyword argument `name` of the call on `line` is written."""
    for node in ast.walk(ast.parse(file.read_text())):
        if isinstance(node, ast.Call) and node.lineno <= line <= (
            node.end_lineno or line
        ):
            for keyword in node.keywords:
                if keyword.arg == name:
                    return keyword.lineno, keyword.col_offset + 1
    return None


def _arguments(name: str) -> list[str]:
    cls = getattr(types, name, None)
    return [f.name for f in fields(cls) if f.name != "_"] if cls is not None else []


def predictors(path: str) -> list[types.Predictor]:
    """Every `gf.Predictor` a file binds to a module-level name, alone or in a list, tuple or dict.

    Raises:
        InvalidInput: For a file that does not load, defines none, or names one twice.
    """
    found: list[types.Predictor] = []
    for name, value in _run(path).items():
        if name.startswith("__"):
            continue
        if isinstance(value, Mapping):
            items = list(value.values())
        else:
            items = list(value) if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, types.Predictor) and not any(
                item is seen for seen in found
            ):
                found.append(item)
    names = [spec.name for spec in found]
    twice = next((name for name in names if names.count(name) > 1), None)
    if twice is not None:
        raise InvalidInput(
            f"{path} defines the predictor {twice} twice",
            hint="give each gf.Predictor a name of its own",
        )
    if not found:
        raise InvalidInput(
            f"{path} defines no gf.Predictor",
            hint="bind each spec to a name at the top of the file, as "
            '`churn = gf.Predictor("churn", ...)`',
        )
    return found


def fine_tune(path: str) -> types.FineTuneSpec:
    """The `run = gf.FineTuneSpec(...)` a file defines.

    Raises:
        InvalidInput: For a file that does not load or defines no `run`.
    """
    run = _run(path).get("run")
    if not isinstance(run, types.FineTuneSpec):
        raise InvalidInput(
            f"{path} defines no `run = gf.FineTuneSpec(...)`",
            hint='write `run = gf.FineTuneSpec("tabicl-v2", data=[...], plan=gf.Plan(steps=200))`',
        )
    return run
