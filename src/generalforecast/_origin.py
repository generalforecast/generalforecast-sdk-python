# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Where a request comes from, for the `X-GF-Origin` header (section 3.7).

`notebook` inside IPython or marimo, with the tool named from its
environment; `script` anywhere else. The CLI sets `cli` itself. Hex runs an
IPython kernel and has no confirmed runtime marker yet, so a Hex notebook
reports `notebook` with the tool `jupyter`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Surface = Literal["console", "cli", "notebook", "script", "api"]


def _version() -> str:
    from generalforecast import __version__

    return __version__


@dataclass(frozen=True)
class Detected:
    """The surface and, in a notebook, the tool running it."""

    surface: Surface
    notebook: str | None = None

    def header(self, client: str | None = None) -> str:
        """The header's value: `notebook; notebook=colab; client=generalforecast/0.2.3`."""
        parts = [self.surface]
        if self.notebook:
            parts.append(f"notebook={self.notebook}")
        parts.append(f"client={client or 'generalforecast/' + _version()}")
        return "; ".join(parts)


def _in_ipython() -> bool:
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        return ipython.get_ipython() is not None
    except (AttributeError, RuntimeError, ValueError):
        return False


def _in_marimo() -> bool:
    marimo = sys.modules.get("marimo")
    if marimo is None:
        return False
    running = getattr(marimo, "running_in_notebook", None)
    try:
        return bool(running()) if callable(running) else True
    except (AttributeError, RuntimeError, ValueError):
        return False


def notebook_tool(environ: Mapping[str, str] | None = None) -> str:
    """The notebook tool, from the markers each one sets."""
    env = os.environ if environ is None else environ
    if env.get("COLAB_RELEASE_TAG"):
        return "colab"
    if env.get("DATABRICKS_RUNTIME_VERSION"):
        return "databricks"
    if _in_marimo():
        return "marimo"
    if env.get("VSCODE_PID"):
        return "vscode"
    return "jupyter"


def detect(environ: Mapping[str, str] | None = None) -> Detected:
    """The origin of requests made from this process."""
    if _in_ipython() or _in_marimo():
        return Detected("notebook", notebook_tool(environ))
    return Detected("script")


def interactive() -> bool:
    """Whether a person is watching: a notebook, or stderr on a terminal."""
    if _in_ipython() or _in_marimo():
        return True
    isatty = getattr(sys.stderr, "isatty", None)
    try:
        return bool(isatty and isatty())
    except ValueError:
        return False
