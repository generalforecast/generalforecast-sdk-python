# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Golden transcripts of the surface spec's CLI samples, at 100 and 44 columns, on a terminal and not.

Each sample in `tests/samples.py` runs against a fresh fake service once
per width and stream, and the four transcripts are kept together in
`tests/golden/<name>.txt`. A terminal's transcript is what the screen
shows: colours are kept, and each frame a progress line drew before a line
replaced it is a line of its own, as the spec's samples show them. The
fake's loopback address reads as the hosted service's. Time is
a clock that moves only when the SDK sleeps or sends a request, so every
duration is the same on every run. `GF_GOLDEN=update pytest` writes them.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from generalforecast import _http, _ops, client, term
from generalforecast.cli import _login, _views, main
from tests import samples
from tests.fake import CONSOLE, TOKEN, Fake, serve

GOLDEN = Path(__file__).parent / "golden"
MODES = ((100, False), (100, True), (44, False), (44, True))
NOW = datetime(2026, 11, 21, 10, 5, tzinfo=UTC)
# Seconds the clock moves for each request sent.
REQUEST_SECONDS = 0.2


class Screen(io.StringIO):
    """A stream that is a terminal, or is not."""

    def __init__(
        self, terminal: bool, text: str = "", echo: io.StringIO | None = None
    ) -> None:
        super().__init__(text)
        self.terminal, self.echo = terminal, echo

    def isatty(self) -> bool:
        return self.terminal

    def readline(self, *args: object) -> str:
        line = super().readline()
        if self.terminal and self.echo is not None:
            self.echo.write(line)
        return line


def shown(raw: str, terminal: bool) -> str:
    """What a terminal shows: each progress frame a line, until the next line replaced it."""
    if not terminal:
        return raw
    lines, line = [], ""
    for piece in re.split(r"(\r\x1b\[K|\n)", raw):
        if piece == "\n" or (piece == "\r\x1b[K" and line):
            lines.append(line)
            line = ""
        elif piece != "\r\x1b[K":
            line += piece
    return "\n".join([*lines, *([line] if line else [])]) + "\n"


def transcript(
    sample: samples.Sample,
    columns: int,
    terminal: bool,
    folder: Path,
    env: pytest.MonkeyPatch,
) -> str:
    """The sample's commands and everything they printed, at this width, on this stream."""
    fake = Fake(chunk_bytes=1 << 24, catalog=list(samples.CATALOG))
    for knob, value in sample.knobs.items():
        setattr(fake, knob, value)
    stop = serve(fake)
    try:
        with env.context() as patch:
            clock = [1000.0]
            sent = _http.Transport.send

            def send(self: _http.Transport, request: _http.Request) -> _http.Reply:
                clock[0] += REQUEST_SECONDS
                return sent(self, request)

            def sleep(seconds: float) -> None:
                clock[0] += seconds

            home = folder / f"home-{columns}-{terminal}"
            work = home / "work"
            work.mkdir(parents=True)
            patch.chdir(work)
            patch.setenv("HOME", str(home))
            patch.delenv("XDG_CONFIG_HOME", raising=False)
            patch.setenv("GF_URL", fake.url)
            if sample.logged_in:
                patch.setenv("GF_API_KEY", TOKEN)
            patch.setenv("COLUMNS", str(columns))
            patch.setenv("TERM", "xterm-256color")
            patch.setenv("TZ", "UTC")
            for name in ("NO_COLOR", "FORCE_COLOR", "SSH_CONNECTION", "SSH_TTY"):
                patch.delenv(name, raising=False)
            time.tzset()
            patch.setattr(term, "clock", lambda: clock[0])
            patch.setattr(term, "TICK_SECONDS", 3600.0)
            patch.setattr(time, "sleep", sleep)
            patch.setattr(_http.Transport, "send", send)
            patch.setattr(_views, "now", lambda: NOW)
            patch.setattr(_ops, "_machine", lambda: "maya-mbp")
            patch.setattr(_login, "_browser", lambda: True)
            patch.setattr("webbrowser.open", lambda url, new=0: True)
            samples.files(sample, work)
            sample.seed(fake)
            client.reset()
            screen = Screen(terminal)
            patch.setattr(sys, "stdout", screen)
            patch.setattr(sys, "stderr", screen)
            patch.setattr(sys, "stdin", Screen(terminal, sample.stdin, echo=screen))
            for command in sample.commands:
                screen.write(f"$ gf {command}\n")
                code = main(shlex.split(command))
                if code:
                    screen.write(f"[exit {code}]\n")
            client.reset()
    finally:
        stop()
        time.tzset()
    assert not fake.errors, fake.errors
    # The fake stands for the hosted service, wherever on loopback it listens.
    return shown(screen.getvalue().replace(fake.url, CONSOLE), terminal)


@pytest.mark.parametrize("sample", samples.SAMPLES, ids=lambda sample: sample.name)
def test_sample(
    sample: samples.Sample, env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parts = []
    for columns, terminal in MODES:
        where = "a terminal" if terminal else "not a terminal"
        parts.append(f"--- {columns} columns, {where}\n")
        parts.append(transcript(sample, columns, terminal, tmp_path, env))
    text = "".join(parts)
    golden = GOLDEN / f"{sample.name}.txt"
    if os.environ.get("GF_GOLDEN") == "update":
        golden.write_text(text)
    assert text == golden.read_text()


def test_every_golden_file_has_its_sample() -> None:
    names = {sample.name for sample in samples.SAMPLES}
    assert len(names) == len(samples.SAMPLES)
    assert {path.stem for path in GOLDEN.glob("*.txt")} == names
