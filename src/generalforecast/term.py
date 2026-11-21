# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Terminal output for the `gf` command.

One renderer draws every surface the CLI shows: status lines with a
right-aligned verb, progress lines for work still going, lock change lines,
errors as cargo writes them, tables, and the interactive prompts `gf init`
uses. Results go to stdout and only `generalforecast.render` writes them;
status lines, prompts and errors go to stderr. Colour is on only when the
stream is a terminal and `NO_COLOR` is unset.

Every line is laid out for the terminal's width (`size`), so the terminal
never breaks one in the middle of a word: `wrap` folds text after a comma or
at a space and `layout` places a row's key, value and quiet note, which
`Output.details`, `Output.collapsed`, the status lines and the last column
of `Output.table` all go through. A list of names takes at most `LIST_LINES`
lines and counts the rest as `and N more`; `-v` shows every name. A stream
that is not a terminal is laid out `WIDTH` columns wide, so a log reads the
same wherever it was written, and there a status line stays one line, to be
searched for.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any

from generalforecast.errors import Error, InvalidInput, printable

# Status verbs are right-aligned in this many columns.
VERB_WIDTH = 12
# Collapsed prompt labels are padded to this width so values start at column 14.
LABEL_WIDTH = 13
# Ends every frame a raw prompt draws. The cursor stays hidden while a frame is
# up, and whatever reads the terminal knows the frame is whole once it arrives.
FRAME_END = "\x1b[?25l"
# The rendering size when the terminal does not report one.
WIDTH = 100
HEIGHT = 24
# A line stops this many columns short of the width: a terminal that wraps as
# soon as its last column is written would put a blank line after a full one.
MARGIN = 1
# A wrapped list of names takes at most this many lines, `and N more` included.
LIST_LINES = 4
# Seconds between redraws of a progress line that shows how long it has run.
TICK_SECONDS = 1.0
# The fewest columns a wrapped value is given. A terminal too narrow for them
# wraps the line itself, which reads better than a column of single words.
MIN_VALUE = 16
# Lock change keys (model, input, selection, target, context) share one column.
LOCK_KEY_WIDTH = len("selection")
# Seconds a raw prompt waits for the rest of an escape sequence.
ESCAPE_SECONDS = 0.05

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
}


def color_enabled(
    stream: IO[str] | None, environ: Mapping[str, str] | None = None
) -> bool:
    """Whether `stream` should get ANSI colour.

    `NO_COLOR` set to anything turns colour off; `FORCE_COLOR` set to a value
    other than `0` turns it on; otherwise colour follows whether the stream is
    a terminal and `TERM` is not `dumb`.
    """
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR"):
        return False
    force = env.get("FORCE_COLOR")
    if force is not None:
        return force != "0"
    if env.get("TERM") == "dumb":
        return False
    return bool(stream is not None and getattr(stream, "isatty", lambda: False)())


def seconds(value: float) -> str:
    """Formats a duration as the status lines show it: `0.21s`, `6.8s`, `41s`."""
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def elapsed(value: float) -> str:
    """Formats a time waited so far, to the second: `45s`, `2m10s`, `1h05m`."""
    total = int(value)
    if total < 60:
        return f"{total}s"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def count(value: int) -> str:
    """Formats a count with thousands separators."""
    return f"{value:,}"


def plural(value: int, noun: str) -> str:
    """`1 row`, `2,052 rows`."""
    return f"{count(value)} {noun}{'' if value == 1 else 's'}"


def size(stream: IO[str] | None = None) -> os.terminal_size:
    """The terminal's columns and lines, or `WIDTH` by `HEIGHT` when unknown."""
    try:
        return os.get_terminal_size((stream or sys.stderr).fileno())
    except (OSError, ValueError, AttributeError):
        return os.terminal_size((WIDTH, HEIGHT))


_ANSI = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*\x1b\\")


def _width(character: str) -> int:
    if unicodedata.combining(character) or unicodedata.category(character) == "Cf":
        return 0
    return 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1


def visible(text: str) -> int:
    """The columns text takes on a terminal, ANSI codes aside."""
    return sum(_width(character) for character in _ANSI.sub("", text))


def pad(text: str, width: int) -> str:
    """Text padded with spaces to `width` printed columns."""
    return text + " " * max(width - visible(text), 0)


def clip(text: str, width: int) -> str:
    """Text cut to at most `width` printed columns, its ANSI codes kept whole."""
    shown, used, at = [], 0, 0
    for match in _ANSI.finditer(text):
        for character in text[at : match.start()]:
            used += _width(character)
            if used > width:
                return "".join(shown) + ("\x1b[0m" if "\x1b[" in text else "")
            shown.append(character)
        shown.append(match.group(0))
        at = match.end()
    for character in text[at:]:
        used += _width(character)
        if used > width:
            return "".join(shown) + ("\x1b[0m" if "\x1b[" in text else "")
        shown.append(character)
    return "".join(shown)


def shorten(text: str, width: int) -> str:
    """Text cut to `width` printed columns, the cut marked with `…`."""
    if visible(text) <= width:
        return text
    return clip(text, max(width - 1, 0)) + "…"


# Text between backticks, and what stands in for its spaces while it folds.
_QUOTED = re.compile(r"`[^`\n]*`")
_HELD = "\x00"
# Where a word wider than a line may be broken: after its last `/` in reach,
# or failing that after its last `-`, `_`, `.` or `:`.
_BREAKS = ("/", "-_.:")


def _separator(word: str, width: int) -> int:
    """How much of a word to keep on a line `width` wide: up to a separator, or 0 for none."""
    head = clip(word, max(width, 0))
    for marks in _BREAKS:
        at = max(head.rfind(mark) for mark in marks) + 1
        if at > 1:
            return at
    return 0


def _words(text: str, width: int, line: str = "") -> list[str]:
    """Text folded at its spaces, after what `line` holds, into lines `width` wide.

    A word wider than a line is broken after a separator in it, starting on
    the line it follows where one is in reach, and where it has none at the
    line's end.
    """
    lines = []
    for word in text.split(" "):
        joined = f"{line} {word}" if line else word
        if visible(joined) <= width:
            line = joined
            continue
        if line.strip() and visible(word) > width:
            at = _separator(word, width - visible(line) - 1)
            if at:
                line, word = f"{line} {word[:at]}", word[at:]
        if line.strip():
            lines.append(line.rstrip())
        while visible(word) > width:
            at = _separator(word, width) or len(clip(word, width)) or 1
            lines.append(word[:at])
            word = word[at:]
        line = word
    return [*lines, line.rstrip()]


def wrap(text: str | Sequence[str], width: int) -> list[str]:
    """Plain text folded into lines of at most `width` printed columns.

    A list of items, or text that holds commas, breaks after a comma where
    it can. An item wider than a line breaks at its spaces, and a word wider
    than a line after a `/`, `-`, `_`, `.` or `:` in it, or at the line's end
    when it has none. In text, such an item carries on from the line before
    it; in a list it starts its own, so a name is never split to fill a
    line. Text between backticks, a command or a name to copy, is one word
    to the fold. Text that fits comes back as one line, unchanged. Style
    the lines after folding: an ANSI code is not kept whole.
    """
    width = max(width, 1)
    if not isinstance(text, str):
        return _folded(list(text), width, listed=True)
    if visible(text) <= width:
        return [text]
    held = _QUOTED.sub(lambda quoted: quoted.group(0).replace(" ", _HELD), text)
    lines = _folded(held.split(", "), width, listed=False)
    return [line.replace(_HELD, " ") for line in lines]


def _folded(items: list[str], width: int, *, listed: bool) -> list[str]:
    """`wrap`'s lines for comma-separated items, of a list or of running text."""
    lines, line = [], ""
    for index, item in enumerate(items):
        piece = item if index == len(items) - 1 else f"{item},"
        joined = f"{line} {piece}" if line else piece
        if visible(joined) <= width:
            line = joined
            continue
        if line and (listed or visible(piece) <= width):
            lines.append(line)
            line = ""
        *whole, line = _words(piece, width, line)
        lines.extend(whole)
    return [*lines, line]


def _shared(pairs: Sequence[tuple[int, int]], room: int, gap: int) -> int:
    """The column the second parts of `(first, second)` widths start in, or -1.

    It follows the widest first part that still leaves room for the second
    part of every pair no wider than it. A pair wider than that is left out,
    so one long first part never pushes the others' second parts away.
    """
    fits = [
        (first, second)
        for first, second in pairs
        if second and first + gap + second <= room
    ]
    for widest in sorted({first for first, _ in fits}, reverse=True):
        if all(
            widest + gap + second <= room for first, second in fits if first <= widest
        ):
            return widest
    return -1


def _more(hidden: int) -> str:
    return f"and {count(hidden)} more"


def fold(
    items: Sequence[str], width: int, limit: int | None = LIST_LINES
) -> tuple[list[str], int]:
    """A list of names folded between names into at most `limit` lines.

    Returns the lines and how many names they leave out. When some are left
    out the last line ends `and N more`, which the lines have room for. With
    no `limit` every name is shown.
    """
    lines = wrap(items, width)
    if limit is None or len(lines) <= limit:
        return lines, 0
    kept = 0
    for shown in range(1, len(items)):
        tail = _more(len(items) - shown)
        if len(wrap([*items[:shown], tail], width)) > limit:
            break
        kept = shown
    hidden = len(items) - kept
    return wrap([*items[:kept], _more(hidden)], width), hidden


def layout(
    rows: Sequence[tuple[str, str | Sequence[str], str]],
    width: int,
    *,
    indent: int = 2,
    gap: int = 2,
    start: int | None = None,
    limit: int | None = LIST_LINES,
    quiet: Callable[[str], str] = lambda text: text,
) -> list[str]:
    """Rows of key, value and quiet note as lines no wider than `width` allows.

    Values start in one column, `start`, by default after the widest key.
    A value that is a list of names is folded between names (`fold`); any
    other is folded as text (`wrap`); the lines after the first hang under
    the value. A note goes beside its value while it fits: the notes of
    one-line values share a column, set by the widest value that leaves each
    of those notes room, so one long value never pushes the others' notes
    away. A folded value's note follows its last line. A note with no room
    beside its value goes on lines of its own under it. Values and notes are
    plain text; `quiet` styles the notes and a list's `and N more`.
    """
    if not rows:
        return []
    if start is None:
        start = indent + max(visible(key) for key, _, _ in rows) + gap
    room = max(width - MARGIN - start, MIN_VALUE)
    blocks = []
    for key, value, note in rows:
        if isinstance(value, str):
            lines, hidden = wrap(value, room), 0
        else:
            lines, hidden = fold(value, room, limit)
        blocks.append((key, lines, hidden, note))

    def beside(at: int, note: str) -> bool:
        return at + gap + visible(note) <= room

    column = _shared(
        [
            (visible(lines[0]), visible(note))
            for _, lines, _, note in blocks
            if len(lines) == 1
        ],
        room,
        gap,
    )
    written = []
    for key, lines, hidden, note in blocks:
        shown = list(lines)
        tail = _more(hidden)
        if hidden and shown[-1].endswith(tail):
            shown[-1] = shown[-1][: len(shown[-1]) - len(tail)] + quiet(tail)
        last = visible(lines[-1])
        under: list[str] = []
        if note and last <= column and beside(column, note):
            shown[-1] = pad(shown[-1], column) + " " * gap + quiet(note)
        elif note and beside(last, note):
            shown[-1] += " " * gap + quiet(note)
        elif note:
            under = [quiet(line) for line in wrap(note, room)]
        head = " " * indent + pad(key, start - indent)
        for index, line in enumerate([*shown, *under]):
            written.append(((" " * start if index else head) + line).rstrip())
    return written


def _drawn(source: str) -> str:
    """A source line as the pointer block draws it: a tab as four spaces, no control characters.

    Each character is drawn by itself, so what is drawn of a line's start is
    the start of what is drawn of the line, and a caret's column follows.
    """
    return "".join(
        "    " if character == "\t" else character
        for character in source
        if character == "\t" or character.isprintable()
    )


def change_lines(
    items: Sequence[tuple[str, str, Any, Any]],
    *,
    style: Callable[..., str] = lambda text, *styles: text,
    width: int | None = None,
) -> list[str]:
    """Lock changes as lines: ` + key  value`, ` - key  value` or ` ~ key  old → new`.

    Keys are padded to the longest in the block; `style` colours the sign.
    With a `width`, a value too long for it is folded under itself.
    """
    if not items:
        return []
    keys = max(LOCK_KEY_WIDTH, *(len(key) for _, key, _, _ in items))
    start = 3 + keys + 2
    lines = []

    def shown(value: Any) -> str:
        # As gf.yaml and gf.lock write it: `true`, not Python's `True`.
        return str(value).lower() if isinstance(value, bool) else f"{value}"

    for op, key, old, new in items:
        colour = {"+": "green", "-": "red", "~": "yellow"}.get(op, "bold")
        if op == "+" or old is None:
            value = shown(new)
        elif op == "-" or new is None:
            value = shown(old)
        else:
            value = f"{shown(old)} → {shown(new)}"
        folded = (
            [value]
            if width is None
            else wrap(value, max(width - MARGIN - start, MIN_VALUE))
        )
        lines.append(f" {style(op, 'bold', colour)} {key.ljust(keys)}  {folded[0]}")
        lines.extend(" " * start + line for line in folded[1:])
    return lines


def suggest(word: str, choices: Iterable[str]) -> str | None:
    """The closest choice to a typo, or None when nothing is close."""
    import difflib

    matches = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return matches[0] if matches else None


@dataclass(frozen=True)
class Pointer:
    """Where in a file an error points, for the `-->` error shape."""

    path: str
    line: int
    column: int
    source: str
    label: str = ""
    width: int = 1


class Aborted(Error):
    """A question was left unanswered where an answer was needed; nothing changed."""

    code = "Aborted"


class Declined(Aborted):
    """The person said no, or pressed Escape: nothing changed, and nothing went wrong.

    `said` is the one line that tells them so in the command's terms, such
    as `Nothing was pushed.`; the message is what `--json` reports.
    """

    def __init__(self, message: str, *, said: str) -> None:
        """Keeps the sentence for a person beside the message for a program."""
        super().__init__(message)
        self.said = said


@dataclass(frozen=True)
class Option:
    """One row of a selector: the value, the shown text and a quiet detail."""

    value: Any
    text: str
    detail: str = ""
    shown: str = ""


class Output:
    """Writes the CLI's human output to `out` (results) and `err` (everything else)."""

    def __init__(
        self,
        *,
        quiet: bool = False,
        verbose: bool = False,
        out: IO[str] | None = None,
        err: IO[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Binds the renderer to its streams and colour setting."""
        self.quiet, self.verbose = quiet, verbose
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self._environ = environ
        self._color = {
            id(self.out): color_enabled(self.out, environ),
            id(self.err): color_enabled(self.err, environ),
        }
        # The verb of the progress line last drawn on stderr, until a line
        # replaces it; on a terminal it has no newline yet.
        self._progress: str | None = None
        # Whether a question is on stderr with its answer still to come, so
        # the line it is on is not ended yet.
        self._asking = False
        # `working` redraws its line from a thread of its own, so a line is
        # written whole or not at all.
        self._lock = threading.RLock()

    def colored(self, stream: IO[str] | None = None) -> bool:
        """Whether a stream (default stderr) gets colour."""
        stream = self.err if stream is None else stream
        return self._color.get(id(stream), color_enabled(stream, self._environ))

    def style(self, text: str, *styles: str, stream: IO[str] | None = None) -> str:
        """Wraps text in ANSI styles when the stream is coloured."""
        if not text or not styles or not self.colored(stream):
            return text
        codes = ";".join(_CODES[name] for name in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"

    def quiet_text(self, text: str, stream: IO[str] | None = None) -> str:
        """Text in the quiet colour."""
        return self.style(text, "dim", stream=stream)

    def _write(self, text: str, stream: IO[str]) -> None:
        with self._lock:
            if stream is self.err and self._progress is not None:
                if self._live():
                    text = "\r\x1b[K" + text
                self._progress = None
            if stream is self.err:
                self._asking = False
            stream.write(text + "\n")
            stream.flush()

    def _live(self) -> bool:
        """Whether stderr is a terminal, where a progress line is redrawn in place."""
        return bool(getattr(self.err, "isatty", lambda: False)())

    def line(self, text: str = "") -> None:
        """One line of a result on stdout; only `generalforecast.render` calls it."""
        self._write(text, self.out)

    def say(self, text: str = "", *, force: bool = False) -> None:
        """One line of commentary on stderr, dropped under `-q` unless forced."""
        if force or not self.quiet:
            self._write(text, self.err)

    def debug(self, text: str) -> None:
        """A quiet line shown only under `-v`."""
        if self.verbose:
            self._write(self.quiet_text(text), self.err)

    def link(self, url: str, text: str | None = None) -> str:
        """A clickable URL on terminals that support OSC 8 hyperlinks.

        Both may come from a server, so neither keeps a terminal escape that
        could end the link early or write anything else.
        """
        url = printable(url)
        shown = printable(text) if text else url
        if not self.colored():
            return shown
        return f"\x1b]8;;{url}\x1b\\{shown}\x1b]8;;\x1b\\"

    def status(self, verb: str, message: str, *, seconds: float | None = None) -> None:
        """A status line: the verb right-aligned in 12 columns, a space, the message.

        A duration ends the line as `in 0.21s`. Dropped under `-q`.
        """
        if self.quiet:
            return
        tail = "" if seconds is None else f" in {globals()['seconds'](seconds)}"
        for line in self._status(verb, message, tail):
            self._write(line, self.err)

    def _status(self, verb: str, message: str, tail: str = "") -> list[str]:
        """A status line's lines: on a terminal, the message folded under itself.

        Anywhere else a status line is one line however long, so a log can
        be searched for it. A message that carries its own escapes, such as
        a link, is left whole, and so is the `tail`, a duration: it ends the
        last line or takes one of its own.
        """
        head = self.style(verb.rjust(VERB_WIDTH), "bold", "green")
        lines = [message]
        if self._live() and "\x1b" not in message:
            room = max(size(self.err).columns - MARGIN - VERB_WIDTH - 1, MIN_VALUE)
            lines = wrap(message, room)
            if visible(lines[-1]) + visible(tail) > room:
                lines.append("")
                tail = tail.lstrip()
        lines[-1] += tail
        under = " " * (VERB_WIDTH + 1)
        return [f"{head} {lines[0]}", *(under + line for line in lines[1:])]

    def progress(
        self, verb: str, message: str, *, log: bool = True, trail: str = ""
    ) -> None:
        """A status line for work still going, such as a wait, drawn as `status` draws one.

        On a terminal each call redraws the line in place and the next line
        written to stderr replaces it; on any other stream the first call
        writes it and the calls after it with the same verb write nothing,
        so a log holds one line. Without `log` it holds none: a step too
        short to be worth a line of a log still shows on a terminal. A
        terminal's line is cut to its width, since a line the terminal
        wrapped could not be redrawn; `trail`, such as the time taken so
        far, ends the line and is never what is cut. Dropped under `-q`.
        """
        if self.quiet:
            return
        head = self.style(verb.rjust(VERB_WIDTH), "bold", "green")
        with self._lock:
            if not self._live():
                if log and self._progress != verb:
                    self._write(f"{head} {message}{trail}", self.err)
                    self._progress = verb
                return
            room = size(self.err).columns - MARGIN - visible(trail)
            line = shorten(f"{head} {message}", max(room, 1)) + trail
            self.err.write(f"\r\x1b[K{line}")
            self.err.flush()
            self._progress = verb

    @contextlib.contextmanager
    def working(self, *, log: bool = True) -> Generator[Activity]:
        """Progress lines for work that takes a while, kept current until the block ends.

        The `Activity` it yields shows a line with `show(verb, message)`. On
        a terminal the line is redrawn every `TICK_SECONDS` with how long
        that verb has been showing, as `Loading the model  12s`, so a person
        sees which step the time is going on. A log gets each verb's line
        once, without the time, or no line without `log`.
        """
        activity = Activity(self, log=log)
        try:
            yield activity
        finally:
            activity.close()

    def verb(self, verb: str, text: str, detail: str | None = None) -> None:
        """`status` with an optional two-space-separated detail."""
        self.status(verb, text if detail is None else f"{text}  {detail}")

    def changes(self, items: Sequence[tuple[str, str, Any, Any]]) -> None:
        """Lock change lines, keys padded to the longest key in the block."""
        if self.quiet:
            return
        width = size(self.err).columns
        for line in change_lines(items, style=self.style, width=width):
            self._write(line, self.err)

    def answer(
        self,
        label: str,
        value: str | Sequence[str],
        note: str = "",
        *,
        noun: str | None = None,
        columns: int | None = None,
        style: Callable[[str], str] | None = None,
    ) -> list[str]:
        """A prompt's answer as lines: the label padded to column 14, the value, a note.

        `layout` places them, so a long list of names folds under itself and
        a note with no room beside the value goes under it. A list too long
        for one line is followed by its count in `noun`s, as `19 columns`,
        when it has no other note. `style` colours the note, quiet by default.
        """
        columns = size(self.err).columns if columns is None else columns
        if noun is not None and not note and not isinstance(value, str):
            room = columns - MARGIN - LABEL_WIDTH
            if visible(", ".join(value)) > room:
                note = plural(len(value), noun)
        return layout(
            [(label, value, note)],
            columns,
            indent=0,
            start=LABEL_WIDTH,
            limit=None if self.verbose else LIST_LINES,
            quiet=self.quiet_text if style is None else style,
        )

    def collapsed(
        self,
        label: str,
        value: str | Sequence[str],
        note: str = "",
        *,
        noun: str | None = None,
    ) -> None:
        """A finished prompt: the label padded to column 14, then the value and a quiet note.

        A summary, so dropped under `-q`; the manifest holds every answer.
        """
        if not self.quiet:
            for line in self.answer(label, value, note, noun=noun):
                self._write(line, self.err)

    def details(
        self,
        rows: Sequence[tuple[str, str | Sequence[str], str]],
        *,
        indent: int = 2,
        stream: IO[str] | None = None,
        force: bool = False,
    ) -> None:
        """Rows of key, value and quiet note, laid out for the terminal's width.

        `layout` says where each part goes. A value given as a list is a list
        of names: past `LIST_LINES` lines the rest are counted as `and N
        more`, and `-v` shows them all. On stderr this is commentary, dropped
        under `-q` unless forced.
        """
        stream = self.err if stream is None else stream
        if stream is self.err and self.quiet and not force:
            return
        lines = layout(
            rows,
            size(stream).columns,
            indent=indent,
            limit=None if self.verbose else LIST_LINES,
            quiet=lambda text: self.quiet_text(text, stream),
        )
        for line in lines:
            self._write(line, stream)

    def _diagnostic(self, lines: list[str]) -> None:
        for line in lines:
            self._write(line, self.err)

    def _styled(self, text: str, kind: str) -> str:
        """A diagnostic label in its colour: errors red, warnings yellow, the rest bold."""
        if kind == "error":
            return self.style(text, "bold", "red")
        if kind == "warning":
            return self.style(text, "bold", "yellow")
        if kind == "location":
            return self.quiet_text(text)
        return self.style(text, "bold")

    def error(
        self,
        message: str,
        *,
        notes: Sequence[str] = (),
        hint: str | None = None,
        tip: str | None = None,
        pointer: Pointer | None = None,
    ) -> None:
        """An error as cargo writes one (`generalforecast.diagnostics`).

        `error: message`, with no code; a `-->` pointer into a file, with
        its source line and carets; a `note:` for each of `notes`, such as
        why, what a service answered and the request's id; a `help:` for
        what to do instead, and one naming a near miss. On a terminal a
        line too long for it folds under its own text.
        """
        from generalforecast.diagnostics import lines

        written = lines("error", self._fold("error", message), style=self._styled)
        self._diagnostic(written)
        if pointer is not None:
            self.pointer(pointer)
        self._diagnostic(
            lines(
                "error",
                "",
                notes=[self._fold("note", note) for note in notes],
                help=[self._fold("help", text or "") for text in (hint, tip)],
                style=self._styled,
            )[1:]
        )

    def _fold(self, label: str, text: str) -> str:
        """A diagnostic's text with a line break wherever a terminal needs one.

        `diagnostics.lines` keeps every line after the first under the
        first, so the fold hangs under the text. A log's lines are left
        whole, and so is text that carries its own escapes.
        """
        if not text or not self._live() or "\x1b" in text:
            return text
        room = size(self.err).columns - MARGIN - len(label) - 2
        return "\n".join(
            line for part in text.split("\n") for line in wrap(part, max(room, 1))
        )

    def pointer(self, where: Pointer) -> None:
        """The `-->` block: location, a bar, the source line and a caret line.

        The bars share a column whatever the line number's width, and the
        carets sit under the text they point at: a tab in the source line
        is drawn as four spaces and a wide character counts its two
        columns, on the source line and under it alike. On a terminal a
        label too long for the caret line folds under its own start.
        """
        number = str(where.line)
        gutter = " " * (len(number) + 1)
        quiet = self.quiet_text
        bar = f"{gutter} {quiet('|')}"
        self._write(
            f"{gutter}{quiet('-->')} {where.path}:{where.line}:{where.column}", self.err
        )
        self._write(bar, self.err)
        self._write(
            f"{quiet(number.rjust(len(number) + 1))} {quiet('|')} "
            f"{_drawn(where.source)}",
            self.err,
        )
        marks = max(where.width, 1)
        carets = self.style("^" * marks, "bold", "red")
        lead = " " * visible(_drawn(where.source[: max(where.column - 1, 0)]))
        label = [where.label] if where.label else []
        start = len(gutter) + 3 + len(lead) + marks + 1
        if label and self._live():
            label = wrap(where.label, max(size(self.err).columns - MARGIN - start, 1))
        self._write(
            f"{bar} {lead}{carets}{''.join(' ' + t for t in label[:1])}", self.err
        )
        for line in label[1:]:
            self._write(f"{bar}{' ' * (start - len(gutter) - 2)}{line}", self.err)

    def end(self) -> None:
        """Leaves the terminal on a line of its own, as after Ctrl-C.

        A progress line in flight, or a question still waiting for its
        answer, has no newline yet: this writes it, so whatever comes next,
        a shell's prompt included, starts a line. Otherwise it writes
        nothing.
        """
        with self._lock:
            if self._asking or (self._progress is not None and self._live()):
                self.err.write("\n")
                self.err.flush()
            self._asking, self._progress = False, None

    def warning(self, text: str, *, help: str | None = None) -> None:
        """A `warning:` diagnostic, dropped under `-q`."""
        from generalforecast.diagnostics import lines

        if not self.quiet:
            self._diagnostic(
                lines(
                    "warning",
                    self._fold("warning", text),
                    help=self._fold("help", help or ""),
                    style=self._styled,
                )
            )

    def hint(self, text: str) -> None:
        """A `help:` line on its own."""
        from generalforecast.diagnostics import lines

        self._diagnostic(
            lines("error", "", help=self._fold("help", text), style=self._styled)[1:]
        )

    def table(
        self,
        rows: Sequence[Sequence[str]],
        *,
        indent: int = 2,
        gap: int = 2,
        align: Sequence[str] | None = None,
        header: Sequence[str] | None = None,
        stream: IO[str] | None = None,
        force: bool = False,
    ) -> None:
        """Rows of cells, each column padded to its widest cell.

        `align` gives `<` or `>` per column (default left). A header row is
        aligned like the values beneath it. A table on stderr is commentary,
        dropped under `-q` unless forced; one on stdout is a result.

        A row wider than the terminal folds its last cell under itself, when
        that cell is left-aligned plain text. A result piped to a program is
        never folded: its lines are what the program reads.
        """
        stream = self.err if stream is None else stream
        if stream is self.err and self.quiet and not force:
            return
        every = [list(header), *map(list, rows)] if header else list(map(list, rows))
        if not every:
            return
        ncol = max(len(row) for row in every)
        widths = [
            max(visible(row[i]) if i < len(row) else 0 for row in every)
            for i in range(ncol)
        ]
        kinds = list(align or []) + ["<"] * (ncol - len(align or []))
        start = indent + sum(widths[:-1]) + gap * (ncol - 1)
        room = size(stream).columns - MARGIN - start
        folds = (
            kinds[-1] == "<"
            and room >= MIN_VALUE
            and (stream is self.err or getattr(stream, "isatty", lambda: False)())
        )
        for row in every:
            cells = []
            for i in range(ncol):
                cell = row[i] if i < len(row) else ""
                fill = " " * (widths[i] - visible(cell))
                cells.append(fill + cell if kinds[i] == ">" else cell + fill)
            last = row[ncol - 1] if len(row) == ncol else ""
            under: list[str] = []
            if folds and visible(last) > room and "\x1b" not in last:
                cells[-1], *under = wrap(last, room)
            self._write(" " * indent + (" " * gap).join(cells).rstrip(), stream)
            for line in under:
                self._write(" " * start + line, stream)

    def fields(self, pairs: Sequence[tuple[str, str]], *, indent: int = 2) -> None:
        """A two-column key/value table on stdout."""
        self.table(pairs, indent=indent, stream=self.out)


class Activity:
    """The progress line of work in hand, from `Output.working`.

    `show` names the work; a thread redraws the line as its time passes.
    `done` ends a piece of work with a status line that stays, and says how
    long it took. A message may be a function of the seconds its verb has
    been showing, to say the time in words of its own: it is given None
    where the line is written once, in a log.
    """

    def __init__(
        self,
        out: Output,
        *,
        log: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Binds the line to its renderer; nothing is drawn until `show`."""
        self.out, self.log, self.clock = out, log, clock
        self._verb: str | None = None
        self._message: str | Callable[[float | None], str] = ""
        self._since = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def show(self, verb: str, message: str | Callable[[float | None], str]) -> None:
        """Draws `verb message`, and keeps its time from the first call with this verb."""
        with self.out._lock:
            if verb != self._verb:
                self._verb, self._since = verb, self.clock()
            self._message = message
            self._draw()
        if self._thread is None and self.out._live() and not self.out.quiet:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def seconds(self) -> float:
        """How long the verb showing has been showing."""
        return self.clock() - self._since if self._verb is not None else 0.0

    def done(self, verb: str, message: str) -> None:
        """Replaces the line with a status line that says how long its work took."""
        with self.out._lock:
            taken = self.seconds()
            self._verb = None
            self.out.status(verb, message, seconds=taken)

    def close(self) -> None:
        """Stops redrawing; the line stays for the next line written to replace."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _draw(self) -> None:
        with self.out._lock:
            if self._verb is None or self._stop.is_set():
                return
            waited, live = int(self.seconds()), self.out._live()
            message = self._message
            if isinstance(message, str):
                trail = f"  {elapsed(waited)}" if waited >= 1 and live else ""
                self.out.progress(self._verb, message, log=self.log, trail=trail)
            else:
                text = message(waited if live else None)
                self.out.progress(self._verb, text, log=self.log)

    def _tick(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            self._draw()


class Prompter:
    """Asks `gf init`'s questions one at a time, and asks to confirm a change.

    On a terminal the prompt is a selector drawn in raw mode: `>` marks the
    cursor, up and down move it, space toggles a multi-select row, Enter
    confirms and Escape aborts. Without a terminal the prompt falls back to
    numbered lines read from stdin. `yes` takes the first option everywhere
    and `no_input` raises instead of asking.
    """

    def __init__(
        self,
        out: Output,
        *,
        stdin: IO[str] | None = None,
        yes: bool = False,
        no_input: bool = False,
        interactive: bool | None = None,
    ) -> None:
        """Binds the prompter to the renderer and the input stream."""
        self.out = out
        self.stdin = stdin if stdin is not None else sys.stdin
        self.yes, self.no_input = yes, no_input
        if interactive is None:
            interactive = (
                getattr(self.stdin, "isatty", lambda: False)()
                and getattr(out.err, "isatty", lambda: False)()
                and os.name == "posix"
            )
        self.interactive = interactive and not yes and not no_input

    def _unavailable(self, label: str, flag: str) -> Error:
        return InvalidInput(
            f"{label.lower()} could not be inferred and no terminal is attached",
            hint=f"pass '{flag}', or run without --no-input in a terminal",
            param=flag.lstrip("-"),
        )

    def choose(
        self, label: str, options: Sequence[Option], *, flag: str, default: int = 0
    ) -> Any:
        """One value from `options`, with the row at `default` pre-selected."""
        if not options:
            raise InvalidInput(
                f"no column can be the {label.lower()}",
                hint=f"pass '{flag}' with a column of the table",
                param=flag.lstrip("-"),
            )
        if self.yes:
            chosen = options[default]
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            chosen = self._plain(label, options, default)
        else:
            chosen = options[self._raw(label, options, default, multi=None)[0]]
        self.out.collapsed(label, chosen.shown or chosen.text)
        return chosen.value

    def choose_many(
        self,
        label: str,
        options: Sequence[Option],
        *,
        flag: str,
        selected: Iterable[int] = (),
        noun: str | None = None,
    ) -> list[Any]:
        """Several values from `options`, with `selected` rows toggled on first.

        With no options there is nothing to ask, and the answer is none.
        `noun` names what is chosen, to count an answer too long for a line.
        """
        picked = set(selected)
        if not options:
            indexes: list[int] = []
        elif self.yes:
            indexes = sorted(picked)
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            indexes = self._plain_many(label, options, picked)
        else:
            indexes = self._raw(label, options, 0, multi=picked)
        values = [options[i] for i in indexes]
        self.out.collapsed(
            label, [option.text for option in values] or "none", noun=noun
        )
        return [option.value for option in values]

    def ask(
        self,
        label: str,
        default: str,
        *,
        flag: str,
        note: Callable[[str], str] | None = None,
        check: Callable[[str], str | None] | None = None,
    ) -> str:
        """One typed line, pre-filled with `default`.

        `check` names what is wrong with a value, or returns None for a good
        one: a terminal shows the problem beside the line and keeps editing,
        and typed input raises it. `note` is a quiet aside on a good value.
        """

        def problem(value: str) -> str | None:
            return check(value) if check is not None else None

        def aside(value: str) -> str:
            return note(value) if note is not None and problem(value) is None else ""

        if self.yes:
            value = default
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            value = self._answer(
                f"{label} [{default}]: ",
                Aborted(
                    f"{label.lower()} was not answered",
                    hint=f"pass '{flag}', or answer on stdin",
                ),
            )
            value = value or default
            wrong = problem(value)
            if wrong is not None:
                raise InvalidInput(
                    f"{label.lower()} {wrong}",
                    hint=f"pass '{flag}'",
                    param=label.lower(),
                )
        else:
            value = self._edit(label, default, aside, problem)
        self.out.collapsed(label, value, aside(value))
        return value

    def confirm(
        self,
        question: str,
        *,
        flag: str = "--yes",
        declined: str = "Nothing was changed.",
    ) -> None:
        """Asks `question [y/N]` on stderr; goes on only if the answer is yes.

        Raises:
            Declined: The answer was anything else; `declined` is the line
                that says what did not happen.
            Aborted: There was nobody to ask, which is an error.
        """
        if self.yes:
            return
        unanswered = Aborted(
            f"`{question}` needs an answer",
            hint=f"pass {flag} to go ahead without asking",
        )
        if self.no_input:
            raise unanswered
        answer = self._answer(f"{question} [y/N] ", unanswered)
        if answer.lower() not in ("y", "yes"):
            raise Declined("cancelled; nothing was changed", said=declined)

    def _answer(self, prompt: str, unanswered: Aborted) -> str:
        """One line read from stdin after `prompt`; `unanswered` when stdin ends.

        The prompt's line is ended either way, so what follows starts a line.
        """
        self.out.err.write(prompt)
        self.out.err.flush()
        # Until Enter ends it, the line the question is on is still open.
        self.out._asking = True
        read = self.stdin.readline()
        self.out._asking = False
        if not read:
            self.out.err.write("\n")
            self.out.err.flush()
            raise unanswered
        return read.strip()

    def _plain(self, label: str, options: Sequence[Option], default: int) -> Option:
        err = self.out.err
        err.write(f"{label}\n")
        for i, option in enumerate(options, 1):
            err.write(
                f"  {i}. {option.text}{'  ' + option.detail if option.detail else ''}\n"
            )
        text = self._answer(
            f"Choose [{default + 1}]: ",
            Aborted(
                f"{label.lower()} was not chosen", hint="answer on stdin, or pass --yes"
            ),
        )
        if not text:
            return options[default]
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        for option in options:
            if option.text == text:
                return option
        raise InvalidInput(f"`{text}` is not one of the {label.lower()} choices")

    def _plain_many(
        self, label: str, options: Sequence[Option], picked: set[int]
    ) -> list[int]:
        err = self.out.err
        err.write(f"{label}\n")
        for i, option in enumerate(options, 1):
            mark = "x" if i - 1 in picked else " "
            err.write(f"  {i}. [{mark}] {option.text}\n")
        text = self._answer(
            "Choose (comma-separated numbers or names, Enter keeps the marked rows): ",
            Aborted(
                f"{label.lower()} were not chosen",
                hint="answer on stdin, or pass --yes",
            ),
        )
        if not text:
            return sorted(picked)
        names = {option.text: i for i, option in enumerate(options)}
        chosen = []
        for part in [p.strip() for p in text.split(",") if p.strip()]:
            if part.isdigit() and 1 <= int(part) <= len(options):
                chosen.append(int(part) - 1)
            elif part in names:
                chosen.append(names[part])
            else:
                raise InvalidInput(
                    f"`{part}` is not one of the {label.lower()} choices"
                )
        return sorted(set(chosen))

    def _byte(self, fd: int, *, wait: bool) -> bytes:
        if not wait and not select.select([fd], [], [], ESCAPE_SECONDS)[0]:
            return b""
        return os.read(fd, 1)

    def _key(self) -> str:
        """One key: a character, `up`, `down`, `escape` or `other`.

        A character is read whole, however many UTF-8 bytes it takes. An
        escape sequence is read to its final byte, so keys such as Delete or
        Home leave nothing behind.
        """
        fd = self.stdin.fileno()
        first = self._byte(fd, wait=True)
        if first != b"\x1b":
            lead = first[0] if first else 0
            extra = (
                3 if lead >= 0xF0 else 2 if lead >= 0xE0 else 1 if lead >= 0xC0 else 0
            )
            data = first
            for _ in range(extra):
                data += self._byte(fd, wait=False)
            return data.decode("utf-8", "replace")
        kind = self._byte(fd, wait=False)
        if not kind:
            return "escape"
        sequence = kind
        if kind in (b"[", b"O"):
            while True:
                byte = self._byte(fd, wait=False)
                sequence += byte
                if not byte or 0x40 <= byte[0] <= 0x7E:
                    break
        return {b"[A": "up", b"OA": "up", b"[B": "down", b"OB": "down"}.get(
            sequence, "other"
        )

    def _raw_session(
        self, draw: Callable[[int, int], list[str]], handle: Callable[[str], bool]
    ) -> None:
        import termios
        import tty

        fd = self.stdin.fileno()
        saved = termios.tcgetattr(fd)
        err = self.out.err
        try:
            tty.setcbreak(fd)
            while True:
                columns, lines = size(err)
                # Every line fits the width, so the cursor climbs exactly
                # as many lines as the frame took.
                frame = [
                    clip(line, max(columns - 1, 1)) for line in draw(columns, lines)
                ]
                err.write("".join(f"{line}\n" for line in frame) + FRAME_END)
                err.flush()
                key = self._key()
                err.write(f"\x1b[{len(frame)}A\r\x1b[J" if frame else "\r\x1b[J")
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in ("escape", "\x04"):
                    raise Declined(
                        "cancelled; nothing was written", said="Nothing was written."
                    )
                if handle(key):
                    break
        finally:
            err.write("\x1b[?25h")
            err.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _raw(
        self,
        label: str,
        options: Sequence[Option],
        default: int,
        *,
        multi: set[int] | None,
    ) -> list[int]:
        state = {"cursor": default}

        def draw(columns: int, lines: int) -> list[str]:
            # The rows that fit under the label, scrolled to keep the cursor in view.
            room = max(lines - 2, 1)
            first = min(
                max(state["cursor"] - room // 2, 0), max(len(options) - room, 0)
            )
            marks = 0 if multi is None else 4
            across = max(columns - MARGIN - 4 - marks, 1)
            # Details share a column, set by the widest name that leaves them
            # room; a name wider than that keeps its detail only if it fits.
            column = _shared(
                [(visible(o.text), visible(o.detail)) for o in options], across, 4
            )
            head = label
            if multi is not None:
                chosen = f"{len(multi)} of {len(options)} selected"
                head = f"{label}  {self.out.quiet_text(chosen)}"
            frame = [head]
            for i in range(first, min(first + room, len(options))):
                option = options[i]
                cursor = ">" if i == state["cursor"] else " "
                mark = "" if multi is None else ("[x] " if i in multi else "[ ] ")
                text = shorten(option.text, across)
                detail = self.out.quiet_text(option.detail)
                if detail and visible(text) <= column:
                    text = f"{pad(text, column)}    {detail}"
                elif detail and visible(text) + 4 + visible(detail) <= across:
                    text = f"{text}    {detail}"
                frame.append(f"  {cursor} {mark}{text}")
            return frame

        def handle(key: str) -> bool:
            if key in ("up", "k"):
                state["cursor"] = (state["cursor"] - 1) % len(options)
            elif key in ("down", "j", "\t"):
                state["cursor"] = (state["cursor"] + 1) % len(options)
            elif key == " " and multi is not None:
                multi.symmetric_difference_update({state["cursor"]})
            elif key in ("\r", "\n"):
                return True
            return False

        self._raw_session(draw, handle)
        return sorted(multi) if multi is not None else [state["cursor"]]

    def _edit(
        self,
        label: str,
        default: str,
        aside: Callable[[str], str],
        problem: Callable[[str], str | None],
    ) -> str:
        state = {"text": default}

        def draw(columns: int, lines: int) -> list[str]:
            wrong = problem(state["text"])
            if wrong is None:
                return self.out.answer(
                    label, state["text"], aside(state["text"]), columns=columns
                )

            def red(text: str) -> str:
                return self.out.style(text, "red", stream=self.out.err)

            return self.out.answer(
                label, state["text"], wrong, columns=columns, style=red
            )

        def handle(key: str) -> bool:
            if key in ("\r", "\n"):
                return problem(state["text"]) is None
            if key in ("\x7f", "\b"):
                state["text"] = state["text"][:-1]
            elif len(key) == 1 and key.isprintable():
                state["text"] += key
            return False

        self._raw_session(draw, handle)
        return state["text"]
