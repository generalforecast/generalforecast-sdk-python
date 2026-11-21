# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Terminal output for the `gf` command.

One renderer draws every surface the CLI shows: status lines with a
right-aligned verb, progress lines for work still going, change lines,
errors as cargo writes them, tables, and the `[y/N]` question a delete
asks. Results go to stdout; status lines, questions and errors go to
stderr. Colour is on only when the stream is a terminal and `NO_COLOR` is
unset, or `FORCE_COLOR` asks for it.

Every line is laid out for the width (`size`): `COLUMNS` when set, else the
terminal's, else `WIDTH`. On a terminal nothing breaks in the middle of a
word: `wrap` folds text after a comma or at a space and `layout` places a
row's key, value and quiet note. A table drops the columns it names, lowest
priority first, while a row does not fit, then folds its last text column.
Off a terminal nothing folds, so a log reads the same wherever it was
written and a status line stays one line, to be searched for.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, NamedTuple

from generalforecast.errors import Error, printable

# Status verbs are right-aligned in this many columns.
VERB_WIDTH = 12
# The rendering size when neither `COLUMNS` nor the terminal gives one.
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
# What progress lines are timed with; a test puts a clock of its own here.
clock: Callable[[], float] = time.monotonic

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
    return tty(stream)


def tty(stream: IO[str] | None) -> bool:
    """Whether a stream is a terminal."""
    try:
        return bool(stream is not None and getattr(stream, "isatty", bool)())
    except ValueError:
        return False


def duration(value: float) -> str:
    """A duration as status lines show it: `0.21s`, `6.8s`, `41s`, `1m12s`, `1h05m`."""
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    if value < 59.5:
        return f"{value:.0f}s"
    return elapsed(round(value))


def elapsed(value: float) -> str:
    """A time waited so far, to the second: `45s`, `2m10s`, `1h05m`."""
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
    """The columns and lines to lay out for: `COLUMNS`, else the terminal's, else `WIDTH` by `HEIGHT`."""
    columns = os.environ.get("COLUMNS", "")
    if columns.isdigit() and int(columns) > 0:
        return os.terminal_size((int(columns), HEIGHT))
    stream = stream or sys.stderr
    if tty(stream):
        try:
            return os.get_terminal_size(stream.fileno())
        except (OSError, ValueError, AttributeError):
            pass
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
    line's end. A command between backticks is never broken: it starts a
    line of its own, and runs past the edge when even that is too narrow.
    """
    lines = []
    for word in text.split(" "):
        joined = f"{line} {word}" if line else word
        if visible(joined) <= width:
            line = joined
            continue
        if "`" in word:
            if line.strip():
                lines.append(line.rstrip())
            line = word
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
    to the fold that is never broken: it starts a line of its own instead,
    and runs past the edge when even that is too narrow. Text that fits comes back as one line, unchanged. Style
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


class Laid(NamedTuple):
    """A table's lines, and how many of its columns did not fit."""

    lines: list[str]
    hidden: int


def columns(
    header: Sequence[str] | None,
    rows: Sequence[Sequence[str]],
    *,
    align: str = "",
    drop: Sequence[int] = (),
    width: int = WIDTH,
    fold: bool = False,
    indent: int = 2,
    gap: int = 2,
) -> Laid:
    """Rows of cells as aligned lines no wider than `width`, where that can be.

    `align` holds `<` or `>` per column (left by default). While a line is
    too wide, the columns in `drop` go, in that order; then, with `fold`, the
    last column folds under itself when it is left-aligned plain text. A
    line that still does not fit runs past the edge. Cells are plain text.
    """
    every = (
        [list(header), *map(list, rows)]
        if header is not None
        else list(map(list, rows))
    )
    if not every:
        return Laid([], 0)
    count_ = max(len(row) for row in every)
    for row in every:
        row.extend([""] * (count_ - len(row)))
    kinds = align.ljust(count_, "<")
    widths = [max(visible(row[i]) for row in every) for i in range(count_)]
    room = width - MARGIN

    def span(kept: list[int]) -> int:
        return indent + sum(widths[i] for i in kept) + gap * (len(kept) - 1)

    kept = list(range(count_))
    for column in drop:
        if span(kept) <= room:
            break
        if column in kept and len(kept) > 1:
            kept.remove(column)
    last = kept[-1]
    start = span(kept[:-1]) + (gap if len(kept) > 1 else 0)
    folds = fold and kinds[last] == "<" and room - start >= MIN_VALUE
    lines = []
    for row in every:
        cells = []
        for i in kept:
            fill = " " * (widths[i] - visible(row[i]))
            cells.append(fill + row[i] if kinds[i] == ">" else row[i] + fill)
        under: list[str] = []
        if folds and visible(row[last]) > room - start:
            cells[-1], *under = wrap(row[last], room - start)
        lines.append((" " * indent + (" " * gap).join(cells)).rstrip())
        lines.extend(" " * start + line for line in under)
    return Laid(lines, count_ - len(kept))


def change_lines(
    items: Sequence[tuple[int, str, str, str]],
    *,
    style: Callable[..., str] = lambda text, *styles: text,
    width: int | None = None,
) -> list[str]:
    """Change lines: `+ key  value`, `~ key  old → new`, `- key  old`, `= key  value`.

    Each item is `(indent, sign, key, value)`; the values of a block share
    one column, after its widest key. `style` colours the sign: `+` green,
    `~` yellow, `-` red, `=` dim. With a `width`, a value too long for it
    folds under itself.
    """
    if not items:
        return []
    start = max(indent + 2 + visible(key) for indent, _, key, _ in items) + 2
    colours = {"+": ("green",), "~": ("yellow",), "-": ("red",), "=": ("dim",)}
    lines = []
    for indent, sign, key, value in items:
        folded = (
            [value]
            if width is None
            else wrap(value, max(width - MARGIN - start, MIN_VALUE))
        )
        head = " " * indent + style(sign, "bold", *colours.get(sign, ()))
        lines.append(pad(f"{head} {key}", start) + folded[0])
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
    """The person said no: nothing changed, and nothing went wrong.

    `said` is the one line that tells them so in the command's terms, such
    as `Nothing was deleted.`; the message is what `--json` reports.
    """

    def __init__(self, message: str, *, said: str) -> None:
        """Keeps the sentence for a person beside the message for a program."""
        super().__init__(message)
        self.said = said


class Output:
    """Writes the CLI's human output to `out` (results) and `err` (everything else)."""

    def __init__(
        self,
        *,
        quiet: bool = False,
        verbose: bool = False,
        out: IO[str] | None = None,
        err: IO[str] | None = None,
        stdin: IO[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Binds the renderer to its streams and colour setting."""
        self.quiet, self.verbose = quiet, verbose
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.stdin = stdin if stdin is not None else sys.stdin
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

    def width(self, stream: IO[str] | None = None) -> int:
        """The columns a stream's lines are laid out for."""
        return size(self.err if stream is None else stream).columns

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
        return tty(self.err)

    def line(self, text: str = "") -> None:
        """One line of a result on stdout."""
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
        tail = "" if seconds is None else f" in {duration(seconds)}"
        for line in self._status(verb, message, tail):
            self._write(line, self.err)

    def _status(self, verb: str, message: str, tail: str = "") -> list[str]:
        """A status line's lines: on a terminal, the message folded under itself.

        Anywhere else a status line is one line however long, so a log can
        be searched for it. A message that carries its own escapes, such as
        a link, is left whole, and so is the `tail`, a duration: it ends the
        last line or takes one of its own.
        """
        head = (
            self.style(verb.rjust(VERB_WIDTH), "bold", "green")
            if verb
            else " " * VERB_WIDTH
        )
        lines = [message]
        if self._live() and "\x1b" not in message:
            room = max(self.width() - MARGIN - VERB_WIDTH - 1, MIN_VALUE)
            lines = wrap(message, room)
            if visible(lines[-1]) + visible(tail) > room:
                lines.append("")
                tail = tail.lstrip()
        lines[-1] += tail
        under = " " * (VERB_WIDTH + 1)
        return [f"{head} {lines[0]}".rstrip(), *(under + line for line in lines[1:])]

    def view(self, url: str | None) -> None:
        """The `View` line that ends a command about one resource: its console URL.

        A URL never folds; on a terminal it is a link, and it may run past the edge.
        """
        if url:
            self.status("View", self.link(url))

    def progress(
        self, verb: str, message: str, *, log: str | None = None, trail: str = ""
    ) -> None:
        """A status line for work still going, such as a wait, drawn as `status` draws one.

        On a terminal each call redraws the line in place and the next line
        written to stderr replaces it; `trail`, such as the time taken so
        far, ends the line and is never what is cut to fit. On any other
        stream the first call with a verb writes `log` (by default the
        message), and the calls after it with the same verb write nothing,
        so a log holds one line. Dropped under `-q`.
        """
        if self.quiet:
            return
        head = self.style(verb.rjust(VERB_WIDTH), "bold", "green")
        with self._lock:
            if not self._live():
                if self._progress != verb:
                    text = message if log is None else log
                    self._write(f"{head} {text}".rstrip(), self.err)
                    self._progress = verb
                return
            room = self.width() - MARGIN - visible(trail)
            line = shorten(f"{head} {message}", max(room, 1)) + trail
            self.err.write(f"\r\x1b[K{line}")
            self.err.flush()
            self._progress = verb

    @contextlib.contextmanager
    def working(self) -> Generator[Activity]:
        """Progress lines for work that takes a while, kept current until the block ends.

        The `Activity` it yields shows a line with `show(verb, message)`. On
        a terminal the line is redrawn every `TICK_SECONDS` with how long
        that verb has been showing, as `Waiting for approval  12s`, so a
        person sees which step the time is going on. A log gets each verb's
        line once, without the time.
        """
        activity = Activity(self)
        try:
            yield activity
        finally:
            activity.close()

    def changes(self, items: Sequence[tuple[int, str, str, str]]) -> None:
        """Change lines on stderr (`change_lines`), folded on a terminal; dropped under `-q`."""
        if self.quiet:
            return
        width = self.width() if self._live() else None
        for line in change_lines(items, style=self.style, width=width):
            self._write(line, self.err)

    def details(
        self,
        rows: Sequence[tuple[str, str | Sequence[str], str]],
        *,
        indent: int = 2,
        stream: IO[str] | None = None,
        force: bool = False,
    ) -> None:
        """Rows of key, value and quiet note, laid out for the width (`layout`).

        A value given as a list is a list of names: on a terminal, past
        `LIST_LINES` lines the rest are counted as `and N more`, and `-v`
        shows them all. Off a terminal nothing folds. On stderr this is
        commentary, dropped under `-q` unless forced.
        """
        stream = self.err if stream is None else stream
        if stream is self.err and self.quiet and not force:
            return
        folds = tty(stream)
        lines = layout(
            rows,
            self.width(stream) if folds else sys.maxsize,
            indent=indent,
            limit=None if self.verbose or not folds else LIST_LINES,
            quiet=lambda text: self.quiet_text(text, stream),
        )
        for line in lines:
            self._write(line, stream)

    def table(
        self,
        header: Sequence[str] | None,
        rows: Sequence[Sequence[str]],
        *,
        align: str = "",
        drop: Sequence[int] = (),
        stream: IO[str] | None = None,
    ) -> int:
        """A table on stdout (`columns`): it drops columns to fit and folds only on a terminal.

        Returns how many columns did not fit.
        """
        stream = self.out if stream is None else stream
        laid = columns(
            header,
            rows,
            align=align,
            drop=drop,
            width=self.width(stream),
            fold=tty(stream),
        )
        for line in laid.lines:
            self._write(line, stream)
        return laid.hidden

    def _diagnostic(self, lines: list[str]) -> None:
        for line in lines:
            self._write(line, self.err)

    def _styler(self, failure: bool) -> Callable[[str, str], str]:
        """Diagnostic labels in their colours: a real failure's `error:` red, a recoverable one's bold."""

        def styled(text: str, kind: str) -> str:
            if kind == "error":
                return self.style(text, "bold", *(("red",) if failure else ()))
            if kind == "warning":
                return self.style(text, "bold", "yellow")
            if kind == "location":
                return self.quiet_text(text)
            return self.style(text, "bold")

        return styled

    def error(
        self,
        message: str,
        *,
        notes: Sequence[str] = (),
        hint: str | None = None,
        tip: str | None = None,
        pointer: Pointer | None = None,
        failure: bool = False,
    ) -> None:
        """An error as cargo writes one (`generalforecast.diagnostics`).

        `error: message`, with no code; a `-->` pointer into a file, with
        its source line and carets; a `note:` for each of `notes`, such as
        why, what a service answered and the request's id; a `help:` naming
        a near miss (`tip`), and one for what to do instead. A real `failure`
        (the service failed, or a run or deploy did) has its `error:` in
        red; one the person can fix, bold. On a terminal a line too long
        for it folds under its own text.
        """
        from generalforecast.diagnostics import lines

        styled = self._styler(failure)
        self._diagnostic(lines("error", self._fold("error", message), style=styled))
        if pointer is not None:
            self.pointer(pointer)
        self._diagnostic(
            lines(
                "error",
                "",
                notes=[self._fold("note", note) for note in notes],
                help=[self._fold("help", text or "") for text in (tip, hint)],
                style=styled,
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
        room = self.width() - MARGIN - len(label) - 2
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
            label = wrap(where.label, max(self.width() - MARGIN - start, 1))
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
                    style=self._styler(False),
                )
            )

    def confirm(self, question: str, *, yes: bool, flag: str = "--yes") -> None:
        """Asks `question [y/N]` on a terminal; goes on only if the answer is yes.

        `yes` goes on without asking. With no terminal to ask on, it is an
        error to go on: the command needs `flag`.

        Raises:
            Declined: The answer was anything but yes.
            Aborted: There was nobody to ask.
        """
        if yes:
            return
        if not (tty(self.stdin) and tty(self.err)):
            raise Aborted(
                "there is no terminal to ask before changing anything",
                hint=f"pass {flag} to go ahead without asking",
            )
        with self._lock:
            if self._progress is not None:
                self.err.write("\r\x1b[K")
                self._progress = None
            self.err.write(f"  {question} [y/N] ")
            self.err.flush()
            self._asking = True
        answer = self.stdin.readline()
        with self._lock:
            # A terminal echoed the Enter that ended the answer; Ctrl-D has none.
            if not answer.endswith("\n"):
                self.err.write("\n")
                self.err.flush()
            self._asking = False
        if answer.strip().lower() not in ("y", "yes"):
            raise Declined(
                "cancelled; nothing was changed", said="Nothing was changed."
            )


class Activity:
    """The progress line of work in hand, from `Output.working`.

    `show` names the work; on a terminal a thread redraws the line as its
    time passes. `done` ends a piece of work with a status line that stays,
    and says how long it took.
    """

    def __init__(self, out: Output) -> None:
        """Binds the line to its renderer; nothing is drawn until `show`."""
        self.out = out
        self._verb: str | None = None
        self._message = ""
        self._log: str | None = None
        self._since = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def show(self, verb: str, message: str, *, log: str | None = None) -> None:
        """Draws `verb message`, and keeps its time from the first call with this verb.

        A log gets `log`, by default the message, once per verb.
        """
        with self.out._lock:
            if verb != self._verb:
                self._verb, self._since = verb, clock()
            self._message, self._log = message, log
            self._draw()
        if self._thread is None and self.out._live() and not self.out.quiet:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def seconds(self) -> float:
        """How long the verb showing has been showing."""
        return clock() - self._since if self._verb is not None else 0.0

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
            waited = int(self.seconds())
            trail = f"  {elapsed(waited)}" if waited >= 1 else ""
            self.out.progress(self._verb, self._message, log=self._log, trail=trail)

    def _tick(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            self._draw()
