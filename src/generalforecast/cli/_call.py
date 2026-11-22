# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""One run of a command: its parsed values, the client, the output and the time taken."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

from generalforecast import term

if TYPE_CHECKING:
    from generalforecast.cli._spec import Command
    from generalforecast.client import Client


class Call:
    """A command being run.

    `args` are its positional arguments and `values` the SDK keyword
    arguments its flags gave, as the SDK takes them; `options` holds the
    flags the CLI keeps for itself (`--yes`, `--output`). `before` is what
    the command's `before` read. `state` keeps what one step of a view
    hands the next, such as an estimate shown when the experiment is created.
    """

    def __init__(
        self,
        command: Command,
        out: term.Output,
        *,
        args: list[Any],
        values: dict[str, Any],
        options: dict[str, Any],
        connect: Callable[[Call], Client],
    ) -> None:
        """Binds the command to its values; the client is made on first use."""
        self.command, self.out = command, out
        self.args, self.values, self.options = args, values, options
        self._connect = connect
        self.started = term.clock()
        self.before: Any = None
        self.state: dict[str, Any] = {}

    @cached_property
    def client(self) -> Client:
        """The client this run calls, made on first use."""
        return self._connect(self)

    @property
    def name(self) -> str:
        """The first argument: the name of what the command is about."""
        return str(self.args[0])

    @property
    def json(self) -> bool:
        """Whether `--json` was given."""
        return bool(self.options.get("json"))

    def method(self, path: str | None = None) -> Callable[..., Any]:
        """The SDK method at `path`, by default the command's own, on this run's client."""
        target: Any = self.client
        for part in (path or self.command.path).split("."):
            target = getattr(target, part)
        return target

    def require(self, *keywords: str) -> None:
        """Refuses a command line that leaves out a flag for a keyword the SDK method needs.

        Raises:
            Usage: Naming the first flag left out.
        """
        from generalforecast.cli._spec import Usage

        for keyword in keywords:
            if keyword not in self.values:
                flag = next(f for f in self.command.flags if f.keyword == keyword)
                raise Usage(
                    f"`{self.command.line}` needs {flag.option}",
                    hint=f"`gf {' '.join(self.command.words)} --help` shows what it takes",
                )

    def invoke(self, **changes: Any) -> Any:
        """Calls the command's SDK method with its arguments, `changes` over its keywords."""
        return self.method()(*self.args, **{**self.values, **changes})

    def seconds(self) -> float:
        """Seconds since the command started."""
        return term.clock() - self.started
