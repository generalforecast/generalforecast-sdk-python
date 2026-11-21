# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`Client`, `AsyncClient`, the module's default client, and login.

A client resolves its server, credential and project once (see
`generalforecast._auth`), sends every request with `X-GF-Origin`, and runs
the operations in `generalforecast._ops`: `Client` in the calling thread,
`AsyncClient` on an event loop. While a call waits it shows one updating
line on stderr, by default only in a notebook or on a terminal.
"""

from __future__ import annotations

import asyncio
import getpass
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Self

from generalforecast import _auth, _http, _ops, _origin
from generalforecast.errors import Error, InvalidInput, Timeout, Unauthorized
from generalforecast.types import Me

__all__ = ["AsyncClient", "Client", "login", "logout", "whoami"]


class _Line:
    """One progress line on stderr, redrawn in place, with the time waited."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.shown = 0
        self.started = time.monotonic()

    def start(self) -> None:
        self.started = time.monotonic()

    def show(self, text: str | None) -> None:
        if not self.enabled:
            return
        if text is None:
            self.clear()
            return
        waited = time.monotonic() - self.started
        if waited >= 1:
            from generalforecast._render import seconds

            text = f"{text}  {seconds(waited)}"
        pad = " " * max(0, self.shown - len(text))
        sys.stderr.write(f"\r{text}{pad}")
        sys.stderr.flush()
        self.shown = len(text)

    def clear(self) -> None:
        if self.enabled and self.shown:
            sys.stderr.write("\r" + " " * self.shown + "\r")
            sys.stderr.flush()
        self.shown = 0


def _not_logged_in() -> Unauthorized:
    return Unauthorized(
        "not logged in",
        hint="run gf.login(), or set GF_API_KEY for unattended use",
    )


class _Base:
    """What both clients share: the resolved connection and what they remember."""

    def __init__(
        self,
        *,
        api_key: str | None,
        project: str | None,
        url: str | None,
        timeout: float,
        retries: int,
        progress: bool | None,
        trust_env: bool,
        origin: str | None,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise InvalidInput(
                "timeout is a positive number of seconds", param="timeout"
            )
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise InvalidInput("retries is a whole number, 0 or more", param="retries")
        resolved = _auth.resolve(api_key, project, url)
        from generalforecast import __version__

        self._connection = _http.Connection(
            url=resolved.url,
            token=resolved.token,
            project=resolved.project,
            origin=origin or _origin.detect().header(),
            timeout=float(timeout),
            retries=retries,
            trust_env=trust_env,
            user_agent=f"generalforecast-python/{__version__}",
        )
        self.credential_source = resolved.source
        self._line = _Line(_origin.interactive() if progress is None else progress)
        self._state = _ops.State()
        self._sleep: Callable[[float], Any] = time.sleep

    @property
    def url(self) -> str:
        """The server this client talks to."""
        return self._connection.url

    @property
    def project(self) -> str | None:
        """The project; for a key with no project given, known after the first call."""
        return self._connection.project

    def _check(self, request: _http.Request) -> bool:
        """Whether the project must be learned from the key before sending."""
        if request.url is not None or (not request.envelope and not request.scoped):
            return False
        if not self._connection.token:
            raise _not_logged_in()
        return request.scoped and not self._connection.project

    def _learned(self, reply: _http.Reply) -> None:
        me = _ops.me_of(reply.result or {}, self._connection.url)
        if not me.project:
            raise InvalidInput(
                "this key names no project", hint="pass project= or set GF_PROJECT"
            )
        self._connection.project = me.project


class Client(_Base):
    """A connection to General Forecast; its resources make every call.

    Credentials resolve in order: `api_key=`, `gf.login()` in this process,
    `GF_API_KEY`, then the credentials file `gf login` writes. `GF_PROJECT`
    and `GF_URL` override the project and server. Requests honour
    `HTTPS_PROXY` and the system CA bundle unless `trust_env=False`. Safe
    requests are retried `retries` times after a transient failure. Use it
    as a context manager, or call `close`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        url: str | None = None,
        timeout: float = 40.0,
        retries: int = 3,
        progress: bool | None = None,
        trust_env: bool = True,
        origin: str | None = None,
    ) -> None:
        """Resolves the connection; nothing is sent until a resource is called."""
        super().__init__(
            api_key=api_key,
            project=project,
            url=url,
            timeout=timeout,
            retries=retries,
            progress=progress,
            trust_env=trust_env,
            origin=origin,
        )
        from generalforecast import resources

        self._transport = _http.Transport(self._connection)
        self.contexts = resources.Contexts(self)
        self.experiments = resources.Experiments(self)
        self.predictors = resources.Predictors(self)
        self.sources = resources.Sources(self)
        self.keys = resources.Keys(self)
        self.models = resources.Models(self)

    def __enter__(self) -> Self:
        """Returns the client."""
        return self

    def __exit__(self, *_: object) -> None:
        """Closes the client."""
        self.close()

    def close(self) -> None:
        """Closes the connection pool; later calls raise `InvalidInput`."""
        self._transport.close()

    def __repr__(self) -> str:
        """The server and project."""
        return f"Client(url={self.url!r}, project={self.project!r})"

    def _send(self, request: _http.Request) -> _http.Reply:
        if self._check(request):
            self._learned(self._transport.send(_http.auth_status()))
        return self._transport.send(request)

    def _step(self, step: Any) -> Any:
        if isinstance(step, _http.Request):
            return self._send(step)
        if isinstance(step, _ops.Sleep):
            self._sleep(step.seconds)
        elif isinstance(step, _ops.Say):
            self._line.show(step.text)
        elif isinstance(step, _ops.Notify):
            step.callback(step.value)
        else:
            raise TypeError(f"unknown step {step!r}")
        return None

    def _run(self, op: _ops.Op) -> Any:
        """Runs an operation to its result."""
        self._line.start()
        try:
            step = next(op)
            while True:
                if isinstance(step, _ops.Emit):
                    raise TypeError("an iterating operation needs _iterate")
                try:
                    result = self._step(step)
                except Error as error:
                    step = op.throw(error)
                    continue
                step = op.send(result)
        except StopIteration as stop:
            return stop.value
        finally:
            self._line.clear()

    def _iterate(self, op: _ops.Op) -> Iterator[Any]:
        """Runs an operation, yielding each item it emits."""
        try:
            step = next(op)
            while True:
                if isinstance(step, _ops.Emit):
                    yield step.item
                    step = op.send(None)
                    continue
                try:
                    result = self._step(step)
                except Error as error:
                    step = op.throw(error)
                    continue
                step = op.send(result)
        except StopIteration:
            return
        finally:
            self._line.clear()


class AsyncClient(_Base):
    """`Client` for asyncio: the same resources, every method `async`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        url: str | None = None,
        timeout: float = 40.0,
        retries: int = 3,
        progress: bool | None = None,
        trust_env: bool = True,
        origin: str | None = None,
    ) -> None:
        """Resolves the connection; nothing is sent until a resource is awaited."""
        super().__init__(
            api_key=api_key,
            project=project,
            url=url,
            timeout=timeout,
            retries=retries,
            progress=progress,
            trust_env=trust_env,
            origin=origin,
        )
        from generalforecast import resources

        self._transport = _http.AsyncTransport(self._connection)
        self._sleep = asyncio.sleep
        self.contexts = resources.AsyncContexts(self)
        self.experiments = resources.AsyncExperiments(self)
        self.predictors = resources.AsyncPredictors(self)
        self.sources = resources.AsyncSources(self)
        self.keys = resources.AsyncKeys(self)
        self.models = resources.AsyncModels(self)

    async def __aenter__(self) -> Self:
        """Returns the client."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Closes the client."""
        await self.close()

    async def close(self) -> None:
        """Closes the connection pool."""
        await self._transport.close()

    def __repr__(self) -> str:
        """The server and project."""
        return f"AsyncClient(url={self.url!r}, project={self.project!r})"

    async def _send(self, request: _http.Request) -> _http.Reply:
        if self._check(request):
            self._learned(await self._transport.send(_http.auth_status()))
        return await self._transport.send(request)

    async def _step(self, step: Any) -> Any:
        if isinstance(step, _http.Request):
            return await self._send(step)
        if isinstance(step, _ops.Sleep):
            await self._sleep(step.seconds)
        elif isinstance(step, _ops.Say):
            self._line.show(step.text)
        elif isinstance(step, _ops.Notify):
            outcome = step.callback(step.value)
            if asyncio.iscoroutine(outcome):
                await outcome
        else:
            raise TypeError(f"unknown step {step!r}")
        return None

    async def _run(self, op: _ops.Op) -> Any:
        """Runs an operation to its result."""
        self._line.start()
        try:
            step = next(op)
            while True:
                try:
                    result = await self._step(step)
                except Error as error:
                    step = op.throw(error)
                    continue
                step = op.send(result)
        except StopIteration as stop:
            return stop.value
        finally:
            self._line.clear()

    async def _iterate(self, op: _ops.Op) -> AsyncIterator[Any]:
        """Runs an operation, yielding each item it emits."""
        try:
            step = next(op)
            while True:
                if isinstance(step, _ops.Emit):
                    yield step.item
                    step = op.send(None)
                    continue
                try:
                    result = await self._step(step)
                except Error as error:
                    step = op.throw(error)
                    continue
                step = op.send(result)
        except StopIteration:
            return
        finally:
            self._line.clear()


_LOCK = threading.Lock()
_DEFAULT: Client | None = None


def default() -> Client:
    """The module's client, built on first use."""
    global _DEFAULT
    with _LOCK:
        if _DEFAULT is None:
            _DEFAULT = Client()
        return _DEFAULT


def reset() -> None:
    """Closes the module's client, so the next use resolves credentials again."""
    global _DEFAULT
    with _LOCK:
        if _DEFAULT is not None:
            _DEFAULT.close()
        _DEFAULT = None


def _drive(transport: _http.Transport, op: _ops.Op) -> Any:
    try:
        step = next(op)
        while True:
            try:
                result = transport.send(step)
            except Error as error:
                step = op.throw(error)
                continue
            step = op.send(result)
    except StopIteration as stop:
        return stop.value


def _connection(
    server: str, token: str | None, project: str | None
) -> _http.Connection:
    from generalforecast import __version__

    return _http.Connection(
        url=server,
        token=token,
        project=project,
        origin=_origin.detect().header(),
        user_agent=f"generalforecast-python/{__version__}",
    )


def login(
    *, project: str | None = None, key: str | None = None, url: str | None = None
) -> Me:
    """Logs this process in, holding the credential in memory only.

    Without `key`, runs the device flow: prints a link and a code to
    approve in a signed-in browser, waits, and keeps a token that lasts 12
    hours. If the wait is interrupted or the code expires, asks for an API
    key instead. Nothing is written to disk.

    Raises:
        Unauthorized: When the login is denied or the key is refused.
    """
    import os

    server = _http.server(url or os.environ.get(_auth.URL_ENV) or _http.DEFAULT_URL)
    expires: Any = None
    minted = False
    if key is None:
        transport = _http.Transport(_connection(server, None, None))
        try:
            granted = _drive(transport, _ops.device_login(time.sleep, print))
            key = str(granted.get("access_token") or "")
            expires = granted.get("expires_at")
            project = project or granted.get("project")
            minted = True
        except (KeyboardInterrupt, Timeout):
            print()
            key = getpass.getpass("Paste an API key instead: ").strip()
        finally:
            transport.close()
    if not key:
        raise Unauthorized(
            "no key was given", hint="run gf.login() again, or pass key="
        )
    transport = _http.Transport(_connection(server, key, project))
    try:
        me = _drive(transport, _ops.whoami(server))
    finally:
        transport.close()
    if expires is not None:
        from dataclasses import replace

        from generalforecast.types import instant

        me = replace(me, token_expires=instant(expires))
    project = project or me.project
    if project != me.project:
        from dataclasses import replace

        me = replace(me, project=project, url=f"{server}/?project={project}")
    _auth.SESSION = _auth.Session(
        server, key, project, me.email, me.token_expires, minted
    )
    reset()
    line = f"Logged in as {me.email or 'this key'}, project {project}."
    if me.token_expires is not None:
        holder = "notebook" if _origin.detect().surface == "notebook" else "session"
        until = me.token_expires.astimezone().strftime("%H:%M")
        line += f" This {holder} holds a token until {until}."
    print(line)
    return me


def logout() -> None:
    """Forgets this process's login; a token `gf.login()` minted is revoked too."""
    session = _auth.SESSION
    _auth.SESSION = None
    reset()
    if session is None or not session.minted:
        return
    transport = _http.Transport(
        _connection(session.url, session.token, session.project)
    )
    try:
        transport.send(_http.auth_logout())
    except Error:
        pass
    finally:
        transport.close()


def whoami() -> Me:
    """Who the module's client acts for."""
    client = default()
    return client._run(_ops.whoami(client.url))
