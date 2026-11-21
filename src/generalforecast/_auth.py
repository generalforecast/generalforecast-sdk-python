# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Where a client's credential, project and server come from.

In order: `api_key=` given to the client, `gf.login()` in this process,
`GF_API_KEY`, then the credentials file `gf login` writes
(`~/.config/gf/credentials.toml`, its key in the macOS keychain when there).
`GF_PROJECT` and `GF_URL` override the project and the server. Nothing
else is read.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from generalforecast import _http
from generalforecast._files import private_lock, read_private, write_private
from generalforecast.errors import Error, InvalidInput

KEY_ENV = "GF_API_KEY"
PROJECT_ENV = "GF_PROJECT"
URL_ENV = "GF_URL"
KEYCHAIN_SERVICE = "generalforecast"
_LIMIT = 65536


@dataclass
class Session:
    """A login held in this process's memory by `gf.login()`; never written to disk."""

    url: str
    token: str
    project: str | None
    email: str | None = None
    expires: datetime | None = None
    minted: bool = False


SESSION: Session | None = None


@dataclass
class Stored:
    """A login the credentials file holds for one server.

    `name` is the key's name; `minted` says `gf login` made the key, so
    `gf logout` revokes it, where a key pasted in is only forgotten.
    """

    server: str
    key_id: str
    project: str | None = None
    email: str | None = None
    workspace: str | None = None
    expires: str | None = None
    store: str = "file"
    key: str | None = None
    name: str | None = None
    minted: bool = False


def config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The gf configuration directory."""
    env = os.environ if environ is None else environ
    home = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(home) / "gf"


def _toml(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


class CredentialStore:
    """Reads and writes logins in credentials.toml and, on macOS, the keychain."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        """Uses the real environment unless a mapping is given."""
        self.env = dict(os.environ if environ is None else environ)
        self.path = config_dir(self.env) / "credentials.toml"

    def keychain(self) -> bool:
        """Whether the macOS keychain is usable."""
        if self.env.get("GF_KEYCHAIN") == "0":
            return False
        return sys.platform == "darwin" and shutil.which("security") is not None

    def _read(self) -> dict[str, dict]:
        raw = read_private(self.path, limit=_LIMIT, missing_ok=True)
        if raw is None:
            return {}
        try:
            data = tomllib.loads(raw.decode())
        except (UnicodeError, tomllib.TOMLDecodeError):
            raise InvalidInput(
                f"{self.path} is not valid TOML",
                hint="fix it, or remove it and run `gf login` again",
            ) from None
        servers = data.get("servers", {})
        return servers if isinstance(servers, dict) else {}

    def _write(self, servers: dict[str, dict]) -> None:
        lines = [
            "# Written by gf login. Keys here are secrets; keep this file private.",
            "",
        ]
        for server, values in sorted(servers.items()):
            lines.append(f"[servers.{_toml(server)}]")
            lines += [
                f"{k} = {str(v).lower() if isinstance(v, bool) else _toml(str(v))}"
                for k, v in values.items()
                if v is not None
            ]
            lines.append("")
        write_private(self.path, "\n".join(lines).encode())

    def load(self, server: str) -> Stored | None:
        """The stored login for a server, with its key when readable."""
        values = self._read().get(server)
        if not isinstance(values, dict) or "key_id" not in values:
            return None
        store = values.get("store", "file")
        key = values.get("key") if store == "file" else self._keychain_read(server)
        return Stored(
            server=server,
            key_id=str(values["key_id"]),
            project=values.get("project"),
            email=values.get("email"),
            workspace=values.get("workspace"),
            expires=values.get("expires"),
            store=store,
            key=key,
            name=values.get("name"),
            minted=values.get("minted") is True,
        )

    def save(self, login: Stored) -> Stored:
        """Stores the key in the keychain when possible, else in the file."""
        if not login.key:
            raise InvalidInput("a stored login needs its key")
        login.store = (
            "keychain" if self.keychain() and self._keychain_write(login) else "file"
        )
        values = {k: v for k, v in asdict(login).items() if k not in ("server", "key")}
        if login.store == "file":
            values["key"] = login.key
        with private_lock(self.path):
            servers = self._read()
            servers[login.server] = values
            self._write(servers)
        return login

    def remove(self, server: str) -> bool:
        """Forgets a server's login; true when there was one."""
        with private_lock(self.path):
            servers = self._read()
            found = servers.pop(server, None)
            if found is not None:
                self._write(servers)
        if found is not None and found.get("store") == "keychain":
            self._security(
                "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", server
            )
        return found is not None

    def _security(
        self, *args: str, secret: str | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["security", *args],
            input=secret,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def _keychain_write(self, login: Stored) -> bool:
        assert login.key is not None
        if any(c in value for value in (login.server, login.key) for c in '"\\\n '):
            return False
        command = (
            f"add-generic-password -U -s {KEYCHAIN_SERVICE} "
            f'-a "{login.server}" -w "{login.key}"\n'
        )
        return self._security("-i", secret=command).returncode == 0

    def _keychain_read(self, server: str) -> str | None:
        if not self.keychain():
            return None
        result = self._security(
            "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", server, "-w"
        )
        return (result.stdout.strip() or None) if result.returncode == 0 else None


@dataclass(frozen=True)
class Resolved:
    """A client's connection: server, credential and where it came from, and the project."""

    url: str
    token: str | None
    project: str | None
    source: str


def resolve(
    api_key: str | None,
    project: str | None,
    url: str | None,
    environ: Mapping[str, str] | None = None,
) -> Resolved:
    """The server, credential and project a new client uses."""
    env = os.environ if environ is None else environ
    session = SESSION
    server = _http.server(
        url
        or env.get(URL_ENV)
        or (session.url if session else None)
        or _http.DEFAULT_URL
    )
    project = project or env.get(PROJECT_ENV) or None
    if api_key:
        return Resolved(server, api_key, project, "api_key")
    if session is not None and session.url == server:
        return Resolved(server, session.token, project or session.project, "login")
    if env.get(KEY_ENV):
        return Resolved(server, env[KEY_ENV], project, KEY_ENV)
    try:
        stored = CredentialStore(env).load(server)
    except Error:
        stored = None
    if stored is not None and stored.key:
        return Resolved(server, stored.key, project or stored.project, "credentials")
    return Resolved(server, None, project, "none")
