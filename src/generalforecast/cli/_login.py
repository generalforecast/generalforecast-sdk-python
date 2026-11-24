# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`gf login`, `gf logout` and `gf whoami`: the login this machine keeps.

`gf login` runs the device flow (section 3.6) as the CLI, which mints a
90-day key named `cli-<machine>`, or takes a key pasted at a prompt or
piped in, and keeps it in the credentials file (`generalforecast._auth`)
for every later command. `gf logout` forgets it, and revokes a key that
`gf login` minted.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Any

from generalforecast import _auth, _http, _ops, term
from generalforecast.cli._call import Call
from generalforecast.client import _drive
from generalforecast.errors import Error, Unauthorized
from generalforecast.types import Me


def server(call: Call) -> str:
    """The server to log in to: `--url`, else `GF_URL`, else the default."""
    return _http.server(
        call.options.get("url") or os.environ.get(_auth.URL_ENV) or _http.DEFAULT_URL
    )


def _transport(url: str, token: str | None, project: str | None) -> _http.Transport:
    from generalforecast._origin import Detected
    from generalforecast.client import connection

    return _http.Transport(connection(url, token, project, Detected("cli").header()))


def _browser() -> bool:
    """Whether a browser can open here: not over SSH, not without a display, not under `GF_NO_BROWSER`."""
    import sys

    env = os.environ
    if env.get("GF_NO_BROWSER") or env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return False
    return not sys.platform.startswith("linux") or bool(
        env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")
    )


def _grant(call: Call, grant: dict[str, Any]) -> None:
    """Shows where to approve the login, opening the browser on a terminal where one can open."""
    import webbrowser

    out = call.out
    url = grant.get("verification_uri_complete") or grant["verification_uri"]
    opened = term.tty(out.err) and _browser() and webbrowser.open(url, new=2)
    out.status("Opening" if opened else "Visit", out.link(url))
    out.status(
        "",
        f"Confirm the code {grant['user_code']} in the browser where you are signed in",
    )


def _pasted(call: Call) -> str:
    """A key typed at a prompt that does not echo it, or piped in."""
    out = call.out
    if term.tty(out.stdin):
        return getpass.getpass("  Paste an API key: ").strip()
    return out.stdin.readline().strip()


def login(call: Call) -> Me:
    """Logs this machine in and keeps the key in the credentials file."""
    url = server(call)
    project = call.options.get("project")
    if call.options.get("key"):
        token, minted = _pasted(call), False
        if not token:
            raise Unauthorized(
                "no key was given", hint="pipe the key in, or paste it at the prompt"
            )
        transport = _transport(url, token, project)
        try:
            me = _drive(transport, _ops.whoami(url))
        finally:
            transport.close()
    else:
        activity = call.state["activity"]
        transport = _transport(url, None, None)
        try:
            granted = _drive(
                transport,
                _ops.device_login("cli"),
                say=lambda step: activity.show(*_verb(step.text or "")),
                reached=lambda grant: _grant(call, grant),
            )
        finally:
            transport.close()
        token, minted = str(granted.get("access_token") or ""), True
        me = _ops.me_of(granted, url, granted.get("expires_at"))
    key = me.key
    expires = me.token_expires or (key.expires if key else None)
    stored = _auth.CredentialStore().save(
        _auth.Stored(
            server=url,
            key_id=key.id if key else "",
            project=project or me.project,
            email=me.email,
            workspace=me.workspace,
            expires=expires.isoformat() if expires else None,
            key=token,
            name=key.name if key else None,
            minted=minted,
        )
    )
    call.state["stored"] = stored
    return me


def _verb(text: str) -> tuple[str, str]:
    verb, _, message = text.partition(" ")
    return verb, message


def where(stored: _auth.Stored) -> str:
    """Where a login is kept: `~/.config/gf/credentials.toml`, or the macOS keychain."""
    if stored.store == "keychain":
        return "the macOS keychain"
    path = str(_auth.CredentialStore().path)
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def logged_in(call: Call, me: Me) -> None:
    """`Logged in maya@acme.com, project acme`, then where the key is kept and until when."""
    from generalforecast.cli._views import day

    stored = call.state["stored"]
    call.out.status(
        "Logged in", f"{me.email or 'with this key'}, project {stored.project}"
    )
    expires = day(me.token_expires or (me.key.expires if me.key else None))
    call.out.status(
        "",
        f"Saved key {stored.name or stored.key_id} to {where(stored)}"
        + (f", expires {expires}" if expires else ""),
    )


def logout(call: Call) -> _auth.Stored | None:
    """Forgets this machine's login, revoking a key `gf login` minted; None when there was none."""
    url = server(call)
    store = _auth.CredentialStore()
    stored = store.load(url)
    if stored is None:
        return None
    if stored.minted and stored.key:
        transport = _transport(url, stored.key, stored.project)
        try:
            transport.send(_http.auth_logout())
        except Error as error:
            call.out.warning(
                f"could not revoke the key {stored.name or stored.key_id}: {error.message}",
                help="it stays valid until it expires; `gf keys revoke` revokes it",
            )
        finally:
            transport.close()
    store.remove(url)
    return stored


def logged_out(call: Call, stored: _auth.Stored | None) -> None:
    """`Logged out maya@acme.com; removed the key cli-maya-mbp from this machine`."""
    if stored is None:
        call.out.status("Logged out", "no login was kept on this machine")
        return
    call.out.status(
        "Logged out",
        f"{stored.email or stored.key_id}; removed the key {stored.name or stored.key_id} "
        "from this machine",
    )


def whoami(call: Call) -> Me:
    """Who the login acts for, as the service says."""
    return call.client._run(_ops.whoami(call.client.url))


def whoami_view(call: Call, me: Me) -> None:
    """Email, workspace, project and its plan, the key and its permissions, and whether the project fine-tunes."""
    from generalforecast.cli._views import day, permissions

    key = held = ""
    if me.key is not None:
        expires = day(me.token_expires or me.key.expires)
        key = me.key.name + (f", expires {expires}" if expires else "")
        held = permissions(me.key) if me.key.permissions else ""
    plan = me.plan.title() if me.plan else ""
    fine_tuning = None
    if me.trains:
        fine_tuning = "included"
    elif me.trains is not None:
        free = me.plan == "free"
        fine_tuning = (
            f"not included in the {plan} plan"
            if free
            else "not available to this project"
        )
    fields = [
        ("Email", me.email),
        ("Workspace", me.workspace),
        ("Project", me.project),
        ("Plan", plan),
        ("Key", key),
        ("Permissions", held),
        ("Fine-tuning", fine_tuning),
    ]
    call.out.details([(k, v or "", "") for k, v in fields if v], stream=call.out.out)
