# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""What `gf login` shows: a table of what to act on, a wait that counts, then who is logged in."""

import argparse
import io
import json
import os

import pytest

from generalforecast import account, cli, errors, login, term
from tests.test_cli_output import Tty, gf

URL = "https://app.generalforecast.com/cli"
TABLE = (
    "  code       BXDW-NHHR\n"
    f"  browser    {URL}?code=BXDW-NHHR\n"
    "  machine    mac\n"
    "  expires    in 10 minutes\n"
)
DONE = (
    "   Logged in as tejas@generalforecast.com to workspace General Forecast\n"
    "      Stored key gf_...Jfwo in ~/.config/gf/credentials.toml (0600)\n"
)


class Flow:
    """A device login the browser approves as soon as it is waited for."""

    expires_in = 600
    outcome: object = None

    def __init__(self, server, *, timeout):
        self.server = server

    def start(self, machine):
        assert machine == "mac"
        return {
            "device_code": "d" * 32,
            "user_code": "BXDW-NHHR",
            "verification_uri": URL,
            "verification_uri_complete": f"{URL}?code=BXDW-NHHR",
            "interval": 5,
            "expires_in": self.expires_in,
        }

    def wait(self, grant):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return {
            "access_token": "gf_" + "k" * 36 + "Jfwo",
            "email": "tejas@generalforecast.com",
            "workspace": {"id": "ws-1", "display_name": "General Forecast"},
            "project": "general-forecast",
            "key": {"id": "k-7ab3"},
        }


@pytest.fixture
def approving(monkeypatch, tmp_path):
    """A machine named `mac` with a browser, no key in its environment and an empty store."""
    opened = []
    for variable in ("GF_API_KEY", "GF_TOKEN", "GF_PROFILE", "GF_URL", "GF_PROJECT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("GF_KEYCHAIN", "0")
    monkeypatch.setattr(term, "TICK_SECONDS", 3600.0)
    monkeypatch.setattr(Flow, "outcome", None)
    monkeypatch.setattr(Flow, "expires_in", 600)
    monkeypatch.setattr(login, "DeviceLogin", Flow)
    monkeypatch.setattr(login, "machine_name", lambda: "mac")
    monkeypatch.setattr(login, "browser_unavailable", lambda: None)
    monkeypatch.setattr(
        "webbrowser.open", lambda url, new=0: opened.append(url) or True
    )
    return opened


def test_login_shows_what_to_act_on_as_a_table_then_who_is_logged_in(
    monkeypatch, approving
):
    code, stdout, stderr = gf("login", monkeypatch=monkeypatch)
    assert code == 0 and stdout == ""
    # A log gets the wait once, without a count.
    assert stderr == TABLE + "     Waiting for approval  (Ctrl-C to cancel)\n" + DONE
    assert approving == [f"{URL}?code=BXDW-NHHR"]


def test_the_wait_counts_in_place_on_a_terminal_and_the_result_replaces_it(
    monkeypatch, approving
):
    code, _, stderr = gf("login", monkeypatch=monkeypatch, tty=True)
    assert code == 0
    assert stderr == (
        TABLE + "\r\x1b[K     Waiting for approval, 0s  (Ctrl-C to cancel)"
        "\r\x1b[K" + DONE
    )
    # The count is the time waited, said as every elapsed time is.
    now = [0.0]
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    work = term.Activity(out, clock=lambda: now[0])
    work.show("Waiting", lambda waited: f"for approval, {term.elapsed(waited or 0)}")
    now[0] = 14.2
    work._draw()
    now[0] = 135.0
    work._draw()
    work.close()
    assert err.getvalue() == (
        "\r\x1b[K     Waiting for approval, 0s"
        "\r\x1b[K     Waiting for approval, 14s"
        "\r\x1b[K     Waiting for approval, 2m15s"
    )


def test_the_code_is_bold_the_address_a_link_and_the_way_out_quiet(
    monkeypatch, approving
):
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={})
    result = account.login(argparse.Namespace(timeout=30.0), out)
    shown = err.getvalue()
    assert "  code       \x1b[1mBXDW-NHHR\x1b[0m\n" in shown
    link = f"\x1b]8;;{URL}?code=BXDW-NHHR\x1b\\{URL}?code=BXDW-NHHR\x1b]8;;\x1b\\"
    assert f"  browser    {link}\n" in shown
    assert "for approval, 0s  \x1b[2m(Ctrl-C to cancel)\x1b[0m" in shown
    assert result["authenticated"] and result["key"]["suffix"] == "gf_...Jfwo"
    # The values start in the column the status lines' messages start in.
    plain = term._ANSI.sub("", shown).splitlines()
    assert plain[0].index("BXDW") == term.VERB_WIDTH + 1


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (600, "in 10 minutes"),
        (60, "in 1 minute"),
        (45, "in 45 seconds"),
        (900, "in 15 minutes"),
    ],
)
def test_the_codes_lifetime_is_said_as_a_duration(
    monkeypatch, approving, seconds, said
):
    monkeypatch.setattr(Flow, "expires_in", seconds)
    _, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert f"  expires    {said}\n" in stderr


def test_without_a_browser_the_table_is_the_same_and_says_what_to_do(
    monkeypatch, approving
):
    monkeypatch.setattr(login, "browser_unavailable", lambda: "SSH session")
    code, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert code == 0 and approving == []
    assert stderr == (
        "warning: no browser available (SSH session)\n"
        "  code       BXDW-NHHR\n"
        # The page that asks for the code, for a browser on any device.
        f"  browser    {URL}\n"
        "  machine    mac\n"
        "  expires    in 10 minutes\n"
        "help: open the browser address on any device and enter the code\n"
        "     Waiting for approval  (Ctrl-C to cancel)\n" + DONE
    )
    # A browser that will not start is the same case.
    monkeypatch.setattr(login, "browser_unavailable", lambda: None)
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: False)
    _, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert stderr.startswith(
        "warning: no browser available (none could be started)\n  code       BXDW-NHHR\n"
    )
    assert "help: open the browser address on any device" in stderr


def test_quiet_and_json_still_show_what_the_person_must_act_on(monkeypatch, approving):
    code, stdout, stderr = gf("-q", "login", monkeypatch=monkeypatch)
    assert code == 0 and stdout == "" and stderr == TABLE
    code, stdout, stderr = gf("--json", "login", monkeypatch=monkeypatch)
    assert code == 0 and stderr == TABLE
    result = json.loads(stdout)
    assert result["email"] == "tejas@generalforecast.com" and result["store"] == "file"
    monkeypatch.setattr(login, "browser_unavailable", lambda: "no display")
    code, _, stderr = gf("-q", "login", monkeypatch=monkeypatch)
    assert stderr == (
        TABLE.replace("?code=BXDW-NHHR", "")
        + "help: open the browser address on any device and enter the code\n"
    )


def test_ctrl_c_during_the_wait_ends_its_line_and_says_nothing(monkeypatch, approving):
    monkeypatch.setattr(Flow, "outcome", KeyboardInterrupt())
    code, stdout, stderr = gf("login", monkeypatch=monkeypatch, tty=True)
    assert code == cli.INTERRUPTED_EXIT and stdout == ""
    assert stderr == (
        TABLE + "\r\x1b[K     Waiting for approval, 0s  (Ctrl-C to cancel)\n"
    )
    assert not os.path.exists(os.path.expanduser("~/.config/gf/credentials.toml"))


def test_a_login_the_service_refuses_says_so_in_the_new_shape(monkeypatch, approving):
    denied = errors.from_wire(
        {"code": "Unauthorized", "reason": "access_denied", "message": "denied"},
        status=400,
    )
    monkeypatch.setattr(Flow, "outcome", denied)
    code, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert code == 1
    assert stderr.endswith(
        "     Waiting for approval  (Ctrl-C to cancel)\n"
        "error: the service refused the login\n"
        "note: Unauthorized (400): denied\n"
        "help: run `gf login` again\n"
    )
    # A code that ran out is this machine's own finding, and reads as one.
    monkeypatch.setattr(Flow, "outcome", login._expired())
    _, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert stderr.endswith("error: the login code expired\nhelp: run `gf login`\n")
