# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`gf admin logs`: what it asks the service for and how it prints what comes back."""

import io
import json
import os
from typing import Any, ClassVar

import pytest

from generalforecast import commands, predictor_commands, render, term
from tests.test_cli_output import gf

PATH = "/predictors/churn/environments/production/logs"
DEPLOYED = {
    "id": "1795034467412000",
    "at": 1795034467.412,
    "level": "info",
    "source": "deploy",
    "code": "Promoted",
    "message": "Deployed churn revision 2 to production in 7.8s",
    "environment": "production",
    "revision": 2,
    "request_id": "r-3abb2401f119428db148927c6a0d358c",
    "fields": {"model": "general-forecast/tabicl-v2-classifier", "generation": 3},
}
REFUSED = {
    "id": "1795034671207000",
    "at": 1795034671.207,
    "level": "warn",
    "source": "replica a1f",
    "code": "PredictionFailed",
    "message": "predict 422 ValidationError: unknown column `Tenure`",
    "environment": "production",
    "revision": 2,
    "request_id": "r-7d10c2e4b6d84f0a9c4e6f8a0b2d4f6a",
    "fields": {
        "key": "gf_...Jfwo",
        "rows": 1,
        "status": 422,
        "help": "did you mean `tenure`?",
    },
}
FAILED = {
    "id": "1795034700000000",
    "at": 1795034700.0,
    "level": "error",
    "source": "plane",
    "code": "PredictionFailed",
    "message": "predict 503 ProviderUnavailable: hosted inference is starting; retry shortly",
    "environment": "production",
    "revision": None,
    "request_id": None,
    "fields": {"status": 503},
}


class Service:
    """A hosted service that answers each log read from `pages` and records it."""

    calls: ClassVar[list] = []
    pages: ClassVar[list] = []

    def _call(self, method, path, kind, ident, payload=None, **options):
        self.calls.append((method, path, options.get("params")))
        page = self.pages.pop(0)
        if isinstance(page, BaseException):
            raise page
        return page


@pytest.fixture
def service(monkeypatch):
    Service.calls, Service.pages = [], []
    monkeypatch.setattr(commands, "client_for", lambda args, settings=None: Service())
    monkeypatch.setattr(predictor_commands.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(predictor_commands.time, "time", lambda: 1795035000.0)
    return Service


def page(*entries, cursor=None):
    return {
        "logs": list(entries),
        "next": cursor or (entries[-1]["id"] if entries else "0"),
        "previous": None,
    }


def test_a_page_prints_oldest_first_and_a_refusal_says_what_it_concerns(
    service, monkeypatch
):
    service.pages = [page(DEPLOYED, REFUSED, FAILED)]
    code, stdout, stderr = gf("admin", "logs", "churn", monkeypatch=monkeypatch)
    assert code == 0 and not stderr
    assert stdout == (
        "2026-11-18 20:41:07.412  info   deploy        "
        "Deployed churn revision 2 to production in 7.8s\n"
        "2026-11-18 20:44:31.207  warn   replica a1f   "
        "predict 422 ValidationError: unknown column `Tenure`\n"
        "                                              "
        "request   r-7d10c2e4b6d84f0a9c4e6f8a0b2d4f6a\n"
        "                                              key       gf_...Jfwo\n"
        "                                              rows      1\n"
        "                                              status    422\n"
        "                                              revision  2\n"
        "                                              help      did you mean `tenure`?\n"
        "2026-11-18 20:45:00.000  error  plane         "
        "predict 503 ProviderUnavailable: hosted inference is starting; retry shortly\n"
        "                                              status  503\n"
    )
    # With no flag nothing is asked of the route but the page, as any service answers it.
    assert service.calls == [("GET", PATH, None)]


def test_verbose_says_what_every_entry_concerns(service, monkeypatch):
    service.pages = [page(DEPLOYED)]
    _, stdout, _ = gf("-v", "admin", "logs", "churn", monkeypatch=monkeypatch)
    assert stdout.splitlines()[1:] == [
        " " * 46 + "request     r-3abb2401f119428db148927c6a0d358c",
        " " * 46 + "revision    2",
        " " * 46 + "generation  3",
        " " * 46 + "model       general-forecast/tabicl-v2-classifier",
    ]


def test_the_flags_select_entries_on_the_service(service, monkeypatch):
    service.pages = [page(), page()]
    code, stdout, _ = gf(
        "admin",
        "logs",
        "churn",
        "--env",
        "staging",
        "--level",
        "warn",
        "--grep",
        "Tenure",
        "--since",
        "15m",
        "--until",
        "2026-11-18T21:30:00Z",
        "-n",
        "50",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and stdout == ""
    assert service.calls == [
        (
            "GET",
            "/predictors/churn/environments/staging/logs",
            {
                "limit": 50,
                "level": "warn",
                "q": "Tenure",
                "since": "2026-11-18T20:35:00Z",
                "until": "2026-11-18T21:30:00Z",
            },
        )
    ]
    gf("admin", "logs", "churn", "--since", "2d", monkeypatch=monkeypatch)
    assert service.calls[-1][2] == {"since": "2026-11-16T20:50:00Z"}


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (("--level", "fatal"), "--level is info, warn or error"),
        (("-n", "0"), "--limit is a number from 1 to 500"),
        (("-n", "501"), "--limit is a number from 1 to 500"),
        (
            ("--since", "last tuesday"),
            (
                "error: --since is a time, such as 2026-10-27T08:00:00Z, "
                "or how long ago, such as 15m, 2h or 1d"
            ),
        ),
        (("--until", "soon"), "error: --until is a time, such as"),
    ],
)
def test_a_flag_the_route_would_refuse_is_refused_here(
    service, monkeypatch, argv, said
):
    code, stdout, stderr = gf("admin", "logs", "churn", *argv, monkeypatch=monkeypatch)
    assert code in (1, 2) and not stdout and said in stderr
    assert service.calls == []


def test_json_is_the_reply_as_the_service_gave_it(service, monkeypatch):
    service.pages = [page(DEPLOYED, REFUSED)]
    code, stdout, _ = gf("--json", "admin", "logs", "churn", monkeypatch=monkeypatch)
    assert code == 0
    assert json.loads(stdout) == page(DEPLOYED, REFUSED)


def test_follow_prints_each_entry_as_it_comes_until_interrupted(service, monkeypatch):
    slept = []
    monkeypatch.setattr(predictor_commands.time, "sleep", slept.append)
    service.pages = [
        page(DEPLOYED),
        page(cursor=DEPLOYED["id"]),
        page(REFUSED),
        KeyboardInterrupt(),
    ]
    code, stdout, stderr = gf(
        "admin", "logs", "churn", "--follow", "--level", "info", monkeypatch=monkeypatch
    )
    # Ctrl-C ends a tail, and is not an error.
    assert code == 130 and stderr == ""
    assert [line[:23] for line in stdout.splitlines() if line[:1] != " "] == [
        "2026-11-18 20:41:07.412",
        "2026-11-18 20:44:31.207",
    ]
    assert [call[2] for call in service.calls] == [
        {"level": "info"},
        {"level": "info", "after": DEPLOYED["id"]},
        {"level": "info", "after": DEPLOYED["id"]},
        {"level": "info", "after": REFUSED["id"]},
    ]
    assert slept == [predictor_commands.LOG_POLL_SECONDS] * 3


def test_follow_reads_on_at_once_while_pages_come_back_full(service, monkeypatch):
    slept = []
    monkeypatch.setattr(predictor_commands.time, "sleep", slept.append)
    service.pages = [page(DEPLOYED, REFUSED), page(FAILED), KeyboardInterrupt()]
    gf("admin", "logs", "churn", "-f", "-n", "2", monkeypatch=monkeypatch)
    # The first page was full, so the second is asked for without a wait.
    assert slept == [predictor_commands.LOG_POLL_SECONDS]
    assert [call[2] for call in service.calls] == [
        {"limit": 2},
        {"limit": 2, "after": REFUSED["id"]},
        {"limit": 2, "after": FAILED["id"]},
    ]


def test_follow_with_json_is_an_entry_a_line(service, monkeypatch):
    service.pages = [page(DEPLOYED), page(REFUSED), KeyboardInterrupt()]
    code, stdout, stderr = gf(
        "--json", "admin", "logs", "churn", "--follow", monkeypatch=monkeypatch
    )
    assert code == 130
    assert [json.loads(line) for line in stdout.splitlines()] == [DEPLOYED, REFUSED]
    assert json.loads(stderr.strip().splitlines()[-1])["error"]["reason"] == (
        "interrupted"
    )


def _drawn(entry: dict[str, Any], columns: int, monkeypatch) -> list[str]:
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    out = term.Output(out=stream, err=io.StringIO(), environ={"NO_COLOR": "1"})
    monkeypatch.setattr(
        term, "size", lambda stream=None: os.terminal_size((columns, 24))
    )
    render.log_entry(entry, out)
    return stream.getvalue().splitlines()


def test_a_long_message_folds_under_itself_on_a_terminal(monkeypatch):
    lines = _drawn(FAILED, 80, monkeypatch)
    assert lines == [
        "2026-11-18 20:45:00.000  error  plane         predict 503 ProviderUnavailable:",
        " " * 46 + "hosted inference is starting;",
        " " * 46 + "retry shortly",
        " " * 46 + "status  503",
    ]
    assert all(len(line) < 80 for line in lines)


def test_nothing_a_service_sends_reaches_the_terminal_as_an_escape(monkeypatch):
    hostile = dict(
        REFUSED,
        message="predict \x1b[2J\x1b]0;owned\x07 422",
        source="replica\x1b[31m",
        fields={"help": "clear \x1b[H"},
    )
    assert "\x1b" not in "\n".join(_drawn(hostile, 120, monkeypatch))
