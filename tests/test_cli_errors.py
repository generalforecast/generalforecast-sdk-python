# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""What a person reads when a command fails, is interrupted or is told no."""

import io
import json
import os
import sys
from typing import Any

import pytest

from generalforecast import cli, errors, term
from tests.test_cli_output import Tty, _write_rows, gf

REQUEST = "r-524a763d64e2"
UPGRADE = (
    "help: this gf may be older than the service; run `pip install -U generalforecast`"
)


def answered(status: int, code: str, message: str, **more) -> errors.Error:
    """The error the SDK raises for a service's reply, as the transport makes it."""
    from generalforecast.transport import response_value

    try:
        response_value(
            status,
            json.dumps({"error": {"code": code, "message": message, **more}}).encode(),
        )
    except errors.Error as error:
        return error
    raise AssertionError("the reply did not raise")


def shown(
    error: BaseException, *, tty: bool = False, json: bool = False
) -> tuple[int, str]:
    """What `present` writes for an error, and the exit code it returns."""
    err = Tty() if tty else io.StringIO()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    return cli.present(error, out, json=json), err.getvalue()


def refused(
    operation: str, error: errors.Error, *, framed: bool = False
) -> errors.Error:
    with pytest.raises(errors.Error) as caught, errors.during(operation, framed=framed):
        raise error
    return caught.value


def test_a_local_error_is_its_message_with_no_code():
    error = errors.InvalidInput(
        "gf.yaml already exists in ./churn",
        hint="pass --force to overwrite it, or run `gf check` to validate it",
    )
    code, text = shown(error)
    assert code == 1 and text == (
        "error: gf.yaml already exists in ./churn\n"
        "help: pass --force to overwrite it, or run `gf check` to validate it\n"
    )
    why = errors.InvalidInput(
        "--until needs a time column", because="rows.csv has none"
    )
    assert shown(why)[1] == (
        "error: --until needs a time column\nnote: rows.csv has none\n"
    )
    # The code is still what a program is told.
    code, text = shown(error, json=True)
    assert code == 1 and text == ""


def test_a_service_refusal_says_what_was_asked_then_what_the_service_said():
    reply = answered(
        422, "ValidationError", "application/json required", request_id=REQUEST
    )
    error = refused("the snapshot upload", reply, framed=True)
    code, text = shown(error)
    assert code == 1 and text == (
        "error: the service refused the snapshot upload\n"
        "note: ValidationError (422): application/json required\n"
        f"note: request {REQUEST}\n"
        f"{UPGRADE}\n"
    )
    assert cli.error_object(error) == {
        "error": {
            "code": "ValidationError",
            "message": "application/json required",
            "request_id": REQUEST,
        }
    }


def enveloped(status: int, error: Any, **envelope) -> errors.Error:
    """The error the SDK raises for a whole reply envelope, as the transport makes it."""
    from generalforecast.transport import response_value

    with pytest.raises(errors.Error) as caught:
        response_value(status, json.dumps({**envelope, "error": error}).encode())
    return caught.value


def test_a_failed_requests_id_is_read_from_the_envelope():
    from generalforecast.service.boundary import Request, Resource, Scope, failure
    from generalforecast.service.contracts import ServiceError

    # The service's own envelope: `request_id` beside `error`, not inside it.
    request = Request(Scope("acme", REQUEST), Resource("predictor", "churn"), {})
    reply: dict[str, Any] = dict(
        failure(
            request, ServiceError("NotFound", "predictor churn does not exist", 404)
        )
    )
    assert reply["request_id"] == REQUEST and "request_id" not in reply["error"]
    error = enveloped(404, reply.pop("error"), **reply)
    assert error.request_id == REQUEST
    assert shown(error)[1] == (
        "error: predictor churn does not exist\n"
        "note: NotFound (404)\n"
        f"note: request {REQUEST}\n"
    )
    assert cli.error_object(error)["error"]["request_id"] == REQUEST
    named = refused("the status of churn", error)
    assert shown(named)[1].endswith(f"note: request {REQUEST}\n")

    # An id inside the error object is still read, and is the one kept
    # when a reply carries both.
    detail = {"code": "NotFound", "message": "no such predictor"}
    assert enveloped(404, {**detail, "request_id": "r-inner"}).request_id == "r-inner"
    both = enveloped(404, {**detail, "request_id": "r-inner"}, request_id="r-outer")
    assert both.request_id == "r-inner"
    assert enveloped(404, detail).request_id is None
    # An envelope's id that is no short string is not an id.
    for unusable in (17, None, "", "r" * 129, ["r-1"]):
        assert enveloped(404, detail, request_id=unusable).request_id is None
    assert enveloped(404, detail, request_id="r-\x1b[31m1").request_id == "r-1"
    # A reply whose error object is malformed still says which request it was.
    malformed = enveloped(500, {"code": 5}, request_id="r-outer")
    assert (malformed.code, malformed.request_id) == ("ProtocolError", "r-outer")


def test_a_failed_downloads_id_is_read_from_the_envelope():
    from generalforecast.transport import _download_failure

    body = {
        "request_id": REQUEST,
        "error": {"code": "NotFound", "message": "no such artifact"},
    }
    error = _download_failure(404, json.dumps(body).encode())
    assert (error.code, error.request_id) == ("NotFound", REQUEST)


def test_a_service_failure_is_not_called_a_refusal():
    error = refused(
        "the promotion of churn revision 3 to staging",
        answered(503, "ModelNotReady", "warming up", reason="wait_timeout"),
    )
    assert shown(error)[1] == (
        "error: the service could not complete the promotion of churn revision 3 to staging\n"
        "note: ModelNotReady (503): warming up\n"
    )


def test_a_services_reason_limit_and_hint_stay_under_its_words():
    error = refused(
        "the prediction",
        answered(
            429,
            "CapacityExceeded",
            "queries exceed the limit for general-forecast/tabicl-v2-classifier",
            limit=100,
            actual=1000,
            hint="split the queries, or push a predictor and call it in batches",
            request_id="r-3abb2401f119428db148927c6a0d358c",
        ),
    )
    assert shown(error)[1] == (
        "error: the service refused the prediction\n"
        "note: CapacityExceeded (429): queries exceed the limit for "
        "general-forecast/tabicl-v2-classifier\n"
        "note: the limit is 100; the request sent 1,000\n"
        "note: request r-3abb2401f119428db148927c6a0d358c\n"
        "help: split the queries, or push a predictor and call it in batches\n"
    )


def test_a_reply_nothing_named_leads_with_the_services_words():
    error = answered(
        404, "NotFound", "predictor churn does not exist", request_id=REQUEST
    )
    assert shown(error)[1] == (
        "error: predictor churn does not exist\n"
        "note: NotFound (404)\n"
        f"note: request {REQUEST}\n"
    )
    # A reply with no words of its own still says what was refused.
    silent = refused(
        "the deletion of churn", answered(403, "Unauthorized", "", reason="forbidden")
    )
    assert shown(silent)[1] == (
        "error: the service refused the deletion of churn\n"
        "note: Unauthorized (403)\n"
        "help: run `gf login` again, or use a key that has this permission\n"
    )


def test_the_innermost_operation_names_the_refusal_and_local_errors_pass_through():
    reply = answered(409, "Conflict", "revision 4 exists")
    with (
        pytest.raises(errors.Error) as caught,
        errors.during("the push"),
        errors.during("revision 4 of churn"),
    ):
        raise reply
    assert caught.value.operation == "revision 4 of churn"
    local = errors.NotFound(
        "no current gf.lock in ./churn", hint="run `gf check` first"
    )
    assert refused("the snapshot upload", local, framed=True).operation is None
    assert shown(local)[1] == (
        "error: no current gf.lock in ./churn\nhelp: run `gf check` first\n"
    )


@pytest.mark.parametrize(
    ("status", "code", "more", "framed", "advised"),
    [
        # gf wrote the request whole and the service found its form wrong.
        (422, "ValidationError", {}, True, True),
        (400, "ValidationError", {}, True, True),
        # The service names a field: that is about what the person supplied.
        (422, "ValidationError", {"param": "inputs.spend"}, True, False),
        # A request the person shaped, refused: theirs to fix.
        (422, "ValidationError", {}, False, False),
        # Refusals of the method, media type or version are never about content.
        (405, "ValidationError", {}, False, True),
        (415, "ValidationError", {}, False, True),
        (426, "ValidationError", {}, False, True),
        # Everything else is what it says it is.
        (404, "NotFound", {}, True, False),
        (401, "Unauthorized", {}, True, False),
        (413, "CapacityExceeded", {}, True, False),
        (409, "Conflict", {}, True, False),
        (503, "ModelNotReady", {}, True, False),
    ],
)
def test_upgrading_is_advised_only_where_the_protocol_was_refused(
    status, code, more, framed, advised
):
    error = refused(
        "the snapshot upload", answered(status, code, "no", **more), framed=framed
    )
    assert (UPGRADE in shown(error)[1]) == advised


def test_the_services_own_advice_comes_before_ours():
    error = refused(
        "the snapshot upload",
        answered(422, "ValidationError", "no", hint="send Arrow IPC"),
        framed=True,
    )
    assert shown(error)[1].endswith("help: send Arrow IPC\n")


def pointed(line: int, column: int, source: str, width: int, label: str = "") -> str:
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err)
    out.error(
        "target column `churned` is not in the context",
        pointer=term.Pointer("gf.yaml", line, column, source, label, width),
        hint="did you mean `churn`?",
    )
    return err.getvalue()


def test_the_pointer_block_lines_its_bars_up_at_every_line_number_width():
    assert pointed(3, 9, "target: churned", 7, "context.csv has spend, churn") == (
        "error: target column `churned` is not in the context\n"
        "  --> gf.yaml:3:9\n"
        "   |\n"
        " 3 | target: churned\n"
        "   |         ^^^^^^^ context.csv has spend, churn\n"
        "help: did you mean `churn`?\n"
    )
    assert pointed(12, 9, "target: churned", 7) == (
        "error: target column `churned` is not in the context\n"
        "   --> gf.yaml:12:9\n"
        "    |\n"
        " 12 | target: churned\n"
        "    |         ^^^^^^^\n"
        "help: did you mean `churn`?\n"
    )
    assert pointed(123, 3, "  target: churned", 6, "here") == (
        "error: target column `churned` is not in the context\n"
        "    --> gf.yaml:123:3\n"
        "     |\n"
        " 123 |   target: churned\n"
        "     |   ^^^^^^ here\n"
        "help: did you mean `churn`?\n"
    )
    for line in (3, 12, 123, 4567):
        block = pointed(line, 9, "target: churned", 7).splitlines()[1:5]
        assert len({row.index("|") for row in block[1:]}) == 1
        assert block[0].index("-->") == block[1].index("|") - 1


def test_the_carets_sit_under_the_text_after_tabs_and_wide_characters():
    # A tab is drawn as four spaces, on the source line and under it alike.
    block = pointed(3, 9, "target:\tchurned", 7).splitlines()
    assert block[3:5] == [
        " 3 | target:    churned",
        "   |            ^^^^^^^",
    ]
    assert block[3].index("churned") == block[4].index("^")
    block = pointed(3, 3, "\t\ttarget: churned", 6).splitlines()
    assert block[3:5] == [
        " 3 |         target: churned",
        "   |         ^^^^^^",
    ]
    # A wide character takes two columns: carets start, and span, by columns.
    source = "inputs: [顧客番号, 解約]"
    column = source.index("解約") + 1
    block = pointed(
        7, column, source, term.visible("解約"), "not in the context"
    ).splitlines()
    assert block[3:5] == [
        " 7 | inputs: [顧客番号, 解約]",
        "   |                    ^^^^ not in the context",
    ]
    lead = term.visible(block[3][: block[3].index("解約")])
    assert lead == block[4].index("^")
    # Nothing in a source line reaches the terminal as an escape.
    block = pointed(3, 9, "target: \x1b[2Jchurned", 7).splitlines()
    assert "\x1b" not in "".join(block) and block[3] == " 3 | target: [2Jchurned"


def test_a_long_label_folds_under_itself_on_a_terminal_and_stays_whole_in_a_log(
    monkeypatch,
):
    names = "customerID, gender, SeniorCitizen, Partner, Dependents, tenure, Churn"
    pointer = term.Pointer(
        "gf.yaml", 3, 9, "target: Chrun", f"churn.csv has {names}", 5
    )
    log = io.StringIO()
    term.Output(out=io.StringIO(), err=log).error("no such column", pointer=pointer)
    assert log.getvalue().splitlines()[4] == f"   |         ^^^^^ churn.csv has {names}"
    monkeypatch.setattr(term, "size", lambda stream=None: os.terminal_size((60, 24)))
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.error(
        "target column `Chrun` is not in the context, which is the table the "
        "predictor learns from",
        pointer=pointer,
        hint="did you mean `Churn`?",
    )
    assert err.getvalue().splitlines() == [
        "error: target column `Chrun` is not in the context,",
        "       which is the table the predictor learns from",
        "  --> gf.yaml:3:9",
        "   |",
        " 3 | target: Chrun",
        "   |         ^^^^^ churn.csv has customerID, gender,",
        "   |               SeniorCitizen, Partner, Dependents,",
        "   |               tenure, Churn",
        "help: did you mean `Churn`?",
    ]


def test_usage_errors_keep_their_shape_without_the_code(monkeypatch):
    code, _, stderr = gf("chekc", monkeypatch=monkeypatch)
    assert code == 2 and stderr == (
        "error: unknown command `chekc`\nhelp: a similar command exists: 'check'\n"
    )
    code, _, stderr = gf("predict", monkeypatch=monkeypatch)
    assert stderr == (
        "error: the following required arguments were not provided:\n"
        "       --queries <FILE>\n"
        "help: run 'gf predict --help' for usage\n"
    )
    code, _, stderr = gf("--json", "chekc", monkeypatch=monkeypatch)
    assert json.loads(stderr)["error"]["reason"] == "usage_error"


def test_a_missing_file_is_named_without_a_code(monkeypatch, tmp_path):
    code, _, stderr = gf(
        "init", "--context", "nowhere.csv", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 1 and stderr == (
        "error: context file `nowhere.csv` does not exist\n"
        "help: pass --context with a .csv, .parquet or .json file\n"
    )


class Service:
    """A hosted service that has no predictor yet and refuses every write."""

    project = "demo"

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def _call(self, method, path, kind, resource, payload=None):
        self.asked.append((method, path))
        if method == "GET":
            raise answered(404, "NotFound", f"{kind} {resource} does not exist")
        raise answered(409, "Conflict", f"{kind} {resource} is being written")

    def upload_snapshot(self, table, *, prefix="table"):
        self.asked.append(("POST", "/snapshots"))
        error = answered(422, "ValidationError", "application/json required")
        error.request_id = REQUEST
        raise error

    @property
    def predictors(self):
        return self

    def check(self, root, *, upgrade=False, keep=False):
        """A check uploads the snapshot first, as the SDK's operation does."""
        with errors.during("the snapshot upload", framed=True):
            self.upload_snapshot(None)


@pytest.fixture
def pushable(monkeypatch, tmp_path):
    """A checked predictor directory and the service its pushes reach."""
    root = tmp_path / "churn"
    root.mkdir()
    _write_rows(root / "rows.csv")
    (root / "gf.yaml").write_text(
        "name: churn\n"
        "model: general-forecast/tabicl-v2-classifier\n"
        "target: y\n"
        "context:\n  snapshot: rows.csv\n"
    )
    (root / "gf.lock").write_text(
        json.dumps(
            {
                "lock_version": 2,
                "digest": "sha256:" + "b" * 64,
                "predictor": {
                    "name": "churn",
                    "model": "general-forecast/tabicl-v2-classifier",
                    "model_digest": "sha256:" + "c" * 64,
                    "context": {"kind": "snapshot", "rows": 60},
                },
            }
        )
    )
    service = Service()
    monkeypatch.setattr(
        "generalforecast.commands.client_for", lambda args, settings=None, **_: service
    )
    return root, service


def test_push_says_which_of_its_requests_the_service_refused(monkeypatch, pushable):
    root, service = pushable
    code, stdout, stderr = gf("push", "--yes", str(root), monkeypatch=monkeypatch)
    assert code == 1 and stdout == ""
    lines = stderr.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("   Uploading"))
    assert lines[at].startswith("   Uploading churn revision 1  snapshot, 60 rows, ")
    assert lines[at + 1 :] == [
        "error: the service refused the snapshot upload",
        "note: ValidationError (422): application/json required",
        f"note: request {REQUEST}",
        UPGRADE,
    ]
    assert service.asked == [("GET", "/predictors/churn"), ("POST", "/snapshots")]
    code, stdout, stderr = gf(
        "--json", "push", "--yes", str(root), monkeypatch=monkeypatch
    )
    assert code == 1 and stdout == ""
    assert json.loads(stderr)["error"] == {
        "code": "ValidationError",
        "message": "application/json required",
        "request_id": REQUEST,
    }


def test_other_commands_name_their_requests_too(monkeypatch, pushable):
    code, _, stderr = gf("status", "churn", monkeypatch=monkeypatch)
    assert code == 1 and stderr == (
        "error: the service refused the request for predictor churn\n"
        "note: NotFound (404): predictor churn does not exist\n"
    )
    code, _, stderr = gf("admin", "delete", "churn", "--yes", monkeypatch=monkeypatch)
    assert stderr == (
        "error: the service refused the deletion of churn\n"
        "note: Conflict (409): predictor churn is being written\n"
    )


def test_answering_no_is_not_an_error(monkeypatch, pushable):
    root, service = pushable
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    code, stdout, stderr = gf("push", str(root), monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT == 2 and stdout == ""
    assert stderr.endswith("Push? [y/N] Nothing was pushed.\n")
    assert "error" not in stderr and service.asked == [("GET", "/predictors/churn")]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    code, _, stderr = gf("admin", "delete", "churn", monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert stderr == "Delete churn and its environments? [y/N] Nothing was deleted.\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO("no\n"))
    code, _, stderr = gf("admin", "key", "revoke", "k-1", monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert stderr == "Revoke key k-1? [y/N] Nothing was revoked.\n"
    # The line is quiet on a terminal, and shows under -q: it answers a question.
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    code, _, stderr = gf("-q", "push", str(root), monkeypatch=monkeypatch)
    assert stderr.endswith("Push? [y/N] Nothing was pushed.\n")
    assert shown(term.Declined("cancelled", said="Nothing was pushed."), tty=True) == (
        2,
        "Nothing was pushed.\n",
    )
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={})
    cli.present(term.Declined("cancelled", said="Nothing was pushed."), out)
    assert err.getvalue() == "\x1b[2mNothing was pushed.\x1b[0m\n"


def test_nobody_to_ask_is_still_an_error(monkeypatch, pushable):
    root, _ = pushable
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, _, stderr = gf("push", str(root), monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert stderr.endswith(
        "Push? [y/N] \n"
        "error: `Push?` needs an answer\n"
        "help: pass --yes to go ahead without asking\n"
    )


def test_escape_at_a_prompt_says_nothing_was_written(monkeypatch, tmp_path):
    _write_rows(tmp_path / "rows.csv")

    class Escaping(term.Prompter):
        def __init__(self, out, **options):
            super().__init__(out, **options, interactive=True)

        def _raw(self, label, options, default, *, multi):
            raise term.Declined(
                "cancelled; nothing was written", said="Nothing was written."
            )

    monkeypatch.setattr(term, "Prompter", Escaping)
    code, _, stderr = gf("init", str(tmp_path), monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert stderr == (
        "       Found rows.csv  60 rows, 3 columns\n\nNothing was written.\n"
    )
    assert not (tmp_path / "gf.yaml").exists()
    code, _, stderr = gf("--json", "init", str(tmp_path), monkeypatch=monkeypatch)
    assert json.loads(stderr)["error"] == {
        "code": "ValidationError",
        "reason": "aborted",
        "message": "cancelled; nothing was written",
    }


def test_ctrl_c_prints_nothing_but_the_end_of_a_line_it_cut_short():
    assert shown(KeyboardInterrupt()) == (cli.INTERRUPTED_EXIT, "")
    # A progress line in flight has no newline yet: it gets one, and stays.
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.progress("Waiting", "for approval, 14s  (Ctrl-C to cancel)")
    assert cli.present(KeyboardInterrupt(), out) == 130
    assert err.getvalue() == (
        "\r\x1b[K     Waiting for approval, 14s  (Ctrl-C to cancel)\n"
    )
    out.status("Locked", "gf.lock")
    assert err.getvalue().endswith("cancel)\n      Locked gf.lock\n")
    # In a log the progress line was a whole line already.
    log = io.StringIO()
    out = term.Output(out=io.StringIO(), err=log)
    out.progress("Waiting", "for approval")
    assert cli.present(KeyboardInterrupt(), out) == 130
    assert log.getvalue() == "     Waiting for approval\n"


def test_ctrl_c_at_a_question_ends_the_questions_line(monkeypatch, pushable):
    root, _ = pushable

    class Interrupted(io.StringIO):
        def readline(self, *args):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Interrupted())
    code, stdout, stderr = gf("push", str(root), monkeypatch=monkeypatch)
    assert code == cli.INTERRUPTED_EXIT and stdout == ""
    assert stderr.endswith("Push? [y/N] \n") and "error" not in stderr
    monkeypatch.setattr(sys, "stdin", Interrupted())
    code, stdout, stderr = gf("--json", "push", str(root), monkeypatch=monkeypatch)
    assert code == cli.INTERRUPTED_EXIT and stdout == ""
    assert json.loads(stderr.partition("[y/N] ")[2]) == {
        "error": {
            "code": "ValidationError",
            "reason": "interrupted",
            "message": "interrupted",
        }
    }


def test_errors_fold_on_a_terminal_and_never_in_a_log(monkeypatch):
    long = errors.InvalidInput(
        "predictor sales has no cadence",
        because="general-forecast/chronos-2 forecasts series, and gf.yaml declares "
        "no `time` column so there is nothing to step along",
        hint="add `time: date` and `frequency: 1d`, or pick a tabular model with "
        "`gf model list --task classification`",
    )
    assert len(shown(long)[1].splitlines()) == 3
    monkeypatch.setattr(term, "size", lambda stream=None: os.terminal_size((60, 24)))
    assert shown(long, tty=True)[1].splitlines() == [
        "error: predictor sales has no cadence",
        "note: general-forecast/chronos-2 forecasts series, and",
        "      gf.yaml declares no `time` column so there is nothing",
        "      to step along",
        # A command in backticks is never split across lines.
        "help: add `time: date` and `frequency: 1d`, or pick a",
        "      tabular model with",
        "      `gf model list --task classification`",
    ]


def test_a_warning_and_a_lone_help_line_fold_the_same_way(monkeypatch):
    monkeypatch.setattr(term, "size", lambda stream=None: os.terminal_size((40, 24)))
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.warning("no browser available (SSH session)")
    out.hint("open the browser address on any device and enter the code")
    assert err.getvalue().splitlines() == [
        "warning: no browser available (SSH",
        "         session)",
        "help: open the browser address on any",
        "      device and enter the code",
    ]
    log = io.StringIO()
    out = term.Output(out=io.StringIO(), err=log)
    out.hint("open the browser address on any device and enter the code")
    assert log.getvalue().count("\n") == 1
