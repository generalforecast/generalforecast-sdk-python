# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The CLI's help, usage errors and output grammar, in process."""

import io
import json
import os
import subprocess
import sys

from generalforecast import cli, term
from generalforecast.command_specs import CLOSING, COMMAND_HELP, COMMANDS, GROUPS


class Tty(io.StringIO):
    def isatty(self):
        return True


def gf(*argv, monkeypatch, env=None, tty=False):
    """Runs `gf argv` in process; returns (exit code, stdout, stderr)."""
    out, err = (Tty(), Tty()) if tty else (io.StringIO(), io.StringIO())
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NO_COLOR", "1")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    try:
        cli.main(list(argv))
    except SystemExit as error:
        code = error.code or 0
    else:
        code = 0
    return code, out.getvalue(), err.getvalue()


def test_root_help_lists_exactly_the_contract_groups(monkeypatch):
    code, stdout, stderr = gf("--help", monkeypatch=monkeypatch)
    assert code == 0 and not stderr
    expected = (
        "Predictors:\n"
        "  init       Write a predictor manifest here\n"
        "  check      Resolve gf.yaml on the service and write gf.lock\n"
        "  eval       Score the held-out rows on the service\n"
        "  predict    Score rows with a hosted predictor or a catalog model\n"
        "\n"
        "Hosted:\n"
        "  login      Log in through your browser\n"
        "  logout     Remove this machine's credentials\n"
        "  whoami     Show the account and project in use\n"
        "  push       Deploy a predictor to an environment\n"
        "  status     Show a predictor's environments\n"
        "  promote    Point an environment at another's revision\n"
        "  rollback   Point an environment at an earlier revision\n"
        "\n"
        "Resources:\n"
        "  model      Catalog models: list, show\n"
        "  context    Live contexts: create, append, label, show\n"
        "  admin      Keys, secrets, projects and hosted predictors\n"
    )
    assert expected in stdout
    assert stdout.rstrip("\n").endswith(CLOSING)
    assert len(COMMANDS) == 14 and [len(names) for _, names in GROUPS] == [4, 7, 3]
    assert all(len(text) < 60 for text in COMMAND_HELP.values())


def test_every_command_help_starts_with_its_summary_and_ends_with_the_closing_line(
    monkeypatch,
):
    for command in COMMANDS:
        code, stdout, _ = gf(command, "--help", monkeypatch=monkeypatch)
        assert code == 0, command
        assert stdout.startswith(COMMAND_HELP[command] + "\n\nUsage: gf " + command)
        assert stdout.rstrip("\n").endswith(CLOSING), command


def test_help_command_matches_dash_dash_help(monkeypatch):
    _, via_flag, _ = gf("eval", "--help", monkeypatch=monkeypatch)
    _, via_help, _ = gf("help", "eval", monkeypatch=monkeypatch)
    assert via_flag == via_help


def test_admin_lists_every_noun_and_nothing_is_hidden(monkeypatch):
    _, stdout, _ = gf("admin", "--help", monkeypatch=monkeypatch)
    for noun in (
        "key",
        "secret",
        "workspace",
        "project",
        "organization",
        "sso",
        "audit",
        "source",
        "logs",
        "diff",
        "delete",
    ):
        assert f"\n  {noun} " in stdout


def test_retired_flags_and_commands_are_gone(monkeypatch):
    for argv in (
        ("check", "--data", "x.csv"),
        ("predict", "--queries", "q.csv", "--features", "a"),
    ):
        code, _, stderr = gf(*argv, monkeypatch=monkeypatch)
        assert code == 2 and stderr.startswith("error: unexpected argument '--")
    code, stdout, stderr = gf(
        "--json", "admin", "membership", "list", monkeypatch=monkeypatch
    )
    assert code == 2 and not stdout
    assert json.loads(stderr)["error"]["message"] == (
        "unknown noun `membership` for `gf admin`"
    )
    for command in ("dataset", "datasets", "bench", "run"):
        code, stdout, stderr = gf("--json", command, monkeypatch=monkeypatch)
        assert code == 2 and not stdout
        assert json.loads(stderr) == {
            "error": {
                "code": "ValidationError",
                "reason": "usage_error",
                "message": f"unknown command `{command}`",
                "hint": "run 'gf --help' for the list of commands",
            }
        }
    _, stdout, _ = gf("--help", monkeypatch=monkeypatch)
    assert "serve" not in stdout and "pull" not in stdout


def test_typos_get_a_tip(monkeypatch):
    code, _, stderr = gf("chekc", monkeypatch=monkeypatch)
    assert code == 2
    assert (
        stderr
        == "error: unknown command `chekc`\nhelp: a similar command exists: 'check'\n"
    )
    code, _, stderr = gf("init", "--contxt", "x", monkeypatch=monkeypatch)
    assert stderr == (
        "error: unexpected argument '--contxt'\nhelp: a similar argument exists: '--context'\n"
    )


def test_required_arguments_align_under_the_label(monkeypatch):
    code, _, stderr = gf("predict", monkeypatch=monkeypatch)
    assert code == 2
    assert stderr == (
        "error: the following required arguments were not provided:\n"
        "       --queries <FILE>\n"
        "help: run 'gf predict --help' for usage\n"
    )


def test_exclusive_flags(monkeypatch):
    code, _, stderr = gf(
        "init", "--until", "2025-01-01", "--holdout", "0.2", monkeypatch=monkeypatch
    )
    assert code == 2
    assert stderr.startswith(
        "error: the argument '--holdout' cannot be used with '--until'"
    )


def test_json_errors_print_only_the_error_object(monkeypatch, tmp_path):
    (tmp_path / "gf.yaml").write_text("name: x\nmodel: m\n")
    code, stdout, stderr = gf(
        "--json",
        "predict",
        str(tmp_path),
        "--queries",
        "q.csv",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and not stdout
    body = json.loads(stderr)
    assert body["error"]["code"] == "ValidationError"
    assert body["error"]["message"] == (
        f"`{tmp_path}` is a predictor directory; predictions run on the hosted service"
    )
    assert body["error"]["hint"] == (
        f"push it with `gf push {tmp_path}`, then run `gf predict x --queries FILE`"
    )
    assert set(body["error"]) <= {
        "code",
        "reason",
        "message",
        "param",
        "because",
        "hint",
        "tip",
        "limit",
        "actual",
        "request_id",
    }


def test_internal_errors_never_show_a_traceback_without_verbose(monkeypatch):
    def boom(args, out):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("generalforecast.commands.execute", boom)
    code, stdout, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert code == 1 and not stdout
    assert stderr == "error: internal error\nhelp: run again with -v\n"
    code, _, stderr = gf("-v", "model", "list", monkeypatch=monkeypatch)
    assert "RuntimeError: secret internals" in stderr
    assert stderr.endswith("error: internal error\nhelp: run again with -v\n")


def test_status_verbs_are_right_aligned_in_twelve_columns():
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.status("Initialized", "predictor churn in ./gf.yaml")
    out.status("Resolved", "churn  classification, 3 inputs", seconds=None)
    out.status("Locked", "gf.lock", seconds=0.21)
    out.status("Compiled", "12 programs", seconds=6.8)
    out.status("Deployed", "churn revision 7 to staging", seconds=41)
    assert err.getvalue() == (
        " Initialized predictor churn in ./gf.yaml\n"
        "    Resolved churn  classification, 3 inputs\n"
        "      Locked gf.lock in 0.21s\n"
        "    Compiled 12 programs in 6.8s\n"
        "    Deployed churn revision 7 to staging in 41s\n"
    )


def test_colour_follows_the_terminal_and_no_color():
    assert not term.color_enabled(io.StringIO(), {})
    assert term.color_enabled(Tty(), {})
    assert not term.color_enabled(Tty(), {"NO_COLOR": "1"})
    assert term.color_enabled(io.StringIO(), {"FORCE_COLOR": "1"})
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={})
    out.status("Locked", "gf.lock")
    out.error("x", hint="y", notes=["request r-1"])
    text = err.getvalue()
    assert "\x1b[1;32m      Locked\x1b[0m gf.lock" in text
    assert "\x1b[1;31merror:\x1b[0m x" in text
    assert "\x1b[1mhelp:\x1b[0m y" in text
    assert "\x1b[1mnote:\x1b[0m request r-1" in text


# Modules no start of the CLI needs: each costs tens of milliseconds to import.
SDK = (
    "pyarrow",
    "numpy",
    "pandas",
    "httpx",
    "generalforecast.client",
    "generalforecast.predictor_commands",
)
# What reading a context table must not load: pandas alone is a third of the
# wait for the first prompt, and the hosted client is not far behind.
HOSTED = (
    "pandas",
    "httpx",
    "asyncio",
    "generalforecast.client",
    "generalforecast.transport",
    "generalforecast.predictor",
    "generalforecast._core",
    "jax",
)


def _loaded(*argv, modules, cwd=None, to_prompt=False):
    """Runs `gf argv` in a fresh interpreter; returns which of `modules` it imported, and stderr.

    With `to_prompt` the run ends where its first question would be drawn,
    so the modules are the ones a person waits for before they can answer.
    """
    code = (
        "import sys; from generalforecast import cli, term\n"
        "def stop(self, label, *args, **options):\n"
        "    raise SystemExit(f'PROMPT={label}')\n"
        f"if {to_prompt!r}:\n    term.Prompter.choose = stop\n"
        "try:\n    cli.main(sys.argv[1:])\n"
        "except SystemExit as error:\n    sys.stderr.write(str(error.code or ''))\n"
        f"heavy = [m for m in {modules!r} if m in sys.modules]; "
        "sys.stdout.write('\\nHEAVY=' + ','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *argv],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        cwd=cwd,
        env={k: v for k, v in os.environ.items() if k != "FORCE_COLOR"},
    )
    assert "\nHEAVY=" in result.stdout, result.stdout[-200:]
    heavy = result.stdout.rsplit("\nHEAVY=", 1)[1]
    return [name for name in heavy.split(",") if name], result.stderr


def test_help_and_version_do_not_import_the_sdk():
    for argv in (["--help"], ["--version"], ["eval", "--help"]):
        assert _loaded(*argv, modules=SDK) == ([], ""), argv
    # The version is package metadata, read only when it is asked for.
    metadata = ("importlib.metadata", "email")
    assert _loaded("--help", modules=metadata)[0] == []
    assert _loaded("init", "--help", modules=metadata)[0] == []
    assert _loaded("--version", modules=metadata)[0] == ["importlib.metadata", "email"]


def _write_mixed(path, rows=120):
    """A context with every kind `gf init` infers: blanks, floats, text and times."""
    lines = ["id,when,store,units,price,promo,note,sales"]
    for i in range(rows):
        price = "" if i % 50 == 0 else f"{1 + i / 7:.2f}"
        note = "  " if i % 9 == 0 else f"n{i}"
        lines.append(
            f"{i},2025-{1 + i % 12:02d}-{1 + i % 28:02d},"
            f"{'north' if i % 2 else 'south'},{i % 7},{price},"
            f"{'true' if i % 3 else 'false'},{note},{'' if i < 5 else i % 4}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_rows(path, rows=60, *, timed=False, blanks=0):
    lines = ["id,x,t,y" if timed else "id,x,y"]
    for i in range(rows):
        y = "" if i < blanks else str(i % 3)
        when = f"2025-01-{1 + i % 28:02d}," if timed else ""
        lines.append(f"{i},{i / 10},{when}{y}")
    path.write_text("\n".join(lines) + "\n")


def _manifest(root):
    import yaml

    return yaml.safe_load((root / "gf.yaml").read_text())


def test_permission_denied_says_what_to_do(monkeypatch):
    from generalforecast import errors

    def denied(args, out):
        raise errors.Unauthorized("permission denied for this scope")

    monkeypatch.setattr("generalforecast.commands.execute", denied)
    code, stdout, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert code == 1 and not stdout
    assert stderr == (
        "error: permission denied for this scope\n"
        "help: run `gf login` again, or use a key that has this permission\n"
    )


def test_whoami_shows_the_workspace_by_name():
    import argparse

    from generalforecast import render

    err, out_stream = io.StringIO(), io.StringIO()
    out = term.Output(out=out_stream, err=err, environ={"NO_COLOR": "1"})
    render._whoami(
        argparse.Namespace(),
        {
            "email": "tejas@generalforecast.com",
            "workspace": {
                "id": "ws-d05c5221daffd41b",
                "display_name": "General Forecast",
            },
            "project": "general-forecast",
            "server": "https://app.generalforecast.com",
            "source": "macOS keychain",
        },
        out,
    )
    text = out_stream.getvalue()
    assert "workspace    General Forecast\n" in text
    assert "ws-d05c" not in text and "{" not in text
