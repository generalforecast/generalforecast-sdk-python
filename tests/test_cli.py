# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The `gf` command: the table mirrors the SDK, and flags, exit codes, JSON and errors behave alike everywhere."""

import inspect
import io
import json
import re
import sys

import pandas as pd
import pytest

import generalforecast as gf
from generalforecast import cli, term
from generalforecast.cli import _commands, _help
from generalforecast.client import Client
from tests import samples

# Section 2.2 of the surface spec, one row per command.
SPEC_COMMANDS = """
login; logout; whoami; deploy
contexts create; contexts upload; contexts refresh; contexts append; contexts label; contexts get;
contexts list; contexts rows; contexts delete
experiments create; experiments estimate; experiments run; experiments update; experiments get;
experiments wait; experiments list; experiments cancel; experiments predictions; experiments slices;
experiments delete
predictors create; predictors update; predictors get; predictors list; predictors revisions;
predictors predict; predictors promote; predictors rollback; predictors logs; predictors delete
sources list; sources get; sources prepare; sources test; sources create; sources tables; sources preview; sources delete
keys list; keys create; keys rotate; keys revoke; keys delete
snapshots create; snapshots stream; snapshots create-synthetic; snapshots synthetic; snapshots list;
snapshots get; snapshots update; snapshots delete
runs create; runs list; runs get; runs cancel; runs lineage; runs wait; runs watch
models list; models get; models predict
"""


class Stream(io.StringIO):
    def __init__(self, text="", *, terminal=False):
        super().__init__(text)
        self.terminal = terminal

    def isatty(self):
        return self.terminal


def gf_cli(monkeypatch, *argv, stdin="", terminal=False):
    """Runs `gf argv`; returns the exit code, stdout and stderr."""
    out, err = Stream(terminal=terminal), Stream(terminal=terminal)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", out)
        patch.setattr(sys, "stderr", err)
        patch.setattr(sys, "stdin", Stream(stdin, terminal=terminal))
        if terminal:
            patch.setenv("TERM", "xterm-256color")
            patch.delenv("NO_COLOR", raising=False)
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_every_command_of_the_spec_is_in_the_table_and_mirrors_an_sdk_method(env):
    wanted = {
        tuple(line.split())
        for line in SPEC_COMMANDS.replace("\n", ";").split(";")
        if line.strip()
    }
    assert set(_commands.COMMANDS) == wanted
    client = Client(api_key="gf_test", project="acme")
    for command in _commands.COMMANDS.values():
        if command.run is not None and command.words[0] in (
            "login",
            "logout",
            "whoami",
            "deploy",
        ):
            continue
        target = client
        for part in command.path.split("."):
            target = getattr(target, part)
        method = inspect.signature(target)
        for flag in command.flags:
            keyword = flag.keyword
            if keyword is None or flag.kind.spread:
                continue
            name = keyword.partition(".")[0]
            assert name in method.parameters or any(
                p.kind is p.VAR_KEYWORD for p in method.parameters.values()
            ), f"{command.line} {flag.option} -> {name}"
    client.close()


def test_help_and_version(monkeypatch, env):
    code, out, _ = gf_cli(monkeypatch, "--help")
    assert code == 0 and out.startswith(_help.TAGLINE) and "snapshots" in out
    assert gf_cli(monkeypatch, "-V")[1] == f"gf {gf.__version__}\n"
    by_flag = gf_cli(monkeypatch, "contexts", "create", "--help")[1]
    assert gf_cli(monkeypatch, "help", "contexts", "create")[1] == by_flag
    assert (
        "Usage: gf contexts create NAME [flags]" in by_flag
        and "--sql-file F" in by_flag
    )
    assert "Python: gf.contexts.create(NAME, **flags)" in by_flag
    group = gf_cli(monkeypatch, "runs", "--help")[1]
    assert "create" in group and "lineage" in group
    code, _, err = gf_cli(monkeypatch, "contexts")
    assert code == 2 and "error: `gf contexts` needs a command" in err


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["contxts", "list"], "help: a similar command exists: `gf contexts`"),
        (
            ["contexts", "create", "x", "--fiel", "a"],
            "help: a similar flag exists: `--file`",
        ),
        (["contexts", "get"], "error: `gf contexts get` needs NAME"),
        (
            ["contexts", "rows", "adult", "--limit", "many"],
            "error: `many` is not a whole number for --limit",
        ),
        (
            ["experiments", "create", "--task", "clustering"],
            "is not classification or regression or forecast",
        ),
        (["runs", "create", "x", "--method", "qlora"], "for --method"),
        (
            ["runs", "create", "x", "--base", "tabicl-v2"],
            "error: `gf runs create NAME` needs --method",
        ),
        (["runs", "create", "x", "--data", "table:x"], "for --data"),
        (["runs", "create", "x", "--budget", "2.5credits"], "for --budget"),
        (
            ["snapshots", "update", "x"],
            "error: `gf snapshots update NAME` needs --tags",
        ),
    ],
)
def test_a_command_line_gf_cannot_read_exits_2(monkeypatch, env, argv, said):
    code, out, err = gf_cli(monkeypatch, *argv)
    assert code == 2 and out == "" and said in err


def test_flags_become_the_sdk_keywords(monkeypatch, fake):
    samples.adult_context(fake)
    assert gf_cli(monkeypatch, "predictors", "create", "p", "--context", "adult", "--model", "tabicl-v2",
                  "--target", "income", "--exclude", "fnlwgt,race", "--type", "age=number",
                  "--type", "sex=category", "--quantiles", "0.1,0.9")[0] == 0  # fmt: skip
    spec = fake.calls_to("POST", r".*/predictors")[0].payload["spec"]
    assert spec["exclude"] == ["fnlwgt", "race"] and spec["types"] == {
        "age": "number",
        "sex": "category",
    }
    assert spec["quantiles"] == [0.1, 0.9]
    assert (
        gf_cli(
            monkeypatch, "predictors", "update", "p", "--clear", "exclude", "--no-wait"
        )[0]
        == 0
    )
    assert fake.calls_to("POST", r".*/predictors/p/update")[0].payload["spec"] == {
        "exclude": None
    }
    samples._snapshot(fake)
    code, _, err = gf_cli(monkeypatch, "runs", "create", "ft", "--base", "tabicl-v2",
                          "--data", "credit-2026-01,tag:nightly,context:adult,context:adult@1",
                          "--target", "income", "--steps", "300", "--learning-rate", "0.0001",
                          "--episode-rows", "1024", "--holdout", "10%", "--method", "lora:8+adapter",
                          "--budget", "2h")  # fmt: skip
    assert code == 0, err
    run = fake.calls_to("POST", r".*/runs")[0].payload
    assert run["plan"] == {
        "steps": 300,
        "learning_rate": 0.0001,
        "episode_sampling": {"rows": 1024},
    }
    assert run["evaluation"] == {"holdout": 0.1} and run["budget"] == {"gpu_hours": 2.0}
    assert run["method"] == {"type": "lora", "rank": 8, "input_adapter": True}
    assert run["data"] == [
        "credit-2026-01",
        {"tag": "nightly"},
        {"context": "adult"},
        {"context": "adult", "version": 1},
    ]
    assert run["target"] == "income"
    assert gf_cli(monkeypatch, "snapshots", "create-synthetic", "warm", "--count", "4",
                  "--rows", "1024..8192", "--missing", "5%", "--set", "graph_nodes=16",
                  "--set", "categorical.share=0.3", "--reject", "trivial")[0] == 0  # fmt: skip
    batch = fake.calls_to("POST", r".*/snapshots/synthetic")[0].payload
    assert batch["rows"] == {"min": 1024, "max": 8192} and batch["missing"] == {
        "rate": 0.05
    }
    assert batch["graph_nodes"] == 16 and batch["categorical"] == {"share": 0.3}


def test_snapshots_and_runs_mirror_their_sdk_methods(monkeypatch, fake, tmp_path):
    table = tmp_path / "t.csv"
    pd.DataFrame({"x": [1.0, 2.0] * 20, "y": [0, 1] * 20}).to_csv(table, index=False)
    code, _, err = gf_cli(monkeypatch, "snapshots", "stream", "t", "--file", str(table),
                          "--target", "y", "--task", "classification")  # fmt: skip
    assert code == 0 and "Created snapshot t" in err
    code, _, err = gf_cli(monkeypatch, "snapshots", "update", "t", "--tags", "")
    assert code == 0 and "Tagged snapshot t  no tags" in err
    assert fake.calls_to("POST", r".*/snapshots/t/update")[0].payload == {"tags": []}
    code, _, err = gf_cli(monkeypatch, "runs", "create", "a", "--base", "tabicl-v2",
                          "--method", "adapter", "--data", "t", "--steps", "10")  # fmt: skip
    assert code == 0 and "Started run a@ft1" in err
    assert fake.calls_to("POST", r".*/runs")[0].payload["method"] == {
        "type": "input_adapter"
    }
    code, out, _ = gf_cli(monkeypatch, "runs", "watch", "a", "--json")
    kinds = [json.loads(line)["kind"] for line in out.splitlines()]
    code, out, _ = gf_cli(monkeypatch, "runs", "lineage", "a@ft1", "--json")
    lineage = json.loads(out)
    assert lineage["model"] == "acme/a@ft1"
    assert [e["kind"] for e in lineage["events"]] == kinds
    out = gf_cli(monkeypatch, "runs", "lineage", "a")[1]
    assert len(out.splitlines()) == len(kinds) and "published" in out
    code, _, err = gf_cli(
        monkeypatch, "runs", "create", "a", "--from", str(table), "--base", "m"
    )
    assert code == 2 and "--from takes every setting from its file, not --base" in err
    for budget, sent in (
        ("500credits", {"credits": 500}),
        ("1.5h", {"gpu_hours": 1.5}),
    ):
        code, _, err = gf_cli(monkeypatch, "runs", "create", "a", "--base", "tabicl-v2",
                              "--method", "adapter", "--data", "t", "--steps", "10",
                              "--budget", budget)  # fmt: skip
        assert code == 0, err
        payload = fake.calls_to("POST", r".*/runs")[-1].payload
        assert (
            payload["budget"] == sent
            and type(payload["budget"].get("credits", 0)) is int
        )


def test_json_prints_one_object_lists_under_their_noun_and_errors_too(
    monkeypatch, fake
):
    samples._sources(fake)
    code, out, _ = gf_cli(monkeypatch, "sources", "list", "--json")
    assert code == 0 and [s["name"] for s in json.loads(out)["sources"]] == [
        "bigquery-acme",
        "snowflake-prod",
    ]
    code, out, _ = gf_cli(
        monkeypatch, "sources", "delete", "bigquery-acme", "--yes", "--json"
    )
    assert json.loads(out) == {"name": "bigquery-acme", "deleted": True}
    code, out, err = gf_cli(monkeypatch, "--json", "contexts", "get", "nope")
    error = json.loads(out)["error"]
    assert (
        code == 1
        and error["code"] == "NotFound"
        and error["recoverable"] is True
        and err == ""
    )
    code, out, _ = gf_cli(monkeypatch, "contexts", "rows", "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "UsageError"


def test_quiet_keeps_results_and_errors_only(monkeypatch, fake):
    samples._keys(fake)
    code, out, err = gf_cli(monkeypatch, "-q", "keys", "rotate", "ci-deploy", "--yes")
    assert code == 0 and err == "" and out.startswith("gf_live_")


def test_a_question_asks_on_a_terminal_and_no_changes_nothing(monkeypatch, fake):
    samples._sources(fake)
    code, _, err = gf_cli(
        monkeypatch, "sources", "delete", "bigquery-acme", stdin="n\n", terminal=True
    )
    assert (
        code == 2
        and "Delete source bigquery-acme? [y/N] " in err
        and "Nothing was changed." in err
    )
    assert "bigquery-acme" in fake.sources
    code, _, err = gf_cli(monkeypatch, "sources", "delete", "bigquery-acme")
    assert code == 2 and "help: pass --yes to go ahead without asking" in err


def test_errors_are_bold_when_recoverable_and_red_when_a_real_failure(
    monkeypatch, fake
):
    fake.catalog = samples.CATALOG
    _, _, err = gf_cli(monkeypatch, "contexts", "get", "nope", terminal=True)
    assert err.startswith("\x1b[1merror:\x1b[0m no context named `nope`")
    samples._failing(fake)
    code, _, err = gf_cli(
        monkeypatch,
        "predictors",
        "update",
        "churn",
        "--model",
        "kumo-medium",
        terminal=True,
    )
    assert (
        code == 1
        and "\x1b[1;31merror:\x1b[0m churn revision 5 failed to start in production"
        in err
    )
    # The progress line the failure stopped stays on screen above the error.
    assert "starting a replica\n\x1b[1;31merror:" in err


def test_sdk_hints_read_as_commands():
    assert (
        cli.for_cli("run gf.login(), or set GF_API_KEY")
        == "run `gf login`, or set GF_API_KEY"
    )
    assert cli.for_cli("gf.experiments.wait('adult-income') picks it up again") == (
        "`gf experiments wait adult-income` picks it up again"
    )
    assert cli.for_cli(
        "see gf.predictors.logs('churn', env='staging', level='error')"
    ) == ("see `gf predictors logs churn --env staging --level error`")
    assert cli.for_cli("gf.snapshots.list() and gf.contexts.list() show them") == (
        "`gf snapshots list` and `gf contexts list` show them"
    )
    assert cli.for_cli("gf.runs.wait('a@ft1') picks it up again") == (
        "`gf runs wait a@ft1` picks it up again"
    )
    assert (
        cli.for_cli("run gf.login() again, or pass key=")
        == "run `gf login` again, or pass --key"
    )
    assert cli.for_cli("pass gf.Selection(core=0.5)") == "pass gf.Selection(core=0.5)"


def test_ctrl_c_exits_130_and_a_closed_pipe_141(monkeypatch, fake):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gf.resources.Contexts, "list", interrupted)
    assert gf_cli(monkeypatch, "contexts", "list")[0] == 130

    def closed(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr(gf.resources.Contexts, "list", closed)
    assert gf_cli(monkeypatch, "contexts", "list")[0] == 141


def test_logs_follow_pages_after_the_last_until_interrupted(monkeypatch, fake):
    fake.catalog = samples.CATALOG
    samples._logs(fake)
    pages = []

    def sleep(seconds):
        pages.append(seconds)
        if len(pages) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(_commands.time, "sleep", sleep)
    code, out, _ = gf_cli(monkeypatch, "predictors", "logs", "churn", "--follow")
    assert code == 130 and out.count("r-3f1a92c0") == 2
    afters = [c.query.get("after") for c in fake.calls_to("GET", r".*/logs")]
    assert afters == [None, "1795255455000000"]


def test_deploy_files_that_name_a_predictor_twice_or_none_are_refused(
    monkeypatch, fake, tmp_path
):
    twice = tmp_path / "twice.py"
    twice.write_text(
        "import generalforecast as gf\n"
        "a = gf.Predictor('p', context='c', model='m', target='t')\n"
        "b = [a, gf.Predictor('p', context='d', model='m', target='t')]\n"
    )
    code, _, err = gf_cli(monkeypatch, "deploy", str(twice))
    assert code == 1 and "defines the predictor p twice" in err
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    assert "defines no gf.Predictor" in gf_cli(monkeypatch, "deploy", str(empty))[2]
    broken = tmp_path / "broken.py"
    broken.write_text("import generalforecast as gf\nx = gf.Predictor('p',\n")
    code, _, err = gf_cli(monkeypatch, "deploy", str(broken))
    assert code == 1 and f"--> {broken}:2:" in err
    assert fake.calls == []


def test_login_with_a_piped_key_keeps_it_and_logout_forgets_it(monkeypatch, fake):
    from generalforecast import _auth

    monkeypatch.delenv("GF_API_KEY")
    code, _, err = gf_cli(monkeypatch, "login", "--key", stdin=fake_token())
    assert code == 0 and "Logged in maya@acme.com, project acme" in err
    stored = _auth.CredentialStore().load(fake.url)
    assert stored.key == fake_token().strip() and not stored.minted
    assert gf_cli(monkeypatch, "contexts", "list")[0] == 0
    code, _, err = gf_cli(monkeypatch, "logout")
    assert code == 0 and "removed the key notebook-maya-mbp" in err
    assert fake.calls_to("POST", "/v1/auth/logout") == [], (
        "a pasted key is forgotten, not revoked"
    )
    assert "no login was kept" in gf_cli(monkeypatch, "logout")[2]


def test_whoami_shows_a_key_that_never_expires_without_a_date(monkeypatch, fake):
    code, out, _ = gf_cli(monkeypatch, "whoami")
    assert code == 0
    assert re.search(r"Key\s+notebook-maya-mbp$", out, re.MULTILINE), out
    fake.trains = False
    out = gf_cli(monkeypatch, "whoami")[1]
    assert re.search(r"Fine-tuning\s+not offered by this service$", out, re.MULTILINE)


def test_sources_list_and_tables_print_what_the_service_says_of_them(monkeypatch, fake):
    from tests import samples

    samples._sources(fake)
    code, out, _ = gf_cli(monkeypatch, "sources", "list", "--json")
    listed = json.loads(out)
    assert code == 0 and listed["available"] is True
    assert listed["sources"][0]["access"] == "service_account"
    code, out, _ = gf_cli(monkeypatch, "sources", "tables", "bigquery-acme", "--json")
    tables = json.loads(out)
    assert (tables["location"], tables["more"]) == ("acme-prod.analytics", False)
    assert tables["tables"][2] == {
        "name": "customers_model",
        "type": "MODEL",
        "readable": False,
    }
    fake.sources.clear()
    fake.sources_available = False
    code, out, err = gf_cli(monkeypatch, "sources", "list")
    assert code == 0 and "This service connects no sources." in err
    assert gf.sources.list().available is False


def fake_token():
    from tests.fake import TOKEN

    return TOKEN + "\n"


def test_rows_show_what_fits_and_count_the_rest(monkeypatch, fake):
    samples.adult_context(fake)
    monkeypatch.setenv("COLUMNS", "30")
    _, out, _ = gf_cli(monkeypatch, "contexts", "rows", "adult", "--limit", "2")
    assert (
        out.splitlines()[-1]
        == "  and 13 more columns; widen the terminal or pass --json"
    )
    frame = pd.read_json(
        io.StringIO(gf_cli(monkeypatch, "contexts", "rows", "adult", "--json")[1])
    )
    assert len(frame) == 3


def test_verbose_shows_every_column_and_request_ids(monkeypatch, fake):
    samples.adult_context(fake)
    _, out, _ = gf_cli(monkeypatch, "-v", "contexts", "get", "adult")
    assert "native_country" in out and "more" not in out
    _, _, err = gf_cli(monkeypatch, "contexts", "get", "nope", "-v")
    assert "note: request r-" in err


def test_the_entry_point_runs_as_a_module():
    import subprocess

    found = subprocess.run(
        [sys.executable, "-m", "generalforecast", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 0 and found.stdout.startswith("gf ")


def test_progress_lines_redraw_on_a_terminal_and_print_once_in_a_log():
    err = Stream(terminal=True)
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    with out.working() as work:
        work.show("Running", "run 1  0 of 3 models finished", log="run 1  3 models")
        work.show("Running", "run 1  2 of 3 models finished", log="run 1  3 models")
    assert err.getvalue().count("\r\x1b[K") == 2 and "2 of 3" in err.getvalue()
    log = io.StringIO()
    out = term.Output(out=io.StringIO(), err=log)
    with out.working() as work:
        work.show("Running", "run 1  0 of 3 models finished", log="run 1  3 models")
        work.show("Running", "run 1  2 of 3 models finished", log="run 1  3 models")
    assert log.getvalue() == "     Running run 1  3 models\n"
