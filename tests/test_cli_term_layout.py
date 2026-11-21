# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Output laid out for the terminal's width: folding, the role table, lists and prompts."""

import io
import os

import pytest

from generalforecast import term


class Tty(io.StringIO):
    def isatty(self):
        return True


# The inputs of the churn table that first showed the role table breaking.
INPUTS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
ROLES = [
    ("id", "customerID", "from gf.yaml"),
    ("inputs", INPUTS, "every other column"),
    ("target", "Churn", "from gf.yaml"),
    (
        "context",
        "6,943 rows",
        "churn.csv; 100 queries are a random 0.0142 of the rows",
    ),
]
WIDTHS = (200, 100, 60, 40)


@pytest.fixture
def columns(monkeypatch):
    """Sets the width every stream reports, as a terminal of that size would."""

    def resize(width, height=24):
        monkeypatch.setattr(
            term, "size", lambda stream=None: os.terminal_size((width, height))
        )

    return resize


def fits(lines, width):
    """Whether no line reaches the terminal's last column."""
    return all(term.visible(line) <= width - term.MARGIN for line in lines)


def test_text_that_fits_is_one_line_unchanged():
    text = "churn  classification, 19 inputs, 6,943 context rows, 100 query rows"
    assert term.wrap(text, 80) == [text]
    assert term.wrap("", 10) == [""]
    assert term.wrap(["a", "b"], 10) == ["a, b"]


def test_text_folds_after_a_comma_before_it_folds_at_a_space():
    text = "churn  classification, 19 inputs, 6,943 context rows, 100 query rows"
    assert term.wrap(text, 46) == [
        "churn  classification, 19 inputs,",
        "6,943 context rows, 100 query rows",
    ]
    # No comma is in reach, so the words fold, and a number keeps its comma.
    assert term.wrap(
        "churn.csv; 100 queries are a random 0.0142 of 6,943 rows", 30
    ) == [
        "churn.csv; 100 queries are a",
        "random 0.0142 of 6,943 rows",
    ]


def test_an_item_wider_than_a_line_folds_at_its_spaces_from_where_it_starts():
    note = (
        "100 rows, the most general-forecast/kumo-tabular-medium-classifier "
        "scores in one call"
    )
    assert term.wrap(note, 86) == [note]
    assert term.wrap(note, 46) == [
        "100 rows, the most general-forecast/",
        "kumo-tabular-medium-classifier scores in one",
        "call",
    ]
    # A name too long for a line starts its own: it is not split to fill one.
    assert term.wrap(["first", "a column whose name is long", "z"], 16) == [
        "first,",
        "a column whose",
        "name is long, z",
    ]


def test_a_word_wider_than_a_line_breaks_after_a_separator_or_at_the_edge():
    model = "general-forecast/kumo-tabular-medium-classifier"
    assert term.wrap(model, 46) == [
        "general-forecast/",
        "kumo-tabular-medium-classifier",
    ]
    assert term.wrap(model, 26) == [
        "general-forecast/",
        "kumo-tabular-medium-",
        "classifier",
    ]
    assert term.wrap("abcdefghij", 4) == ["abcd", "efgh", "ij"]
    assert term.wrap("/", 1) == ["/"]


@pytest.mark.parametrize("width", [1, 2, 3, 7, 16, 40, 59, 99])
def test_no_folded_line_is_wider_than_asked_whatever_the_characters(width):
    texts = [
        ", ".join(INPUTS),
        "顧客番号, 性別, 高齢者, 契約形態, 月額料金, 支払方法, 解約",
        "日本語の列名がとても長い場合でも折り返す",
        "é" * 50 + " " + "a/b-c_d.e:f" * 9,
        "x  y   z, , ,",
    ]
    for text in texts:
        lines = term.wrap(text, width)
        # A wide character takes two columns, so one alone can pass a width of 1.
        assert all(term.visible(line) <= max(width, 2) for line in lines), lines
        kept = "".join(lines).replace(" ", "")
        assert kept == text.replace(" ", "")


def test_wide_characters_fold_by_the_columns_they_take():
    names = ["顧客番号", "性別", "高齢者", "契約形態", "月額料金", "支払方法"]
    assert term.wrap(names, 20) == [
        "顧客番号, 性別,",
        "高齢者, 契約形態,",
        "月額料金, 支払方法",
    ]
    assert [term.visible(line) for line in term.wrap(names, 20)] == [15, 17, 18]


def test_a_long_list_is_cut_to_a_few_lines_and_counts_the_rest():
    assert term.LIST_LINES == 4
    lines, hidden = term.fold(INPUTS, 88)
    assert hidden == 0 and len(lines) == 4
    lines, hidden = term.fold(INPUTS, 48)
    assert lines == [
        "gender, SeniorCitizen, Partner, Dependents,",
        "tenure, PhoneService, MultipleLines,",
        "InternetService, OnlineSecurity, OnlineBackup,",
        "DeviceProtection, TechSupport, and 7 more",
    ]
    assert hidden == 7
    lines, hidden = term.fold(INPUTS, 28)
    assert lines[-1] == "InternetService, and 11 more" and hidden == 11
    for width in (16, 28, 48, 88, 300):
        for limit in (1, 2, 4, 9):
            lines, hidden = term.fold(INPUTS, width, limit)
            shown = sum(name in " ".join(lines).split(", ") for name in INPUTS)
            assert shown + hidden == len(INPUTS)
            assert not hidden or lines[-1].endswith(f"and {hidden} more")
            assert all(term.visible(line) <= width for line in lines)
    every, hidden = term.fold(INPUTS, 28, None)
    assert hidden == 0 and len(every) == 12


def test_the_role_table_at_one_hundred_columns_keeps_notes_beside_short_values():
    assert term.layout(ROLES, 100) == [
        "  id       customerID  from gf.yaml",
        "  inputs   gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,",
        "           InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,",
        "           StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,",
        "           TotalCharges  every other column",
        "  target   Churn       from gf.yaml",
        "  context  6,943 rows  churn.csv; 100 queries are a random 0.0142 of the rows",
    ]


def test_the_role_table_at_sixty_columns_counts_the_rest_and_drops_notes_under():
    assert term.layout(ROLES, 60) == [
        "  id       customerID  from gf.yaml",
        "  inputs   gender, SeniorCitizen, Partner, Dependents,",
        "           tenure, PhoneService, MultipleLines,",
        "           InternetService, OnlineSecurity, OnlineBackup,",
        "           DeviceProtection, TechSupport, and 7 more",
        "           every other column",
        "  target   Churn       from gf.yaml",
        "  context  6,943 rows",
        "           churn.csv; 100 queries are a random 0.0142 of",
        "           the rows",
    ]


def test_the_role_table_at_forty_columns_is_still_a_table():
    assert term.layout(ROLES, 40) == [
        "  id       customerID  from gf.yaml",
        "  inputs   gender, SeniorCitizen,",
        "           Partner, Dependents, tenure,",
        "           PhoneService, MultipleLines,",
        "           InternetService, and 11 more",
        "           every other column",
        "  target   Churn       from gf.yaml",
        "  context  6,943 rows",
        "           churn.csv; 100 queries are a",
        "           random 0.0142 of the rows",
    ]


def test_a_wide_terminal_keeps_a_long_value_on_one_line_without_moving_the_notes():
    wide = term.layout(ROLES, 300)
    assert wide == [
        "  id       customerID  from gf.yaml",
        f"  inputs   {', '.join(INPUTS)}  every other column",
        "  target   Churn       from gf.yaml",
        "  context  6,943 rows  churn.csv; 100 queries are a random 0.0142 of the rows",
    ]
    # Every note fits beside the widest value here, so they do share a column.
    short = [
        ("id", "order_id", "from gf.yaml"),
        ("inputs", ["amount", "items", "signup_source"], "every other column"),
        ("target", "returned", "from gf.yaml"),
        ("context", "10,000 rows", "context.csv"),
    ]
    assert term.layout(short, 100) == [
        "  id       order_id                      from gf.yaml",
        "  inputs   amount, items, signup_source  every other column",
        "  target   returned                      from gf.yaml",
        "  context  10,000 rows                   context.csv",
    ]


@pytest.mark.parametrize("width", WIDTHS)
def test_no_role_table_line_reaches_the_last_column(width):
    for limit in (term.LIST_LINES, None):
        lines = term.layout(ROLES, width, limit=limit)
        assert fits(lines, width), lines
        assert not any(line != line.rstrip() for line in lines)
        notes = [line.index("from gf.yaml") for line in lines if "from gf" in line]
        assert len(set(notes)) == 1
    assert ", ".join(INPUTS) in " ".join(
        line.strip() for line in term.layout(ROLES, width, limit=None)
    )


def test_a_note_with_no_room_beside_its_value_goes_under_it():
    rows = [
        ("past", ["price", "promo"], "every other column but id, store and weekday"),
        ("future", "none", ""),
    ]
    assert term.layout(rows, 44) == [
        "  past    price, promo",
        "          every other column but id,",
        "          store and weekday",
        "  future  none",
    ]
    # A folded value's short last line takes its note at the shared column.
    rows = [("id", "order", "from gf.yaml"), ("inputs", ["alpha", "beta", "c"], "x")]
    assert term.layout(rows, 32) == [
        "  id      order  from gf.yaml",
        "  inputs  alpha, beta, c  x",
    ]
    assert term.layout(rows, 20, start=10) == [
        "  id      order",
        "          from gf.yaml",
        "  inputs  alpha, beta, c",
        "          x",
    ]


def test_a_role_table_of_wide_characters_lines_up_by_columns():
    rows = [
        ("id", "顧客番号", "from gf.yaml"),
        ("inputs", ["性別", "高齢者", "契約形態", "月額料金", "支払方法"], "listed"),
        ("target", "解約", "from gf.yaml"),
    ]
    lines = term.layout(rows, 40)
    assert lines == [
        "  id      顧客番号  from gf.yaml",
        "  inputs  性別, 高齢者, 契約形態,",
        "          月額料金, 支払方法  listed",
        "  target  解約      from gf.yaml",
    ]
    assert fits(lines, 40)
    assert term.visible(lines[0].split("from")[0]) == term.visible(
        lines[3].split("from")[0]
    )


def test_details_colour_the_notes_and_the_count_and_verbose_shows_every_name(
    columns,
):
    columns(60)
    err = Tty()
    term.Output(out=io.StringIO(), err=err, environ={}).details(ROLES)
    shown = err.getvalue().splitlines()
    assert shown[0] == "  id       customerID  \x1b[2mfrom gf.yaml\x1b[0m"
    assert shown[4] == (
        "           DeviceProtection, TechSupport, \x1b[2mand 7 more\x1b[0m"
    )
    assert shown[5] == "           \x1b[2mevery other column\x1b[0m"
    err = io.StringIO()
    term.Output(out=io.StringIO(), err=err, verbose=True).details(ROLES)
    text = err.getvalue()
    assert "more" not in text and "TotalCharges\n" in text
    assert fits(text.splitlines(), 60)
    err = io.StringIO()
    term.Output(out=io.StringIO(), err=err, quiet=True).details(ROLES)
    assert err.getvalue() == ""


def test_a_stream_that_is_not_a_terminal_is_laid_out_one_hundred_columns_wide():
    assert term.WIDTH == 100 and term.size(io.StringIO()).columns == 100
    err = io.StringIO()
    term.Output(out=io.StringIO(), err=err).details(ROLES)
    assert err.getvalue().splitlines() == term.layout(ROLES, 100)


def test_a_status_line_folds_on_a_terminal_and_stays_one_line_in_a_log(columns):
    columns(40)
    message = "churn  classification, 19 inputs, 6,943 context rows, 100 query rows"
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.status("Resolved", message)
    out.status("Locked", "gf.lock", seconds=0.54)
    assert err.getvalue().splitlines() == [
        "    Resolved churn  classification,",
        "             19 inputs,",
        "             6,943 context rows,",
        "             100 query rows",
        "      Locked gf.lock in 0.54s",
    ]
    log = io.StringIO()
    term.Output(out=io.StringIO(), err=log).status("Resolved", message)
    assert log.getvalue() == f"    Resolved {message}\n"
    # A message with its own escapes, such as a link, is never cut into.
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={})
    out.status("Opening", out.link("https://app.generalforecast.com/a/long/path/to"))
    assert err.getvalue().count("\n") == 1


def test_a_progress_line_is_cut_to_the_terminal_so_it_can_be_redrawn(columns):
    columns(30)
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.progress("Predicting", "1,200 of 9,000 rows with a long model name  12s")
    assert err.getvalue() == "\r\x1b[K  Predicting 1,200 of 9,000 …"
    assert term.visible(err.getvalue()[4:]) == 29
    # What trails the line, such as the time so far, is never what is cut.
    out.progress("Predicting", "1,200 of 9,000 rows", trail="  2m10s")
    assert err.getvalue().endswith("\r\x1b[K  Predicting 1,200 of…  2m10s")


def test_an_answer_folds_its_list_and_says_how_many(columns):
    columns(60)
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err)
    out.collapsed("Inputs", ["spend", "plan", "signups"], noun="column")
    out.collapsed("Inputs", INPUTS, noun="column")
    out.collapsed("Task", "classification", "Churn has 2 distinct bool values")
    out.collapsed("Target", "Churn")
    out.collapsed("Series", "none", noun="column")
    assert err.getvalue().splitlines() == [
        "Inputs       spend, plan, signups",
        "Inputs       gender, SeniorCitizen, Partner, Dependents,",
        "             tenure, PhoneService, MultipleLines,",
        "             InternetService, OnlineSecurity, OnlineBackup,",
        "             DeviceProtection, TechSupport, and 7 more",
        "             19 columns",
        "Task         classification",
        "             Churn has 2 distinct bool values",
        "Target       Churn",
        "Series       none",
    ]
    columns(100)
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err)
    out.collapsed("Inputs", INPUTS, noun="column")
    out.collapsed("Task", "classification", "Churn has 2 distinct bool values")
    assert err.getvalue().splitlines() == [
        "Inputs       gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,",
        "             InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,",
        "             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,",
        "             MonthlyCharges, TotalCharges  19 columns",
        "Task         classification  Churn has 2 distinct bool values",
    ]


def test_a_table_folds_its_last_column_but_never_a_piped_result(columns):
    columns(40)
    plan = [
        ("name", "churn"),
        ("model", "general-forecast/tabicl-v2-classifier@7a01c2e"),
        ("first call", "about $0.0004 after a cold start of about 40s"),
    ]
    err, piped = io.StringIO(), io.StringIO()
    out = term.Output(out=piped, err=err)
    out.table(plan, indent=2)
    assert err.getvalue().splitlines() == [
        "  name        churn",
        "  model       general-forecast/",
        "              tabicl-v2-",
        "              classifier@7a01c2e",
        "  first call  about $0.0004 after a",
        "              cold start of about 40s",
    ]
    out.fields(plan)
    assert piped.getvalue().splitlines() == [
        "  name        churn",
        "  model       general-forecast/tabicl-v2-classifier@7a01c2e",
        "  first call  about $0.0004 after a cold start of about 40s",
    ]
    shown = Tty()
    term.Output(out=shown, err=err, environ={"NO_COLOR": "1"}).fields(plan)
    assert fits(shown.getvalue().splitlines(), 40)
    # Right-aligned numbers are never folded.
    numbers = io.StringIO()
    term.Output(out=io.StringIO(), err=numbers).table(
        [["accuracy", "0." + "9" * 40]], align=["<", ">"]
    )
    assert numbers.getvalue().count("\n") == 1


def test_lock_changes_fold_under_their_value():
    changes = [
        ("+", "input", None, "signup_source"),
        ("~", "selection", None, "core 1,000 → 1,500, neighbors 3,000 → 4,500"),
    ]
    assert term.change_lines(changes) == [
        " + input      signup_source",
        " ~ selection  core 1,000 → 1,500, neighbors 3,000 → 4,500",
    ]
    assert term.change_lines(changes, width=100) == term.change_lines(changes)
    assert term.change_lines(changes, width=40) == [
        " + input      signup_source",
        " ~ selection  core 1,000 → 1,500,",
        "              neighbors 3,000 → 4,500",
    ]


class Frames(term.Prompter):
    """A prompter on a terminal of a given size: keeps each frame, then answers Enter."""

    def __init__(self, out, width, height=24):
        super().__init__(out, stdin=io.StringIO(""), interactive=True)
        self.size, self.frames = (width, height), []

    def _raw_session(self, draw, handle):
        self.frames.append(draw(*self.size))
        handle("\r")


def _options(names, detail="category"):
    return [term.Option(name, name, detail) for name in names]


@pytest.mark.parametrize("width", WIDTHS)
def test_a_selector_fits_the_terminal_and_counts_what_is_chosen(width, columns):
    columns(width)
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err)
    prompter = Frames(out, width)
    chosen = prompter.choose_many(
        "Inputs",
        _options(INPUTS),
        flag="--inputs",
        selected=range(len(INPUTS)),
        noun="column",
    )
    assert chosen == INPUTS
    frame = prompter.frames[0]
    assert frame[0] == "Inputs  19 of 19 selected"
    assert frame[1] == "  > [x] gender              category"
    assert frame[-1] == "    [x] TotalCharges        category"
    assert fits(frame, width)
    assert fits(err.getvalue().splitlines(), width)


def test_a_selector_keeps_names_whole_before_details_and_marks_a_cut_name():
    models = [
        "general-forecast/kumo-tabular-medium-classifier",
        "general-forecast/mitra-classifier",
        "general-forecast/tabicl-v2-classifier",
    ]
    out = term.Output(out=io.StringIO(), err=io.StringIO())
    wide, narrow, tiny = Frames(out, 100), Frames(out, 60), Frames(out, 40)
    for prompter in (wide, narrow, tiny):
        prompter.choose("Model", _options(models, "classification"), flag="--model")
    assert wide.frames[0] == [
        "Model",
        "  > general-forecast/kumo-tabular-medium-classifier    classification",
        "    general-forecast/mitra-classifier                  classification",
        "    general-forecast/tabicl-v2-classifier              classification",
    ]
    # The longest name leaves its detail no room; the others keep theirs, in line.
    assert narrow.frames[0] == [
        "Model",
        "  > general-forecast/kumo-tabular-medium-classifier",
        "    general-forecast/mitra-classifier        classification",
        "    general-forecast/tabicl-v2-classifier    classification",
    ]
    assert tiny.frames[0] == [
        "Model",
        "  > general-forecast/kumo-tabular-medi…",
        "    general-forecast/mitra-classifier",
        "    general-forecast/tabicl-v2-classif…",
    ]
    assert fits(tiny.frames[0], 40) and term.shorten("abcdef", 4) == "abc…"
    assert term.shorten("日本語の列", 6) == "日本…" and term.shorten("abc", 3) == "abc"


def test_an_edited_answer_shows_its_note_under_it_when_there_is_no_room():
    out = term.Output(out=io.StringIO(), err=Tty(), environ={})
    prompter = Frames(out, 60)
    note = "100 rows, the most general-forecast/tabicl-v2-classifier scores in one call"
    prompter.ask("Holdout", "0.0142", flag="--holdout", note=lambda value: note)
    plain = [term._ANSI.sub("", line) for line in prompter.frames[0]]
    assert plain == [
        "Holdout      0.0142",
        "             100 rows, the most",
        "             general-forecast/tabicl-v2-classifier scores",
        "             in one call",
    ]
    assert prompter.frames[0][1].startswith("             \x1b[2m100 rows")
    wrong = Frames(out, 40)
    wrong.ask(
        "Holdout",
        "2",
        flag="--holdout",
        check=lambda value: "must be between 0 and 1, like 0.2",
    )
    assert wrong.frames[0] == [
        "Holdout      2",
        "             \x1b[31mmust be between 0 and 1,\x1b[0m",
        "             \x1b[31mlike 0.2\x1b[0m",
    ]


def test_a_lock_change_writes_a_boolean_as_gf_yaml_does():
    changes = [("+", "context.holdout.stratify", None, True), ("~", "x", False, True)]
    assert term.change_lines(changes) == [
        " + context.holdout.stratify  true",
        " ~ x                         false → true",
    ]
