# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The CLI samples of the surface spec (sections 2.3 to 2.7), and the service state each starts from.

Each `Sample` is the commands of one sample block, the files they read, and
a seed that puts the fake service in the state the block shows: the UCI
Adult context, the churn and demand predictors, the sources and keys.
`tests/test_cli_samples.py` runs each at 100 and 44 columns, on a terminal
and not, against `tests/golden/<name>.txt`.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests import contract
from tests import fake as service
from tests.fake import NOW, TOKEN, Fake


def _model(full: str, short: str, task: str, rows: int) -> dict[str, Any]:
    """A catalog entry with the engine catalog's limits: a forecaster takes 8 inputs, a tabular model 10,000."""
    family = short.split("-")[0]
    inputs = 8 if task == "forecast" else 10000
    return service.model(f"general-forecast/{full}", short, task, family=family,
                         context_rows=rows, inputs=inputs)  # fmt: skip


CATALOG = [
    _model("kumo-tabular-small-classifier", "kumo-small", "classification", 60000),
    _model("kumo-tabular-small-regressor", "kumo-small", "regression", 60000),
    _model("kumo-tabular-medium-classifier", "kumo-medium", "classification", 60000),
    _model("kumo-tabular-medium-regressor", "kumo-medium", "regression", 60000),
    _model("kumo-tabular-large-classifier", "kumo-large", "classification", 60000),
    _model("kumo-tabular-large-regressor", "kumo-large", "regression", 60000),
    _model("tabicl-v2-classifier", "tabicl-v2", "classification", 100000),
    _model("tabicl-v2-regressor", "tabicl-v2", "regression", 100000),
    _model("mitra-2-classifier", "mitra-2", "classification", 5120),
    _model("mitra-2-regressor", "mitra-2", "regression", 5120),
    _model("chronos-2", "chronos-2", "forecast", 100000),
    _model("timestral-209m", "timestral-209m", "forecast", 100000),
]  # fmt: skip

ADULT_COLUMNS = [
    ("age", "number", 0), ("workclass", "category", 1836), ("fnlwgt", "number", 0),
    ("education", "category", 0), ("education_num", "number", 0), ("marital_status", "category", 0),
    ("occupation", "category", 1843), ("relationship", "category", 0), ("race", "category", 0),
    ("sex", "category", 0), ("capital_gain", "number", 0), ("capital_loss", "number", 0),
    ("hours_per_week", "number", 0), ("native_country", "category", 583), ("income", "category", 0),
]  # fmt: skip
ADULT_ROWS = [
    {"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education_num": 13,
     "marital_status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family",
     "race": "White", "sex": "Male", "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
     "native_country": "United-States", "income": "<=50K"},
    {"age": 50, "workclass": "Self-emp-not-inc", "fnlwgt": 83311, "education": "Bachelors",
     "education_num": 13, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
     "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0,
     "hours_per_week": 13, "native_country": "United-States", "income": "<=50K"},
    {"age": 38, "workclass": "Private", "fnlwgt": 215646, "education": "HS-grad", "education_num": 9,
     "marital_status": "Divorced", "occupation": "Handlers-cleaners", "relationship": "Not-in-family",
     "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
     "native_country": "United-States", "income": "<=50K"},
]  # fmt: skip


def adult(rows: int) -> pd.DataFrame:
    """Adult-shaped rows: the first three are the dataset's, the rest drawn from a fixed seed."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame([ADULT_ROWS[i % 3] for i in range(rows)])
    frame["age"] = rng.integers(17, 90, rows)
    frame["fnlwgt"] = rng.integers(12285, 1484705, rows)
    frame["income"] = rng.choice(["<=50K", ">50K"], rows, p=[0.76, 0.24])
    frame.iloc[:3] = pd.DataFrame(ADULT_ROWS)
    return frame


def context(fake: Fake, name: str, **values: Any) -> dict[str, Any]:
    """A ready snapshot context, version 1, unless `values` say otherwise."""
    columns = values.pop("columns", [("id", "number", 0)])
    ctx = {
        "name": name, "kind": "snapshot", "state": "ready", "version": 1, "versions": 1,
        "digest": service.digest(name), "rows": 32561, "id": None, "time": None, "source": None,
        "freshness_seconds": None, "read_at": None, "error": None, "predictors": [], "experiments": [],
        "fine_tunes": [],
        "origin": service.origin(), "created": NOW, "updated": NOW, "file": None,
        "url": fake.link(f"data/contexts/{name}"), "_versions": [], "_rows": [],
        "columns": [{"name": n, "type": t, "missing": m, "distinct": None} for n, t, m in columns],
    }  # fmt: skip
    ctx.update(values)
    fake.contexts[name] = ctx
    return ctx


def adult_context(fake: Fake, **values: Any) -> dict[str, Any]:
    return context(fake, "adult", columns=ADULT_COLUMNS, file="adult.csv", _rows=ADULT_ROWS,
                   updated="2026-11-21T09:12:04Z", **values)  # fmt: skip


def customers(fake: Fake, version: int = 8) -> dict[str, Any]:
    names = ["customer_id", "tenure_m", "plan", "monthly_usd", "seats", "region", "signup_date",
             "support_tickets", "churned"]  # fmt: skip
    return context(
        fake, "customers", version=version, versions=version, rows=48210,
        digest=service.digest(f"customers@{version}"),
        source={"source": "bigquery-acme", "kind": "bigquery", "table": "acme.analytics.customers", "sql": None},
        columns=[(n, "number", 0) for n in names], updated="2026-11-20T17:03:00Z",
        _versions=[{"version": v, "digest": service.digest(f"customers@{v}"), "rows": 48210}
                   for v in range(1, version + 1)],
    )  # fmt: skip


def experiment(
    fake: Fake, name: str, ctx: str, target: str, runs: list[str], **values: Any
) -> dict[str, Any]:
    """An experiment whose runs ended as `runs` says; the last may still be going."""
    made = []
    for number, state in enumerate(runs, 1):
        models = (
            ["tabicl-v2", "kumo-medium", "mitra-2"] if number == 1 else ["kumo-large"]
        )
        run = service._run(fake, name, number, models, state)
        if state == "finished":
            count = len(models)
            run.update(leaderboard=copy.deepcopy(service._entries(models)), rows=6513 * count,
                       credits=13.0 * count, usd=0.013 * count, started="2026-11-21T09:10:52Z",
                       finished=NOW, _polls=0)  # fmt: skip
        made.append(run)
    fake.experiments[name] = {
        "name": name, "title": "", "context": ctx, "context_version": 1, "target": target,
        "task": "classification", "id": None, "time": None, "inputs": [], "types": {}, "sql": None,
        "holdout": {"kind": "random", "fraction": 0.2}, "metric": "auroc", "origin": service.origin(),
        "created": NOW, "updated": "2026-11-21T09:20:00Z", "url": fake.link(f"experiments/{name}"),
        "_runs": made, "_setup": {}, **values,
    }  # fmt: skip
    return fake.experiments[name]


def predictor(
    fake: Fake,
    name: str,
    revisions: list[dict[str, Any]],
    envs: dict[str, tuple[int, int, float | None]],
) -> None:
    """A predictor with these revisions' specs, each environment serving `(revision, calls today, p50)`."""
    made = []
    for number, spec in enumerate(revisions, 1):
        resolved = service._resolved(fake, spec)
        lock = json.dumps({"spec": spec, "resolved": resolved}, sort_keys=True)
        made.append({"predictor": name, "number": number, "status": "ready",
                     "lock_digest": service.digest(lock), "spec": spec, "resolved": resolved,
                     "origin": service.origin("console" if number < len(revisions) else "cli"),
                     "created": "2026-11-18T14:40:00Z" if number < len(revisions) else "2026-11-21T10:02:00Z"})  # fmt: skip
    environments = {}
    for env, (number, calls, p50) in envs.items():
        environments[env] = {
            "name": env, "revision": number, "state": "ready", "generation": number,
            "resolved": made[number - 1]["resolved"], "calls_today": calls,
            "p50_seconds": p50, "updated": NOW, "url": fake.link(f"predictors/{name}"),
            "_previous": number - 1 if number > 1 else None,
        }  # fmt: skip
    fake.predictors[name] = {"name": name, "_revisions": made, "environments": environments,
                             "created": NOW, "url": fake.link(f"predictors/{name}"),
                             "origin": service.origin("console")}  # fmt: skip
    ctx = fake.contexts[str(revisions[-1]["context"]).partition("@")[0]]
    ctx["predictors"] = [*ctx["predictors"], name]


def adult_income(fake: Fake) -> None:
    """adult-income on revision 2, kumo-medium, in production and staging."""
    spec = {"context": "adult", "model": "tabicl-v2", "target": "income"}
    predictor(fake, "adult-income", [spec, dict(spec, model="kumo-medium")],
              {"production": (2, 1204, 0.9), "staging": (2, 0, None)})  # fmt: skip


def churn(fake: Fake, *, staging: int = 5) -> None:
    """churn: revision 4 kumo-small on customers 7, revision 5 kumo-medium on customers 8."""
    customers(fake)
    base = {
        "model": "kumo-small",
        "target": "churned",
        "id": "customer_id",
        "exclude": ["signup_date"],
    }
    revisions = [dict(base, context="customers@7") for _ in range(4)]
    revisions.append(dict(base, model="kumo-medium", context="customers@8"))
    predictor(
        fake,
        "churn",
        revisions,
        {"production": (4, 18330, 0.6), "staging": (staging, 210, 0.7)},
    )


def demand(fake: Fake) -> None:
    """demand-us on revision 12 and demand-eu on revision 9, as the CI example's file declares them."""
    for region, count in (("us", 12), ("eu", 9)):
        context(fake, f"sales-{region}", rows=104000, columns=[("week", "date", 0), ("sku", "category", 0),
                                                                ("units", "number", 0)])  # fmt: skip
        spec = {"context": f"sales-{region}", "model": "chronos-2", "target": "units", "time": "week",
                "series": ["sku"], "horizon": 8, "frequency": "7d"}  # fmt: skip
        predictor(
            fake, f"demand-{region}", [spec] * count, {"production": (count, 5200, 1.4)}
        )


PREDICTORS_PY = """import generalforecast as gf

adult = gf.Predictor("adult-income", context="adult", model="tabicl-v2", target="income")

churn = gf.Predictor(
    "churn",
    context="customers",
    model="kumo-medium",
    target="churned",
    id="customer_id",
    exclude=["signup_date"],
)

demand = [
    gf.Predictor(f"demand-{region}", context=f"sales-{region}", model="chronos-2",
                 target="units", time="week", series=["sku"], horizon=8, frequency="7d")
    for region in ("us", "eu")
]
"""

TYPO_PY = """import generalforecast as gf

adult = gf.Predictor("adult-income", context="adult", model="tabicl-v2", target="income")

churn = gf.Predictor(
    "churn",
    context="customers",
    target="churned",
    modle="kumo-medium",
)
"""

RUN_PY = """import generalforecast as gf

run = gf.FineTuneSpec(
    "tabicl-v2",
    phases=[
        gf.Phase("warmup", data=[gf.Tag("synthetic-credit-warmup")], steps=200),
        gf.Phase("credit", data=["credit-2026-01", "adult@1"], steps=800, episode_rows=4096),
    ],
    target="income",
    method=gf.LoRA(rank=16, input_adapter=True),
    architecture=[gf.ArchitectureChange("row_encoder", to={"layers": 4})],
    budget=gf.Budget(credits=500),
)
"""


@dataclass(frozen=True)
class Sample:
    """One sample block: its commands, the files they read, and the service state it starts from."""

    name: str
    commands: tuple[str, ...]
    seed: Callable[[Fake], None] = lambda fake: None
    files: dict[str, Callable[[], str | pd.DataFrame]] = field(default_factory=dict)
    stdin: str = ""
    logged_in: bool = True
    knobs: dict[str, Any] = field(default_factory=dict)


def _sources(fake: Fake) -> None:
    for name, kind, access, config, tested, tables, contexts in (
        ("bigquery-acme", "bigquery", "service_account", {"project": "acme-prod", "dataset": "analytics"},
         "2026-11-21T08:02:00Z", 42, 3),
        ("snowflake-prod", "snowflake", "secret", {"account": "acme-xy12", "warehouse": "ANALYTICS_XS"},
         "2026-11-20T16:10:00Z", 18, 0),
    ):  # fmt: skip
        fake.sources[name] = {"name": name, "kind": kind, "access": access, "state": "connected",
                              "config": config, "secret_expires": None, "tested_at": tested,
                              "tables": tables, "contexts": contexts, "error": None,
                              "origin": service.origin("console"), "created": NOW,
                              "url": fake.link(f"data/sources/{name}")}  # fmt: skip


# The keys of the samples, by the ids the CLI is given.
CLI_KEY, CI_KEY, OLD_KEY = (
    "key-19ab0c1d2e3f40516273",
    "key-8f2c0c1d2e3f40516273",
    "key-02cd0c1d2e3f40516273",
)


def _keys(fake: Fake) -> None:
    from generalforecast.types import STANDARD_PERMISSIONS

    maya = service.OWNER
    raj = {**maya, "subject": "108234567890123456790", "email": "raj@acme.com"}
    for key_id, name, prefix, expires, used, revoked, owner, machine in (
        (CLI_KEY, "cli-maya-mbp", "gf_19ab", 1803029524.0, 1795255470.0, None, maya, "maya-mbp"),
        (CI_KEY, "ci-deploy", "gf_8f2c", None, 1795255320.0, None, raj, None),
        (OLD_KEY, "old-laptop", "gf_02cd", 1780272000.0, None, 1780272000.0, None, None),
    ):  # fmt: skip
        fake.keys[key_id] = {"id": key_id, "project": fake.project, "name": name,
                             "permissions": list(STANDARD_PERMISSIONS), "created_at": 1785542400.0,
                             "expires_at": expires, "revoked_at": revoked,
                             "state": "revoked" if revoked else "active", "prefix": prefix,
                             "last_used_at": used, "owner": owner, "service": False,
                             "legacy": owner is None, "machine": machine}  # fmt: skip


def _all_contexts(fake: Fake) -> None:
    adult_context(fake, experiments=["adult-income"], predictors=["adult-income"])
    customers(fake)
    fake.contexts["customers"].update(
        experiments=["churn-nov"], predictors=["churn"], version=1, versions=1
    )
    fake.contexts["customers"]["predictors"] = ["churn", "churn-nov"]
    context(fake, "events", kind="live", rows=12804, version=None, versions=0, freshness_seconds=300,
            source={"source": "snowflake-prod", "kind": "snowflake", "table": None, "sql": "select ..."},
            updated="2026-11-21T10:03:00Z", predictors=["conversion"])  # fmt: skip
    context(fake, "signups", kind="live", rows=1904, version=None, versions=0, updated="2026-11-21T10:04:40Z",
            columns=[("plan", "category", 0), ("seats", "number", 0), ("converted", "boolean", 0)])  # fmt: skip


def _adult_experiments(fake: Fake) -> None:
    adult_context(fake)
    experiment(fake, "adult-income", "adult", "income", ["finished"])


def _refused(fake: Fake) -> None:
    """snowflake-prod refuses its test, as the contract's example has it."""
    _sources(fake)
    op = next(op for op in contract.operations() if op.id == "sources.test")
    refused = op.body["responses"]["422"]["content"]["application/json"]["examples"]
    fake.source_errors["snowflake-prod"] = next(iter(refused.values()))["value"][
        "error"
    ]


def _logs(fake: Fake) -> None:
    churn(fake)
    shown = contract.example("environments.logs")["logs"]
    fake.log_entries = [
        {**shown[0], "id": "1795255452000000", "at": 1795255452.0, "source": "predict", "code": "Predicted",
         "message": "100 rows  0.92s  key ci-deploy", "environment": "production", "revision": 4,
         "request_id": "r-3f1a92c0", "fields": {"rows": 100}},
        {**shown[1], "id": "1795255455000000", "at": 1795255455.0, "source": "predict",
         "code": "MissingColumn", "message": "missing column plan, filled empty",
         "environment": "production", "revision": 4, "request_id": "r-77b0e1aa", "fields": {}},
    ]  # fmt: skip


def _cancel(fake: Fake) -> None:
    adult_context(fake)
    experiment(fake, "adult-income", "adult", "income", ["finished", "running"])


def _two_experiments(fake: Fake) -> None:
    _all_contexts(fake)
    experiment(fake, "adult-income", "adult", "income", ["finished", "finished"])
    found = experiment(
        fake,
        "churn-nov",
        "customers",
        "churned",
        ["finished"] * 3,
        origin=service.origin("console"),
    )
    for run in found["_runs"]:
        run.update(_best="kumo-medium", _score=0.871)


def _deploy(fake: Fake) -> None:
    adult_context(fake)
    _failing(fake)
    fake.deploy_ends, fake.polls_until_done = "ready", 2
    demand(fake)


def _unchanged(fake: Fake) -> None:
    adult_context(fake)
    churn(fake)
    demand(fake)
    spec = {"context": "adult", "model": "tabicl-v2", "target": "income"}
    predictor(fake, "adult-income", [spec, spec], {"production": (2, 1204, 0.9)})
    fake.predictors["churn"]["environments"]["production"]["revision"] = 5


def _login(fake: Fake) -> None:
    fake.key = {"id": CLI_KEY, "name": "cli-maya-mbp", "expires_at": 1803029524.0}


def _logout(fake: Fake) -> None:
    from generalforecast import _auth

    _login(fake)
    login = _auth.Stored(fake.url, CLI_KEY, "acme", "maya@acme.com", key=TOKEN)
    login.name, login.minted = "cli-maya-mbp", True
    _auth.CredentialStore().save(login)


def _failing(fake: Fake) -> None:
    churn(fake)
    fake.predictors["churn"]["environments"].pop("staging")
    fake.predictors["churn"]["_revisions"].pop()
    fake.deploy_ends = "failed"
    fake.polls_until_done = 3


def _snapshot(fake: Fake) -> None:
    import pyarrow as pa

    service._store_snapshot(
        fake, "credit-2026-01", pa.table({"income": [float(i) for i in range(40)], "defaulted": [i % 2 for i in range(40)]}),
        {"target": "defaulted", "task": "classification"}, ["nightly"],
    )  # fmt: skip


def _fine_tune(fake: Fake) -> None:
    _snapshot(fake)
    adult_context(fake)
    fake.batches["credit-warmup"] = {"name": "credit-warmup", "tag": "synthetic-credit-warmup", "count": 8,
                                     "seed": 7, "options": {}, "state": "done", "counts": {"ready": 8},
                                     "tables": [], "created": 1761200000.0, "_polls": 0}  # fmt: skip


def _ran(fake: Fake) -> None:
    import generalforecast as gf

    _snapshot(fake)
    gf.fine_tunes.runs.create("credit-default", base="tabicl-v2", data=["credit-2026-01"],
                              plan=gf.Plan(steps=1000), wait=True)  # fmt: skip
    gf.fine_tunes.runs.create(
        "credit-default",
        base="tabicl-v2",
        data=["credit-2026-01"],
        plan=gf.Plan(steps=1000),
    )


def _queries(rows: int) -> Callable[[], pd.DataFrame]:
    return lambda: adult(rows).drop(columns="income")


SAMPLES = [
    Sample("login", ("login",), _login, logged_in=False, knobs={"pending_logins": 2}),
    Sample("whoami", ("whoami",), _login),
    Sample("contexts-create-file", ("contexts create adult --file adult.csv",),
           files={"adult.csv": lambda: adult(32561)}),
    Sample("contexts-upload-unchanged", ("contexts upload adult adult.csv",),
           lambda fake: _upload_adult(fake), files={"adult.csv": lambda: adult(2000)}),
    Sample("contexts-create-source",
           ("contexts create customers --source bigquery-acme --table acme.analytics.customers --id customer_id",),
           knobs={"polls_until_done": 3}),
    Sample("contexts-create-live-source",
           ("contexts create events --source snowflake-prod --sql-file events.sql --live --freshness 5m --time event_at",),
           files={"events.sql": lambda: "select * from events where event_at > now() - interval '30 days'\n"}),
    Sample("contexts-live-api", (
        "contexts create signups --live --columns plan=category,seats=number,converted=boolean --id account_id --targets converted",
        "contexts append signups rows.csv",
        "contexts label signups outcomes.csv --target converted",
    ), files={
        "rows.csv": lambda: pd.DataFrame({"account_id": range(250), "plan": ["team"] * 250, "seats": [5] * 250}),
        "outcomes.csv": lambda: pd.DataFrame({"account_id": range(41), "converted": [True] * 41}),
    }),
    Sample("contexts-list", ("contexts list",), _all_contexts),
    Sample("contexts-get", ("contexts get adult",), lambda fake: adult_context(
        fake, experiments=["adult-income"], predictors=["adult-income"])),
    Sample("sources-list", ("sources list",), _sources),
    Sample("keys-create", ("keys create ci-deploy --expires 90d",)),
    Sample("models-list", ("models list",)),
    Sample("experiments-create", ("experiments create --context adult --target income --models kumo-medium,tabicl-v2,mitra-2",),
           adult_context, knobs={"polls_until_done": 4}),
    Sample("experiments-estimate", ("experiments estimate --context adult --target income",), adult_context),
    Sample("experiments-list", ("experiments list",), _two_experiments),
    Sample("experiments-cancel", ("experiments cancel adult-income",), _cancel),
    Sample("experiments-get", ("experiments get adult-income",), _adult_experiments),
    Sample("experiments-get-json", ("experiments get adult-income --json",), _adult_experiments),
    Sample("predictors-create", ("predictors create adult-income --context adult --model tabicl-v2 --target income",),
           adult_context, knobs={"polls_until_done": 3}),
    Sample("predictors-update", ("predictors update adult-income --model kumo-medium",),
           lambda fake: (adult_context(fake), predictor(fake, "adult-income", [
               {"context": "adult", "model": "tabicl-v2", "target": "income"}], {"production": (1, 0, None)}))),
    Sample("predictors-get", ("predictors get adult-income",), lambda fake: (adult_context(fake), adult_income(fake))),
    Sample("predictors-predict", ("predictors predict adult-income --queries applicants.csv --output scored.csv",),
           lambda fake: (adult_context(fake), adult_income(fake)), files={"applicants.csv": _queries(1000)}),
    Sample("predictors-promote-rollback", ("predictors promote churn --from staging --env production",
                                           "predictors rollback churn"), churn),
    Sample("deploy", ("deploy predictors.py --env production",), _deploy, files={"predictors.py": lambda: PREDICTORS_PY}),
    Sample("deploy-dry-run", ("deploy predictors.py --dry-run",), _deploy, files={"predictors.py": lambda: PREDICTORS_PY}),
    Sample("deploy-unchanged", ("deploy predictors.py",), _unchanged, files={"predictors.py": lambda: PREDICTORS_PY}),
    Sample("error-not-logged-in", ("contexts list",), logged_in=False),
    Sample("error-no-model", ("predictors create churn --context customers --model kumo-med --target churned",),
           customers),
    Sample("error-load", ("deploy predictors.py",), files={"predictors.py": lambda: TYPO_PY}),
    Sample("error-no-column", ("experiments create --context adult --target incme",), adult_context),
    Sample("error-failed-deploy", ("predictors update churn --model kumo-medium --context customers",), _failing),
    Sample("logout", ("logout",), _logout, logged_in=False),
    Sample("contexts-refresh", ("contexts refresh customers",), customers),
    Sample("contexts-rows", ("contexts rows adult --limit 2",), adult_context),
    Sample("contexts-delete", ("contexts delete signups --yes",), _all_contexts),
    Sample("experiments-run", ("experiments run adult-income --models kumo-large",), _adult_experiments,
           knobs={"polls_until_done": 3}),
    Sample("experiments-update", ('experiments update adult-income --holdout "20% stratified"',), _two_experiments),
    Sample("experiments-wait", ("experiments wait adult-income",),
           lambda fake: (adult_context(fake), experiment(fake, "adult-income", "adult", "income",
                                                         ["finished", "finished", "running"]))),
    Sample("experiments-predictions", ("experiments predictions adult-income --model tabicl-v2 --output held.csv",),
           _adult_experiments),
    Sample("experiments-slices", ("experiments slices adult-income --model tabicl-v2",), _adult_experiments),
    Sample("experiments-delete", ("experiments delete churn-old --yes",),
           lambda fake: (customers(fake), experiment(fake, "churn-old", "customers", "churned", ["finished"] * 3))),
    Sample("predictors-list", ("predictors list",), lambda fake: (adult_context(fake), adult_income(fake), churn(fake))),
    Sample("predictors-revisions", ("predictors revisions churn",), churn),
    Sample("predictors-logs", ("predictors logs churn --limit 2",), _logs),
    Sample("predictors-delete", ("predictors delete churn",), churn, stdin="y\n"),
    Sample("sources-get", ("sources get bigquery-acme",), _sources),
    Sample("sources-test", ("sources test snowflake-prod",), _refused),
    Sample("sources-create", ("sources create pg-replica --kind postgres --config host=10.2.0.4,port=5432,database=app,user=gf_read --secret-stdin",),
           stdin="hunter2\n"),
    Sample("sources-preview", ("sources preview bigquery-acme --table acme.analytics.customers",), _sources),
    Sample("sources-delete", ("sources delete pg-replica --yes",), lambda fake: fake.sources.update(
        {"pg-replica": {"name": "pg-replica", "kind": "postgres", "state": "ready", "config": {}}})),
    Sample("keys-list", ("keys list",), _keys),
    Sample("keys-rotate", (f"keys rotate {CI_KEY} --yes",), _keys),
    Sample("keys-delete", (f"keys delete {OLD_KEY} --yes",), _keys),
    Sample("models-get", ("models get kumo-medium --task regression",)),
    Sample("models-predict", ("models predict tabicl-v2 --context train.csv --queries new.csv --target income",),
           files={"train.csv": lambda: adult(26048), "new.csv": _queries(3)}),
    Sample("fine-tunes-snapshots", (
        "fine-tunes snapshots create credit-2026-01 --file credit.csv --tag nightly --target defaulted --task classification",
        "fine-tunes snapshots list",
    ), files={"credit.csv": lambda: pd.DataFrame({"income": [float(i) for i in range(40)],
                                                  "defaulted": [i % 2 for i in range(40)]})}),
    Sample("fine-tunes-runs-create", (
        "fine-tunes runs create credit-default --base tabicl-v2 --data credit-2026-01 --steps 1000 --wait",
    ), _snapshot),
    Sample("fine-tunes-runs-from", ("fine-tunes runs create credit-default --from run.py --dry-run",),
           _fine_tune, files={"run.py": lambda: RUN_PY}),
    Sample("fine-tunes-runs-list", ("fine-tunes runs list", "fine-tunes runs get credit-default@ft1"), _ran),
]  # fmt: skip


def _upload_adult(fake: Fake) -> None:
    import generalforecast as gf

    gf.contexts.create("adult", data="adult.csv")


def files(sample: Sample, folder: Path) -> None:
    """Writes the files a sample's commands read."""
    for name, make in sample.files.items():
        made = make()
        if isinstance(made, pd.DataFrame):
            made.to_csv(folder / name, index=False)
        else:
            (folder / name).write_text(made)
