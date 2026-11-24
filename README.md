# General Forecast

The Python SDK for [General Forecast](https://generalforecast.com): contexts,
experiments and predictors on tabular and time series foundation models, run
on the hosted service.

- Docs: https://docs.generalforecast.com
- API reference: https://docs.generalforecast.com/api

```sh
uv pip install generalforecast
```

Python 3.12 or later. The package is pure Python; every prediction runs on the
service.

## Quickstart: UCI Adult in a notebook

```python
import generalforecast as gf
import pandas as pd

gf.login()
# Open https://app.generalforecast.com/device and enter the code WXHG-PTRB
# Logged in as maya@acme.com, project acme. This notebook holds a token until 21:14.

cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"]
adult = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                    names=cols, skipinitialspace=True)

ctx = gf.contexts.create("adult", adult)
ctx        # Context adult  snapshot v1  32,561 rows  15 columns

gf.experiments.estimate(context="adult", target="income",
                        models=["kumo-medium", "tabicl-v2", "mitra-2"])
# Estimate(calls=6, rows=19539, credits=39.1, usd=0.0391)

exp = gf.experiments.create(context="adult", target="income", holdout="20%",
                            models=["kumo-medium", "tabicl-v2", "mitra-2"])
exp        # leaderboard, as HTML in Jupyter, Colab, Hex, marimo
print(exp)
# Experiment adult-income  run 1  classification, target income, 6,513 held-out rows
#      Model        AUROC  Interval        Time  Cost
#   1  tabicl-v2    0.931  0.925 to 0.937  38s   $0.013
#   2  kumo-medium  0.929  0.923 to 0.935  51s   $0.013
#   3  mitra-2      0.918  0.911 to 0.925  44s   $0.013
#   https://app.generalforecast.com/?project=acme#/experiments/adult-income

gf.experiments.slices(exp, model=exp.best.model)[:3]     # error by slice
gf.experiments.run(exp, models=["kumo-large"])           # run 2, same setup

p = gf.predictors.create("adult-income", context="adult",
                         model=exp.best.model, target="income")
p.environments["production"].state                        # 'ready'

out = gf.predictors.predict("adult-income", adult.drop(columns="income").head(5))
out.to_pandas()            # label, p_<=50K, p_>50K
out.credits                # 0.01

clf = gf.Classifier("tabicl-v2").fit(adult.drop(columns="income")[:26048],
                                     adult.income[:26048])
clf.estimate(adult.drop(columns="income")[26048:])       # Estimate(calls=2, rows=6513, ...)
```

The experiment appears in the console under Experiments with origin
`notebook`.

## Logging in

`gf.login()` holds a 12-hour token in memory and never writes a file. Outside
a notebook, set `GF_API_KEY` (and `GF_PROJECT` if the key spans projects).
Credentials resolve in this order: `gf.Client(api_key=...)`, `gf.login()` in
this process, `GF_API_KEY`, then the credentials file `gf login` writes;
`GF_PROJECT` and `GF_URL` override the project and the server.

Requests honour `HTTPS_PROXY` and the system CA bundle, so the SDK works behind
a corporate proxy; `gf.Client(trust_env=False)` opts out.

## Deploying from code

`predictors.deploy` is idempotent: per spec it creates the predictor, makes a
revision when the resolved spec differs from what the environment serves, or
does nothing.

```python
import generalforecast as gf

adult = gf.Predictor(
    "adult-income", context="adult", model="tabicl-v2", target="income"
)
churn = gf.Predictor(
    "churn",
    context="customers",
    model="kumo-medium",
    target="churned",
    id="customer_id",
    exclude=["signup_date"],
)

for deploy in gf.predictors.deploy(adult, churn, env="staging"):
    print(deploy.predictor, deploy.action)
gf.predictors.promote("churn", from_env="staging", env="production")
```

## The `gf` command

`gf <resource> <verb> [NAME] [flags]` is the Python API from a shell, word for
word: `gf experiments wait adult-income --run 2` calls
`gf.experiments.wait("adult-income", run=2)`. A file of rows is the argument
after the name, as the SDK takes it: `gf predictors predict churn new.csv` calls
`gf.predictors.predict("churn", "new.csv")`, and `gf models get tabicl-v2` shows
each of a family's models. Status lines go to stderr and results to stdout;
`--json` prints the result as one JSON object.

```console
$ gf login
$ gf contexts create adult adult.csv
   Uploading adult.csv  32,561 rows, 15 columns, 1.1 MiB
     Created context adult version 1  32,561 rows, 15 columns in 3.4s
        View https://app.generalforecast.com/?project=acme#/data/contexts/adult
$ gf experiments create --context adult --target income
$ gf deploy predictors.py --env staging
```

`gf deploy FILE` runs a Python file of `gf.Predictor` definitions and deploys
each, printing the plan first; it never prompts and never deletes a predictor
the file leaves out. In CI:

```yaml
- run: uvx --from generalforecast gf contexts upload customers data/customers.parquet
- run: uvx --from generalforecast gf deploy predictors.py --env staging
- run: uvx --from generalforecast gf predictors promote churn --from staging --env production
```

`gf help <command>` lists a command's flags. Exit codes: 0 done (nothing to
change included), 1 failed, 2 a command line `gf` cannot read or a question
declined, 130 Ctrl-C, 141 a closed pipe.

## Contexts

A context is uploaded (its rows, `data`), imported from a source (`source=` with
`table=` or `sql=`), read live from a source (`live=True`), or fed through the
API (`live=True` with `columns=`, then `append` and `label`). Uploads add a
version only when the rows changed, so a CI re-run is a no-op. Any reference
can pin a version: `"adult@2"`.

## Models and estimators

`gf.models.predict(model, queries, context=..., target=...)` sends the
context once, as an Arrow snapshot named by its digest, and the queries as
Arrow in parts by bytes. `Classifier`, `Regressor` and `Forecaster` are
hosted and in-context: `fit` uploads your rows and trains nothing; `predict`
reads them as context. A context too large to send fails at `fit`, at once,
with `CapacityExceeded`.

```python
clf = gf.Classifier("tabicl-v2").fit(X_train, y_train)
clf        # Classifier(model='tabicl-v2')  # 26,048 context rows uploaded; nothing trained
clf.estimate(X_test)      # Estimate(calls=2, rows=6513, credits=13, usd=0.013)
clf.predict(X_test)
# Predicted 6,513 rows with tabicl-v2-classifier from 26,048 context rows: 2 calls, 13.0 credits ($0.013) in 9.4s
clf.usage_                # every call since fit; clf.last_usage_ is the last one's
```

Rows go up once, named by their digest, so `cross_val_score` and
`GridSearchCV` send each fold's rows only the first time; `verbose=False`
keeps the estimators quiet.

## Fine-tuning

`gf.snapshots` holds training snapshots and synthetic batches, and `gf.runs`
the runs that train on them. Each run of a name publishes its next version,
`<name>@ft<n>`, trained afresh from the same base; a published version works
anywhere a model goes. Settings are dicts, as the service takes them; results
are typed.

```python
snap = gf.snapshots.create("credit-2026-01", rows, target="defaulted",
                           task="classification")
warmup = gf.snapshots.create_synthetic("credit-warmup", count=64, seed=0,
                                       task="classification",
                                       rows={"min": 1024, "max": 8192})
run = gf.runs.create("credit-default", base="tabicl-v2",
                     method={"type": "lora", "rank": 16},
                     data=[{"tag": warmup.tag}, snap.name,
                           {"context": "credit", "version": 3}],
                     target="defaulted", plan={"steps": 1000})
run = gf.runs.wait(run)
run        # credit-default@ft1  finished  step 1,000/1,000  loss 0.410  ...
run.model  # 'acme/credit-default@ft1', the model to pass anywhere a model goes
gf.predictors.create("credit-default", context="credit", model=run.model,
                     target="defaulted")
```

A fine-tune's id is `<project>/<name>@ft<n>`; in its own project
`credit-default@ft1` names it too.

`snapshots.create` uploads in chunks; `snapshots.stream` sends a small table
as one Arrow stream. A run's `data` names snapshots, `{"tag": T}` for every
snapshot carrying a tag, and `{"context": NAME, "version": N}` for a context's
version, which needs `target=`. `phases=[{"name", "data", "plan"}, ...]`
trains a curriculum, and `architecture`, `evaluation` and `budget` take the
service's settings. `runs.watch` yields a run's events as they happen,
`runs.lineage` lists them all, and `runs.cancel(run, save=True)` publishes
what a training run has kept.

## Clients and errors

`gf.Client` and `gf.AsyncClient` carry the same resources: `contexts`,
`experiments`, `predictors`, `snapshots`, `runs`, `sources`, `keys` and
`models`. Every error is a `gf.Error` with `message`, `hint`, `code`,
`reason`, `request_id` and `recoverable`, false only for `Internal`,
`Unavailable` and failed runs or deploys. Reads, and writes named by their id, are sent again after a lost
connection, a 502, 503 or 504, or a 429 that passes, such as `rate_limited`;
a 429 for a limit that lifts only when you end a run or delete data
(`run_limit_reached`, `quota_exceeded`, `snapshot_quota_exceeded`) is raised
at once.

## License

MIT; see [LICENSE](LICENSE).
