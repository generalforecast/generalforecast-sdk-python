# General Forecast

The Python SDK and `gf` command line for [General Forecast](https://generalforecast.com):
predictors on tabular and time series foundation models, run on the hosted service.

- Docs: https://docs.generalforecast.com
- API reference: https://docs.generalforecast.com/api

```sh
uv pip install generalforecast
```

Python 3.12 or later. The package is pure Python; every prediction, check and
evaluation runs on the service.

## Log in

```sh
gf login
```

`gf login` opens your browser and saves a key for this machine. In CI or a
notebook, set `GF_API_KEY` (and `GF_PROJECT` if the key spans projects) instead.

## A predictor from a table

A predictor is a directory with a manifest, `gf.yaml`, and the table its context
comes from.

```sh
mkdir churn && cp customers.csv churn/
gf init churn          # asks for the target and model, writes gf.yaml
gf check churn         # resolves gf.yaml on the service, writes gf.lock
gf eval churn          # scores the held-out rows on the service
gf push churn          # deploys it to production
gf predict churn --queries new_customers.csv
```

`gf check` uploads the manifest's snapshot and writes the `gf.lock` the service
answers with (`--upgrade` accepts a newer model digest). `gf eval` needs a
snapshot context with a `holdout` or `until`, which `gf init` writes. `gf push`
has the service check the directory again, store the snapshot's rows and
register them as a new revision; a lock that no longer
matches is refused until `gf check` runs again.
`gf status`, `gf promote` and `gf rollback` move environments between
revisions.

To score rows with a catalog model without a predictor directory:

```sh
gf predict --model general-forecast/tabicl-v2-classifier \
  --target churned --context customers.csv --queries new_customers.csv
```

`gf model list` shows the catalog with each model's task and limits.

## Python

```python
import generalforecast as gf

churn = gf.Predictor("churn")  # a pushed predictor
predictions = churn.predict("new_customers.csv")

adhoc = gf.Predictor(
    "general-forecast/tabicl-v2-classifier",
    target="churned",
    context="customers.csv",
)
predictions = adhoc.predict("new_customers.csv")
```

### Estimators

`Classifier`, `Regressor` and `Forecaster` follow scikit-learn's conventions.
`fit` keeps the labeled rows; `predict` sends them as the context of a hosted
prediction.

```python
import numpy as np
import pandas as pd
import generalforecast as gf

clf = gf.Classifier(model="general-forecast/tabicl-v2-classifier")
clf.fit(X_context, y_context)
probabilities = clf.predict_proba(X_queries)

days = pd.date_range("2025-01-01", periods=60, freq="D")
history = pd.DataFrame(
    {
        "date": np.tile(days, 2),
        "sku": np.repeat(["a", "b"], 60),
        "units": np.r_[np.arange(60.0) % 7, 10 + np.arange(60.0) % 5],
    }
)
fc = gf.Forecaster(target="units", time="date", series="sku", horizon=7)
fc.fit(history)
quantiles = fc.forecast()  # 0.1, 0.5 and 0.9 for each sku and step
```

A model reads a bounded number of context rows per request. When the training
rows are more than one request carries, push them as a context with a predictor
that reads it, or pass `selection=` to draw each query's context from them.

## Live contexts

A live context is a table on the service that grows as rows arrive; a predictor
whose manifest names it (`context: {live: events}`) always predicts with the
latest rows.

```sh
gf context create events --file events.yaml
gf context append events rows.csv --id event_id
gf context label events labels.csv --id event_id --target converted
```

```python
events = gf.Predictor("conversion").context
events.append(new_rows)
events.label(ids, outcomes)
```

## The client

`gf.Client` and `gf.AsyncClient` cover the whole API: predictors, revisions and
environments, contexts, snapshots, models, keys and secrets.

```python
with gf.Client() as client:
    checked = client.predictors.check("churn")
    scores = client.predictors.evaluate("churn")
    client.predictors.push("churn", env="staging", wait=True)
```

## License

MIT; see [LICENSE](LICENSE).
