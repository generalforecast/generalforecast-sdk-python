# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Score rows with a catalog model; the context goes up once, by digest."""

from pathlib import Path

import pandas as pd

import generalforecast as gf

rows = pd.read_csv(Path(__file__).parent / "churn" / "customers.csv")
context, queries = rows.iloc[:200], rows.iloc[200:].drop(columns=["churned"])

out = gf.models.predict(
    "tabicl-v2", context=context, queries=queries, target="churned", id="customer_id"
)
print(out.to_pandas().head())
print(f"{out.rows} rows, {out.credits} credits (${out.usd})")
