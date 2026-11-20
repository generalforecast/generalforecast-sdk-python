# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Score rows with a catalog model, the context sent with each call."""

from pathlib import Path

import pandas as pd

import generalforecast as gf

rows = pd.read_csv(Path(__file__).parent / "churn" / "customers.csv")
context, queries = rows.iloc[:200], rows.iloc[200:].drop(columns=["churned"])

predictor = gf.Predictor(
    "general-forecast/tabicl-v2-classifier",
    target="churned",
    id="customer_id",
    context=context,
)
print(predictor.predict(queries).head())
