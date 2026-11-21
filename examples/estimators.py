# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The scikit-learn style estimators, predicting on the hosted service."""

from pathlib import Path

import numpy as np
import pandas as pd

import generalforecast as gf

rows = pd.read_csv(Path(__file__).parent / "churn" / "customers.csv")
features = rows[["tenure_months", "monthly_spend", "support_tickets"]]
X_context, X_queries = features.iloc[:200], features.iloc[200:]
y_context, y_queries = rows["churned"].iloc[:200], rows["churned"].iloc[200:]

classifier = gf.Classifier().fit(X_context, y_context)
print("accuracy", classifier.score(X_queries, y_queries))

days = pd.date_range("2025-01-01", periods=60, freq="D")
history = pd.DataFrame(
    {
        "date": np.tile(days, 2),
        "sku": np.repeat(["a", "b"], 60),
        "units": np.r_[np.arange(60.0) % 7, 10 + np.arange(60.0) % 5],
    }
)
future = pd.DataFrame(
    {
        "date": np.tile(pd.date_range("2025-03-02", periods=7, freq="D"), 2),
        "sku": np.repeat(["a", "b"], 7),
    }
)
forecaster = gf.Forecaster(target="units", time="date", series=["sku"], horizon=7)
print(forecaster.fit(history).forecast(future))
