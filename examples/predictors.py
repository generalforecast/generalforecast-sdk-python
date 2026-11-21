# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Predictors as code: `gf.predictors.deploy(adult, churn, *demand)` makes production match."""

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

demand = [
    gf.Predictor(
        f"demand-{region}",
        context=f"sales-{region}",
        model="chronos-2",
        target="units",
        time="week",
        series=["sku"],
        horizon=8,
        frequency="7d",
    )
    for region in ("us", "eu")
]

if __name__ == "__main__":
    for deploy in gf.predictors.deploy(adult, churn, *demand, env="staging"):
        print(deploy.predictor, deploy.action)
