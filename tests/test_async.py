# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import inspect

import pandas as pd

import generalforecast as gf
from generalforecast import resources


def test_every_sync_method_has_an_async_twin():
    pairs = [
        (resources.Contexts, resources.AsyncContexts),
        (resources.Experiments, resources.AsyncExperiments),
        (resources.Predictors, resources.AsyncPredictors),
        (resources.Sources, resources.AsyncSources),
        (resources.Keys, resources.AsyncKeys),
        (resources.Models, resources.AsyncModels),
    ]
    for sync, asynchronous in pairs:
        for name, method in inspect.getmembers(sync, inspect.isfunction):
            if name.startswith("_"):
                continue
            twin = getattr(asynchronous, name)
            assert inspect.iscoroutinefunction(twin), (sync.__name__, name)
            assert inspect.signature(twin) == inspect.signature(method), (
                sync.__name__,
                name,
            )


def test_the_async_client_runs_the_same_operations(fake):
    async def go():
        async with gf.AsyncClient() as client:
            frame = pd.DataFrame({"age": [1, 2, 3], "income": ["a", "b", "a"]})
            ctx = await client.contexts.create("adult", data=frame)
            exp = await client.experiments.create(context=ctx, target="income")
            p = await client.predictors.create(
                "adult-income", context="adult", model=exp.best.model, target="income"
            )
            out = await client.predictors.predict(p, frame.drop(columns="income"))
            direct = await client.models.predict(
                "tabicl-v2",
                context=frame,
                queries=frame.drop(columns="income"),
                target="income",
            )
            return ctx, exp, p, out, direct

    ctx, exp, p, out, direct = asyncio.run(go())
    assert ctx.state == "ready" and exp.run.state == "finished"
    assert p.environments["production"].state == "ready"
    assert out.rows == 3 and direct.rows == 3
