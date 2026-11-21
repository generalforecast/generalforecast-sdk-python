# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The fake service, a clean environment, and waits that take no time."""

import asyncio
import time

import pytest

from generalforecast import _auth, client
from tests.fake import TOKEN, Fake, serve


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No credentials, profiles or proxies leak in from the machine running the tests."""
    for name in ("GF_API_KEY", "GF_PROJECT", "GF_URL", "HTTPS_PROXY", "HTTP_PROXY",
                 "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("GF_KEYCHAIN", "0")
    monkeypatch.setattr(_auth, "SESSION", None)
    client.reset()
    yield monkeypatch
    client.reset()


@pytest.fixture
def fake(env):
    """The fake service, with `GF_URL` and `GF_API_KEY` pointing at it."""
    service = Fake()
    stop = serve(service)
    env.setenv("GF_URL", service.url)
    env.setenv("GF_API_KEY", TOKEN)
    slept = []
    real_async_sleep = asyncio.sleep

    async def no_wait(seconds, *args, **kwargs):
        slept.append(seconds)
        await real_async_sleep(0)

    env.setattr(time, "sleep", slept.append)
    env.setattr(asyncio, "sleep", no_wait)
    service.slept = slept
    yield service
    client.reset()
    stop()
    assert not service.errors, service.errors
