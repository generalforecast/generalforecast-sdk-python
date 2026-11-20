# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""`predictors.logs` asks the service for a page of an environment's log."""

import datetime

import pytest

from generalforecast import _operations as ops
from generalforecast.client import Client
from generalforecast.errors import InvalidInput
from generalforecast.transport import Profile

URL = "https://service.example"


def call(**selected):
    client = Client(
        profile=Profile(URL, "test", environment="staging"), api_key="fixture-key"
    )
    operation = ops.predictor_report(
        client, "risk", None, "logs", ops.log_selection(**selected)
    )
    return next(operation)


def test_a_plain_read_sends_no_query_so_any_service_answers_it():
    sent = call()
    assert (sent.method, sent.path) == (
        "GET",
        "/predictors/risk/environments/staging/logs",
    )
    assert sent.params is None


def test_a_read_sends_its_page_its_cursor_and_its_filters():
    sent = call(
        limit=50,
        after="1795034671207000",
        level="warn",
        q="Tenure",
        since=datetime.datetime(2026, 11, 18, 20, 0, tzinfo=datetime.UTC),
        until=1795035600,
    )
    assert sent.params == {
        "limit": 50,
        "after": "1795034671207000",
        "level": "warn",
        "q": "Tenure",
        "since": "2026-11-18T20:00:00Z",
        "until": "2026-11-18T21:00:00Z",
    }
    assert call(before="17", since="2026-11-18T20:00:00+02:00").params == {
        "before": "17",
        "since": "2026-11-18T18:00:00Z",
    }


@pytest.mark.parametrize("param", ["since", "until"])
def test_a_time_that_is_not_one_names_its_own_parameter(param):
    with pytest.raises(InvalidInput) as refused:
        call(**{param: "last tuesday"})
    assert refused.value.param == param
    assert refused.value.message.startswith(f"{param} is a datetime")


def test_the_reply_is_read_as_the_service_gave_it():
    client = Client(
        profile=Profile(URL, "test", environment="staging"), api_key="fixture-key"
    )
    operation = ops.predictor_report(client, "risk", "production", "logs")
    next(operation)
    page = {
        "logs": [
            {
                "id": "17",
                "at": 1795034671.207,
                "level": "warn",
                "source": "replica a1f",
                "code": "PredictionFailed",
                "message": "predict 422 ValidationError: unknown column `Tenure`",
                "environment": "production",
                "revision": 2,
                "request_id": "r-1",
                "fields": {"key": "gf_...Jfwo", "rows": 1, "status": 422},
            }
        ],
        "next": "17",
        "previous": None,
    }
    with pytest.raises(StopIteration) as done:
        operation.send(page)
    read = done.value.value
    assert read["next"] == "17" and read["previous"] is None
    assert read["logs"][0]["fields"]["key"] == "gf_...Jfwo"
