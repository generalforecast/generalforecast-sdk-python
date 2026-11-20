# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Predictions the hosted service accepts to answer later, in both clients: asking, polling, waiting."""

import asyncio
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import generalforecast as gf
from generalforecast import _operations
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.retry import (
    DEFAULT_RETRY,
    PREDICT,
    READ,
    WAIT_EXPIRED,
    PredictionWait,
    RetryEvent,
    RetryPolicy,
)
from generalforecast.service.boundary import Request, Resource, Scope, success
from generalforecast.service.contracts import ServiceError
from generalforecast.transport import AsyncTransport, Profile, Reply, Transport
from tests.test_sdk_retry import MODEL, PREDICTION, fast, lost, unavailable

ID = "pred-0123456789abcdef01234567"
LOCATION = f"/v1/projects/test/predictions/{ID}"
RESPOND_ASYNC = "respond-async, wait=20"

both = pytest.mark.parametrize("asynchronous", [False, True])


def accepted(state: str = "starting", wait: float | None = 5.0) -> Reply:
    """The 202 a hosted plane answers for a prediction it is still working on."""
    body = {
        "api_version": "gf/v1",
        "project": "test",
        "request_id": "r-x",
        "resource": {"kind": "prediction", "name": ID},
        "result": {"id": ID, "state": state, "created": 1761200000, "retry_after": 5},
    }
    return Reply(202, body, wait, LOCATION)


class Script:
    """A transport that answers each request with the next outcome, and records it."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[tuple[str, str, dict[str, str]]] = []

    def reply(self, method, path, body, headers, *, deadline=None):
        self.sent.append((method, path.removeprefix("/v1/projects/test"), headers))
        assert self.outcomes, f"unexpected {method} {path}"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome if isinstance(outcome, Reply) else Reply(200, outcome)


class Clock:
    """Time that passes only while a client sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def run(asynchronous: bool, script: Script, body, monkeypatch, **options):
    """Runs `body(client)` against `script`; returns its value, the waits and the events."""
    monkeypatch.setattr(_operations, "answer", lambda _prepared, value: value)
    clock = Clock()
    events: list[PredictionWait] = []
    options.setdefault("retry", DEFAULT_RETRY)

    async def go():
        profile = Profile("https://service.example", "test")
        cls = AsyncClient if asynchronous else Client
        client: Any = cls(
            profile=profile, api_key="fixture-key", on_wait=events.append, **options
        )
        if asynchronous:

            async def exchange(*args, **kwargs):
                return script.reply(*args, **kwargs)

            async def sleep(seconds):
                clock.sleep(seconds)

        else:

            def exchange(*args, **kwargs):
                return script.reply(*args, **kwargs)

            sleep = clock.sleep
        client._transport.exchange = exchange
        client._retry_sleep = sleep
        client._clock = clock
        try:
            value = body(client)
            return await value if inspect.isawaitable(value) else value
        finally:
            if isinstance(client, AsyncClient):
                await client.aclose()
            else:
                client.close()

    return asyncio.run(go()), clock.slept, events


def predict(client):
    return client.predictors.predict("risk", {"x": [3]})


def ad_hoc(client):
    return client.predict(
        "general-forecast/tabicl-v2-classifier",
        "y",
        {"x": [1, 2], "y": [True, False]},
        {"x": [3]},
    )


@both
@pytest.mark.parametrize("call", [predict, ad_hoc])
def test_a_prediction_asks_to_be_accepted_and_nothing_else_does(
    asynchronous, call, monkeypatch
):
    script = Script(
        *([MODEL] if call is ad_hoc else []),
        PREDICTION,
        {"name": "risk"},
        {"models": []},
    )

    async def chain(client):
        for value in (
            call(client),
            client._call("POST", "/predictors", "predictor", "risk", {"name": "risk"}),
            client._call("GET", "/models", "model", "models"),
        ):
            if inspect.isawaitable(value):
                await value

    run(asynchronous, script, chain, monkeypatch)
    sent = [sent[2] for sent in script.sent]
    if call is ad_hoc:
        assert "Prefer" not in sent.pop(0)
    prediction, create, listing = sent
    assert prediction["Prefer"] == RESPOND_ASYNC
    assert "Prefer" not in create and "Prefer" not in listing


@both
def test_a_known_model_server_is_not_asked_to_accept(asynchronous, monkeypatch):
    script = Script(PREDICTION)

    def body(client):
        client._transport.platform = False
        return predict(client)

    run(asynchronous, script, body, monkeypatch)
    assert "Prefer" not in script.sent[0][2]


@both
def test_a_prediction_answered_at_once_is_returned_as_before(asynchronous, monkeypatch):
    script = Script(PREDICTION)
    result, slept, events = run(asynchronous, script, predict, monkeypatch)
    assert result.model == PREDICTION["model"]
    assert [sent[:2] for sent in script.sent] == [("POST", "/predictors/risk/predict")]
    assert slept == [] and events == []


@both
def test_an_accepted_prediction_is_polled_until_its_answer(asynchronous, monkeypatch):
    retries: list[RetryEvent] = []
    script = Script(accepted(), accepted("running", 7.0), PREDICTION)
    result, slept, events = run(
        asynchronous,
        script,
        predict,
        monkeypatch,
        retry=RetryPolicy(on_retry=retries.append),
    )
    assert result.model == PREDICTION["model"]
    assert [sent[:2] for sent in script.sent] == [
        ("POST", "/predictors/risk/predict"),
        ("GET", f"/predictions/{ID}"),
        ("GET", f"/predictions/{ID}"),
    ]
    post, *polls = (sent[2] for sent in script.sent)
    for poll in polls:
        assert "Prefer" not in poll
        assert poll["X-GF-Request-ID"] == post["X-GF-Request-ID"]
        assert poll["X-GF-Project"] == "test"
    assert slept == [5.0, 7.0]
    # Waiting is not retrying.
    assert retries == []
    assert [(e.prediction, e.state, e.elapsed, e.wait) for e in events] == [
        (ID, "starting", 0.0, 5.0),
        (ID, "running", 5.0, 7.0),
    ]
    assert events[0].path == "/v1/projects/test/predictors/risk/predict"
    assert events[0].request_id == post["X-GF-Request-ID"]


@both
def test_an_accepted_ad_hoc_prediction_is_polled_too(asynchronous, monkeypatch):
    script = Script(MODEL, accepted(), PREDICTION)
    result, slept, _ = run(asynchronous, script, ad_hoc, monkeypatch)
    assert result.model == PREDICTION["model"] and slept == [5.0]
    assert [sent[1] for sent in script.sent] == [
        "/models/tabicl-v2-classifier",
        "/predict",
        f"/predictions/{ID}",
    ]


@both
def test_a_final_error_is_raised_as_the_prediction_would_have_raised_it(
    asynchronous, monkeypatch
):
    refused = ServiceError("ValidationError", "declared by the predictor", 422)
    script = Script(accepted(), refused)
    with pytest.raises(ServiceError) as raised:
        run(asynchronous, script, predict, monkeypatch)
    assert raised.value is refused
    assert len(script.sent) == 2


@both
def test_a_final_retryable_error_retries_the_prediction(asynchronous, monkeypatch):
    retries: list[RetryEvent] = []
    no_read_retries = RetryPolicy(
        predict=fast(), control=fast(attempts=1), on_retry=retries.append
    )
    script = Script(accepted(), unavailable(wait=3.0), PREDICTION)
    result, slept, _ = run(
        asynchronous, script, predict, monkeypatch, retry=no_read_retries
    )
    assert result.model == PREDICTION["model"]
    assert [sent[0] for sent in script.sent] == ["POST", "GET", "POST"]
    # The prediction is sent again as itself, under the same request id.
    assert script.sent[0][2] == script.sent[2][2]
    assert slept == [5.0, 3.0]
    assert [(e.retry_class, e.method) for e in retries] == [
        (READ, "GET"),
        (PREDICT, "POST"),
    ]


@both
@pytest.mark.parametrize(
    ("asked", "waited"), [(0.0, 1.0), (12.0, 12.0), (120.0, 30.0), (None, 5.0)]
)
def test_retry_after_paces_the_polls_from_one_to_thirty_seconds(
    asynchronous, asked, waited, monkeypatch
):
    script = Script(accepted(wait=asked), accepted(wait=asked), PREDICTION)
    _, slept, events = run(asynchronous, script, predict, monkeypatch)
    assert slept == [waited, waited]
    assert [event.wait for event in events] == [waited, waited]


@both
def test_the_wait_runs_out_with_a_timeout_naming_the_prediction(
    asynchronous, monkeypatch
):
    script = Script(accepted(), accepted(), accepted(), accepted())
    with pytest.raises(gf.Unavailable) as raised:
        run(asynchronous, script, predict, monkeypatch, wait_seconds=12)
    error = raised.value
    assert (error.code, error.status, error.reason) == ("Timeout", 504, WAIT_EXPIRED)
    assert (
        error.message
        == f"prediction {ID} did not finish within 12s; it may still finish"
    )
    assert error.request_id == script.sent[0][2]["X-GF-Request-ID"]
    # Polls at 5 and 10 seconds, the last at 12; the prediction is not sent again.
    assert [sent[0] for sent in script.sent] == ["POST", "GET", "GET", "GET"]
    assert script.outcomes == []


@both
def test_the_default_wait_is_fifteen_minutes_from_the_first_accept(
    asynchronous, monkeypatch
):
    script = Script(*[accepted(wait=30.0)] * 32)
    with pytest.raises(gf.Unavailable) as raised:
        run(asynchronous, script, predict, monkeypatch)
    assert raised.value.message.endswith("within 900s; it may still finish")
    assert len(script.sent) == 1 + 30 and len(script.outcomes) == 1


@both
def test_a_poll_that_fails_on_the_way_is_retried_as_a_read(asynchronous, monkeypatch):
    retries: list[RetryEvent] = []
    script = Script(accepted(), lost(), PREDICTION)
    result, slept, _ = run(
        asynchronous,
        script,
        predict,
        monkeypatch,
        retry=RetryPolicy(predict=fast(), control=fast(), on_retry=retries.append),
    )
    assert result.model == PREDICTION["model"]
    assert [sent[0] for sent in script.sent] == ["POST", "GET", "GET"]
    assert slept == [5.0, 1.0]
    assert [(e.retry_class, e.method, e.path) for e in retries] == [
        (READ, "GET", LOCATION)
    ]


@both
def test_an_accept_with_no_location_is_a_protocol_error(asynchronous, monkeypatch):
    script = Script(Reply(202, accepted().value, 5.0, None))
    with pytest.raises(ServiceError) as raised:
        run(asynchronous, script, predict, monkeypatch, retry=None)
    assert raised.value.code == "ProtocolError"


def test_an_async_client_awaits_a_coroutine_on_wait(monkeypatch):
    told: list[str] = []

    async def on_wait(event: PredictionWait) -> None:
        await asyncio.sleep(0)
        told.append(event.state)

    script = Script(accepted(), PREDICTION)

    def body(client):
        client._on_wait = on_wait
        return predict(client)

    run(True, script, body, monkeypatch)
    assert told == ["starting"]


def test_wait_seconds_is_checked():
    with pytest.raises(ServiceError) as raised:
        Client("https://service.example", api_key="fixture-key", wait_seconds=-1)
    assert raised.value.code == "ValidationError"


@pytest.fixture
def plane():
    """A hosted plane that accepts a prediction, then answers its poll with `.final`."""

    class State:
        def __init__(self) -> None:
            self.final: tuple[int, dict[str, Any]] = (200, {})
            self.location = LOCATION
            self.paths: list[tuple[str, str, str | None]] = []
            self.url = ""

    state = State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def answer(self, status, body, headers):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state.paths.append(("POST", self.path, self.headers.get("Prefer")))
            self.answer(
                202,
                accepted().value,
                {"Location": state.location, "Retry-After": "0"},
            )

        def do_GET(self):
            state.paths.append(("GET", self.path, self.headers.get("Prefer")))
            status, result = state.final
            request = Request(
                Scope("test", self.headers["X-GF-Request-ID"]),
                Resource("predictor", "risk"),
                {},
            )
            body = success(request, result) if status == 200 else {"error": result}
            self.answer(status, body, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    worker.join()


def over_the_wire(asynchronous: bool, url: str) -> tuple[Any, list[float]]:
    """A prediction through the real transport, the waits recorded instead of slept."""
    slept: list[float] = []

    async def go():
        cls = AsyncClient if asynchronous else Client
        client: Any = cls(base_url=url, project="test", api_key="fixture-key")

        def sleep(seconds):
            slept.append(seconds)

        async def asleep(seconds):
            slept.append(seconds)

        client._retry_sleep = asleep if asynchronous else sleep
        try:
            value = predict(client)
            return await value if inspect.isawaitable(value) else value
        finally:
            if asynchronous:
                await client.aclose()
            else:
                client.close()

    return asyncio.run(go()), slept


@both
def test_the_answer_over_the_wire_is_decoded_as_a_synchronous_one(asynchronous, plane):
    plane.final = (200, PREDICTION)
    result, slept = over_the_wire(asynchronous, plane.url)
    assert result.model == PREDICTION["model"]
    assert result.table.num_rows == 1 and result.task == "classification"
    assert plane.paths == [
        ("POST", "/v1/projects/test/predictors/risk/predict", RESPOND_ASYNC),
        ("GET", LOCATION, None),
    ]
    # A Retry-After of 0 is held to the one-second floor.
    assert slept == [1.0]


@both
def test_an_error_over_the_wire_is_mapped_as_a_synchronous_one(asynchronous, plane):
    plane.final = (
        422,
        {
            "code": "ValidationError",
            "message": "declared by the predictor",
            "param": "target",
        },
    )
    with pytest.raises(gf.InvalidInput) as raised:
        over_the_wire(asynchronous, plane.url)
    error = raised.value
    assert (error.code, error.status, error.message, error.param) == (
        "ValidationError",
        422,
        "declared by the predictor",
        "target",
    )


@both
def test_an_accept_naming_another_origin_is_refused(asynchronous, plane):
    plane.location = f"https://elsewhere.example{LOCATION}"
    with pytest.raises(ServiceError) as raised:
        over_the_wire(asynchronous, plane.url)
    assert raised.value.code == "ProtocolError"
    assert [path[0] for path in plane.paths] == ["POST"]


@both
def test_the_transport_returns_a_202_with_its_location_and_retry_after(
    asynchronous, plane
):
    async def go():
        transport: Any = (AsyncTransport if asynchronous else Transport)(plane.url)
        reply = transport.exchange(
            "POST", "/v1/projects/test/predict", {"x": 1}, {"Prefer": RESPOND_ASYNC}
        )
        reply = await reply if inspect.isawaitable(reply) else reply
        if asynchronous:
            await transport.aclose()
        return reply

    plane.location = f"{plane.url}{LOCATION}"
    reply = asyncio.run(go())
    assert (reply.status, reply.location, reply.retry_after) == (202, LOCATION, 0.0)
    assert reply.value["result"]["id"] == ID
