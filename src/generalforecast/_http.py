# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The HTTP layer: every route the SDK calls, the envelope, and the transports.

This module is the one place that knows the wire. Each route in sections 3.1
to 3.9 of the surface spec is a function below returning a `Request`; the
resources build payloads and read results, nothing else. Aligning the SDK
with a new `api/openapi.yaml` means changing this file.

Choices the spec leaves open, made here:

- A query part is an Arrow IPC stream whose schema metadata holds the
  request envelope as JSON under `gf.request`, so one body carries both.
  A reply may be JSON, or an Arrow stream with the result under `gf.result`.
- `PUT /snapshots?digest=` with an empty body asks whether the service holds
  that snapshot (200) or not (404); the same PUT with the Arrow stream
  stores it.
- A context upload declares the table's `digest` beside `format` and
  `bytes`, so an unchanged table is answered without its bytes.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import random
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

from generalforecast.errors import (
    Error,
    InvalidInput,
    Unavailable,
    from_wire,
)

if TYPE_CHECKING:
    import pyarrow as pa

API_VERSION = "gf/v1"
DEFAULT_URL = "https://app.generalforecast.com"
ARROW = "application/vnd.apache.arrow.stream"
OCTETS = "application/octet-stream"
RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRY_AFTER = 60.0
MAX_REPLY_BYTES = 512 * 1024 * 1024

LOG = logging.getLogger("generalforecast.http")


@dataclass(frozen=True)
class Request:
    """One HTTP request, before the connection's project and credential are added.

    `path` is under `/v1/projects/{project}` when `scoped`, else from the
    server's root; `url` instead sends to another origin (a signed upload
    URL) with no credential. `payload` goes in the envelope, or as plain
    JSON when `envelope` is false; `arrow` sends a table with the envelope
    in its schema metadata; `body` sends bytes as they are.
    """

    method: str
    path: str
    kind: str = "project"
    name: str = "project"
    payload: Any = None
    params: Mapping[str, Any] = field(default_factory=dict)
    scoped: bool = True
    envelope: bool = True
    arrow: Any = None
    body: bytes | None = None
    content_type: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    retry: bool | None = None
    respond_async: bool = False
    accept: frozenset[int] = frozenset()

    @property
    def safe(self) -> bool:
        """Whether a try may be sent again after a transient failure."""
        if self.retry is not None:
            return self.retry
        return self.method in ("GET", "PUT", "DELETE")


@dataclass(frozen=True)
class Reply:
    """A successful answer: its status, the result, and its headers."""

    status: int
    result: Any
    headers: Mapping[str, str]
    request_id: str | None = None
    table: Any = None


def _segment(value: str) -> str:
    return quote(str(value), safe="@")


def _query(**params: Any) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def auth_status() -> Request:
    """Who the credential acts for."""
    return Request("GET", "/v1/auth/status", "workspace", "auth", scoped=False)


def auth_logout() -> Request:
    """Revokes the key making the call."""
    return Request(
        "POST", "/v1/auth/logout", "workspace", "auth", payload={}, scoped=False
    )


def device_login_start(machine: str, client: str) -> Request:
    """Starts a device login (section 3.6)."""
    return Request(
        "POST",
        "/v1/device-logins",
        payload={"machine": machine, "client": client},
        scoped=False,
        envelope=False,
        retry=False,
    )


def device_login_token(device_code: str) -> Request:
    """Redeems an approved device login, once."""
    return Request(
        "POST",
        "/v1/device-logins/token",
        payload={"device_code": device_code},
        scoped=False,
        envelope=False,
        retry=False,
    )


def contexts_create(payload: Mapping[str, Any]) -> Request:
    """Creates a context from one origin (3.1)."""
    return Request("POST", "/contexts", "context", payload["name"], payload=payload)


def contexts_version(name: str, payload: Mapping[str, Any]) -> Request:
    """Opens an upload of a new version of a snapshot context."""
    return Request(
        "POST", f"/contexts/{_segment(name)}/versions", "context", name, payload=payload
    )


def contexts_bytes(name: str, version: int, offset: int, chunk: bytes) -> Request:
    """Sends one chunk of an upload when the service gave no signed URL."""
    return Request(
        "POST",
        f"/contexts/{_segment(name)}/versions/{version}/bytes",
        "context",
        name,
        params={"offset": offset},
        body=chunk,
        content_type=OCTETS,
        retry=True,
    )


def upload_chunk(url: str, chunk: bytes, offset: int, total: int) -> Request:
    """Sends one chunk to a signed resumable upload URL, without a credential."""
    end = offset + len(chunk) - 1
    return Request(
        "PUT",
        "",
        url=url,
        body=chunk,
        content_type=OCTETS,
        headers={"Content-Range": f"bytes {offset}-{end}/{total}"},
        envelope=False,
        accept=frozenset({308}),
    )


def contexts_refresh(name: str) -> Request:
    """Imports a new version of a source snapshot context."""
    return Request(
        "POST", f"/contexts/{_segment(name)}/refresh", "context", name, payload={}
    )


def contexts_write(name: str, verb: str, payload: Mapping[str, Any]) -> Request:
    """Appends rows to, or labels rows of, an API-fed live context."""
    return Request(
        "POST",
        f"/contexts/{_segment(name)}/{verb}",
        "context",
        name,
        payload=payload,
        retry="write_id" in payload,
    )


def contexts_get(name: str, version: int | None = None) -> Request:
    """Reads a context, or one of its versions."""
    return Request(
        "GET",
        f"/contexts/{_segment(name)}",
        "context",
        name,
        params=_query(version=version),
    )


def contexts_list() -> Request:
    """Lists the project's contexts."""
    return Request("GET", "/contexts", "context", "contexts")


def contexts_rows(
    name: str, *, limit: int, version: int | None, after: str | None = None
) -> Request:
    """Reads a page of a context's rows."""
    return Request(
        "GET",
        f"/contexts/{_segment(name)}/rows",
        "context",
        name,
        params=_query(limit=limit, version=version, after=after),
    )


def contexts_delete(name: str) -> Request:
    """Deletes a context; 409 while a predictor serves it."""
    return Request(
        "POST", f"/contexts/{_segment(name)}/delete", "context", name, payload={}
    )


def snapshot_probe(digest: str) -> Request:
    """Asks whether the service holds the snapshot with this digest."""
    return Request(
        "PUT",
        "/snapshots",
        "snapshot",
        "snapshots",
        params={"digest": digest},
        body=b"",
        content_type=ARROW,
    )


def snapshot_put(digest: str, body: bytes) -> Request:
    """Stores an Arrow snapshot under its digest; storing it again changes nothing."""
    return Request(
        "PUT",
        "/snapshots",
        "snapshot",
        "snapshots",
        params={"digest": digest},
        body=body,
        content_type=ARROW,
    )


def experiments_create(payload: Mapping[str, Any]) -> Request:
    """Creates an experiment and starts run 1, or with `dry_run` estimates it."""
    return Request(
        "POST",
        "/experiments",
        "experiment",
        payload.get("name") or "experiments",
        payload=payload,
        retry=bool(payload.get("dry_run")),
    )


def experiments_list(
    context: str | None = None, after: str | None = None, limit: int = 100
) -> Request:
    """Lists experiments, optionally on one context."""
    return Request(
        "GET",
        "/experiments",
        "experiment",
        "experiments",
        params=_query(context=context, after=after, limit=limit),
    )


def experiments_get(name: str, run: int | None = None) -> Request:
    """Reads an experiment with its latest run, or run `run`."""
    return Request(
        "GET",
        f"/experiments/{_segment(name)}",
        "experiment",
        name,
        params=_query(run=run),
    )


def experiments_update(name: str, payload: Mapping[str, Any]) -> Request:
    """Changes an experiment's setup; earlier runs become stale."""
    return Request(
        "POST",
        f"/experiments/{_segment(name)}/update",
        "experiment",
        name,
        payload=payload,
    )


def experiments_run(name: str, payload: Mapping[str, Any]) -> Request:
    """Starts a run of an experiment."""
    return Request(
        "POST",
        f"/experiments/{_segment(name)}/runs",
        "experiment",
        name,
        payload=payload,
    )


def experiments_cancel(name: str, run: int) -> Request:
    """Cancels a run."""
    return Request(
        "POST",
        f"/experiments/{_segment(name)}/runs/{run}/cancel",
        "experiment",
        name,
        payload={},
        retry=True,
    )


def experiments_rows(
    name: str, run: int, *, model: str | None, after: str | None, limit: int = 10_000
) -> Request:
    """Reads a page of a run's held-out predictions."""
    return Request(
        "GET",
        f"/experiments/{_segment(name)}/runs/{run}/rows",
        "experiment",
        name,
        params=_query(model=model, after=after, limit=limit),
    )


def experiments_slices(name: str, run: int, *, model: str | None) -> Request:
    """Reads a run's error by slice."""
    return Request(
        "GET",
        f"/experiments/{_segment(name)}/runs/{run}/slices",
        "experiment",
        name,
        params=_query(model=model),
    )


def experiments_delete(name: str) -> Request:
    """Deletes an experiment and its runs."""
    return Request(
        "POST",
        f"/experiments/{_segment(name)}/delete",
        "experiment",
        name,
        payload={},
    )


def predictors_create(payload: Mapping[str, Any]) -> Request:
    """Creates a predictor from a spec and deploys it (3.3)."""
    return Request("POST", "/predictors", "predictor", payload["name"], payload=payload)


def predictors_update(name: str, payload: Mapping[str, Any]) -> Request:
    """Makes a revision from a partial spec and deploys it."""
    return Request(
        "POST",
        f"/predictors/{_segment(name)}/update",
        "predictor",
        name,
        payload=payload,
    )


def predictors_deploy(name: str, payload: Mapping[str, Any]) -> Request:
    """Creates, revises or leaves a predictor as the spec says; idempotent."""
    return Request(
        "POST",
        f"/predictors/{_segment(name)}/deploy",
        "predictor",
        name,
        payload=payload,
        retry=True,
    )


def predictors_get(name: str) -> Request:
    """Reads a predictor."""
    return Request("GET", f"/predictors/{_segment(name)}", "predictor", name)


def predictors_list() -> Request:
    """Lists predictors."""
    return Request("GET", "/predictors", "predictor", "predictors")


def predictors_revisions(name: str) -> Request:
    """Lists a predictor's revisions."""
    return Request("GET", f"/predictors/{_segment(name)}/revisions", "predictor", name)


def predictors_environment(name: str, env: str) -> Request:
    """Reads one environment of a predictor."""
    return Request(
        "GET",
        f"/predictors/{_segment(name)}/environments/{_segment(env)}",
        "predictor",
        name,
    )


def predictors_move(
    name: str, env: str, verb: str, payload: Mapping[str, Any]
) -> Request:
    """Promotes or rolls back an environment."""
    return Request(
        "POST",
        f"/predictors/{_segment(name)}/environments/{_segment(env)}/{verb}",
        "predictor",
        name,
        payload=payload,
        retry=True,
    )


def predictors_logs(name: str, env: str, params: Mapping[str, Any]) -> Request:
    """Reads a page of an environment's logs."""
    return Request(
        "GET",
        f"/predictors/{_segment(name)}/environments/{_segment(env)}/logs",
        "predictor",
        name,
        params=_query(**params),
    )


def predictors_delete(name: str) -> Request:
    """Deletes a predictor and its revisions."""
    return Request(
        "POST", f"/predictors/{_segment(name)}/delete", "predictor", name, payload={}
    )


def predictors_predict(
    name: str, payload: Mapping[str, Any], queries: pa.Table
) -> Request:
    """Scores one part of the queries with a predictor."""
    return Request(
        "POST",
        f"/predictors/{_segment(name)}/predict",
        "predictor",
        name,
        payload=payload,
        arrow=queries,
        retry=True,
        respond_async=True,
    )


def predict(payload: Mapping[str, Any], queries: pa.Table) -> Request:
    """Scores one part of the queries with a model and a context by reference."""
    return Request(
        "POST",
        "/predict",
        "predictor",
        "ad-hoc",
        payload=payload,
        arrow=queries,
        retry=True,
        respond_async=True,
    )


def prediction_get(location: str) -> Request:
    """Polls a prediction the service accepted to answer later."""
    prediction = location.rstrip("/").rsplit("/", 1)[-1]
    return Request(
        "GET", f"/predictions/{_segment(prediction)}", "prediction", prediction
    )


def estimate(payload: Mapping[str, Any]) -> Request:
    """Prices calls without making them (3.4)."""
    return Request(
        "POST", "/estimate", "model", "estimate", payload=payload, retry=True
    )


def sources_list() -> Request:
    """Lists sources."""
    return Request("GET", "/sources", "source", "sources")


def sources_get(name: str) -> Request:
    """Reads a source."""
    return Request("GET", f"/sources/{_segment(name)}", "source", name)


def sources_create(payload: Mapping[str, Any]) -> Request:
    """Connects a source; its secret is stored as `source-<name>`."""
    return Request("POST", "/sources", "source", payload["name"], payload=payload)


def sources_action(name: str, verb: str, payload: Mapping[str, Any]) -> Request:
    """Tests, previews or deletes a source."""
    return Request(
        "POST",
        f"/sources/{_segment(name)}/{verb}",
        "source",
        name,
        payload=payload,
        retry=verb in ("test", "preview"),
    )


def sources_tables(name: str, prefix: str | None) -> Request:
    """Lists a source's tables."""
    return Request(
        "GET",
        f"/sources/{_segment(name)}/tables",
        "source",
        name,
        params=_query(prefix=prefix),
    )


def keys_list() -> Request:
    """Lists the project's keys."""
    return Request("GET", "/keys", "api-key", "keys")


def keys_create(payload: Mapping[str, Any]) -> Request:
    """Creates a key; its token is in this reply only."""
    return Request("POST", "/keys", "api-key", payload["name"], payload=payload)


def keys_action(key: str, verb: str) -> Request:
    """Rotates, revokes or deletes a key."""
    return Request(
        "POST", f"/keys/{_segment(key)}/{verb}", "api-key", key, payload={"id": key}
    )


def models_list(task: str | None, all: bool) -> Request:
    """Lists catalog models and the project's published fine-tune versions."""
    return Request(
        "GET",
        "/models",
        "model",
        "models",
        params=_query(task=task, all="true" if all else None),
    )


def models_get(name: str, task: str | None) -> Request:
    """Reads one model by short name or full id."""
    return Request(
        "GET",
        f"/models/{quote(name, safe='@')}",
        "model",
        name,
        params=_query(task=task),
    )


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def server(url: str) -> str:
    """A server URL checked: HTTPS, or HTTP on loopback only, with no path.

    Raises:
        InvalidInput: For any other URL, so a credential never goes in clear.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise InvalidInput(f"`{url}` is not a server URL", hint="pass https://host")
    if parts.scheme == "http" and not _loopback(parts.hostname):
        raise InvalidInput(
            f"`{url}` is plain HTTP", hint="use https://; plain HTTP is for localhost"
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise InvalidInput(f"`{url}` must not carry credentials, a query or fragment")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def ssl_context(trust_env: bool) -> ssl.SSLContext:
    """The trust store: the system's CA bundle and certifi's, plus any named in the environment.

    With `trust_env`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` and
    `CURL_CA_BUNDLE` add their bundles, so a corporate proxy's CA works.
    """
    import certifi

    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    if trust_env:
        for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            path = os.environ.get(name)
            if path and os.path.isfile(path):
                context.load_verify_locations(path)
        directory = os.environ.get("SSL_CERT_DIR")
        if directory and os.path.isdir(directory):
            context.load_verify_locations(capath=directory)
    return context


def request_id() -> str:
    """A fresh request id."""
    return "r-" + os.urandom(16).hex()


@dataclass
class Connection:
    """Where requests go and what they carry: server, credential, project, origin."""

    url: str
    token: str | None
    project: str | None
    origin: str
    timeout: float = 40.0
    retries: int = 3
    trust_env: bool = True
    user_agent: str = "generalforecast-python"

    def prepare(
        self, request: Request, rid: str
    ) -> tuple[str, str, dict[str, str], bytes | None, dict[str, Any]]:
        """The method, URL, headers, body and query of one try."""
        headers = {
            "Accept": "application/json, " + ARROW,
            "User-Agent": self.user_agent,
            "X-GF-Origin": self.origin,
            "X-GF-Request-ID": rid,
        }
        headers.update(request.headers)
        if request.url is not None:
            url = request.url
            authorize = False
        elif request.scoped:
            if not self.project:
                raise InvalidInput(
                    "no project chosen",
                    hint="pass project= or set GF_PROJECT",
                )
            url = f"{self.url}/v1/projects/{_segment(self.project)}{request.path}"
            authorize = True
        else:
            url = self.url + request.path
            authorize = True
        if authorize and self.token:
            headers["Authorization"] = "Bearer " + self.token
        if authorize and self.project:
            headers["X-GF-Project"] = self.project
        if request.respond_async:
            headers["Prefer"] = "respond-async"
        body: bytes | None = None
        if request.arrow is not None:
            from generalforecast import _data

            envelope = self.envelope(request, rid)
            body = _data.stream(request.arrow, {"gf.request": json.dumps(envelope)})
            headers["Content-Type"] = ARROW
        elif request.body is not None:
            body = request.body
            headers["Content-Type"] = request.content_type or OCTETS
        elif request.payload is not None:
            value = self.envelope(request, rid) if request.envelope else request.payload
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        return request.method, url, headers, body, dict(request.params)

    def envelope(self, request: Request, rid: str) -> dict[str, Any]:
        """The `gf/v1` envelope around a request's payload."""
        return {
            "api_version": API_VERSION,
            "project": self.project,
            "request_id": rid,
            "resource": {"kind": request.kind, "name": request.name},
            "payload": request.payload if request.payload is not None else {},
        }


def interpret(
    status: int, headers: Mapping[str, str], content: bytes, rid: str
) -> Reply:
    """A successful reply's result, or the error the reply names.

    Raises:
        Error: Of the category the error's code, or else the status, names.
    """
    kind = headers.get("content-type", "").split(";")[0].strip()
    if kind == ARROW and status < 300:
        from generalforecast import _data

        table = _data.read_stream(content)
        meta = (table.schema.metadata or {}).get(b"gf.result")
        result = json.loads(meta) if meta else {}
        return Reply(status, result, headers, rid, table.replace_schema_metadata(None))
    value: Any = None
    if content:
        try:
            value = json.loads(content)
        except ValueError:
            value = None
    if status >= 300 or (isinstance(value, Mapping) and "error" in value):
        error = value.get("error") if isinstance(value, Mapping) else None
        if status < 400 and not error:
            raise Unavailable(
                f"the server answered {status}, a redirect this client does not follow",
                hint="check GF_URL",
                status=status,
                request_id=rid,
            )
        echoed = value.get("request_id") if isinstance(value, Mapping) else None
        raise from_wire(error, status if status >= 400 else 500, echoed or rid)
    if isinstance(value, Mapping) and "result" in value and "api_version" in value:
        return Reply(status, value["result"], headers, value.get("request_id") or rid)
    return Reply(status, value, headers, rid)


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    raw = headers.get("retry-after")
    if raw:
        try:
            return min(max(float(raw), 0.0), MAX_RETRY_AFTER)
        except ValueError:
            pass
    return min(0.5 * 2**attempt, 8.0) * (0.5 + random.random() / 2)


def _unreachable(error: Exception, url: str, rid: str) -> Error:
    host = urlsplit(url).netloc
    return Unavailable(
        f"could not reach {host}: {type(error).__name__}",
        hint="check the network, HTTPS_PROXY, and GF_URL; then try again",
        request_id=rid,
    )


class Transport:
    """Sends requests in the calling thread, retrying what is safe to retry."""

    def __init__(self, connection: Connection) -> None:
        """Opens an HTTP client that honours HTTPS_PROXY unless `trust_env` is false."""
        import httpx

        self.connection = connection
        self.closed = False
        self._client = httpx.Client(
            timeout=connection.timeout,
            verify=ssl_context(connection.trust_env),
            trust_env=connection.trust_env,
            follow_redirects=False,
        )
        self.sleep = time.sleep

    def close(self) -> None:
        """Closes the connection pool."""
        self.closed = True
        self._client.close()

    def send(self, request: Request) -> Reply:
        """Sends one request, again after a transient failure when it is safe to."""
        import httpx

        if self.closed:
            raise InvalidInput("this client is closed")
        rid = request_id()
        attempts = self.connection.retries + 1 if request.safe else 1
        for attempt in range(attempts):
            method, url, headers, body, params = self.connection.prepare(request, rid)
            try:
                response = self._client.request(
                    method, url, headers=headers, content=body, params=params
                )
            except httpx.TransportError as error:
                LOG.debug("%s %s failed: %s", method, url, error)
                if attempt + 1 < attempts:
                    self.sleep(_retry_after({}, attempt))
                    continue
                raise _unreachable(error, url, rid) from None
            LOG.debug("%s %s %s %s", method, url, response.status_code, rid)
            if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                self.sleep(_retry_after(response.headers, attempt))
                continue
            if response.status_code in request.accept:
                return Reply(response.status_code, None, response.headers, rid)
            return interpret(
                response.status_code, response.headers, response.content, rid
            )
        raise AssertionError("unreachable")


class AsyncTransport:
    """Sends requests on an event loop, retrying what is safe to retry."""

    def __init__(self, connection: Connection) -> None:
        """Opens an async HTTP client that honours HTTPS_PROXY unless `trust_env` is false."""
        import httpx

        self.connection = connection
        self.closed = False
        self._client = httpx.AsyncClient(
            timeout=connection.timeout,
            verify=ssl_context(connection.trust_env),
            trust_env=connection.trust_env,
            follow_redirects=False,
        )
        self.sleep = asyncio.sleep

    async def close(self) -> None:
        """Closes the connection pool."""
        self.closed = True
        await self._client.aclose()

    async def send(self, request: Request) -> Reply:
        """Sends one request, again after a transient failure when it is safe to."""
        import httpx

        if self.closed:
            raise InvalidInput("this client is closed")
        rid = request_id()
        attempts = self.connection.retries + 1 if request.safe else 1
        for attempt in range(attempts):
            method, url, headers, body, params = self.connection.prepare(request, rid)
            try:
                response = await self._client.request(
                    method, url, headers=headers, content=body, params=params
                )
            except httpx.TransportError as error:
                if attempt + 1 < attempts:
                    await self.sleep(_retry_after({}, attempt))
                    continue
                raise _unreachable(error, url, rid) from None
            if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                await self.sleep(_retry_after(response.headers, attempt))
                continue
            if response.status_code in request.accept:
                return Reply(response.status_code, None, response.headers, rid)
            return interpret(
                response.status_code, response.headers, response.content, rid
            )
        raise AssertionError("unreachable")
