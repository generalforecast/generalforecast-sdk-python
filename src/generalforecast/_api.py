# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The HTTP API's request and reply types, generated from api/openapi.yaml.

Do not edit: run `python api/generate.py`. Every schema is a frozen dataclass
(or a type alias for scalars, arrays and mappings) with `from_payload`, which
checks the JSON shape and names the offending field in `PayloadError.param`,
and `to_payload`, which writes the JSON back without the optional fields left
as None. `ROUTES` lists every route with its request and reply types, and
the media type of its request body.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal

VERSION = "gf/v1"


class PayloadError(ValueError):
    """A payload that does not match the spec; `param` names the field."""

    def __init__(self, message: str, param: str) -> None:
        """Keep the message and the dotted path of the field."""
        super().__init__(f"{param}: {message}" if param else message)
        self.message, self.param = message, param


Decoder = Callable[[Any, str], Any]


def _at(param: str, key: str) -> str:
    return f"{param}.{key}" if param else key


def _text(value: Any, param: str) -> str:
    if type(value) is not str:
        raise PayloadError("expected a string", param)
    return value


def _integer(value: Any, param: str) -> int:
    if type(value) is not int:
        raise PayloadError("expected an integer", param)
    return value


def _number(value: Any, param: str) -> float:
    if type(value) not in (int, float):
        raise PayloadError("expected a number", param)
    return value


def _boolean(value: Any, param: str) -> bool:
    if type(value) is not bool:
        raise PayloadError("expected a boolean", param)
    return value


def _null(value: Any, param: str) -> None:
    if value is not None:
        raise PayloadError("expected null", param)


def _any(value: Any, param: str) -> Any:
    return value


def _pattern(expression: str) -> Decoder:
    compiled = re.compile(expression)

    def decode(value: Any, param: str) -> str:
        text = _text(value, param)
        if compiled.search(text) is None:
            raise PayloadError(f"expected a string matching {expression}", param)
        return text

    return decode


def _literal(*options: Any) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        if type(value) is bool and bool not in {type(option) for option in options}:
            raise PayloadError(f"expected one of {list(options)}", param)
        if value not in options:
            raise PayloadError(f"expected one of {list(options)}", param)
        return value

    return decode


def _items(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> tuple:
        if type(value) is not list:
            raise PayloadError("expected an array", param)
        return tuple(item(element, f"{param}[{i}]") for i, element in enumerate(value))

    return decode


def _mapping(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> dict:
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        return {key: item(element, _at(param, key)) for key, element in value.items()}

    return decode


def _union(*options: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        errors = []
        for option in options:
            try:
                return option(value, param)
            except PayloadError as error:
                errors.append(error.message)
        raise PayloadError(" or ".join(errors), param)

    return decode


def _record(cls: type[Payload]) -> Decoder:
    return cls.from_payload


def _enum(cls: type[StrEnum]) -> Decoder:
    def decode(value: Any, param: str) -> StrEnum:
        try:
            return cls(_text(value, param))
        except ValueError:
            raise PayloadError(
                f"expected one of {[m.value for m in cls]}", param
            ) from None

    return decode


def _encode(value: Any) -> Any:
    if isinstance(value, Payload):
        return value.to_payload()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Field:
    """How one JSON key of a record maps to a dataclass field."""

    key: str
    name: str
    required: bool
    decode: Decoder


@dataclass(frozen=True, kw_only=True)
class Payload:
    """A record schema: JSON in through `from_payload`, out through `to_payload`."""

    FIELDS: ClassVar[tuple[Field, ...]] = ()
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    EXTRA: ClassVar[bool] = False

    @classmethod
    def from_payload(cls, value: Any, param: str = ""):
        """Decode a JSON object, naming the first field that does not fit."""
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        known = {spec.key for spec in cls.FIELDS}
        kwargs: dict[str, Any] = {}
        for spec in cls.FIELDS:
            if spec.key in value:
                kwargs[spec.name] = spec.decode(value[spec.key], _at(param, spec.key))
            elif spec.required:
                raise PayloadError("required", _at(param, spec.key))
        unknown = [key for key in value if key not in known]
        if cls.EXTRA:
            kwargs["extra"] = {key: value[key] for key in unknown}
        elif unknown:
            raise PayloadError("unknown field", _at(param, unknown[0]))
        return cls(**kwargs)

    def to_payload(self) -> dict[str, Any]:
        """The JSON object, without optional fields left as None."""
        payload: dict[str, Any] = {}
        for spec in self.FIELDS:
            value = getattr(self, spec.name)
            if value is None and not spec.required:
                continue
            payload[spec.key] = _encode(value)
        if self.EXTRA:
            payload.update(_encode(self.__dict__["extra"]))
        return payload


@dataclass(frozen=True)
class Route:
    """A route: its request and reply types, and its request body's media type.

    `request` is None for a GET, which has no body, and for a body that is not
    JSON, such as an Arrow stream, which `body` names.
    """

    name: str
    method: str
    path: str
    request: type[Payload] | None
    reply: type[Payload]
    body: str | None


Name = str
_decode_Name: Decoder = _pattern("^[a-z][a-z0-9-]{0,62}$")


ModelName = str
_decode_ModelName: Decoder = _pattern("^[a-z0-9][a-z0-9.-]{0,62}$")


RunModel = str
_decode_RunModel: Decoder = _pattern(
    "^[a-z][a-z0-9-]{0,62}/[a-z][a-z0-9-]{0,62}@ft[1-9]\\d*$",
)


RunRef = str
_decode_RunRef: Decoder = _pattern("^[a-z][a-z0-9-]{0,62}(?:@ft[1-9]\\d*)?$")


FineTuneName = str
_decode_FineTuneName: Decoder = _pattern("^[a-z][a-z0-9-]{0,62}@ft[1-9]\\d*$")


ModelId = str
_decode_ModelId: Decoder = _pattern(
    "^[a-z][a-z0-9-]{0,62}/[a-z0-9][a-z0-9.-]{0,62}(?:@ft[1-9]\\d*)?(?:@sha256:[0-9a-f]{64})?(?:@\\d+)?(?:\\*\\d+)?$",
)


Digest = str
_decode_Digest: Decoder = _pattern("^sha256:[0-9a-f]{64}$")


Task = Literal["classification", "regression", "forecast"]
_decode_Task: Decoder = _literal("classification", "regression", "forecast")


ResourceKind = Literal[
    "predictor",
    "model",
    "project",
    "api-key",
    "secret",
    "context",
    "source",
    "workspace",
    "snapshot",
    "run",
    "prediction",
    "experiment",
    "estimate",
]
_decode_ResourceKind: Decoder = _literal(
    "predictor",
    "model",
    "project",
    "api-key",
    "secret",
    "context",
    "source",
    "workspace",
    "snapshot",
    "run",
    "prediction",
    "experiment",
    "estimate",
)


@dataclass(frozen=True, kw_only=True)
class Resource(Payload):
    """What the envelope is about; GET lists use the collection name."""

    kind: ResourceKind
    name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _decode_ResourceKind),
        Field("name", "name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Envelope(Payload):
    """The fields every request and reply carries."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
    )


class ErrorCode(StrEnum):
    """The ErrorCode values.

    ValidationError 422, NotFound 404, Unauthorized 401, CapacityExceeded 422,
    ModelNotReady 503, Internal 500, Conflict 409 (a compare-and-swap on a stale
    generation), Unsupported 501 (an operation this server does not offer, such as a
    hosted route on gf serve), PaymentRequired 402 (the workspace's plan or credit does
    not cover the request), AmbiguousReference 422 (a short model name whose task
    nothing decides: pass `task`, or the full id). Hosted fine-tuning's gates, with
    their `reason`: `PaymentRequired` 402 `plan_required` (fine-tuning needs a Pro or
    Enterprise plan) and `credits_exhausted` (an enforced workspace has used its
    credits, which also refuses predictions); `Unauthorized` 403
    `fine_tuning_unavailable` (fine-tuning is not turned on for the project);
    `CapacityExceeded` 429 `run_limit_reached` (the project has as many runs queued or
    running as it may), `quota_exceeded` (the project's training data would pass its
    quota) and `snapshot_quota_exceeded`. Those three 429s carry `limit` and `actual`,
    and lift only when the caller ends a run or deletes data, so clients do not retry
    them; a 429 `rate_limited` passes on its own and may be retried. A run's create is
    refused with `Conflict` 409 `run_active` (another run of the fine-tune is still
    queued or running) or `base_mismatch` (the fine-tune's versions train from another
    base), and with `ValidationError` 422 `base_is_fine_tune` (a fine-tune sent as the
    base).
    """

    ValidationError = "ValidationError"
    NotFound = "NotFound"
    Unauthorized = "Unauthorized"
    CapacityExceeded = "CapacityExceeded"
    ModelNotReady = "ModelNotReady"
    Internal = "Internal"
    Conflict = "Conflict"
    Unsupported = "Unsupported"
    PaymentRequired = "PaymentRequired"
    AmbiguousReference = "AmbiguousReference"


@dataclass(frozen=True, kw_only=True)
class Error(Payload):
    """The Error schema.

    `param` names the payload field (`target`, `inputs.plan`, `queries`); every limit
    error carries `limit` and `actual` and names the model in `message`. The request id
    is in the envelope.
    """

    code: ErrorCode
    message: str
    reason: str | None = None
    param: str | None = None
    hint: str | None = None
    limit: int | None = None
    actual: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("code", "code", True, _enum(ErrorCode)),
        Field("message", "message", True, _text),
        Field("reason", "reason", False, _pattern("^[a-z][a-z0-9_]{0,63}$")),
        Field("param", "param", False, _text),
        Field("hint", "hint", False, _text),
        Field("limit", "limit", False, _integer),
        Field("actual", "actual", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ErrorReply(Payload):
    """The ErrorReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("error", "error", True, _record(Error)),
    )


Row = dict[str, str | float | bool | None]
_decode_Row: Decoder = _mapping(_union(_text, _number, _boolean, _null))


Kind = Literal["int", "float", "0/1", "category", "text", "timestamp", "bool"]
_decode_Kind: Decoder = _literal(
    "int",
    "float",
    "0/1",
    "category",
    "text",
    "timestamp",
    "bool",
)


@dataclass(frozen=True, kw_only=True)
class Inputs(Payload):
    """The Inputs schema.

    inputs by rule rather than by list: `exclude` removes columns from the inferred
    set, `types` overrides inferred types; forecast predictors use `past` and `future`
    instead
    """

    exclude: tuple[str, ...] | None = None
    types: dict[str, Kind] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_Kind)),
        Field("past", "past", False, _items(_text)),
        Field("future", "future", False, _items(_text)),
    )


ColumnType = Literal["number", "category", "text", "boolean", "date", "timestamp"]
_decode_ColumnType: Decoder = _literal(
    "number",
    "category",
    "text",
    "boolean",
    "date",
    "timestamp",
)


@dataclass(frozen=True, kw_only=True)
class Selection(Payload):
    """The Selection schema.

    which context rows a query sees; fractions below 1 are of the model's row cap,
    integers are rows; `core: null` is the whole table
    """

    core: float | None = None
    neighbors: float | None = None
    kind: Literal["stratified", "random"] | None = None
    recent: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("core", "core", False, _union(_number, _null)),
        Field("neighbors", "neighbors", False, _number),
        Field("kind", "kind", False, _literal("stratified", "random")),
        Field("recent", "recent", False, _union(_integer, _null)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "core": 0.25,
        "neighbors": 0.75,
        "kind": "stratified",
    }


ModelField = ModelId | ModelName
_decode_ModelField: Decoder = _union(_decode_ModelId, _decode_ModelName)


@dataclass(frozen=True, kw_only=True)
class SnapshotReference(Payload):
    """A context the service holds by its digest."""

    snapshot: Digest

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("snapshot", "snapshot", True, _decode_Digest),
    )


@dataclass(frozen=True, kw_only=True)
class Manifest(Payload):
    """The Manifest schema.

    the ad hoc predictor: a predictor's fields, with `context` as rows or by reference
    to a snapshot the service holds (`{"snapshot": "<digest>"}`, uploaded with `PUT
    /snapshots?digest=`), and `queries` as rows, or as the Arrow stream the request is.
    `task` decides a short model name's entry; `exclude`, `types`, `past` and `future`
    take a predictor spec's forms beside `inputs`'
    """

    model: ModelField
    target: str
    context: tuple[Row, ...] | SnapshotReference
    queries: tuple[Row, ...]
    task: Task | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None
    id: str | None = None
    inputs: tuple[str, ...] | Inputs | None = None
    time: str | None = None
    series: tuple[str, ...] | None = None
    related: Literal["all"] | tuple[str, ...] | None = None
    frequency: str | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None
    outputs: (
        tuple[Literal["label", "value", "probabilities", "quantiles"], ...] | None
    ) = None
    selection: Selection | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("task", "task", False, _decode_Task),
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("past", "past", False, _items(_text)),
        Field("future", "future", False, _items(_text)),
        Field("model", "model", True, _decode_ModelField),
        Field("target", "target", True, _text),
        Field("id", "id", False, _text),
        Field("inputs", "inputs", False, _union(_items(_text), _record(Inputs))),
        Field("time", "time", False, _text),
        Field("series", "series", False, _items(_text)),
        Field("related", "related", False, _union(_literal("all"), _items(_text))),
        Field(
            "frequency",
            "frequency",
            False,
            _union(_pattern("^[1-9][0-9]*(s|min|h|d|w)$"), _number),
        ),
        Field("horizon", "horizon", False, _integer),
        Field("quantiles", "quantiles", False, _items(_number)),
        Field(
            "outputs",
            "outputs",
            False,
            _items(_literal("label", "value", "probabilities", "quantiles")),
        ),
        Field("selection", "selection", False, _record(Selection)),
        Field(
            "context",
            "context",
            True,
            _union(_items(_decode_Row), _record(SnapshotReference)),
        ),
        Field("queries", "queries", True, _items(_decode_Row)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"quantiles": [0.1, 0.5, 0.9]}


@dataclass(frozen=True, kw_only=True)
class Usage(Payload):
    """The Usage schema.

    what a call is billed: its scored rows, and past the large-context line
    (`large_context_cells` in the price book) the GPU seconds of its own execution.
    `calls` is how many model calls the engine made, each as many queries as the model
    takes at once, as `/estimate` counts them
    """

    rows: int
    credits: float
    usd: float
    calls: int | None = None
    gpu_seconds: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("calls", "calls", False, _integer),
        Field("rows", "rows", True, _integer),
        Field("credits", "credits", True, _number),
        Field("usd", "usd", True, _number),
        Field("gpu_seconds", "gpu_seconds", False, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Timing(Payload):
    """The Timing schema.

    where a call's time went, queueing and cold start apart from compute;
    `compile_seconds` is the time its steps spent compiling or loading programs they
    waited on, which `compute_seconds`, its own execution, leaves out
    """

    queued_seconds: float
    cold_start_seconds: float
    compute_seconds: float
    compile_seconds: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("queued_seconds", "queued_seconds", True, _number),
        Field("cold_start_seconds", "cold_start_seconds", True, _number),
        Field("compile_seconds", "compile_seconds", False, _number),
        Field("compute_seconds", "compute_seconds", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class HeldSnapshot(Payload):
    """The HeldSnapshot schema."""

    digest: Digest
    rows: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("digest", "digest", True, _decode_Digest),
        Field("rows", "rows", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class HeldSnapshotReply(Payload):
    """The HeldSnapshotReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: HeldSnapshot

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(HeldSnapshot)),
    )


@dataclass(frozen=True, kw_only=True)
class ManifestRequest(Payload):
    """The ManifestRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Manifest

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Manifest)),
    )


AsOfRequest = str | float
_decode_AsOfRequest: Decoder = _union(_text, _number)


@dataclass(frozen=True, kw_only=True)
class Queries(Payload):
    """A predictor's payload: the queries, and `as_of` for a live context."""

    queries: tuple[Row, ...]
    as_of: AsOfRequest | None = None
    env: Name | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("queries", "queries", True, _items(_decode_Row)),
        Field("as_of", "as_of", False, _decode_AsOfRequest),
        Field("env", "env", False, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class QueriesRequest(Payload):
    """The QueriesRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Queries

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Queries)),
    )


ContextKind = Literal["snapshot", "request", "live"]
_decode_ContextKind: Decoder = _literal("snapshot", "request", "live")


AsOf = str | None
_decode_AsOf: Decoder = _union(_pattern("Z$"), _null)


@dataclass(frozen=True, kw_only=True)
class ContextInfo(Payload):
    """The ContextInfo schema.

    the context a prediction actually used, the same shape from gf serve and the hosted
    platform. `source` is the context's kind; `name` names a live context; `version` is
    a live context's accepted sequence; `digest` is a snapshot's content digest; `rows`
    counts the rows read; `as_of` is the instant a live context was read as of. Each
    field a source does not have is null.
    """

    source: ContextKind
    name: Name | None
    version: int | None
    digest: Digest | None
    rows: int
    as_of: AsOf
    used: Literal["exact", "retrieved"] | None = None
    core: int | None = None
    cells: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _decode_ContextKind),
        Field("name", "name", True, _union(_decode_Name, _null)),
        Field("version", "version", True, _union(_integer, _null)),
        Field("digest", "digest", True, _union(_decode_Digest, _null)),
        Field("rows", "rows", True, _integer),
        Field("as_of", "as_of", True, _decode_AsOf),
        Field("used", "used", False, _literal("exact", "retrieved")),
        Field("core", "core", False, _union(_integer, _null)),
        Field("cells", "cells", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Prediction(Payload):
    """The Prediction schema.

    the query's id under the predictor's id column name when it names one (omitted when
    it does not); `label` and `probabilities` for classification, `value` for
    regression, `value` and `quantiles` for forecasts
    """

    label: str | int | bool | None = None
    value: float | None = None
    probabilities: dict[str, float] | None = None
    quantiles: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("label", "label", False, _union(_text, _integer, _boolean)),
        Field("value", "value", False, _number),
        Field("probabilities", "probabilities", False, _mapping(_number)),
        Field("quantiles", "quantiles", False, _mapping(_number)),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class Result(Payload):
    """The Result schema."""

    model: ModelId
    task: Task
    context: ContextInfo
    predictions: tuple[Prediction, ...]
    predictor: Name | None = None
    revision: int | None = None
    environment: Name | None = None
    usage: Usage | None = None
    timing: Timing | None = None
    notes: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelId),
        Field("task", "task", True, _decode_Task),
        Field("context", "context", True, _record(ContextInfo)),
        Field("predictor", "predictor", False, _union(_decode_Name, _null)),
        Field("revision", "revision", False, _union(_integer, _null)),
        Field("environment", "environment", False, _union(_decode_Name, _null)),
        Field("predictions", "predictions", True, _items(_record(Prediction))),
        Field("usage", "usage", False, _record(Usage)),
        Field("timing", "timing", False, _record(Timing)),
        Field("notes", "notes", False, _items(_text)),
    )


@dataclass(frozen=True, kw_only=True)
class ResultReply(Payload):
    """The ResultReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Result

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Result)),
    )


PredictionId = str
_decode_PredictionId: Decoder = _pattern("^pred-[0-9a-f]{24}$")


@dataclass(frozen=True, kw_only=True)
class PredictionStatus(Payload):
    """The PredictionStatus schema.

    a prediction the hosted plane accepted and is still working on: `starting` while
    hosted inference starts, `running` once it has taken the work
    """

    id: PredictionId
    state: Literal["starting", "running"]
    created: float
    retry_after: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_PredictionId),
        Field("state", "state", True, _literal("starting", "running")),
        Field("created", "created", True, _number),
        Field("retry_after", "retry_after", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class PredictionAccepted(Payload):
    """The PredictionAccepted schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictionStatus

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictionStatus)),
    )


@dataclass(frozen=True, kw_only=True)
class Limits(Payload):
    """The Limits schema.

    the sizes a model was trained and measured to; a call past them is admitted by the
    device's memory and carries a note
    """

    context_rows: int
    queries: int
    inputs: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context_rows", "context_rows", True, _integer),
        Field("queries", "queries", True, _integer),
        Field("inputs", "inputs", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Shape(Payload):
    """A compiled program's shape."""

    context_rows: int
    queries: int
    inputs: int
    classes: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context_rows", "context_rows", True, _integer),
        Field("queries", "queries", True, _integer),
        Field("inputs", "inputs", True, _integer),
        Field("classes", "classes", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class TrainedOn(Payload):
    """The TrainedOn schema.

    what a catalog model saw in training: its upstream's stated context rows, inputs
    and classes, each null where none is stated. A note shown beside results, never a
    limit; null for a forecaster
    """

    rows: int | None
    inputs: int | None
    classes: int | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("rows", "rows", True, _union(_integer, _null)),
        Field("inputs", "inputs", True, _union(_integer, _null)),
        Field("classes", "classes", True, _union(_integer, _null)),
    )


FineTuneKind = Literal["lora", "input_adapter", "full"]
_decode_FineTuneKind: Decoder = _literal("lora", "input_adapter", "full")


KeptStep = int | None
_decode_KeptStep: Decoder = _union(_integer, _null)


RunArchitectureComponent = Literal[
    "attention.kv_heads",
    "attention.query_scale",
    "attention.qk_norm",
    "attention.output_gate",
    "positional_encoding",
    "norm",
    "norm_placement",
    "feed_forward",
    "attention.temperature",
    "feed_forward_branch",
    "cell_embedding.numeric",
    "cell_embedding.categorical",
]
_decode_RunArchitectureComponent: Decoder = _literal(
    "attention.kv_heads",
    "attention.query_scale",
    "attention.qk_norm",
    "attention.output_gate",
    "positional_encoding",
    "norm",
    "norm_placement",
    "feed_forward",
    "attention.temperature",
    "feed_forward_branch",
    "cell_embedding.numeric",
    "cell_embedding.categorical",
)


@dataclass(frozen=True, kw_only=True)
class RunArchitectureOption(Payload):
    """The RunArchitectureOption schema.

    What a component becomes: `{"type": <option>}` with the option's settings beside
    it, as `{"type": "grouped", "heads": 2}` or `{"type": "rope", "theta": 10000}`. A
    catalog model's `architecture` lists each component's options, their settings and
    defaults; a setting left out takes its default, and an unknown one is refused with
    422.
    """

    type: str
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("type", "type", True, _text),)
    EXTRA: ClassVar[bool] = True


RunArchitectureInitType = Literal[
    "default",
    "zeros",
    "ones",
    "constant",
    "normal",
    "truncated_normal",
    "uniform",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
]
_decode_RunArchitectureInitType: Decoder = _literal(
    "default",
    "zeros",
    "ones",
    "constant",
    "normal",
    "truncated_normal",
    "uniform",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
)


@dataclass(frozen=True, kw_only=True)
class RunArchitectureInit(Payload):
    """The RunArchitectureInit schema.

    How a change's new parameters start, each matching its `torch.nn.init` /
    `jax.nn.initializers` namesake: `default` (the option's documented init), `zeros`,
    `ones`, `constant` (`value`), `normal` (`mean` 0 and `std` 0.02 when left out),
    `truncated_normal` (`std` 0.02, cut at 2 standard deviations), `uniform` (`low` 0
    and `high` 1), and `xavier_uniform`, `xavier_normal`, `kaiming_uniform`,
    `kaiming_normal` (fan in, ReLU gain) and `orthogonal`, which draw the change's
    matrices while its vectors take the option's default. Draws are deterministic in
    the run's `seed`, the component and the stage. Every init is accepted for every
    option with parameters.
    """

    type: RunArchitectureInitType
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("type", "type", True, _decode_RunArchitectureInitType),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class RunResolvedArchitectureChange(Payload):
    """The RunResolvedArchitectureChange schema.

    A change as the run resolved it and its version records it: the stages it applies
    in, the option with every setting filled in (a null `hidden` is the block's own
    width), and its init with every setting filled in, null for an option without
    parameters
    """

    component: RunArchitectureComponent
    stages: tuple[str, ...]
    to: RunArchitectureOption
    init: RunArchitectureInit | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("component", "component", True, _decode_RunArchitectureComponent),
        Field("stages", "stages", True, _items(_text)),
        Field("to", "to", True, _record(RunArchitectureOption)),
        Field("init", "init", True, _union(_record(RunArchitectureInit), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class DerivedModel(Payload):
    """The DerivedModel schema.

    what a fine-tune is made of, as a predictor's lock records it: its base, pinned,
    and its weights; present on the project's fine-tunes alone, which otherwise
    describe themselves as their base does
    """

    base: ModelId
    base_digest: Digest
    digest: Digest
    bags: int
    kind: FineTuneKind
    version: int
    created: float
    kept_step: KeptStep
    architecture: tuple[RunResolvedArchitectureChange, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("base", "base", True, _decode_ModelId),
        Field("base_digest", "base_digest", True, _decode_Digest),
        Field("digest", "digest", True, _decode_Digest),
        Field("bags", "bags", True, _integer),
        Field("kind", "kind", True, _decode_FineTuneKind),
        Field("version", "version", True, _integer),
        Field("created", "created", True, _number),
        Field("kept_step", "kept_step", True, _decode_KeptStep),
        Field(
            "architecture",
            "architecture",
            False,
            _items(_record(RunResolvedArchitectureChange)),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class FineTuneVersion(Payload):
    """The FineTuneVersion schema.

    one published version of a fine-tune: its id, its weights' digest and the step its
    run kept. The run that trained it has the same id as its `model`. A version never
    changes: its id names the same weights for as long as it exists.
    """

    name: ModelId
    digest: Digest
    version: int
    kind: FineTuneKind
    created: float
    kept_step: KeptStep

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_ModelId),
        Field("digest", "digest", True, _decode_Digest),
        Field("version", "version", True, _integer),
        Field("kind", "kind", True, _decode_FineTuneKind),
        Field("created", "created", True, _number),
        Field("kept_step", "kept_step", True, _decode_KeptStep),
    )


@dataclass(frozen=True, kw_only=True)
class Artifact(Payload):
    """The Artifact schema."""

    repository: str
    revision: str
    digest: Digest
    license: str
    files: dict[str, str]
    digest_format: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("repository", "repository", True, _text),
        Field("revision", "revision", True, _text),
        Field("digest", "digest", True, _decode_Digest),
        Field("license", "license", True, _text),
        Field("digest_format", "digest_format", False, _text),
        Field("files", "files", True, _mapping(_text)),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class ModelVariant(Payload):
    """A catalog variant's ensemble, `@K` codes and `*N` estimators."""

    codes: int
    estimators: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("codes", "codes", True, _integer),
        Field("estimators", "estimators", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Capability(Payload):
    """The Capability schema."""

    name: str
    state: Literal["supported", "unsupported", "unconfigured"]
    code: str | None = None
    reason: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("supported", "unsupported", "unconfigured"),
        ),
        Field("code", "code", False, _union(_text, _null)),
        Field("reason", "reason", False, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ArchitectureSetting(Payload):
    """The ArchitectureSetting schema.

    one setting of an option or an init; `default` is its value when left out (null for
    a width the block gives), absent for a `required` one
    """

    name: str
    type: Literal["integer", "number", "boolean"]
    description: str
    default: int | float | bool | None = None
    minimum: float | None = None
    required: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("type", "type", True, _literal("integer", "number", "boolean")),
        Field("description", "description", True, _text),
        Field("default", "default", False, _union(_integer, _number, _boolean, _null)),
        Field("minimum", "minimum", False, _number),
        Field("required", "required", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ArchitectureOption(Payload):
    """The ArchitectureOption schema.

    one option of a component, its settings, and how its new parameters start under the
    `default` init; `default_init` is null for an option without parameters, which
    takes no init
    """

    type: str
    description: str
    settings: tuple[ArchitectureSetting, ...]
    default_init: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("type", "type", True, _text),
        Field("description", "description", True, _text),
        Field("settings", "settings", True, _items(_record(ArchitectureSetting))),
        Field("default_init", "default_init", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ArchitectureUnavailable(Payload):
    """The ArchitectureUnavailable schema.

    options of a component the model cannot take in some stages, and why; empty
    `options` is every option, empty `stages` every stage
    """

    options: tuple[str, ...]
    stages: tuple[str, ...]
    reason: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("options", "options", True, _items(_text)),
        Field("stages", "stages", True, _items(_text)),
        Field("reason", "reason", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class ArchitectureComponent(Payload):
    """A component kind, its options and what this model cannot take of it."""

    component: RunArchitectureComponent
    description: str
    options: tuple[ArchitectureOption, ...]
    unavailable: tuple[ArchitectureUnavailable, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("component", "component", True, _decode_RunArchitectureComponent),
        Field("description", "description", True, _text),
        Field("options", "options", True, _items(_record(ArchitectureOption))),
        Field(
            "unavailable",
            "unavailable",
            True,
            _items(_record(ArchitectureUnavailable)),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class ArchitectureInit(Payload):
    """The ArchitectureInit schema.

    an init type; a `fan_based` one draws matrices alone, a change's vectors then
    taking the option's default
    """

    type: str
    description: str
    settings: tuple[ArchitectureSetting, ...]
    fan_based: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("type", "type", True, _text),
        Field("description", "description", True, _text),
        Field("settings", "settings", True, _items(_record(ArchitectureSetting))),
        Field("fan_based", "fan_based", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ModelArchitecture(Payload):
    """The ModelArchitecture schema.

    what a run's `architecture` may change of a catalog model: its stages, every
    standard component kind with its options, their settings and default inits and what
    the model cannot take, and the init types. Present on the catalog models a run may
    change, the ones that can be fine-tuned.
    """

    stages: tuple[str, ...]
    components: tuple[ArchitectureComponent, ...]
    inits: tuple[ArchitectureInit, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("stages", "stages", True, _items(_text)),
        Field("components", "components", True, _items(_record(ArchitectureComponent))),
        Field("inits", "inits", True, _items(_record(ArchitectureInit))),
    )


@dataclass(frozen=True, kw_only=True)
class Model(Payload):
    """The Model schema.

    the catalog's description of a served model: `model` is the resolved id with its
    content digest, `limits` what one call may send here, `warm_shapes` the shapes
    compiled at startup
    """

    short: str | None
    kind: Literal["catalog", "fine-tune"]
    base: str | None
    trained_on: TrainedOn | None
    model: str
    name: ModelId
    task: Task
    repository: str
    revision: str
    license: str
    runtime: Literal["cpu", "gpu"]
    artifact: Artifact
    bounds: dict[str, tuple[float, ...] | int | float | None]
    capabilities: tuple[Capability, ...]
    limits: Limits
    warm_shapes: tuple[Shape, ...]
    family: str | None = None
    hostable: bool | None = None
    derived: DerivedModel | None = None
    versions: tuple[FineTuneVersion, ...] | None = None
    artifact_verification: str | None = None
    license_source: str | None = None
    variant: ModelVariant | None = None
    architecture: ModelArchitecture | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("short", "short", True, _union(_text, _null)),
        Field("kind", "kind", True, _literal("catalog", "fine-tune")),
        Field("family", "family", False, _text),
        Field("base", "base", True, _union(_text, _null)),
        Field("trained_on", "trained_on", True, _union(_record(TrainedOn), _null)),
        Field("hostable", "hostable", False, _boolean),
        Field("model", "model", True, _text),
        Field("name", "name", True, _decode_ModelId),
        Field("derived", "derived", False, _record(DerivedModel)),
        Field("versions", "versions", False, _items(_record(FineTuneVersion))),
        Field("task", "task", True, _decode_Task),
        Field("repository", "repository", True, _text),
        Field("revision", "revision", True, _text),
        Field("license", "license", True, _text),
        Field("runtime", "runtime", True, _literal("cpu", "gpu")),
        Field("artifact", "artifact", True, _record(Artifact)),
        Field("artifact_verification", "artifact_verification", False, _text),
        Field("license_source", "license_source", False, _text),
        Field("variant", "variant", False, _record(ModelVariant)),
        Field(
            "bounds",
            "bounds",
            True,
            _mapping(_union(_items(_number), _union(_integer, _number, _null))),
        ),
        Field("capabilities", "capabilities", True, _items(_record(Capability))),
        Field("limits", "limits", True, _record(Limits)),
        Field("warm_shapes", "warm_shapes", True, _items(_record(Shape))),
        Field("architecture", "architecture", False, _record(ModelArchitecture)),
    )


@dataclass(frozen=True, kw_only=True)
class ModelList(Payload):
    """The ModelList schema."""

    models: tuple[Model, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("models", "models", True, _items(_record(Model))),
    )


@dataclass(frozen=True, kw_only=True)
class ModelReply(Payload):
    """The ModelReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Model

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Model)),
    )


@dataclass(frozen=True, kw_only=True)
class ModelListReply(Payload):
    """The ModelListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ModelList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ModelList)),
    )


@dataclass(frozen=True, kw_only=True)
class Build(Payload):
    """The Build schema."""

    commit: str | None
    version: str | None
    gf_commit: str | None
    gf_version: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("commit", "commit", True, _union(_text, _null)),
        Field("version", "version", True, _union(_text, _null)),
        Field("gf_commit", "gf_commit", True, _union(_text, _null)),
        Field("gf_version", "gf_version", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Health(Payload):
    """The Health schema."""

    status: Literal["ok"] = "ok"
    build: Build | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("ok")),
        Field("build", "build", False, _record(Build)),
    )


@dataclass(frozen=True, kw_only=True)
class HealthError(Payload):
    """The HealthError schema."""

    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("error", "error", True, _record(Error)),
    )


@dataclass(frozen=True, kw_only=True)
class Empty(Payload):
    """The Empty schema."""

    FIELDS: ClassVar[tuple[Field, ...]] = ()


@dataclass(frozen=True, kw_only=True)
class EmptyRequest(Payload):
    """The EmptyRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Empty

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Empty)),
    )


@dataclass(frozen=True, kw_only=True)
class Project(Payload):
    """The Project schema."""

    name: Name
    organization: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("organization", "organization", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectReply(Payload):
    """The ProjectReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Project

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Project)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectList(Payload):
    """The ProjectList schema."""

    projects: tuple[Project, ...]
    organizations: tuple[Name, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("projects", "projects", True, _items(_record(Project))),
        Field("organizations", "organizations", True, _items(_decode_Name)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectListReply(Payload):
    """The ProjectListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ProjectList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ProjectList)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectCreate(Payload):
    """The ProjectCreate schema."""

    name: Name
    organization: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("organization", "organization", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectCreateRequest(Payload):
    """The ProjectCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ProjectCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ProjectCreate)),
    )


Permission = (
    Literal[
        "predictors:read",
        "predictors:write",
        "predictors:predict",
        "contexts:read",
        "contexts:write",
        "contexts:history",
        "sources:read",
        "sources:write",
        "sources:manage",
        "models:read",
        "snapshots:read",
        "snapshots:write",
        "runs:read",
        "runs:write",
        "keys:read",
        "keys:write",
        "secrets:read",
        "secrets:write",
        "members:read",
        "members:write",
        "sso:read",
        "sso:write",
        "audit:read",
        "operators:read",
        "operators:write",
        "projects:read",
    ]
    | str
)
_decode_Permission: Decoder = _union(
    _literal(
        "predictors:read",
        "predictors:write",
        "predictors:predict",
        "contexts:read",
        "contexts:write",
        "contexts:history",
        "sources:read",
        "sources:write",
        "sources:manage",
        "models:read",
        "snapshots:read",
        "snapshots:write",
        "runs:read",
        "runs:write",
        "keys:read",
        "keys:write",
        "secrets:read",
        "secrets:write",
        "members:read",
        "members:write",
        "sso:read",
        "sso:write",
        "audit:read",
        "operators:read",
        "operators:write",
        "projects:read",
    ),
    _pattern("^[a-z][a-z0-9-]*:[a-z][a-z0-9-]*$"),
)


@dataclass(frozen=True, kw_only=True)
class Key(Payload):
    """`token` is present only in the create and rotate replies."""

    id: str
    project: Name
    name: Name
    permissions: tuple[Permission, ...]
    created_at: float
    expires_at: float | None
    revoked_at: float | None
    state: Literal["active", "expired", "revoked"] | None = None
    prefix: str | None = None
    last_used_at: float | None = None
    ephemeral: bool | None = None
    token: str | None = None
    replaces: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
        Field("project", "project", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("permissions", "permissions", True, _items(_decode_Permission)),
        Field("created_at", "created_at", True, _number),
        Field("expires_at", "expires_at", True, _union(_number, _null)),
        Field("revoked_at", "revoked_at", True, _union(_number, _null)),
        Field("state", "state", False, _literal("active", "expired", "revoked")),
        Field("prefix", "prefix", False, _union(_text, _null)),
        Field("last_used_at", "last_used_at", False, _union(_number, _null)),
        Field("ephemeral", "ephemeral", False, _boolean),
        Field("token", "token", False, _pattern("^gf_[A-Za-z0-9_-]{32,}$")),
        Field("replaces", "replaces", False, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class KeyReply(Payload):
    """The KeyReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Key

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Key)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyList(Payload):
    """The KeyList schema."""

    keys: tuple[Key, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("keys", "keys", True, _items(_record(Key))),
    )


@dataclass(frozen=True, kw_only=True)
class KeyListReply(Payload):
    """The KeyListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: KeyList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(KeyList)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyCreate(Payload):
    """The KeyCreate schema."""

    name: Name
    permissions: tuple[Permission, ...]
    expires_in: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("permissions", "permissions", True, _items(_decode_Permission)),
        Field("expires_in", "expires_in", False, _union(_number, _null)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"expires_in": 2592000}


@dataclass(frozen=True, kw_only=True)
class KeyCreateRequest(Payload):
    """The KeyCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: KeyCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(KeyCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRef(Payload):
    """The KeyRef schema."""

    id: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRefRequest(Payload):
    """The KeyRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: KeyRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(KeyRef)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRevoked(Payload):
    """The KeyRevoked schema."""

    id: str
    revoked: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _text),
        Field("revoked", "revoked", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRevokedReply(Payload):
    """The KeyRevokedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: KeyRevoked

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(KeyRevoked)),
    )


@dataclass(frozen=True, kw_only=True)
class Secret(Payload):
    """Metadata only; a value is never returned."""

    project: Name
    name: Name
    version: int
    created_at: float
    updated_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("project", "project", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("version", "version", True, _integer),
        Field("created_at", "created_at", True, _number),
        Field("updated_at", "updated_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SecretReply(Payload):
    """The SecretReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Secret

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Secret)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretList(Payload):
    """The SecretList schema."""

    secrets: tuple[Secret, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("secrets", "secrets", True, _items(_record(Secret))),
    )


@dataclass(frozen=True, kw_only=True)
class SecretListReply(Payload):
    """The SecretListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SecretList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SecretList)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretCreate(Payload):
    """The SecretCreate schema."""

    name: Name
    value: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("value", "value", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SecretCreateRequest(Payload):
    """The SecretCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRotate(Payload):
    """The SecretRotate schema."""

    name: Name
    value: str
    expected_version: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("value", "value", True, _text),
        Field("expected_version", "expected_version", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRotateRequest(Payload):
    """The SecretRotateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretRotate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretRotate)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRef(Payload):
    """The SecretRef schema."""

    name: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("name", "name", True, _decode_Name),)


@dataclass(frozen=True, kw_only=True)
class SecretRefRequest(Payload):
    """The SecretRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretRef)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretDeleted(Payload):
    """The SecretDeleted schema."""

    name: Name
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretDeletedReply(Payload):
    """The SecretDeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SecretDeleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SecretDeleted)),
    )


RunState = Literal[
    "queued",
    "setting_up",
    "training",
    "preempted",
    "finished",
    "cancelling",
    "cancelled",
    "failed",
]
_decode_RunState: Decoder = _literal(
    "queued",
    "setting_up",
    "training",
    "preempted",
    "finished",
    "cancelling",
    "cancelled",
    "failed",
)


@dataclass(frozen=True, kw_only=True)
class RunData(Payload):
    """The RunData schema.

    a snapshot a run reads, by its name and the digest the name had when the run was
    created
    """

    name: Name
    digest: Digest
    phase: Name | None = None
    context: Name | None = None
    version: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("digest", "digest", True, _decode_Digest),
        Field("phase", "phase", False, _decode_Name),
        Field("context", "context", False, _decode_Name),
        Field("version", "version", False, _integer),
    )


RunComponent = Literal[
    "column_encoder",
    "row_encoder",
    "in_context_transformer",
    "prediction_head",
]
_decode_RunComponent: Decoder = _literal(
    "column_encoder",
    "row_encoder",
    "in_context_transformer",
    "prediction_head",
)


@dataclass(frozen=True, kw_only=True)
class RunMethod(Payload):
    """The RunMethod schema.

    How a run changes the network: `{"type": "lora", "rank": 16}`, with
    `"input_adapter": true` to train an input adapter in front of it; `{"type":
    "input_adapter"}`; or `{"type": "full", "components": [...]}`, some of
    `column_encoder`, `row_encoder`, `in_context_transformer` and `prediction_head`. A
    method the base does not take is refused with 422.
    """

    type: Literal["lora", "input_adapter", "full"]
    rank: int | None = None
    input_adapter: bool | None = None
    components: tuple[RunComponent, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("type", "type", True, _literal("lora", "input_adapter", "full")),
        Field("rank", "rank", False, _integer),
        Field("input_adapter", "input_adapter", False, _boolean),
        Field("components", "components", False, _items(_decode_RunComponent)),
    )


@dataclass(frozen=True, kw_only=True)
class RunArchitectureChange(Payload):
    """The RunArchitectureChange schema.

    One change to the base's architecture, as a create sends it: `{"component":
    "attention.query_scale", "stages": ["icl"], "to": {"type": "gated_log_scale"},
    "init": {"type": "normal", "std": 0.02}}`. `stages` left out is every stage of the
    base that can take the option; `init` left out is `{"type": "default"}`, and an
    option without parameters (`none`, `pre`, `all`, `grouped`, `rope`) takes none.
    """

    component: RunArchitectureComponent
    to: RunArchitectureOption
    stages: tuple[str, ...] | None = None
    init: RunArchitectureInit | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("component", "component", True, _decode_RunArchitectureComponent),
        Field("stages", "stages", False, _items(_text)),
        Field("to", "to", True, _record(RunArchitectureOption)),
        Field("init", "init", False, _record(RunArchitectureInit)),
    )


@dataclass(frozen=True, kw_only=True)
class RunWarning(Payload):
    """Something about the run to know, which does not stop it.

    `uncalibrated_architecture`: the run changes the base's architecture, so the
    changed model is not calibrated to the base and starts from `start_loss` where the
    base scores `base_loss`, on the evaluation rows; both are null until step 0
    measures them.
    """

    code: Literal["uncalibrated_architecture"]
    message: str
    base_loss: float | None
    start_loss: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("code", "code", True, _literal("uncalibrated_architecture")),
        Field("message", "message", True, _text),
        Field("base_loss", "base_loss", True, _union(_number, _null)),
        Field("start_loss", "start_loss", True, _union(_number, _null)),
    )


SyntheticBatchName = str
_decode_SyntheticBatchName: Decoder = _pattern("^[a-z][a-z0-9-]{0,53}$")


@dataclass(frozen=True, kw_only=True)
class RunWaiting(Payload):
    """The RunWaiting schema.

    the synthetic batches a run waits for before it trains, and how many of their
    tables are still pending. `data` holds the snapshots already pinned, and the
    batches' ready tables join it once every batch is done; absent once the run is not
    waiting.
    """

    batches: tuple[SyntheticBatchName, ...]
    pending: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("batches", "batches", True, _items(_decode_SyntheticBatchName)),
        Field("pending", "pending", True, _integer),
    )


RunTag = str
_decode_RunTag: Decoder = _pattern("^[a-z0-9][a-z0-9._:/-]{0,63}$")


@dataclass(frozen=True, kw_only=True)
class RunDataTag(Payload):
    """Every ready snapshot in the project carrying `tag`, sorted by name.

    The run expands it when it is created, or once the synthetic batches it names are
    done, and keeps the names and digests it found, so `runs.get` shows exactly what it
    trains on and tagging snapshots later changes nothing. A tag no ready or generating
    snapshot carries is refused with 422.
    """

    tag: RunTag

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("tag", "tag", True, _decode_RunTag),)


@dataclass(frozen=True, kw_only=True)
class RunDataContext(Payload):
    """The RunDataContext schema.

    a context version a run trains on: `version`, or the latest when it is left out,
    pinned to that version's snapshot digest when the run is created
    """

    context: Name
    version: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context", "context", True, _decode_Name),
        Field("version", "version", False, _integer),
    )


RunDataItem = Name | RunDataTag | RunDataContext
_decode_RunDataItem: Decoder = _union(
    _decode_Name,
    _record(RunDataTag),
    _record(RunDataContext),
)


@dataclass(frozen=True, kw_only=True)
class Origin(Payload):
    """The Origin schema.

    where a resource was made: the surface from the request's `X-GF-Origin` header
    (`api` when it sends none) and who made it
    """

    surface: Literal["console", "cli", "notebook", "script", "api"]
    notebook: str | None
    client: str | None
    by: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "surface",
            "surface",
            True,
            _literal("console", "cli", "notebook", "script", "api"),
        ),
        Field("notebook", "notebook", True, _union(_text, _null)),
        Field("client", "client", True, _union(_text, _null)),
        Field("by", "by", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class EpisodeSampling(Payload):
    """How the trainer draws an episode from the table's training rows."""

    rows: int | Literal["auto"] | None = None
    context_fraction: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("rows", "rows", False, _union(_integer, _literal("auto"))),
        Field("context_fraction", "context_fraction", False, _number),
    )


@dataclass(frozen=True, kw_only=True)
class RunPhasePlan(Payload):
    """The RunPhasePlan schema.

    A phase's own settings: its `steps`, how it samples episodes and its episodes per
    step, `{"steps": 400, "episode_sampling": {"rows": 1024}}`. A setting left out
    takes the service's default, never the phase before's. Every other plan setting is
    set for the whole run, in the run's `plan` (`RunCurriculumPlan`), and is refused
    here with 422.
    """

    steps: int
    episode_sampling: EpisodeSampling | None = None
    episodes_per_step: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("steps", "steps", True, _integer),
        Field("episode_sampling", "episode_sampling", False, _record(EpisodeSampling)),
        Field("episodes_per_step", "episodes_per_step", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class RunPinnedPhase(Payload):
    """The RunPinnedPhase schema.

    a curriculum run's phase as the run keeps it, each snapshot pinned to its digest
    """

    name: Name
    data: tuple[RunData, ...]
    plan: RunPhasePlan

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("data", "data", True, _items(_record(RunData))),
        Field("plan", "plan", True, _record(RunPhasePlan)),
    )


@dataclass(frozen=True, kw_only=True)
class RunPhaseProgress(Payload):
    """One phase's training so far."""

    name: Name
    step: int
    loss: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("step", "step", True, _integer),
        Field("loss", "loss", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class RunProgress(Payload):
    """The RunProgress schema.

    Where training is: the steps trained over the whole run and the latest training
    loss. A curriculum run adds the phase it is in, by name and by its place in
    `phases` (from 0), the steps trained within that phase, and each phase reached so
    far with its own steps and latest training loss.
    """

    step: int
    loss: float | None
    phase: Name | None = None
    phase_index: int | None = None
    phase_step: int | None = None
    phases: tuple[RunPhaseProgress, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("step", "step", True, _integer),
        Field("loss", "loss", True, _union(_number, _null)),
        Field("phase", "phase", False, _decode_Name),
        Field("phase_index", "phase_index", False, _integer),
        Field("phase_step", "phase_step", False, _integer),
        Field("phases", "phases", False, _items(_record(RunPhaseProgress))),
    )


@dataclass(frozen=True, kw_only=True)
class EarlyStoppingThreshold(Payload):
    """How far above 0 a step's paired gain over the base must be to keep it."""

    standard_errors: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("standard_errors", "standard_errors", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class RunEvaluation(Payload):
    """How the run holds out rows and keeps a step.

    `holdout` of the table's rows, clamped to 2,000..50,000 rows, are held out by its
    time column, its group column, or stratified by class or target; `report_share` of
    them are report rows, the rest select rows. Every `eval_steps`, a check scores
    every select row against `contexts` fixed contexts of `context_rows` training rows,
    the base's trained row cap when null, and a step becomes the new best only if its
    gain over the base, row by row, is more than
    `early_stopping_threshold.standard_errors` standard errors above 0 and beats the
    best so far. Training stops early once the plan's `early_stopping_patience` checks
    in a row have not lowered the run's lowest select loss, significant or not, so a
    run whose gain is still rising goes on. A create may leave any setting out; the run
    says each.
    """

    holdout: float
    context_rows: int | None
    contexts: int
    early_stopping_threshold: EarlyStoppingThreshold
    report_share: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("holdout", "holdout", True, _number),
        Field("context_rows", "context_rows", True, _union(_integer, _null)),
        Field("contexts", "contexts", True, _integer),
        Field(
            "early_stopping_threshold",
            "early_stopping_threshold",
            True,
            _record(EarlyStoppingThreshold),
        ),
        Field("report_share", "report_share", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class RunPlan(Payload):
    """The RunPlan schema.

    How a run trains: its `steps`, how it samples episodes, the optimizer's settings
    and its checkpoint settings, `{"steps": 200, "learning_rate": 5e-5,
    "episode_sampling": {"rows": 1024}}`. A setting the trainer does not read is
    refused with 422.
    """

    steps: int
    episode_sampling: EpisodeSampling | None = None
    learning_rate: float | None = None
    optimizer: Literal["adamw"] | None = None
    schedule: Literal["cosine"] | None = None
    warmup_steps: int | None = None
    weight_decay: float | None = None
    beta2: float | None = None
    grad_clip: float | None = None
    episodes_per_step: int | None = None
    eval_steps: int | None = None
    early_stopping_patience: int | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    save_only_model: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("steps", "steps", True, _integer),
        Field("episode_sampling", "episode_sampling", False, _record(EpisodeSampling)),
        Field("learning_rate", "learning_rate", False, _number),
        Field("optimizer", "optimizer", False, _literal("adamw")),
        Field("schedule", "schedule", False, _literal("cosine")),
        Field("warmup_steps", "warmup_steps", False, _integer),
        Field("weight_decay", "weight_decay", False, _number),
        Field("beta2", "beta2", False, _number),
        Field("grad_clip", "grad_clip", False, _number),
        Field("episodes_per_step", "episodes_per_step", False, _integer),
        Field("eval_steps", "eval_steps", False, _integer),
        Field("early_stopping_patience", "early_stopping_patience", False, _integer),
        Field("save_steps", "save_steps", False, _integer),
        Field("save_total_limit", "save_total_limit", False, _integer),
        Field("save_only_model", "save_only_model", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class RunCurriculumPlan(Payload):
    """The RunCurriculumPlan schema.

    The settings of a curriculum run that span the whole run: the optimizer's, the
    learning-rate schedule's, the checks' and the checkpoints', `{"learning_rate":
    1e-4, "warmup_steps": 20, "eval_steps": 25}`. One optimizer and one schedule run
    over the phases' steps together, and the trained weights and the optimizer's
    moments carry from one phase into the next. Checks, `early_stopping_patience` and
    the kept step belong to the last phase alone, from its first step; checkpoints are
    saved in every phase. `steps`, `episode_sampling` and `episodes_per_step` are set
    per phase (`RunPhasePlan`) and are refused here with 422.
    """

    learning_rate: float | None = None
    optimizer: Literal["adamw"] | None = None
    schedule: Literal["cosine"] | None = None
    warmup_steps: int | None = None
    weight_decay: float | None = None
    beta2: float | None = None
    grad_clip: float | None = None
    eval_steps: int | None = None
    early_stopping_patience: int | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    save_only_model: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("learning_rate", "learning_rate", False, _number),
        Field("optimizer", "optimizer", False, _literal("adamw")),
        Field("schedule", "schedule", False, _literal("cosine")),
        Field("warmup_steps", "warmup_steps", False, _integer),
        Field("weight_decay", "weight_decay", False, _number),
        Field("beta2", "beta2", False, _number),
        Field("grad_clip", "grad_clip", False, _number),
        Field("eval_steps", "eval_steps", False, _integer),
        Field("early_stopping_patience", "early_stopping_patience", False, _integer),
        Field("save_steps", "save_steps", False, _integer),
        Field("save_total_limit", "save_total_limit", False, _integer),
        Field("save_only_model", "save_only_model", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class RunUsage(Payload):
    """The RunUsage schema."""

    gpu_seconds: float
    charged_gpu_seconds: float
    voided_gpu_seconds: float
    credits: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("gpu_seconds", "gpu_seconds", True, _number),
        Field("charged_gpu_seconds", "charged_gpu_seconds", True, _number),
        Field("voided_gpu_seconds", "voided_gpu_seconds", True, _number),
        Field("credits", "credits", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class RunReportMetrics(Payload):
    """The RunReportMetrics schema.

    the task metrics of the base's and the kept step's predictions on the report rows
    """

    base: dict[str, float]
    kept: dict[str, float]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("base", "base", True, _mapping(_number)),
        Field("kept", "kept", True, _mapping(_number)),
    )


@dataclass(frozen=True, kw_only=True)
class RunTableReport(Payload):
    """The RunTableReport schema.

    Report rows, scored once for the base and once for the kept step against the check
    contexts: their mean losses, the gain (the base's loss less the kept step's, above
    0 when the kept step is better) with its 95% interval from the per-row gains, and
    task metrics for each: log loss, accuracy and, for two classes, ROC AUC; or CRPS,
    RMSE, MAE and R².
    """

    rows: int
    base: float
    kept: float
    gain: float
    ci95: tuple[float, ...]
    metrics: RunReportMetrics

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("rows", "rows", True, _integer),
        Field("base", "base", True, _number),
        Field("kept", "kept", True, _number),
        Field("gain", "gain", True, _number),
        Field("ci95", "ci95", True, _items(_number)),
        Field("metrics", "metrics", True, _record(RunReportMetrics)),
    )


@dataclass(frozen=True, kw_only=True)
class RunReport(Payload):
    """The RunReport schema.

    the report rows' scores under `data`, the report rows of every snapshot the run
    holds rows out from (a curriculum run's last phase's) scored together
    """

    data: RunTableReport

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("data", "data", True, _record(RunTableReport)),
    )


@dataclass(frozen=True, kw_only=True)
class RunPublished(Payload):
    """The RunPublished schema.

    What a run published: the version, its weights' digest, the step the weights were
    kept at and how they scored on the rows the run held out, against the base on the
    same rows. The same as the lineage's `published` note. A version kept at step 0
    predicts exactly as its base does: no check during training beat the base.
    """

    model: ModelId
    digest: Digest
    kept_step: KeptStep
    validation_loss: float | None
    base_validation_loss: float | None
    diverged_at_step: int | None = None
    check_context_rows: int | None = None
    check_contexts: int | None = None
    report: RunReport | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelId),
        Field("digest", "digest", True, _decode_Digest),
        Field("kept_step", "kept_step", True, _decode_KeptStep),
        Field("validation_loss", "validation_loss", True, _union(_number, _null)),
        Field(
            "base_validation_loss",
            "base_validation_loss",
            True,
            _union(_number, _null),
        ),
        Field("diverged_at_step", "diverged_at_step", False, _integer),
        Field("check_context_rows", "check_context_rows", False, _integer),
        Field("check_contexts", "check_contexts", False, _integer),
        Field("report", "report", False, _union(_record(RunReport), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Run(Payload):
    """The Run schema."""

    origin: Origin
    url: str
    model: RunModel
    name: Name
    version: int
    state: RunState
    reason: str | None
    attempt: int
    base: ModelId
    data: tuple[RunData, ...]
    data_digest: Digest | None
    progress: RunProgress | None
    method: RunMethod
    evaluation: RunEvaluation
    budget: dict[str, Any] | None
    seed: int
    tags: tuple[RunTag, ...]
    plan: RunPlan | RunCurriculumPlan
    created: float
    updated: float
    published: RunPublished | None
    waiting: RunWaiting | None = None
    phases: tuple[RunPinnedPhase, ...] | None = None
    architecture: tuple[RunResolvedArchitectureChange, ...] | None = None
    warnings: tuple[RunWarning, ...] | None = None
    usage: RunUsage | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("origin", "origin", True, _record(Origin)),
        Field("url", "url", True, _text),
        Field("model", "model", True, _decode_RunModel),
        Field("name", "name", True, _decode_Name),
        Field("version", "version", True, _integer),
        Field("state", "state", True, _decode_RunState),
        Field("reason", "reason", True, _union(_text, _null)),
        Field("attempt", "attempt", True, _integer),
        Field("base", "base", True, _decode_ModelId),
        Field("data", "data", True, _items(_record(RunData))),
        Field("data_digest", "data_digest", True, _union(_decode_Digest, _null)),
        Field("waiting", "waiting", False, _record(RunWaiting)),
        Field("phases", "phases", False, _items(_record(RunPinnedPhase))),
        Field(
            "architecture",
            "architecture",
            False,
            _items(_record(RunResolvedArchitectureChange)),
        ),
        Field("warnings", "warnings", False, _items(_record(RunWarning))),
        Field("progress", "progress", True, _union(_record(RunProgress), _null)),
        Field("method", "method", True, _record(RunMethod)),
        Field("evaluation", "evaluation", True, _record(RunEvaluation)),
        Field("budget", "budget", True, _union(_mapping(_any), _null)),
        Field("seed", "seed", True, _integer),
        Field("tags", "tags", True, _items(_decode_RunTag)),
        Field(
            "plan",
            "plan",
            True,
            _union(_record(RunPlan), _record(RunCurriculumPlan)),
        ),
        Field("created", "created", True, _number),
        Field("updated", "updated", True, _number),
        Field("usage", "usage", False, _union(_record(RunUsage), _null)),
        Field("published", "published", True, _union(_record(RunPublished), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class RunPhase(Payload):
    """The RunPhase schema.

    One phase of a curriculum run, as a create sends it: its `name`, unique in the run;
    the ready snapshots it trains on, unique in the phase, each its own table with its
    own encoding and holdout; and its `plan` (`RunPhasePlan`), its steps and how it
    samples episodes.
    """

    name: Name
    data: tuple[RunDataItem, ...]
    plan: RunPhasePlan

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("data", "data", True, _items(_decode_RunDataItem)),
        Field("plan", "plan", True, _record(RunPhasePlan)),
    )


@dataclass(frozen=True, kw_only=True)
class RunEvaluationRequest(Payload):
    """The RunEvaluationRequest schema.

    a create's evaluation settings; each left out takes its default (`RunEvaluation`)
    """

    holdout: float | None = None
    context_rows: int | None = None
    contexts: int | None = None
    early_stopping_threshold: EarlyStoppingThreshold | None = None
    report_share: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("holdout", "holdout", False, _number),
        Field("context_rows", "context_rows", False, _integer),
        Field("contexts", "contexts", False, _integer),
        Field(
            "early_stopping_threshold",
            "early_stopping_threshold",
            False,
            _record(EarlyStoppingThreshold),
        ),
        Field("report_share", "report_share", False, _number),
    )


@dataclass(frozen=True, kw_only=True)
class RunReply(Payload):
    """The RunReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Run

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Run)),
    )


@dataclass(frozen=True, kw_only=True)
class RunList(Payload):
    """The RunList schema."""

    runs: tuple[Run, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("runs", "runs", True, _items(_record(Run))),
    )


@dataclass(frozen=True, kw_only=True)
class RunListReply(Payload):
    """The RunListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: RunList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(RunList)),
    )


@dataclass(frozen=True, kw_only=True)
class RunCreate(Payload):
    """A run's settings.

    It trains on `data` with a `plan` (`RunPlan`), or on a curriculum of `phases`, with
    an optional `plan` of the settings that span the whole run (`RunCurriculumPlan`); a
    create that sends both `data` and `phases`, or neither, is refused with 422.
    `method`, `evaluation`, `budget`, `seed` and `tags` apply to the whole run either
    way. Every snapshot is its own table, never pooled with another, and has the base's
    task. With an input adapter every snapshot has exactly the last phase's columns in
    the same order (with `data`, every snapshot has the same columns).
    """

    name: Name
    base: ModelField
    method: RunMethod
    data: tuple[RunDataItem, ...] | None = None
    plan: RunPlan | RunCurriculumPlan | None = None
    phases: tuple[RunPhase, ...] | None = None
    architecture: tuple[RunArchitectureChange, ...] | None = None
    evaluation: RunEvaluationRequest | None = None
    budget: dict[str, Any] | None = None
    seed: int | None = None
    tags: tuple[RunTag, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("base", "base", True, _decode_ModelField),
        Field("data", "data", False, _items(_decode_RunDataItem)),
        Field("method", "method", True, _record(RunMethod)),
        Field(
            "plan",
            "plan",
            False,
            _union(_record(RunPlan), _record(RunCurriculumPlan)),
        ),
        Field("phases", "phases", False, _items(_record(RunPhase))),
        Field(
            "architecture",
            "architecture",
            False,
            _items(_record(RunArchitectureChange)),
        ),
        Field("evaluation", "evaluation", False, _record(RunEvaluationRequest)),
        Field("budget", "budget", False, _mapping(_any)),
        Field("seed", "seed", False, _integer),
        Field("tags", "tags", False, _items(_decode_RunTag)),
    )


@dataclass(frozen=True, kw_only=True)
class RunCreateRequest(Payload):
    """The RunCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: RunCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(RunCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class RunCancel(Payload):
    """How a run is cancelled; an empty payload cancels without saving."""

    save: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("save", "save", False, _boolean),)
    DEFAULTS: ClassVar[dict[str, Any]] = {"save": False}


@dataclass(frozen=True, kw_only=True)
class RunCancelRequest(Payload):
    """The RunCancelRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: RunCancel

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(RunCancel)),
    )


@dataclass(frozen=True, kw_only=True)
class RunEvent(Payload):
    """The RunEvent schema."""

    at: float
    kind: str
    body: dict[str, Any]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("at", "at", True, _number),
        Field("kind", "kind", True, _text),
        Field("body", "body", True, _mapping(_any)),
    )


@dataclass(frozen=True, kw_only=True)
class RunLineage(Payload):
    """The RunLineage schema."""

    model: RunModel
    events: tuple[RunEvent, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_RunModel),
        Field("events", "events", True, _items(_record(RunEvent))),
    )


@dataclass(frozen=True, kw_only=True)
class RunLineageReply(Payload):
    """The RunLineageReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: RunLineage

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(RunLineage)),
    )


SnapshotState = Literal["open", "pending", "ready", "rejected", "expired", "deleted"]
_decode_SnapshotState: Decoder = _literal(
    "open",
    "pending",
    "ready",
    "rejected",
    "expired",
    "deleted",
)


@dataclass(frozen=True, kw_only=True)
class SnapshotColumn(Payload):
    """The SnapshotColumn schema."""

    name: str
    type: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("type", "type", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class UploadedOrigin(Payload):
    """A table someone uploaded or streamed; an upload's origin when it sends none."""

    kind: Literal["uploaded"] = "uploaded"

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("uploaded")),
    )


@dataclass(frozen=True, kw_only=True)
class ContextOrigin(Payload):
    """A table made from a live context's rows, as they stood at `as_of`."""

    kind: Literal["context"] = "context"
    context: Name
    as_of: str | float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("context")),
        Field("context", "context", True, _decode_Name),
        Field("as_of", "as_of", True, _union(_text, _number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticOrigin(Payload):
    """The SyntheticOrigin schema.

    A table a synthetic batch generated: the batch, the seed the table was drawn from,
    the generator's version, and every option as this table used it, with each range
    replaced by its draw and `schema_from` expanded to the schema it copied, with the
    source snapshot's name and digest. The service sets it; an upload that sends it is
    refused with 422.
    """

    kind: Literal["synthetic"] = "synthetic"
    batch: SyntheticBatchName
    seed: int
    generator: str
    options: dict[str, Any]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("synthetic")),
        Field("batch", "batch", True, _decode_SyntheticBatchName),
        Field("seed", "seed", True, _integer),
        Field("generator", "generator", True, _text),
        Field("options", "options", True, _mapping(_any)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotMetadata(Payload):
    """The SnapshotMetadata schema.

    A table's training metadata: its `target` and `task`, and optionally its `time` and
    `group` columns, its column types and its origin. A run refuses a snapshot without
    a target and task. The columns set how a run holds rows out of the snapshot: its
    latest rows with `time`, otherwise whole groups with `group`, otherwise a share
    stratified by class or target.
    """

    target: str
    task: Literal["classification", "regression"]
    column_types: dict[str, Literal["numerical", "categorical"]] | None = None
    time: str | None = None
    group: str | None = None
    origin: UploadedOrigin | ContextOrigin | SyntheticOrigin | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("target", "target", True, _text),
        Field("task", "task", True, _literal("classification", "regression")),
        Field(
            "column_types",
            "column_types",
            False,
            _mapping(_literal("numerical", "categorical")),
        ),
        Field("time", "time", False, _text),
        Field("group", "group", False, _text),
        Field(
            "origin",
            "origin",
            False,
            _union(
                _record(UploadedOrigin),
                _record(ContextOrigin),
                _record(SyntheticOrigin),
            ),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Snapshot(Payload):
    """A named, immutable table.

    `digest`, `rows` and `columns` are set once it is `ready`; `origin` says where it
    was uploaded from. A create that opens an upload also carries where its bytes go:
    `url`, `method` and `chunk_bytes`.
    """

    name: Name
    state: SnapshotState
    digest: Digest | None
    bytes: int
    rows: int | None
    columns: tuple[SnapshotColumn, ...] | None
    metadata: SnapshotMetadata | None
    tags: tuple[RunTag, ...]
    error: str | None
    expires: float | None
    created: float
    origin: Origin | None = None
    url: str | None = None
    method: Literal["PUT"] | None = None
    chunk_bytes: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("origin", "origin", False, _record(Origin)),
        Field("name", "name", True, _decode_Name),
        Field("state", "state", True, _decode_SnapshotState),
        Field("digest", "digest", True, _union(_decode_Digest, _null)),
        Field("bytes", "bytes", True, _integer),
        Field("rows", "rows", True, _union(_integer, _null)),
        Field(
            "columns",
            "columns",
            True,
            _union(_items(_record(SnapshotColumn)), _null),
        ),
        Field("metadata", "metadata", True, _union(_record(SnapshotMetadata), _null)),
        Field("tags", "tags", True, _items(_decode_RunTag)),
        Field("error", "error", True, _union(_text, _null)),
        Field("expires", "expires", True, _union(_number, _null)),
        Field("created", "created", True, _number),
        Field("url", "url", False, _union(_text, _null)),
        Field("method", "method", False, _literal("PUT")),
        Field("chunk_bytes", "chunk_bytes", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotReply(Payload):
    """The SnapshotReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Snapshot

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Snapshot)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotList(Payload):
    """The SnapshotList schema."""

    snapshots: tuple[Snapshot, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("snapshots", "snapshots", True, _items(_record(Snapshot))),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotListReply(Payload):
    """The SnapshotListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SnapshotList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SnapshotList)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotCreate(Payload):
    """The SnapshotCreate schema."""

    name: Name
    bytes: int
    format: Literal["parquet"] = "parquet"
    metadata: SnapshotMetadata
    tags: tuple[RunTag, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("bytes", "bytes", True, _integer),
        Field("format", "format", True, _literal("parquet")),
        Field("metadata", "metadata", True, _record(SnapshotMetadata)),
        Field("tags", "tags", False, _items(_decode_RunTag)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotCreateRequest(Payload):
    """The SnapshotCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SnapshotCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SnapshotCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotUpdate(Payload):
    """The SnapshotUpdate schema.

    A snapshot's new tags, which replace the tags a user gave it; an empty list clears
    them. The reserved tags (`uploaded`, `synthetic` and any starting `synthetic-`)
    stay as they are and are refused here with 422. Nothing else about a snapshot
    changes, so any other field is refused with 422.
    """

    tags: tuple[RunTag, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("tags", "tags", True, _items(_decode_RunTag)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotUpdateRequest(Payload):
    """The SnapshotUpdateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SnapshotUpdate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SnapshotUpdate)),
    )


SyntheticBatchState = Literal["pending", "done"]
_decode_SyntheticBatchState: Decoder = _literal("pending", "done")


SyntheticFunction = Literal[
    "linear",
    "mlp",
    "tree",
    "gp",
    "discretize",
    "quadratic",
    "cluster",
]
_decode_SyntheticFunction: Decoder = _literal(
    "linear",
    "mlp",
    "tree",
    "gp",
    "discretize",
    "quadratic",
    "cluster",
)


SyntheticMissingPattern = Literal[
    "random",
    "rows",
    "blocks",
    "by_target",
    "censored",
    "latent",
    "near_empty",
]
_decode_SyntheticMissingPattern: Decoder = _literal(
    "random",
    "rows",
    "blocks",
    "by_target",
    "censored",
    "latent",
    "near_empty",
)


@dataclass(frozen=True, kw_only=True)
class SyntheticRowsRange(Payload):
    """A range of rows per table, drawn log-uniformly."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticColumnsRange(Payload):
    """A range of feature columns per table."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticClassesRange(Payload):
    """A range of classes per table."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticCategoriesRange(Payload):
    """The SyntheticCategoriesRange schema.

    a range for the most categories a categorical column may have, drawn log-uniformly
    """

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticGraphNodesRange(Payload):
    """A range of hidden nodes in a table's causal graph, drawn log-uniformly."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticGroupsRange(Payload):
    """A range of groups of correlated columns per table."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticLevelsRange(Payload):
    """A range of levels a coarsened column is cut into."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticHighCardinalityRange(Payload):
    """A range of high-cardinality categorical columns per table."""

    min: int
    max: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _integer),
        Field("max", "max", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticShareRange(Payload):
    """A range of shares from 0 to 1."""

    min: float
    max: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _number),
        Field("max", "max", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticShare90Range(Payload):
    """A range of shares from 0 to 0.9."""

    min: float
    max: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _number),
        Field("max", "max", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticShare50Range(Payload):
    """A range of shares from 0 to 0.5."""

    min: float
    max: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("min", "min", True, _number),
        Field("max", "max", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticCategorical(Payload):
    """The SyntheticCategorical schema.

    how many of a table's feature columns are categorical, and how many categories each
    may have
    """

    share: float | SyntheticShareRange | None = None
    max_categories: int | SyntheticCategoriesRange | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("share", "share", False, _union(_number, _record(SyntheticShareRange))),
        Field(
            "max_categories",
            "max_categories",
            False,
            _union(_integer, _record(SyntheticCategoriesRange)),
        ),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "share": {"min": 0, "max": 0.5},
        "max_categories": {"min": 2, "max": 50},
    }


@dataclass(frozen=True, kw_only=True)
class SyntheticCorrelated(Payload):
    """The SyntheticCorrelated schema.

    groups of feature columns made to move together through a shared hidden factor
    """

    share: float | SyntheticShareRange | None = None
    groups: int | SyntheticGroupsRange | None = None
    strength: float | SyntheticShareRange | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("share", "share", False, _union(_number, _record(SyntheticShareRange))),
        Field(
            "groups",
            "groups",
            False,
            _union(_integer, _record(SyntheticGroupsRange)),
        ),
        Field(
            "strength",
            "strength",
            False,
            _union(_number, _record(SyntheticShareRange)),
        ),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "share": 0,
        "groups": 1,
        "strength": {"min": 0.2, "max": 0.7},
    }


@dataclass(frozen=True, kw_only=True)
class SyntheticCoarsened(Payload):
    """The SyntheticCoarsened schema.

    numeric columns cut into a few ordered levels, as a rounded or bucketed measurement
    is
    """

    share: float | SyntheticShareRange | None = None
    levels: int | SyntheticLevelsRange | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("share", "share", False, _union(_number, _record(SyntheticShareRange))),
        Field(
            "levels",
            "levels",
            False,
            _union(_integer, _record(SyntheticLevelsRange)),
        ),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"share": 0, "levels": {"min": 2, "max": 8}}


@dataclass(frozen=True, kw_only=True)
class SyntheticMissing(Payload):
    """Missing cells, null in categorical columns and NaN in numeric ones."""

    rate: float | SyntheticShare90Range | None = None
    patterns: tuple[SyntheticMissingPattern, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("rate", "rate", False, _union(_number, _record(SyntheticShare90Range))),
        Field("patterns", "patterns", False, _items(_decode_SyntheticMissingPattern)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"rate": 0, "patterns": ["random"]}


@dataclass(frozen=True, kw_only=True)
class SyntheticTarget(Payload):
    """How noisy, imbalanced or heavy-tailed the target `y` is."""

    noise: float | SyntheticShareRange | None = None
    label_noise: float | SyntheticShare50Range | None = None
    imbalance: float | SyntheticShareRange | None = None
    heavy_tail: float | SyntheticShareRange | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("noise", "noise", False, _union(_number, _record(SyntheticShareRange))),
        Field(
            "label_noise",
            "label_noise",
            False,
            _union(_number, _record(SyntheticShare50Range)),
        ),
        Field(
            "imbalance",
            "imbalance",
            False,
            _union(_number, _record(SyntheticShareRange)),
        ),
        Field(
            "heavy_tail",
            "heavy_tail",
            False,
            _union(_number, _record(SyntheticShareRange)),
        ),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "noise": {"min": 0.01, "max": 0.1},
        "label_noise": 0,
        "imbalance": 0,
        "heavy_tail": 0,
    }


@dataclass(frozen=True, kw_only=True)
class SyntheticBatchCreate(Payload):
    """The SyntheticBatchCreate schema.

    A synthetic batch: its name, how many tables, the seed, and the options every table
    is drawn with. Each numeric option is an exact value or a range `{"min": a, "max":
    b}`, a <= b, both within the option's bounds, drawn once per table; an option left
    out takes its default. `task` is required without `schema_from`.
    """

    name: SyntheticBatchName
    count: int | None = None
    seed: int | None = None
    schema_from: Name | None = None
    task: Literal["classification", "regression"] | None = None
    rows: int | SyntheticRowsRange | None = None
    columns: int | SyntheticColumnsRange | None = None
    classes: int | SyntheticClassesRange | None = None
    categorical: SyntheticCategorical | None = None
    functions: tuple[SyntheticFunction, ...] | None = None
    graph_nodes: int | SyntheticGraphNodesRange | None = None
    irrelevant_columns: float | SyntheticShare90Range | None = None
    correlated_columns: SyntheticCorrelated | None = None
    coarsened_columns: SyntheticCoarsened | None = None
    high_cardinality_columns: int | SyntheticHighCardinalityRange | None = None
    missing: SyntheticMissing | None = None
    target: SyntheticTarget | None = None
    reject: tuple[Literal["trivial", "unlearnable"], ...] | None = None
    tags: tuple[RunTag, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_SyntheticBatchName),
        Field("count", "count", False, _integer),
        Field("seed", "seed", False, _integer),
        Field("schema_from", "schema_from", False, _decode_Name),
        Field("task", "task", False, _literal("classification", "regression")),
        Field("rows", "rows", False, _union(_integer, _record(SyntheticRowsRange))),
        Field(
            "columns",
            "columns",
            False,
            _union(_integer, _record(SyntheticColumnsRange)),
        ),
        Field(
            "classes",
            "classes",
            False,
            _union(_integer, _record(SyntheticClassesRange)),
        ),
        Field("categorical", "categorical", False, _record(SyntheticCategorical)),
        Field("functions", "functions", False, _items(_decode_SyntheticFunction)),
        Field(
            "graph_nodes",
            "graph_nodes",
            False,
            _union(_integer, _record(SyntheticGraphNodesRange)),
        ),
        Field(
            "irrelevant_columns",
            "irrelevant_columns",
            False,
            _union(_number, _record(SyntheticShare90Range)),
        ),
        Field(
            "correlated_columns",
            "correlated_columns",
            False,
            _record(SyntheticCorrelated),
        ),
        Field(
            "coarsened_columns",
            "coarsened_columns",
            False,
            _record(SyntheticCoarsened),
        ),
        Field(
            "high_cardinality_columns",
            "high_cardinality_columns",
            False,
            _union(_integer, _record(SyntheticHighCardinalityRange)),
        ),
        Field("missing", "missing", False, _record(SyntheticMissing)),
        Field("target", "target", False, _record(SyntheticTarget)),
        Field("reject", "reject", False, _items(_literal("trivial", "unlearnable"))),
        Field("tags", "tags", False, _items(_decode_RunTag)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "count": 1,
        "rows": 4096,
        "columns": {"min": 2, "max": 32},
        "classes": {"min": 2, "max": 10},
        "functions": [
            "linear",
            "mlp",
            "tree",
            "gp",
            "discretize",
            "quadratic",
            "cluster",
        ],
        "graph_nodes": {"min": 2, "max": 32},
        "irrelevant_columns": 0,
        "high_cardinality_columns": 0,
        "reject": [],
    }


@dataclass(frozen=True, kw_only=True)
class SyntheticBatchCreateRequest(Payload):
    """The SyntheticBatchCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SyntheticBatchCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SyntheticBatchCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticBatchCreated(Payload):
    """A batch as its create answers it, every table pending."""

    name: SyntheticBatchName
    tag: RunTag
    count: int
    snapshots: tuple[Name, ...]
    state: SyntheticBatchState

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_SyntheticBatchName),
        Field("tag", "tag", True, _decode_RunTag),
        Field("count", "count", True, _integer),
        Field("snapshots", "snapshots", True, _items(_decode_Name)),
        Field("state", "state", True, _decode_SyntheticBatchState),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticTable(Payload):
    """The SyntheticTable schema.

    one table of a batch, by its snapshot name, with its state and, once rejected, why
    """

    name: Name
    state: SnapshotState
    error: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("state", "state", True, _decode_SnapshotState),
        Field("error", "error", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticCounts(Payload):
    """How many of a batch's tables are in each state."""

    pending: int
    ready: int
    rejected: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("pending", "pending", True, _integer),
        Field("ready", "ready", True, _integer),
        Field("rejected", "rejected", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticBatch(Payload):
    """The SyntheticBatch schema.

    a batch as `snapshots.synthetic` reads it: its create's answer, and the seed,
    options, tables and counts. Its fields are listed whole, as `SyntheticBatchCreated`
    allows no other fields.
    """

    name: SyntheticBatchName
    tag: RunTag
    count: int
    snapshots: tuple[Name, ...]
    state: SyntheticBatchState
    seed: int
    options: SyntheticBatchCreate
    tables: tuple[SyntheticTable, ...]
    counts: SyntheticCounts

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_SyntheticBatchName),
        Field("tag", "tag", True, _decode_RunTag),
        Field("count", "count", True, _integer),
        Field("snapshots", "snapshots", True, _items(_decode_Name)),
        Field("state", "state", True, _decode_SyntheticBatchState),
        Field("seed", "seed", True, _integer),
        Field("options", "options", True, _record(SyntheticBatchCreate)),
        Field("tables", "tables", True, _items(_record(SyntheticTable))),
        Field("counts", "counts", True, _record(SyntheticCounts)),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticBatchCreatedReply(Payload):
    """The SyntheticBatchCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SyntheticBatchCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SyntheticBatchCreated)),
    )


@dataclass(frozen=True, kw_only=True)
class SyntheticBatchReply(Payload):
    """The SyntheticBatchReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SyntheticBatch

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SyntheticBatch)),
    )


State = Literal["pending", "building", "ready", "failed", "stopped"]
_decode_State: Decoder = _literal("pending", "building", "ready", "failed", "stopped")


@dataclass(frozen=True, kw_only=True)
class Spec(Payload):
    """A predictor's spec: `gf.Predictor`'s fields but its name, as JSON."""

    context: str
    model: ModelField
    target: str
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    series: tuple[str, ...] | None = None
    frequency: str | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None
    selection: Selection | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "context",
            "context",
            True,
            _pattern("^[a-z][a-z0-9-]{0,62}(@[1-9][0-9]{0,8})?$"),
        ),
        Field("model", "model", True, _decode_ModelField),
        Field("target", "target", True, _text),
        Field("task", "task", False, _decode_Task),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("inputs", "inputs", False, _items(_text)),
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("series", "series", False, _items(_text)),
        Field("frequency", "frequency", False, _union(_text, _number)),
        Field("horizon", "horizon", False, _integer),
        Field("quantiles", "quantiles", False, _items(_number)),
        Field("past", "past", False, _items(_text)),
        Field("future", "future", False, _items(_text)),
        Field("selection", "selection", False, _record(Selection)),
    )


@dataclass(frozen=True, kw_only=True)
class Resolved(Payload):
    """The Resolved schema.

    what a spec resolved to against its context version's rows, by the engine's
    `resolve`: the full model id and digest, the task, inputs and their types. A deploy
    compares these, so a short name and its full id are one predictor
    """

    model: ModelField
    short: str | None
    model_digest: Digest
    base_digest: Digest | None
    task: Task
    context: Name
    context_version: int | None
    context_digest: Digest | None
    inputs: tuple[str, ...]
    types: dict[str, ColumnType]
    context_rows: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelField),
        Field("short", "short", True, _union(_text, _null)),
        Field("model_digest", "model_digest", True, _decode_Digest),
        Field("base_digest", "base_digest", True, _union(_decode_Digest, _null)),
        Field("task", "task", True, _decode_Task),
        Field("context", "context", True, _decode_Name),
        Field("context_version", "context_version", True, _union(_integer, _null)),
        Field("context_digest", "context_digest", True, _union(_decode_Digest, _null)),
        Field("context_rows", "context_rows", False, _union(_integer, _null)),
        Field("inputs", "inputs", True, _items(_text)),
        Field("types", "types", True, _mapping(_decode_ColumnType)),
    )


@dataclass(frozen=True, kw_only=True)
class Revision(Payload):
    """The Revision schema.

    an immutable spec with what it resolved to; the plane keeps its lock beside it
    """

    predictor: Name
    number: int
    status: Literal["pending", "building", "ready", "failed", "stopped"]
    lock_digest: Digest
    spec: Spec
    resolved: Resolved
    origin: Origin
    created: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictor", "predictor", True, _decode_Name),
        Field("number", "number", True, _integer),
        Field(
            "status",
            "status",
            True,
            _literal("pending", "building", "ready", "failed", "stopped"),
        ),
        Field("lock_digest", "lock_digest", True, _decode_Digest),
        Field("spec", "spec", True, _record(Spec)),
        Field("resolved", "resolved", True, _record(Resolved)),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionReply(Payload):
    """The RevisionReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Revision

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Revision)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionList(Payload):
    """The RevisionList schema."""

    revisions: tuple[Revision, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revisions", "revisions", True, _items(_record(Revision))),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionListReply(Payload):
    """The RevisionListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: RevisionList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(RevisionList)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentFailure(Payload):
    """The EnvironmentFailure schema.

    why an environment's revision failed to start: the failure in words, the request
    whose deploy or promote started it, and the revision the environment still serves
    (null when it served none)
    """

    message: str
    request_id: str | None
    serving_revision: int | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("message", "message", True, _text),
        Field("request_id", "request_id", True, _union(_text, _null)),
        Field("serving_revision", "serving_revision", True, _union(_integer, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Environment(Payload):
    """A pointer to a revision; `generation` guards promote and rollback."""

    name: Name
    revision: int | None
    state: Literal["starting", "ready", "failed", "stopped"]
    generation: int
    url: str
    calls_today: int | None = None
    p50_seconds: float | None = None
    updated: str | None = None
    error: EnvironmentFailure | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("revision", "revision", True, _union(_integer, _null)),
        Field(
            "state",
            "state",
            True,
            _literal("starting", "ready", "failed", "stopped"),
        ),
        Field("generation", "generation", True, _integer),
        Field("calls_today", "calls_today", False, _integer),
        Field("p50_seconds", "p50_seconds", False, _union(_number, _null)),
        Field("updated", "updated", False, _text),
        Field("url", "url", True, _text),
        Field("error", "error", False, _union(_record(EnvironmentFailure), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentReply(Payload):
    """The EnvironmentReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Environment

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Environment)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentList(Payload):
    """The EnvironmentList schema."""

    environments: tuple[Environment, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("environments", "environments", True, _items(_record(Environment))),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentListReply(Payload):
    """The EnvironmentListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: EnvironmentList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(EnvironmentList)),
    )


@dataclass(frozen=True, kw_only=True)
class Promote(Payload):
    """The Promote schema.

    the revision to point at, or `from_env`, the environment whose revision to take;
    `expected_generation` refuses a change past a stale read
    """

    revision: int | None = None
    from_env: Name | None = None
    expected_generation: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revision", "revision", False, _integer),
        Field("from_env", "from_env", False, _decode_Name),
        Field("expected_generation", "expected_generation", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class PromoteRequest(Payload):
    """The PromoteRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Promote

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Promote)),
    )


@dataclass(frozen=True, kw_only=True)
class ServingContext(Payload):
    """The ServingContext schema.

    the context an environment serves: `source` is `snapshot`, `request` or
    `live/<context>`; the version fields and `freshness_lag_seconds` are set for live
    contexts
    """

    source: str
    version: int | str | None
    rows: int
    as_of: AsOf
    accepted_version: int | str | None
    serving_version: int | str | None
    freshness_lag_seconds: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _text),
        Field("version", "version", True, _union(_integer, _text, _null)),
        Field("rows", "rows", True, _integer),
        Field("as_of", "as_of", True, _decode_AsOf),
        Field(
            "accepted_version",
            "accepted_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "serving_version",
            "serving_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "freshness_lag_seconds",
            "freshness_lag_seconds",
            True,
            _union(_number, _null),
        ),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class Metrics(Payload):
    """The Metrics schema.

    serving facts: `context` is the context the environment is serving now,
    `accepted_version`/`serving_version` compare the newest accepted write of a live
    context with the one being served, `fresh` says whether they agree (null without a
    context)
    """

    status: State
    context: ServingContext | None
    accepted_version: int | str | None
    serving_version: int | str | None
    fresh: bool | None
    requests: int
    errors: int
    mean_latency_seconds: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _decode_State),
        Field("context", "context", True, _union(_record(ServingContext), _null)),
        Field(
            "accepted_version",
            "accepted_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "serving_version",
            "serving_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field("fresh", "fresh", True, _union(_boolean, _null)),
        Field("requests", "requests", True, _integer),
        Field("errors", "errors", True, _integer),
        Field(
            "mean_latency_seconds",
            "mean_latency_seconds",
            True,
            _union(_number, _null),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class MetricsReply(Payload):
    """The MetricsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Metrics

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Metrics)),
    )


LogCursor = str
_decode_LogCursor: Decoder = _pattern("^[0-9]{1,18}$")


LogLevel = Literal["info", "warn", "error"]
_decode_LogLevel: Decoder = _literal("info", "warn", "error")


@dataclass(frozen=True, kw_only=True)
class LogEntry(Payload):
    """One thing that happened.

    `id` is the entry's cursor, which `after` and `before` take.
    """

    id: LogCursor
    at: float
    level: LogLevel
    source: str
    code: str
    message: str
    environment: str | None
    revision: int | None
    request_id: str | None
    fields: dict[str, str | float | bool]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_LogCursor),
        Field("at", "at", True, _number),
        Field("level", "level", True, _decode_LogLevel),
        Field("source", "source", True, _text),
        Field("code", "code", True, _text),
        Field("message", "message", True, _text),
        Field("environment", "environment", True, _union(_text, _null)),
        Field("revision", "revision", True, _union(_integer, _null)),
        Field("request_id", "request_id", True, _union(_text, _null)),
        Field("fields", "fields", True, _mapping(_union(_text, _number, _boolean))),
    )


@dataclass(frozen=True, kw_only=True)
class Logs(Payload):
    """A page of a log, oldest first.

    `next` is the cursor to pass as `after` for the entries that follow the page.
    """

    logs: tuple[LogEntry, ...]
    next: LogCursor
    previous: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("logs", "logs", True, _items(_record(LogEntry))),
        Field("next", "next", True, _decode_LogCursor),
        Field("previous", "previous", True, _union(_pattern("^[0-9]{1,18}$"), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class LogsReply(Payload):
    """The LogsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Logs

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Logs)),
    )


@dataclass(frozen=True, kw_only=True)
class LogBucket(Payload):
    """The LogBucket schema."""

    at: float
    info: int
    warn: int
    error: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("at", "at", True, _number),
        Field("info", "info", True, _integer),
        Field("warn", "warn", True, _integer),
        Field("error", "error", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class LogHistogram(Payload):
    """The LogHistogram schema."""

    since: float
    until: float
    bucket_seconds: float
    buckets: tuple[LogBucket, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("since", "since", True, _number),
        Field("until", "until", True, _number),
        Field("bucket_seconds", "bucket_seconds", True, _number),
        Field("buckets", "buckets", True, _items(_record(LogBucket))),
    )


@dataclass(frozen=True, kw_only=True)
class LogHistogramReply(Payload):
    """The LogHistogramReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: LogHistogram

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(LogHistogram)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeleted(Payload):
    """The PredictorDeleted schema."""

    name: Name
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeletedReply(Payload):
    """The PredictorDeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorDeleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorDeleted)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextAppend(Payload):
    """The LiveContextAppend schema."""

    write_id: str
    rows: tuple[Row, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("write_id", "write_id", True, _text),
        Field("rows", "rows", True, _items(_decode_Row)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextAppendRequest(Payload):
    """The LiveContextAppendRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: LiveContextAppend

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(LiveContextAppend)),
    )


@dataclass(frozen=True, kw_only=True)
class Label(Payload):
    """The Label schema."""

    id: str | int
    value: str | float | bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _union(_text, _integer)),
        Field("value", "value", True, _union(_text, _number, _boolean)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextLabel(Payload):
    """The LiveContextLabel schema."""

    write_id: str
    target: str
    labels: tuple[Label, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("write_id", "write_id", True, _text),
        Field("target", "target", True, _text),
        Field("labels", "labels", True, _items(_record(Label))),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextLabelRequest(Payload):
    """The LiveContextLabelRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: LiveContextLabel

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(LiveContextLabel)),
    )


@dataclass(frozen=True, kw_only=True)
class Receipt(Payload):
    """Durable acceptance of an append or label write."""

    context: Name
    write_id: str
    operation: Literal["append", "label"]
    rows: int
    sequence: int
    accepted_at: float
    durably_accepted: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context", "context", True, _decode_Name),
        Field("write_id", "write_id", True, _text),
        Field("operation", "operation", True, _literal("append", "label")),
        Field("rows", "rows", True, _integer),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _number),
        Field("durably_accepted", "durably_accepted", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ReceiptReply(Payload):
    """The ReceiptReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Receipt

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Receipt)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorCreate(Payload):
    """The PredictorCreate schema."""

    name: Name
    spec: Spec
    env: Name
    dry_run: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("spec", "spec", True, _record(Spec)),
        Field("env", "env", True, _decode_Name),
        Field("dry_run", "dry_run", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorCreateRequest(Payload):
    """The PredictorCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: PredictorCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(PredictorCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePrepare(Payload):
    """The SourcePrepare schema."""

    kind: Literal["bigquery", "gcs", "snowflake"]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("bigquery", "gcs", "snowflake")),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePrepareRequest(Payload):
    """The SourcePrepareRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SourcePrepare

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SourcePrepare)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceIdentity(Payload):
    """The workspace's Google service account."""

    email: str
    unique_id: str
    principal: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("email", "email", True, _text),
        Field("unique_id", "unique_id", True, _pattern("^[0-9]+$")),
        Field("principal", "principal", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SourceGrant(Payload):
    """The SourceGrant schema."""

    role: str
    on: Literal["dataset", "bucket"]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("role", "role", True, _text),
        Field("on", "on", True, _literal("dataset", "bucket")),
    )


@dataclass(frozen=True, kw_only=True)
class SourceTrust(Payload):
    """The SourceTrust schema.

    what a Snowflake source trusts, the issuer and subject of the account's ID tokens
    """

    issuer: str
    subject: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePrepared(Payload):
    """The SourcePrepared schema."""

    kind: Literal["bigquery", "gcs", "snowflake"]
    identity: SourceIdentity
    grants: tuple[SourceGrant, ...]
    oidc: SourceTrust | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("bigquery", "gcs", "snowflake")),
        Field("identity", "identity", True, _record(SourceIdentity)),
        Field("grants", "grants", True, _items(_record(SourceGrant))),
        Field("oidc", "oidc", False, _record(SourceTrust)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePreparedReply(Payload):
    """The SourcePreparedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SourcePrepared

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SourcePrepared)),
    )


SourceKind = Literal["bigquery", "gcs", "snowflake", "databricks", "s3", "postgres"]
_decode_SourceKind: Decoder = _literal(
    "bigquery",
    "gcs",
    "snowflake",
    "databricks",
    "s3",
    "postgres",
)


@dataclass(frozen=True, kw_only=True)
class Source(Payload):
    """A workspace connection to a warehouse or bucket.

    Config keys by kind: bigquery `project, dataset`; gcs `bucket, prefix`; snowflake
    `account, warehouse, database, role`; databricks `host, warehouse, client_id`; s3
    `role_arn, region, bucket, prefix`; postgres `host, port, database, user, tls`
    """

    name: Name
    kind: SourceKind
    state: Literal["ready", "failing", "expired"]
    config: dict[str, str]
    secret_expires: str | None
    tested_at: str | None
    error: str | None
    origin: Origin
    created: str
    url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("kind", "kind", True, _decode_SourceKind),
        Field("state", "state", True, _literal("ready", "failing", "expired")),
        Field("config", "config", True, _mapping(_text)),
        Field("secret_expires", "secret_expires", True, _union(_text, _null)),
        Field("tested_at", "tested_at", True, _union(_text, _null)),
        Field("error", "error", True, _union(_text, _null)),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
        Field("url", "url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SourceList(Payload):
    """The SourceList schema."""

    sources: tuple[Source, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("sources", "sources", True, _items(_record(Source))),
    )


@dataclass(frozen=True, kw_only=True)
class SourceListReply(Payload):
    """The SourceListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SourceList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SourceList)),
    )


@dataclass(frozen=True, kw_only=True)
class PermissionGrant(Payload):
    """A permission keys may hold; one that is not enforced authorizes nothing yet."""

    name: Permission
    description: str
    enforced: bool
    roles: tuple[Literal["admin", "editor", "viewer", "scorer"], ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Permission),
        Field("description", "description", True, _text),
        Field("enforced", "enforced", True, _boolean),
        Field(
            "roles",
            "roles",
            True,
            _items(_literal("admin", "editor", "viewer", "scorer")),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class KeyIdentity(Payload):
    """The KeyIdentity schema."""

    id: str
    name: Name
    project: Name
    expires_at: float | None
    ephemeral: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
        Field("name", "name", True, _decode_Name),
        Field("project", "project", True, _decode_Name),
        Field("expires_at", "expires_at", True, _union(_number, _null)),
        Field("ephemeral", "ephemeral", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceIdentity(Payload):
    """The WorkspaceIdentity schema."""

    id: str
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _text),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Principal(Payload):
    """The credential's kind and id, with what the server knows of it."""

    kind: str
    id: str
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _text),
        Field("id", "id", True, _text),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class AuthStatus(Payload):
    """The AuthStatus schema."""

    authenticated: Literal[True] = True
    principal: Principal
    google: Literal["configured", "unconfigured"]
    sign_in: tuple[Literal["google", "microsoft", "github"], ...]
    secrets: Literal["configured", "unconfigured"]
    capabilities: tuple[Capability, ...]
    permission_inventory: tuple[PermissionGrant, ...]
    bootstrap: Literal["unconfigured", "available", "consumed"]
    key: KeyIdentity | None = None
    email: str | None = None
    machine: str | None = None
    workspace: WorkspaceIdentity | None = None
    url: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("authenticated", "authenticated", True, _literal(True)),
        Field("principal", "principal", True, _record(Principal)),
        Field("google", "google", True, _literal("configured", "unconfigured")),
        Field(
            "sign_in",
            "sign_in",
            True,
            _items(_literal("google", "microsoft", "github")),
        ),
        Field("secrets", "secrets", True, _literal("configured", "unconfigured")),
        Field("capabilities", "capabilities", True, _items(_record(Capability))),
        Field(
            "permission_inventory",
            "permission_inventory",
            True,
            _items(_record(PermissionGrant)),
        ),
        Field(
            "bootstrap",
            "bootstrap",
            True,
            _literal("unconfigured", "available", "consumed"),
        ),
        Field("key", "key", False, _record(KeyIdentity)),
        Field("email", "email", False, _union(_text, _null)),
        Field("machine", "machine", False, _union(_text, _null)),
        Field(
            "workspace",
            "workspace",
            False,
            _union(_record(WorkspaceIdentity), _null),
        ),
        Field("url", "url", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class AuthStatusReply(Payload):
    """The AuthStatusReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: AuthStatus

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(AuthStatus)),
    )


@dataclass(frozen=True, kw_only=True)
class LoggedOut(Payload):
    """The LoggedOut schema."""

    revoked: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revoked", "revoked", True, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class LoggedOutReply(Payload):
    """The LoggedOutReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: LoggedOut

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(LoggedOut)),
    )


@dataclass(frozen=True, kw_only=True)
class Organization(Payload):
    """The Organization schema."""

    name: Name
    account: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("account", "account", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationReply(Payload):
    """The OrganizationReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Organization

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Organization)),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationCreateRequest(Payload):
    """The OrganizationCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Organization

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Organization)),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationList(Payload):
    """The OrganizationList schema."""

    organizations: tuple[Organization, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("organizations", "organizations", True, _items(_record(Organization))),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationListReply(Payload):
    """The OrganizationListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: OrganizationList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(OrganizationList)),
    )


@dataclass(frozen=True, kw_only=True)
class Sso(Payload):
    """The Sso schema."""

    organization: Name
    provider: Literal["google"] = "google"
    providers: tuple[Literal["google", "microsoft", "github"], ...]
    enabled: bool
    configured: bool
    membership_required: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("organization", "organization", True, _decode_Name),
        Field("provider", "provider", True, _literal("google")),
        Field(
            "providers",
            "providers",
            True,
            _items(_literal("google", "microsoft", "github")),
        ),
        Field("enabled", "enabled", True, _boolean),
        Field("configured", "configured", True, _boolean),
        Field("membership_required", "membership_required", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class SsoReply(Payload):
    """The SsoReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Sso

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Sso)),
    )


@dataclass(frozen=True, kw_only=True)
class SsoConfigure(Payload):
    """The SsoConfigure schema."""

    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("enabled", "enabled", True, _boolean),)


@dataclass(frozen=True, kw_only=True)
class SsoConfigureRequest(Payload):
    """The SsoConfigureRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SsoConfigure

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SsoConfigure)),
    )


@dataclass(frozen=True, kw_only=True)
class AuditEvent(Payload):
    """The AuditEvent schema."""

    sequence: int
    at: float
    actor: str
    project: Name
    action: str
    target: str
    outcome: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("sequence", "sequence", True, _integer),
        Field("at", "at", True, _number),
        Field("actor", "actor", True, _text),
        Field("project", "project", True, _decode_Name),
        Field("action", "action", True, _text),
        Field("target", "target", True, _text),
        Field("outcome", "outcome", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class AuditLog(Payload):
    """The AuditLog schema."""

    events: tuple[AuditEvent, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("events", "events", True, _items(_record(AuditEvent))),
    )


@dataclass(frozen=True, kw_only=True)
class AuditLogReply(Payload):
    """The AuditLogReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: AuditLog

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(AuditLog)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginStart(Payload):
    """The DeviceLoginStart schema."""

    machine: str
    client: Literal["cli", "notebook"]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("machine", "machine", True, _text),
        Field("client", "client", True, _literal("cli", "notebook")),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginGrant(Payload):
    """The DeviceLoginGrant schema."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "device_code",
            "device_code",
            True,
            _pattern("^gf_dc_[A-Za-z0-9_-]{43}$"),
        ),
        Field("user_code", "user_code", True, _pattern("^[A-Z0-9]{4}-[A-Z0-9]{4}$")),
        Field("verification_uri", "verification_uri", True, _text),
        Field("verification_uri_complete", "verification_uri_complete", True, _text),
        Field("expires_in", "expires_in", True, _integer),
        Field("interval", "interval", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginPoll(Payload):
    """The DeviceLoginPoll schema."""

    device_code: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "device_code",
            "device_code",
            True,
            _pattern("^gf_dc_[A-Za-z0-9_-]{43}$"),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginToken(Payload):
    """The DeviceLoginToken schema."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: float | None
    key: KeyIdentity
    email: str
    workspace: WorkspaceIdentity
    project: Name
    url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("access_token", "access_token", True, _text),
        Field("token_type", "token_type", True, _literal("bearer")),
        Field("expires_at", "expires_at", True, _union(_number, _null)),
        Field("key", "key", True, _record(KeyIdentity)),
        Field("email", "email", True, _text),
        Field("workspace", "workspace", True, _record(WorkspaceIdentity)),
        Field("project", "project", True, _decode_Name),
        Field("url", "url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class LoginErrorReply(Payload):
    """Why a device login cannot go on, outside the envelope as the login is.

    `reason` says which: `authorization_pending` (`Unauthorized`, poll again),
    `slow_down` (`ValidationError`, poll less often), `expired_token` and
    `access_denied` (`Unauthorized`, start again) or `rate_limited`
    (`CapacityExceeded`).
    """

    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("error", "error", True, _record(Error)),
    )


WorkspaceRole = Literal["admin", "editor", "viewer", "scorer"]
_decode_WorkspaceRole: Decoder = _literal("admin", "editor", "viewer", "scorer")


JoinRequestId = str
_decode_JoinRequestId: Decoder = _pattern("^jr-[0-9a-f]{16}$")


InvitationId = str
_decode_InvitationId: Decoder = _pattern("^inv-[0-9a-f]{16}$")


WaitlistId = str
_decode_WaitlistId: Decoder = _pattern("^wl-[0-9a-f]{16}$")


WorkspaceName = str
_decode_WorkspaceName: Decoder = _text


UserCode = str
_decode_UserCode: Decoder = _text


@dataclass(frozen=True, kw_only=True)
class WorkspaceSummary(Payload):
    """A workspace the caller belongs to: its role there and the projects it may use."""

    workspace: Name
    display_name: str
    role: WorkspaceRole | None
    admin: bool
    default_project: Name | None
    projects: tuple[Name, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("admin", "admin", True, _boolean),
        Field("default_project", "default_project", True, _union(_decode_Name, _null)),
        Field("projects", "projects", True, _items(_decode_Name)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceDetails(Payload):
    """The WorkspaceDetails schema.

    a workspace as `workspaces.get` reads it: its summary, its organization and when it
    was made. Its fields are listed whole, as `WorkspaceSummary` allows no other
    fields.
    """

    workspace: Name
    display_name: str
    role: WorkspaceRole | None
    admin: bool
    default_project: Name | None
    projects: tuple[Name, ...]
    organization: Name
    created_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("admin", "admin", True, _boolean),
        Field("default_project", "default_project", True, _union(_decode_Name, _null)),
        Field("projects", "projects", True, _items(_decode_Name)),
        Field("organization", "organization", True, _decode_Name),
        Field("created_at", "created_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequest(Payload):
    """A request to join a workspace, as the person who asked sees it."""

    id: JoinRequestId
    workspace: Name
    display_name: str
    state: Literal["pending", "approved", "denied", "cancelled"]
    created_at: float
    decided_at: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "approved", "denied", "cancelled"),
        ),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestAdmin(Payload):
    """A request to join a workspace, as its admins see it."""

    id: JoinRequestId
    workspace: Name
    subject: str
    email: str
    name: str | None
    domain: str
    state: Literal["pending", "approved", "denied", "cancelled"]
    role: WorkspaceRole | None
    created_at: float
    decided_at: float | None
    decided_by: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("subject", "subject", True, _text),
        Field("email", "email", True, _text),
        Field("name", "name", True, _union(_text, _null)),
        Field("domain", "domain", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "approved", "denied", "cancelled"),
        ),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
        Field("decided_by", "decided_by", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class OnboardingProfile(Payload):
    """The signed-in person's verified email, all null until they sign in again."""

    email: str | None
    email_domain: str | None
    display_name: str | None
    personal_domain: bool | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("email", "email", True, _union(_text, _null)),
        Field("email_domain", "email_domain", True, _union(_text, _null)),
        Field("display_name", "display_name", True, _union(_text, _null)),
        Field("personal_domain", "personal_domain", True, _union(_boolean, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class DiscoveredWorkspace(Payload):
    """A workspace the person's email domain may ask to join."""

    workspace: Name
    display_name: str
    domain: str
    request: JoinRequest | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("domain", "domain", True, _text),
        Field("request", "request", True, _union(_record(JoinRequest), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Onboarding(Payload):
    """The Onboarding schema.

    where a signed-in person stands: their workspaces, those their domain may join, and
    whether they may create one
    """

    state: Literal["member", "onboarding", "reauthentication_required"]
    profile: OnboardingProfile
    workspaces: tuple[WorkspaceSummary, ...]
    discovered: tuple[DiscoveredWorkspace, ...]
    requests: tuple[JoinRequest, ...]
    can_create_workspace: bool
    signup: Literal["open", "waitlist", "pending"]
    workspace_name: str | None
    sign_in_url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "state",
            "state",
            True,
            _literal("member", "onboarding", "reauthentication_required"),
        ),
        Field("profile", "profile", True, _record(OnboardingProfile)),
        Field("workspaces", "workspaces", True, _items(_record(WorkspaceSummary))),
        Field("discovered", "discovered", True, _items(_record(DiscoveredWorkspace))),
        Field("requests", "requests", True, _items(_record(JoinRequest))),
        Field("can_create_workspace", "can_create_workspace", True, _boolean),
        Field("signup", "signup", True, _literal("open", "waitlist", "pending")),
        Field("workspace_name", "workspace_name", True, _union(_text, _null)),
        Field("sign_in_url", "sign_in_url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceList(Payload):
    """The WorkspaceList schema."""

    workspaces: tuple[WorkspaceSummary, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspaces", "workspaces", True, _items(_record(WorkspaceSummary))),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreate(Payload):
    """The WorkspaceCreate schema."""

    display_name: WorkspaceName

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("display_name", "display_name", True, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreated(Payload):
    """`created` is false when the request id already made this workspace."""

    workspace: WorkspaceSummary
    organization: Name
    created: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _record(WorkspaceSummary)),
        Field("organization", "organization", True, _decode_Name),
        Field("created", "created", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRef(Payload):
    """The WorkspaceRef schema."""

    workspace: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRename(Payload):
    """The WorkspaceRename schema."""

    workspace: Name
    display_name: WorkspaceName

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceJoin(Payload):
    """The WorkspaceJoin schema.

    `workspace` is set once the caller is a member, `request` while a join request is
    pending or was denied lately
    """

    status: Literal["already_member", "pending", "denied"]
    workspace: WorkspaceSummary | None
    request: JoinRequest | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "status",
            "status",
            True,
            _literal("already_member", "pending", "denied"),
        ),
        Field("workspace", "workspace", True, _union(_record(WorkspaceSummary), _null)),
        Field("request", "request", True, _union(_record(JoinRequest), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMember(Payload):
    """One membership in the workspace, its organizations or projects."""

    issuer: str
    subject: str
    email: str | None
    name: str | None
    scope_kind: Literal["account", "organization", "project"]
    scope_id: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("email", "email", True, _union(_text, _null)),
        Field("name", "name", True, _union(_text, _null)),
        Field(
            "scope_kind",
            "scope_kind",
            True,
            _literal("account", "organization", "project"),
        ),
        Field("scope_id", "scope_id", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMemberList(Payload):
    """The WorkspaceMemberList schema."""

    members: tuple[WorkspaceMember, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("members", "members", True, _items(_record(WorkspaceMember))),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChange(Payload):
    """The MemberRoleChange schema."""

    workspace: Name
    issuer: str
    subject: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRole(Payload):
    """The MemberRole schema."""

    issuer: str
    subject: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChanged(Payload):
    """The MemberRoleChanged schema."""

    member: MemberRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("member", "member", True, _record(MemberRole)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemove(Payload):
    """The MemberRemove schema."""

    workspace: Name
    issuer: str
    subject: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class MemberGone(Payload):
    """The MemberGone schema."""

    issuer: str
    subject: str
    removed: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("removed", "removed", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemoved(Payload):
    """The MemberRemoved schema."""

    member: MemberGone

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("member", "member", True, _record(MemberGone)),
    )


@dataclass(frozen=True, kw_only=True)
class Invitation(Payload):
    """An invitation to a workspace; `state` is `expired` once a pending one lapses."""

    id: InvitationId
    workspace: Name
    email: str
    role: WorkspaceRole
    state: Literal["pending", "expired", "accepted", "revoked"]
    created_at: float
    expires_at: float
    created_by: str
    accepted_at: float | None
    revoked_at: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_InvitationId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("email", "email", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "expired", "accepted", "revoked"),
        ),
        Field("created_at", "created_at", True, _number),
        Field("expires_at", "expires_at", True, _number),
        Field("created_by", "created_by", True, _text),
        Field("accepted_at", "accepted_at", True, _union(_number, _null)),
        Field("revoked_at", "revoked_at", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationList(Payload):
    """The InvitationList schema."""

    invitations: tuple[Invitation, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitations", "invitations", True, _items(_record(Invitation))),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreate(Payload):
    """`role` is `viewer` and `expires_in` a week unless given."""

    workspace: Name
    email: str
    role: WorkspaceRole | None = None
    expires_in: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("email", "email", True, _text),
        Field("role", "role", False, _decode_WorkspaceRole),
        Field("expires_in", "expires_in", False, _number),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreated(Payload):
    """The InvitationCreated schema.

    `token` and `invitation_url` only when the server has no mailer and `delivery` is
    `link`; `replaces` names a pending invitation to the same email it revoked
    """

    invitation: Invitation
    delivery: Literal["email", "link"]
    replaces: InvitationId | None
    token: str | None = None
    invitation_url: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitation", "invitation", True, _record(Invitation)),
        Field("delivery", "delivery", True, _literal("email", "link")),
        Field("replaces", "replaces", True, _union(_decode_InvitationId, _null)),
        Field("token", "token", False, _pattern("^gf_inv_[A-Za-z0-9_-]{43}$")),
        Field("invitation_url", "invitation_url", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationRevoke(Payload):
    """The InvitationRevoke schema."""

    workspace: Name
    id: InvitationId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_InvitationId),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationView(Payload):
    """The InvitationView schema."""

    invitation: Invitation

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitation", "invitation", True, _record(Invitation)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationToken(Payload):
    """The token from the invitation link."""

    token: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("token", "token", True, _pattern("^gf_inv_[A-Za-z0-9_-]{43}$")),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationPreview(Payload):
    """The InvitationPreview schema."""

    workspace: Name
    display_name: str
    role: WorkspaceRole
    expires_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
        Field("expires_at", "expires_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationAccepted(Payload):
    """The InvitationAccepted schema."""

    status: Literal["joined", "already_member"]
    workspace: WorkspaceSummary

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("joined", "already_member")),
        Field("workspace", "workspace", True, _record(WorkspaceSummary)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestRef(Payload):
    """The JoinRequestRef schema."""

    id: JoinRequestId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestView(Payload):
    """The JoinRequestView schema."""

    request: JoinRequest

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequest)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestList(Payload):
    """The JoinRequestList schema."""

    requests: tuple[JoinRequestAdmin, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("requests", "requests", True, _items(_record(JoinRequestAdmin))),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApprove(Payload):
    """`role` is `viewer` unless given."""

    workspace: Name
    id: JoinRequestId
    role: WorkspaceRole | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_JoinRequestId),
        Field("role", "role", False, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDeny(Payload):
    """The JoinRequestDeny schema."""

    workspace: Name
    id: JoinRequestId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_JoinRequestId),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMembership(Payload):
    """The WorkspaceMembership schema.

    the account membership an approval granted, or the one the person already held
    """

    issuer: str
    subject: str
    scope_kind: Literal["account"] = "account"
    scope_id: Name
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("scope_kind", "scope_kind", True, _literal("account")),
        Field("scope_id", "scope_id", True, _decode_Name),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApproved(Payload):
    """The JoinRequestApproved schema."""

    request: JoinRequestAdmin
    membership: WorkspaceMembership

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequestAdmin)),
        Field("membership", "membership", True, _record(WorkspaceMembership)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDenied(Payload):
    """The JoinRequestDenied schema."""

    request: JoinRequestAdmin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequestAdmin)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomain(Payload):
    """An email domain whose people may ask to join the workspace."""

    workspace: Name
    domain: str
    enabled_by: str
    enabled_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled_by", "enabled_by", True, _text),
        Field("enabled_at", "enabled_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainList(Payload):
    """The JoinDomainList schema.

    `email_domain` is the caller's own company domain, the one it may enable; null for
    a personal one
    """

    domains: tuple[JoinDomain, ...]
    email_domain: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("domains", "domains", True, _items(_record(JoinDomain))),
        Field("email_domain", "email_domain", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigure(Payload):
    """Enabling takes only the domain of the admin's own verified email."""

    workspace: Name
    domain: str
    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled", "enabled", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigured(Payload):
    """The JoinDomainConfigured schema."""

    workspace: Name
    domain: str
    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled", "enabled", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistJoin(Payload):
    """`display_name` is the workspace the person means to create."""

    display_name: WorkspaceName | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("display_name", "display_name", False, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistStatus(Payload):
    """The WaitlistStatus schema."""

    status: Literal["approved", "pending"]
    message: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("approved", "pending")),
        Field("message", "message", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistEntry(Payload):
    """The WaitlistEntry schema.

    a waitlist sign-up; approving one from a company domain approves the whole domain
    """

    id: WaitlistId
    email: str
    display_name: str | None
    state: Literal["pending", "approved"]
    approves: Literal["person", "domain"]
    domain: str
    created_at: float
    decided_at: float | None
    workspace_name: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_WaitlistId),
        Field("email", "email", True, _text),
        Field("display_name", "display_name", True, _union(_text, _null)),
        Field("state", "state", True, _literal("pending", "approved")),
        Field("approves", "approves", True, _literal("person", "domain")),
        Field("domain", "domain", True, _text),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
        Field("workspace_name", "workspace_name", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistRef(Payload):
    """The WaitlistRef schema."""

    id: WaitlistId

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("id", "id", True, _decode_WaitlistId),)


@dataclass(frozen=True, kw_only=True)
class WaitlistEntryView(Payload):
    """The WaitlistEntryView schema."""

    entry: WaitlistEntry

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("entry", "entry", True, _record(WaitlistEntry)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginCode(Payload):
    """The DeviceLoginCode schema."""

    user_code: UserCode

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("user_code", "user_code", True, _decode_UserCode),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginWorkspace(Payload):
    """The DeviceLoginWorkspace schema."""

    workspace: Name
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginPreview(Payload):
    """The DeviceLoginPreview schema.

    what the sign-in page confirms: the machine and client asking ("Approve the CLI on
    maya-mbp", "Approve a notebook on maya-mbp") and the workspaces with a project the
    key may serve
    """

    user_code: str
    machine: str
    client: Literal["cli", "notebook"]
    expires_at: float
    email: str
    workspaces: tuple[DeviceLoginWorkspace, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("machine", "machine", True, _text),
        Field("client", "client", True, _literal("cli", "notebook")),
        Field("expires_at", "expires_at", True, _number),
        Field("email", "email", True, _text),
        Field("workspaces", "workspaces", True, _items(_record(DeviceLoginWorkspace))),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginApprove(Payload):
    """The DeviceLoginApprove schema."""

    user_code: UserCode
    workspace: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("user_code", "user_code", True, _decode_UserCode),
        Field("workspace", "workspace", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginGranted(Payload):
    """The DeviceLoginGranted schema."""

    id: Name
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginApproved(Payload):
    """The DeviceLoginApproved schema."""

    user_code: str
    state: Literal["approved"] = "approved"
    workspace: DeviceLoginGranted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("state", "state", True, _literal("approved")),
        Field("workspace", "workspace", True, _record(DeviceLoginGranted)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginDenied(Payload):
    """The DeviceLoginDenied schema."""

    user_code: str
    state: Literal["denied"] = "denied"

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("state", "state", True, _literal("denied")),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginApproveRequest(Payload):
    """The DeviceLoginApproveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: DeviceLoginApprove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(DeviceLoginApprove)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginCodeRequest(Payload):
    """The DeviceLoginCodeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: DeviceLoginCode

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(DeviceLoginCode)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreateRequest(Payload):
    """The InvitationCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationRevokeRequest(Payload):
    """The InvitationRevokeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationRevoke

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationRevoke)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationTokenRequest(Payload):
    """The InvitationTokenRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationToken

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationToken)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigureRequest(Payload):
    """The JoinDomainConfigureRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinDomainConfigure

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinDomainConfigure)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApproveRequest(Payload):
    """The JoinRequestApproveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestApprove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestApprove)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDenyRequest(Payload):
    """The JoinRequestDenyRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestDeny

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestDeny)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestRefRequest(Payload):
    """The JoinRequestRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestRef)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemoveRequest(Payload):
    """The MemberRemoveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: MemberRemove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(MemberRemove)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChangeRequest(Payload):
    """The MemberRoleChangeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: MemberRoleChange

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(MemberRoleChange)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistJoinRequest(Payload):
    """The WaitlistJoinRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WaitlistJoin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WaitlistJoin)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistRefRequest(Payload):
    """The WaitlistRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WaitlistRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WaitlistRef)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreateRequest(Payload):
    """The WorkspaceCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRefRequest(Payload):
    """The WorkspaceRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceRef)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRenameRequest(Payload):
    """The WorkspaceRenameRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceRename

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceRename)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginApprovedReply(Payload):
    """The DeviceLoginApprovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: DeviceLoginApproved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(DeviceLoginApproved)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginDeniedReply(Payload):
    """The DeviceLoginDeniedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: DeviceLoginDenied

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(DeviceLoginDenied)),
    )


@dataclass(frozen=True, kw_only=True)
class DeviceLoginPreviewReply(Payload):
    """The DeviceLoginPreviewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: DeviceLoginPreview

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(DeviceLoginPreview)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationAcceptedReply(Payload):
    """The InvitationAcceptedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationAccepted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationAccepted)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreatedReply(Payload):
    """The InvitationCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationCreated)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationListReply(Payload):
    """The InvitationListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationList)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationPreviewReply(Payload):
    """The InvitationPreviewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationPreview

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationPreview)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationViewReply(Payload):
    """The InvitationViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationView)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfiguredReply(Payload):
    """The JoinDomainConfiguredReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinDomainConfigured

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinDomainConfigured)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainListReply(Payload):
    """The JoinDomainListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinDomainList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinDomainList)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApprovedReply(Payload):
    """The JoinRequestApprovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestApproved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestApproved)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDeniedReply(Payload):
    """The JoinRequestDeniedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestDenied

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestDenied)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestListReply(Payload):
    """The JoinRequestListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestList)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestViewReply(Payload):
    """The JoinRequestViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestView)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemovedReply(Payload):
    """The MemberRemovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: MemberRemoved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(MemberRemoved)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChangedReply(Payload):
    """The MemberRoleChangedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: MemberRoleChanged

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(MemberRoleChanged)),
    )


@dataclass(frozen=True, kw_only=True)
class OnboardingReply(Payload):
    """The OnboardingReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Onboarding

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Onboarding)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistEntryViewReply(Payload):
    """The WaitlistEntryViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WaitlistEntryView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WaitlistEntryView)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistStatusReply(Payload):
    """The WaitlistStatusReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WaitlistStatus

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WaitlistStatus)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreatedReply(Payload):
    """The WorkspaceCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceCreated)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceDetailsReply(Payload):
    """The WorkspaceDetailsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceDetails

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceDetails)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceJoinReply(Payload):
    """The WorkspaceJoinReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceJoin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceJoin)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceListReply(Payload):
    """The WorkspaceListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceList)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMemberListReply(Payload):
    """The WorkspaceMemberListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceMemberList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceMemberList)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceSummaryReply(Payload):
    """The WorkspaceSummaryReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceSummary

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceSummary)),
    )


@dataclass(frozen=True, kw_only=True)
class Code(Payload):
    """The Code schema.

    the same resource made in code: the SDK call, the `gf` command and a curl request
    """

    python: str
    cli: str
    curl: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("python", "python", True, _text),
        Field("cli", "cli", True, _text),
        Field("curl", "curl", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Deleted(Payload):
    """The Deleted schema."""

    name: Name
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class LargeContext(Payload):
    """The LargeContext schema.

    the large-context line: GPU seconds past `large_context_cells` in the price book,
    and their cost
    """

    gpu_seconds: float
    credits: float
    usd: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("gpu_seconds", "gpu_seconds", True, _number),
        Field("credits", "credits", True, _number),
        Field("usd", "usd", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class Estimate(Payload):
    """The Estimate schema.

    what a call or a run would cost from the live price book: model calls, scored rows,
    credits (1 credit is $0.001) and dollars
    """

    calls: int
    rows: int
    credits: float
    usd: float
    large_context: LargeContext | None = None
    warnings: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("calls", "calls", True, _integer),
        Field("rows", "rows", True, _integer),
        Field("credits", "credits", True, _number),
        Field("usd", "usd", True, _number),
        Field("large_context", "large_context", False, _record(LargeContext)),
        Field("warnings", "warnings", False, _items(_text)),
    )


@dataclass(frozen=True, kw_only=True)
class Column(Payload):
    """The Column schema."""

    name: str
    type: ColumnType
    missing: int
    distinct: int | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("type", "type", True, _decode_ColumnType),
        Field("missing", "missing", True, _integer),
        Field("distinct", "distinct", True, _union(_integer, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextData(Payload):
    """The ContextData schema.

    an upload's announcement; the bytes follow in chunks unless the rows are unchanged
    """

    format: Literal["parquet", "arrow", "csv"]
    bytes: int
    digest: Digest
    rows: int | None = None
    file: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("format", "format", True, _literal("parquet", "arrow", "csv")),
        Field("bytes", "bytes", True, _integer),
        Field("digest", "digest", True, _decode_Digest),
        Field("rows", "rows", False, _integer),
        Field("file", "file", False, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSourceRead(Payload):
    """What a context reads from a source: one `table` or a `sql` query."""

    source: Name
    table: str | None = None
    sql: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _decode_Name),
        Field("table", "table", False, _text),
        Field("sql", "sql", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSource(Payload):
    """The ContextSource schema."""

    source: Name
    kind: SourceKind
    table: str | None
    sql: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _decode_Name),
        Field("kind", "kind", True, _decode_SourceKind),
        Field("table", "table", True, _union(_text, _null)),
        Field("sql", "sql", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextCreate(Payload):
    """The ContextCreate schema.

    one origin: `data` (an upload, version 1), `source` (an import, or with `kind:
    live` a read at call time cached for `freshness_seconds`), or `kind: live` with
    `columns` for rows that arrive through `append`
    """

    name: Name
    kind: Literal["snapshot", "live"]
    data: ContextData | None = None
    source: ContextSourceRead | None = None
    columns: dict[str, ColumnType] | None = None
    targets: tuple[str, ...] | None = None
    id: str | None = None
    time: str | None = None
    freshness_seconds: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("kind", "kind", True, _literal("snapshot", "live")),
        Field("data", "data", False, _record(ContextData)),
        Field("source", "source", False, _record(ContextSourceRead)),
        Field("columns", "columns", False, _mapping(_decode_ColumnType)),
        Field("targets", "targets", False, _items(_text)),
        Field("id", "id", False, _union(_text, _null)),
        Field("time", "time", False, _union(_text, _null)),
        Field("freshness_seconds", "freshness_seconds", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ContextUploadTarget(Payload):
    """The ContextUploadTarget schema."""

    version: int
    url: str | None
    chunk_bytes: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("version", "version", True, _integer),
        Field("url", "url", True, _union(_text, _null)),
        Field("chunk_bytes", "chunk_bytes", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Context(Payload):
    """The Context schema.

    the data a model learns from; `version` is the latest (or the one read), `upload`
    where an upload's chunks go
    """

    name: Name
    kind: Literal["snapshot", "live"]
    state: Literal["uploading", "importing", "ready", "failed"]
    version: int | None
    versions: int
    digest: Digest | None
    rows: int | None
    columns: tuple[Column, ...]
    id: str | None
    time: str | None
    source: ContextSource | None
    freshness_seconds: int | None
    read_at: str | None
    error: str | None
    predictors: tuple[Name, ...]
    experiments: tuple[Name, ...]
    origin: Origin
    created: str
    updated: str
    url: str
    targets: tuple[str, ...] | None = None
    file: str | None = None
    upload: ContextUploadTarget | None = None
    unchanged: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("kind", "kind", True, _literal("snapshot", "live")),
        Field(
            "state",
            "state",
            True,
            _literal("uploading", "importing", "ready", "failed"),
        ),
        Field("version", "version", True, _union(_integer, _null)),
        Field("versions", "versions", True, _integer),
        Field("digest", "digest", True, _union(_decode_Digest, _null)),
        Field("rows", "rows", True, _union(_integer, _null)),
        Field("columns", "columns", True, _items(_record(Column))),
        Field("id", "id", True, _union(_text, _null)),
        Field("time", "time", True, _union(_text, _null)),
        Field("targets", "targets", False, _items(_text)),
        Field("source", "source", True, _union(_record(ContextSource), _null)),
        Field("freshness_seconds", "freshness_seconds", True, _union(_integer, _null)),
        Field("read_at", "read_at", True, _union(_text, _null)),
        Field("error", "error", True, _union(_text, _null)),
        Field("predictors", "predictors", True, _items(_decode_Name)),
        Field("experiments", "experiments", True, _items(_decode_Name)),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
        Field("updated", "updated", True, _text),
        Field("url", "url", True, _text),
        Field("file", "file", False, _union(_text, _null)),
        Field("upload", "upload", False, _record(ContextUploadTarget)),
        Field("unchanged", "unchanged", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ContextUpload(Payload):
    """The ContextUpload schema."""

    data: ContextData

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("data", "data", True, _record(ContextData)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextList(Payload):
    """The ContextList schema."""

    contexts: tuple[Context, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("contexts", "contexts", True, _items(_record(Context))),
    )


@dataclass(frozen=True, kw_only=True)
class ContextVersionInfo(Payload):
    """The ContextVersionInfo schema."""

    version: int
    digest: Digest
    rows: int
    created: str
    origin: Origin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("version", "version", True, _integer),
        Field("digest", "digest", True, _decode_Digest),
        Field("rows", "rows", True, _integer),
        Field("created", "created", True, _text),
        Field("origin", "origin", True, _record(Origin)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextVersions(Payload):
    """The ContextVersions schema."""

    versions: tuple[ContextVersionInfo, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("versions", "versions", True, _items(_record(ContextVersionInfo))),
    )


@dataclass(frozen=True, kw_only=True)
class Rows(Payload):
    """A page of rows; `next` is the cursor for the page after, or null at the end."""

    columns: tuple[Column, ...]
    rows: tuple[Row, ...]
    next: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("columns", "columns", True, _items(_record(Column))),
        Field("rows", "rows", True, _items(_decode_Row)),
        Field("next", "next", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Received(Payload):
    """The bytes of the version the service holds so far."""

    received: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("received", "received", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Holdout(Payload):
    """The Holdout schema.

    the rows held out to score: a random share or count (optionally stratified), whole
    groups of a column, the latest by time, or none
    """

    kind: Literal["random", "group", "time", "none"]
    fraction: float | None = None
    rows: int | None = None
    column: str | None = None
    until: str | None = None
    stratify: bool | None = None
    seed: int | None = None
    repeats: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _literal("random", "group", "time", "none")),
        Field("fraction", "fraction", False, _number),
        Field("rows", "rows", False, _integer),
        Field("column", "column", False, _text),
        Field("until", "until", False, _text),
        Field("stratify", "stratify", False, _boolean),
        Field("seed", "seed", False, _integer),
        Field("repeats", "repeats", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentSetup(Payload):
    """The ExperimentSetup schema.

    an experiment's setup; `name` defaults to `<context>-<target>`, the task is
    inferred from the target, the holdout is the latest 20% by `time` when one is
    declared, else 20% at random
    """

    context: Name
    target: str
    name: Name | None = None
    title: str | None = None
    version: int | None = None
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    holdout: Holdout | None = None
    sql: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", False, _decode_Name),
        Field("title", "title", False, _text),
        Field("context", "context", True, _decode_Name),
        Field("version", "version", False, _integer),
        Field("target", "target", True, _text),
        Field("task", "task", False, _decode_Task),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("inputs", "inputs", False, _items(_text)),
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("holdout", "holdout", False, _record(Holdout)),
        Field("sql", "sql", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentCreate(Payload):
    """The setup, and the models of run 1: the recommended set when left out."""

    context: Name
    target: str
    name: Name | None = None
    title: str | None = None
    version: int | None = None
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    holdout: Holdout | None = None
    sql: str | None = None
    models: tuple[ModelField, ...] | None = None
    dry_run: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", False, _decode_Name),
        Field("title", "title", False, _text),
        Field("context", "context", True, _decode_Name),
        Field("version", "version", False, _integer),
        Field("target", "target", True, _text),
        Field("task", "task", False, _decode_Task),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("inputs", "inputs", False, _items(_text)),
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("holdout", "holdout", False, _record(Holdout)),
        Field("sql", "sql", False, _text),
        Field("models", "models", False, _items(_decode_ModelField)),
        Field("dry_run", "dry_run", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentUpdate(Payload):
    """The setup fields that change; earlier runs become `stale`."""

    title: str | None = None
    context: Name | None = None
    version: int | None = None
    target: str | None = None
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    holdout: Holdout | None = None
    sql: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("title", "title", False, _text),
        Field("context", "context", False, _decode_Name),
        Field("version", "version", False, _integer),
        Field("target", "target", False, _text),
        Field("task", "task", False, _decode_Task),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("inputs", "inputs", False, _items(_text)),
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("holdout", "holdout", False, _record(Holdout)),
        Field("sql", "sql", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Entry(Payload):
    """The Entry schema."""

    model: ModelField
    short: str | None
    state: Literal["queued", "running", "finished", "failed", "cancelled"]
    score: float | None
    interval: tuple[float, ...] | None
    metrics: dict[str, float]
    seconds: float
    rows: int
    credits: float
    usd: float
    error: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelField),
        Field("short", "short", True, _union(_text, _null)),
        Field(
            "state",
            "state",
            True,
            _literal("queued", "running", "finished", "failed", "cancelled"),
        ),
        Field("score", "score", True, _union(_number, _null)),
        Field("interval", "interval", True, _union(_items(_number), _null)),
        Field("metrics", "metrics", True, _mapping(_number)),
        Field("seconds", "seconds", True, _number),
        Field("rows", "rows", True, _integer),
        Field("credits", "credits", True, _number),
        Field("usd", "usd", True, _number),
        Field("error", "error", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRun(Payload):
    """The ExperimentRun schema.

    a run: its models scored on the held-out rows, best first; it pins the context
    digest, setup digest and model digests, and is `stale` once the setup moved
    """

    experiment: Name
    number: int
    state: Literal["queued", "running", "finished", "failed", "cancelled"]
    stale: bool
    models: tuple[ModelId, ...]
    metric: str
    lower_is_better: bool
    held_rows: int
    context_rows: int
    repeats: int
    leaderboard: tuple[Entry, ...]
    rows: int
    credits: float
    usd: float
    created: str
    started: str | None
    finished: str | None
    origin: Origin
    error: str | None
    url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("experiment", "experiment", True, _decode_Name),
        Field("number", "number", True, _integer),
        Field(
            "state",
            "state",
            True,
            _literal("queued", "running", "finished", "failed", "cancelled"),
        ),
        Field("stale", "stale", True, _boolean),
        Field("models", "models", True, _items(_decode_ModelId)),
        Field("metric", "metric", True, _text),
        Field("lower_is_better", "lower_is_better", True, _boolean),
        Field("held_rows", "held_rows", True, _integer),
        Field("context_rows", "context_rows", True, _integer),
        Field("repeats", "repeats", True, _integer),
        Field("leaderboard", "leaderboard", True, _items(_record(Entry))),
        Field("rows", "rows", True, _integer),
        Field("credits", "credits", True, _number),
        Field("usd", "usd", True, _number),
        Field("created", "created", True, _text),
        Field("started", "started", True, _union(_text, _null)),
        Field("finished", "finished", True, _union(_text, _null)),
        Field("origin", "origin", True, _record(Origin)),
        Field("error", "error", True, _union(_text, _null)),
        Field("url", "url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunSummary(Payload):
    """The ExperimentRunSummary schema."""

    number: int
    state: Literal["queued", "running", "finished", "failed", "cancelled"]
    models: tuple[ModelId, ...]
    best: str | None
    score: float | None
    origin: Origin
    created: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("number", "number", True, _integer),
        Field(
            "state",
            "state",
            True,
            _literal("queued", "running", "finished", "failed", "cancelled"),
        ),
        Field("models", "models", True, _items(_decode_ModelId)),
        Field("best", "best", True, _union(_text, _null)),
        Field("score", "score", True, _union(_number, _null)),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Experiment(Payload):
    """The Experiment schema.

    models evaluated on one context for one target; `run` is the latest, or the one
    asked for
    """

    name: Name
    title: str
    context: Name
    context_version: int
    target: str
    task: Task
    id: str | None
    time: str | None
    inputs: tuple[str, ...]
    types: dict[str, ColumnType]
    holdout: Holdout
    sql: str | None
    metric: str
    runs: tuple[ExperimentRunSummary, ...]
    run: ExperimentRun | None
    origin: Origin
    created: str
    updated: str
    url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("title", "title", True, _text),
        Field("context", "context", True, _decode_Name),
        Field("context_version", "context_version", True, _integer),
        Field("target", "target", True, _text),
        Field("task", "task", True, _decode_Task),
        Field("id", "id", True, _union(_text, _null)),
        Field("time", "time", True, _union(_text, _null)),
        Field("inputs", "inputs", True, _items(_text)),
        Field("types", "types", True, _mapping(_decode_ColumnType)),
        Field("holdout", "holdout", True, _record(Holdout)),
        Field("sql", "sql", True, _union(_text, _null)),
        Field("metric", "metric", True, _text),
        Field("runs", "runs", True, _items(_record(ExperimentRunSummary))),
        Field("run", "run", True, _union(_record(ExperimentRun), _null)),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
        Field("updated", "updated", True, _text),
        Field("url", "url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentPlan(Payload):
    """A dry run: the setup as it resolves, and what run 1 would cost."""

    setup: ExperimentSetup
    estimate: Estimate
    warnings: tuple[str, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("setup", "setup", True, _record(ExperimentSetup)),
        Field("estimate", "estimate", True, _record(Estimate)),
        Field("warnings", "warnings", True, _items(_text)),
    )


ExperimentCreated = Experiment | ExperimentPlan
_decode_ExperimentCreated: Decoder = _union(
    _record(Experiment),
    _record(ExperimentPlan),
)


@dataclass(frozen=True, kw_only=True)
class ExperimentList(Payload):
    """The experiments, each without its run's leaderboard."""

    experiments: tuple[Experiment, ...]
    next: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("experiments", "experiments", True, _items(_record(Experiment))),
        Field("next", "next", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunCreate(Payload):
    """The ExperimentRunCreate schema."""

    models: tuple[ModelField, ...] | None = None
    dry_run: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("models", "models", False, _items(_decode_ModelField)),
        Field("dry_run", "dry_run", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunEstimate(Payload):
    """The ExperimentRunEstimate schema."""

    estimate: Estimate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("estimate", "estimate", True, _record(Estimate)),
    )


ExperimentRunCreated = ExperimentRun | ExperimentRunEstimate
_decode_ExperimentRunCreated: Decoder = _union(
    _record(ExperimentRun),
    _record(ExperimentRunEstimate),
)


@dataclass(frozen=True, kw_only=True)
class ExperimentRunList(Payload):
    """The ExperimentRunList schema."""

    runs: tuple[ExperimentRun, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("runs", "runs", True, _items(_record(ExperimentRun))),
    )


@dataclass(frozen=True, kw_only=True)
class Slice(Payload):
    """The Slice schema."""

    column: str
    value: str
    rows: int
    score: float
    delta: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("column", "column", True, _text),
        Field("value", "value", True, _text),
        Field("rows", "rows", True, _integer),
        Field("score", "score", True, _number),
        Field("delta", "delta", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SliceList(Payload):
    """The SliceList schema."""

    slices: tuple[Slice, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("slices", "slices", True, _items(_record(Slice))),
    )


@dataclass(frozen=True, kw_only=True)
class HeldRows(Payload):
    """The held-out rows with one model's predictions beside them."""

    model: ModelField
    rows: tuple[Row, ...]
    next: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelField),
        Field("rows", "rows", True, _items(_decode_Row)),
        Field("next", "next", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class SpecPatch(Payload):
    """The spec fields that change; `null` clears one."""

    context: str | None = None
    model: ModelField | None = None
    target: str | None = None
    task: Task | None = None
    id: str | None = None
    time: str | None = None
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    series: tuple[str, ...] | None = None
    frequency: str | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None
    selection: Selection | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "context",
            "context",
            False,
            _union(_pattern("^[a-z][a-z0-9-]{0,62}(@[1-9][0-9]{0,8})?$"), _null),
        ),
        Field("model", "model", False, _union(_decode_ModelField, _null)),
        Field("target", "target", False, _union(_text, _null)),
        Field("task", "task", False, _union(_decode_Task, _null)),
        Field("id", "id", False, _union(_text, _null)),
        Field("time", "time", False, _union(_text, _null)),
        Field("inputs", "inputs", False, _union(_items(_text), _null)),
        Field("exclude", "exclude", False, _union(_items(_text), _null)),
        Field("types", "types", False, _union(_mapping(_decode_ColumnType), _null)),
        Field("series", "series", False, _union(_items(_text), _null)),
        Field("frequency", "frequency", False, _union(_union(_text, _number), _null)),
        Field("horizon", "horizon", False, _union(_integer, _null)),
        Field("quantiles", "quantiles", False, _union(_items(_number), _null)),
        Field("past", "past", False, _union(_items(_text), _null)),
        Field("future", "future", False, _union(_items(_text), _null)),
        Field("selection", "selection", False, _union(_record(Selection), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorUpdate(Payload):
    """The PredictorUpdate schema."""

    spec: SpecPatch
    env: Name
    dry_run: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("spec", "spec", True, _record(SpecPatch)),
        Field("env", "env", True, _decode_Name),
        Field("dry_run", "dry_run", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeploy(Payload):
    """The PredictorDeploy schema."""

    spec: Spec
    env: Name
    dry_run: bool | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("spec", "spec", True, _record(Spec)),
        Field("env", "env", True, _decode_Name),
        Field("dry_run", "dry_run", False, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class Change(Payload):
    """The Change schema."""

    field: str
    before: str | None
    after: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("field", "field", True, _text),
        Field("before", "before", True, _union(_text, _null)),
        Field("after", "after", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Deploy(Payload):
    """What a create, update or deploy did; `unchanged` wrote nothing.

    The environment is `starting` until its replica is ready, which `environments.get`
    reports
    """

    predictor: Name
    env: Name
    action: Literal["created", "updated", "unchanged"]
    changes: tuple[Change, ...]
    revision: Revision
    environment: Environment | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictor", "predictor", True, _decode_Name),
        Field("env", "env", True, _decode_Name),
        Field("action", "action", True, _literal("created", "updated", "unchanged")),
        Field("changes", "changes", True, _items(_record(Change))),
        Field("revision", "revision", True, _record(Revision)),
        Field("environment", "environment", True, _union(_record(Environment), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorInfo(Payload):
    """A named, versioned endpoint: its latest revision's spec and each environment."""

    name: Name
    spec: Spec
    resolved: Resolved
    revision: int
    environments: dict[str, Environment]
    origin: Origin
    created: str
    url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("spec", "spec", True, _record(Spec)),
        Field("resolved", "resolved", True, _record(Resolved)),
        Field("revision", "revision", True, _integer),
        Field("environments", "environments", True, _mapping(_record(Environment))),
        Field("origin", "origin", True, _record(Origin)),
        Field("created", "created", True, _text),
        Field("url", "url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorInfoList(Payload):
    """The PredictorInfoList schema."""

    predictors: tuple[PredictorInfo, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictors", "predictors", True, _items(_record(PredictorInfo))),
    )


@dataclass(frozen=True, kw_only=True)
class EstimateQuery(Payload):
    """The EstimateQuery schema."""

    model: ModelField
    context_rows: int
    queries: int
    task: Task | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelField),
        Field("task", "task", False, _decode_Task),
        Field("context_rows", "context_rows", True, _integer),
        Field("queries", "queries", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SourceCreate(Payload):
    """The SourceCreate schema."""

    name: Name
    kind: SourceKind
    config: dict[str, str]
    secret: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("kind", "kind", True, _decode_SourceKind),
        Field("config", "config", True, _mapping(_text)),
        Field("secret", "secret", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SourceTable(Payload):
    """The SourceTable schema."""

    name: str
    rows: int | None
    bytes: int | None
    updated: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("rows", "rows", True, _union(_integer, _null)),
        Field("bytes", "bytes", True, _union(_integer, _null)),
        Field("updated", "updated", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceTables(Payload):
    """The SourceTables schema."""

    tables: tuple[SourceTable, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("tables", "tables", True, _items(_record(SourceTable))),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePreview(Payload):
    """One `table` or a `sql` query; the answer is its first 20 rows."""

    table: str | None = None
    sql: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("table", "table", False, _text),
        Field("sql", "sql", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class KeyDeleted(Payload):
    """The KeyDeleted schema."""

    id: str
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _text),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextReply(Payload):
    """The ContextReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Context

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Context)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextListReply(Payload):
    """The ContextListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ContextList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ContextList)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextVersionsReply(Payload):
    """The ContextVersionsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ContextVersions

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ContextVersions)),
    )


@dataclass(frozen=True, kw_only=True)
class RowsReply(Payload):
    """The RowsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Rows

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Rows)),
    )


@dataclass(frozen=True, kw_only=True)
class ReceivedReply(Payload):
    """The ReceivedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Received

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Received)),
    )


@dataclass(frozen=True, kw_only=True)
class DeletedReply(Payload):
    """The DeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Deleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Deleted)),
    )


@dataclass(frozen=True, kw_only=True)
class CodeReply(Payload):
    """The CodeReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Code

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Code)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentReply(Payload):
    """The ExperimentReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Experiment

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Experiment)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentCreatedReply(Payload):
    """The ExperimentCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ExperimentCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _decode_ExperimentCreated),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentListReply(Payload):
    """The ExperimentListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ExperimentList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ExperimentList)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunReply(Payload):
    """The ExperimentRunReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ExperimentRun

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ExperimentRun)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunCreatedReply(Payload):
    """The ExperimentRunCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ExperimentRunCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _decode_ExperimentRunCreated),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunListReply(Payload):
    """The ExperimentRunListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ExperimentRunList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ExperimentRunList)),
    )


@dataclass(frozen=True, kw_only=True)
class SliceListReply(Payload):
    """The SliceListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SliceList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SliceList)),
    )


@dataclass(frozen=True, kw_only=True)
class HeldRowsReply(Payload):
    """The HeldRowsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: HeldRows

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(HeldRows)),
    )


@dataclass(frozen=True, kw_only=True)
class DeployReply(Payload):
    """The DeployReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Deploy

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Deploy)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorInfoReply(Payload):
    """The PredictorInfoReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorInfo

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorInfo)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorInfoListReply(Payload):
    """The PredictorInfoListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorInfoList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorInfoList)),
    )


@dataclass(frozen=True, kw_only=True)
class EstimateReply(Payload):
    """The EstimateReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Estimate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Estimate)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceReply(Payload):
    """The SourceReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Source

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Source)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceTablesReply(Payload):
    """The SourceTablesReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SourceTables

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SourceTables)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyDeletedReply(Payload):
    """The KeyDeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: KeyDeleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(KeyDeleted)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextCreateRequest(Payload):
    """The ContextCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ContextCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ContextCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextUploadRequest(Payload):
    """The ContextUploadRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ContextUpload

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ContextUpload)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentCreateRequest(Payload):
    """The ExperimentCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ExperimentCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ExperimentCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentUpdateRequest(Payload):
    """The ExperimentUpdateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ExperimentUpdate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ExperimentUpdate)),
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentRunCreateRequest(Payload):
    """The ExperimentRunCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ExperimentRunCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ExperimentRunCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorUpdateRequest(Payload):
    """The PredictorUpdateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: PredictorUpdate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(PredictorUpdate)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeployRequest(Payload):
    """The PredictorDeployRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: PredictorDeploy

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(PredictorDeploy)),
    )


@dataclass(frozen=True, kw_only=True)
class EstimateQueryRequest(Payload):
    """The EstimateQueryRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: EstimateQuery

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(EstimateQuery)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceCreateRequest(Payload):
    """The SourceCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SourceCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SourceCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePreviewRequest(Payload):
    """The SourcePreviewRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SourcePreview

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SourcePreview)),
    )


ROUTES: tuple[Route, ...] = (
    Route(
        "predictors.predict",
        "POST",
        "/v1/projects/{project}/predictors/{name}/predict",
        QueriesRequest,
        ResultReply,
        "application/json",
    ),
    Route(
        "predict",
        "POST",
        "/v1/projects/{project}/predict",
        ManifestRequest,
        ResultReply,
        "application/json",
    ),
    Route(
        "predictions.get",
        "GET",
        "/v1/projects/{project}/predictions/{id}",
        None,
        ResultReply,
        None,
    ),
    Route(
        "environments.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments",
        None,
        EnvironmentListReply,
        None,
    ),
    Route(
        "environments.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}",
        None,
        EnvironmentReply,
        None,
    ),
    Route(
        "environments.promote",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote",
        PromoteRequest,
        EnvironmentReply,
        "application/json",
    ),
    Route(
        "environments.rollback",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback",
        PromoteRequest,
        EnvironmentReply,
        "application/json",
    ),
    Route(
        "environments.metrics",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics",
        None,
        MetricsReply,
        None,
    ),
    Route(
        "environments.logs",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs",
        None,
        LogsReply,
        None,
    ),
    Route(
        "environments.logs.histogram",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs/histogram",
        None,
        LogHistogramReply,
        None,
    ),
    Route(
        "predictors.delete",
        "POST",
        "/v1/projects/{project}/predictors/{name}/delete",
        EmptyRequest,
        PredictorDeletedReply,
        "application/json",
    ),
    Route(
        "contexts.list",
        "GET",
        "/v1/projects/{project}/contexts",
        None,
        ContextListReply,
        None,
    ),
    Route(
        "contexts.create",
        "POST",
        "/v1/projects/{project}/contexts",
        ContextCreateRequest,
        ContextReply,
        "application/json",
    ),
    Route(
        "contexts.get",
        "GET",
        "/v1/projects/{project}/contexts/{name}",
        None,
        ContextReply,
        None,
    ),
    Route(
        "contexts.versions",
        "GET",
        "/v1/projects/{project}/contexts/{name}/versions",
        None,
        ContextVersionsReply,
        None,
    ),
    Route(
        "contexts.upload",
        "POST",
        "/v1/projects/{project}/contexts/{name}/versions",
        ContextUploadRequest,
        ContextReply,
        "application/json",
    ),
    Route(
        "contexts.bytes",
        "POST",
        "/v1/projects/{project}/contexts/{name}/versions/{version}/bytes",
        None,
        ReceivedReply,
        "application/octet-stream",
    ),
    Route(
        "contexts.refresh",
        "POST",
        "/v1/projects/{project}/contexts/{name}/refresh",
        EmptyRequest,
        ContextReply,
        "application/json",
    ),
    Route(
        "contexts.append",
        "POST",
        "/v1/projects/{project}/contexts/{name}/append",
        LiveContextAppendRequest,
        ReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.label",
        "POST",
        "/v1/projects/{project}/contexts/{name}/label",
        LiveContextLabelRequest,
        ReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.rows",
        "GET",
        "/v1/projects/{project}/contexts/{name}/rows",
        None,
        RowsReply,
        None,
    ),
    Route(
        "contexts.delete",
        "POST",
        "/v1/projects/{project}/contexts/{name}/delete",
        EmptyRequest,
        DeletedReply,
        "application/json",
    ),
    Route(
        "contexts.code",
        "GET",
        "/v1/projects/{project}/contexts/{name}/code",
        None,
        CodeReply,
        None,
    ),
    Route(
        "experiments.list",
        "GET",
        "/v1/projects/{project}/experiments",
        None,
        ExperimentListReply,
        None,
    ),
    Route(
        "experiments.create",
        "POST",
        "/v1/projects/{project}/experiments",
        ExperimentCreateRequest,
        ExperimentCreatedReply,
        "application/json",
    ),
    Route(
        "experiments.get",
        "GET",
        "/v1/projects/{project}/experiments/{name}",
        None,
        ExperimentReply,
        None,
    ),
    Route(
        "experiments.update",
        "POST",
        "/v1/projects/{project}/experiments/{name}/update",
        ExperimentUpdateRequest,
        ExperimentReply,
        "application/json",
    ),
    Route(
        "experiments.runs",
        "GET",
        "/v1/projects/{project}/experiments/{name}/runs",
        None,
        ExperimentRunListReply,
        None,
    ),
    Route(
        "experiments.run",
        "POST",
        "/v1/projects/{project}/experiments/{name}/runs",
        ExperimentRunCreateRequest,
        ExperimentRunCreatedReply,
        "application/json",
    ),
    Route(
        "experiments.runs.get",
        "GET",
        "/v1/projects/{project}/experiments/{name}/runs/{number}",
        None,
        ExperimentRunReply,
        None,
    ),
    Route(
        "experiments.cancel",
        "POST",
        "/v1/projects/{project}/experiments/{name}/runs/{number}/cancel",
        EmptyRequest,
        ExperimentRunReply,
        "application/json",
    ),
    Route(
        "experiments.rows",
        "GET",
        "/v1/projects/{project}/experiments/{name}/runs/{number}/rows",
        None,
        HeldRowsReply,
        None,
    ),
    Route(
        "experiments.slices",
        "GET",
        "/v1/projects/{project}/experiments/{name}/runs/{number}/slices",
        None,
        SliceListReply,
        None,
    ),
    Route(
        "experiments.delete",
        "POST",
        "/v1/projects/{project}/experiments/{name}/delete",
        EmptyRequest,
        DeletedReply,
        "application/json",
    ),
    Route(
        "experiments.code",
        "GET",
        "/v1/projects/{project}/experiments/{name}/code",
        None,
        CodeReply,
        None,
    ),
    Route(
        "predictors.list",
        "GET",
        "/v1/projects/{project}/predictors",
        None,
        PredictorInfoListReply,
        None,
    ),
    Route(
        "predictors.create",
        "POST",
        "/v1/projects/{project}/predictors",
        PredictorCreateRequest,
        DeployReply,
        "application/json",
    ),
    Route(
        "predictors.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}",
        None,
        PredictorInfoReply,
        None,
    ),
    Route(
        "predictors.update",
        "POST",
        "/v1/projects/{project}/predictors/{name}/update",
        PredictorUpdateRequest,
        DeployReply,
        "application/json",
    ),
    Route(
        "predictors.deploy",
        "POST",
        "/v1/projects/{project}/predictors/{name}/deploy",
        PredictorDeployRequest,
        DeployReply,
        "application/json",
    ),
    Route(
        "revisions.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions",
        None,
        RevisionListReply,
        None,
    ),
    Route(
        "revisions.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions/{number}",
        None,
        RevisionReply,
        None,
    ),
    Route(
        "predictors.code",
        "GET",
        "/v1/projects/{project}/predictors/{name}/code",
        None,
        CodeReply,
        None,
    ),
    Route(
        "estimate",
        "POST",
        "/v1/projects/{project}/estimate",
        EstimateQueryRequest,
        EstimateReply,
        "application/json",
    ),
    Route(
        "sources.list",
        "GET",
        "/v1/projects/{project}/sources",
        None,
        SourceListReply,
        None,
    ),
    Route(
        "sources.create",
        "POST",
        "/v1/projects/{project}/sources",
        SourceCreateRequest,
        SourceReply,
        "application/json",
    ),
    Route(
        "sources.get",
        "GET",
        "/v1/projects/{project}/sources/{source}",
        None,
        SourceReply,
        None,
    ),
    Route(
        "sources.tables",
        "GET",
        "/v1/projects/{project}/sources/{source}/tables",
        None,
        SourceTablesReply,
        None,
    ),
    Route(
        "sources.preview",
        "POST",
        "/v1/projects/{project}/sources/{source}/preview",
        SourcePreviewRequest,
        RowsReply,
        "application/json",
    ),
    Route(
        "sources.test",
        "POST",
        "/v1/projects/{project}/sources/{source}/test",
        EmptyRequest,
        SourceReply,
        "application/json",
    ),
    Route(
        "sources.delete",
        "POST",
        "/v1/projects/{project}/sources/{source}/delete",
        EmptyRequest,
        DeletedReply,
        "application/json",
    ),
    Route(
        "keys.delete",
        "POST",
        "/v1/projects/{project}/keys/{key}/delete",
        KeyRefRequest,
        KeyDeletedReply,
        "application/json",
    ),
    Route(
        "models.list",
        "GET",
        "/v1/projects/{project}/models",
        None,
        ModelListReply,
        None,
    ),
    Route(
        "models.get",
        "GET",
        "/v1/projects/{project}/models/{model}",
        None,
        ModelReply,
        None,
    ),
    Route("ready", "GET", "/readyz", None, Health, None),
    Route("live", "GET", "/livez", None, Health, None),
    Route("projects.list", "GET", "/v1/projects", None, ProjectListReply, None),
    Route(
        "projects.create",
        "POST",
        "/v1/projects",
        ProjectCreateRequest,
        ProjectReply,
        "application/json",
    ),
    Route("keys.list", "GET", "/v1/projects/{project}/keys", None, KeyListReply, None),
    Route(
        "keys.create",
        "POST",
        "/v1/projects/{project}/keys",
        KeyCreateRequest,
        KeyReply,
        "application/json",
    ),
    Route(
        "keys.revoke",
        "POST",
        "/v1/projects/{project}/keys/{key}/revoke",
        KeyRefRequest,
        KeyRevokedReply,
        "application/json",
    ),
    Route(
        "keys.rotate",
        "POST",
        "/v1/projects/{project}/keys/{key}/rotate",
        KeyRefRequest,
        KeyReply,
        "application/json",
    ),
    Route(
        "secrets.list",
        "GET",
        "/v1/projects/{project}/secrets",
        None,
        SecretListReply,
        None,
    ),
    Route(
        "secrets.create",
        "POST",
        "/v1/projects/{project}/secrets",
        SecretCreateRequest,
        SecretReply,
        "application/json",
    ),
    Route(
        "secrets.get",
        "GET",
        "/v1/projects/{project}/secrets/{secret}",
        None,
        SecretReply,
        None,
    ),
    Route(
        "secrets.rotate",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/rotate",
        SecretRotateRequest,
        SecretReply,
        "application/json",
    ),
    Route(
        "secrets.delete",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/delete",
        SecretRefRequest,
        SecretDeletedReply,
        "application/json",
    ),
    Route(
        "sources.prepare",
        "POST",
        "/v1/projects/{project}/sources:prepare",
        SourcePrepareRequest,
        SourcePreparedReply,
        "application/json",
    ),
    Route(
        "snapshots.list",
        "GET",
        "/v1/projects/{project}/snapshots",
        None,
        SnapshotListReply,
        None,
    ),
    Route(
        "snapshots.create",
        "POST",
        "/v1/projects/{project}/snapshots",
        SnapshotCreateRequest,
        SnapshotReply,
        "application/json",
    ),
    Route(
        "snapshots.put",
        "PUT",
        "/v1/projects/{project}/snapshots",
        None,
        HeldSnapshotReply,
        "application/vnd.apache.arrow.stream",
    ),
    Route(
        "snapshots.create_synthetic",
        "POST",
        "/v1/projects/{project}/snapshots/synthetic",
        SyntheticBatchCreateRequest,
        SyntheticBatchCreatedReply,
        "application/json",
    ),
    Route(
        "snapshots.synthetic",
        "GET",
        "/v1/projects/{project}/snapshots/synthetic/{name}",
        None,
        SyntheticBatchReply,
        None,
    ),
    Route(
        "snapshots.get",
        "GET",
        "/v1/projects/{project}/snapshots/{snapshot}",
        None,
        SnapshotReply,
        None,
    ),
    Route(
        "snapshots.update",
        "POST",
        "/v1/projects/{project}/snapshots/{snapshot}/update",
        SnapshotUpdateRequest,
        SnapshotReply,
        "application/json",
    ),
    Route(
        "snapshots.delete",
        "POST",
        "/v1/projects/{project}/snapshots/{snapshot}/delete",
        EmptyRequest,
        SnapshotReply,
        "application/json",
    ),
    Route("runs.list", "GET", "/v1/projects/{project}/runs", None, RunListReply, None),
    Route(
        "runs.create",
        "POST",
        "/v1/projects/{project}/runs",
        RunCreateRequest,
        RunReply,
        "application/json",
    ),
    Route("runs.get", "GET", "/v1/projects/{project}/runs/{run}", None, RunReply, None),
    Route(
        "runs.cancel",
        "POST",
        "/v1/projects/{project}/runs/{run}/cancel",
        RunCancelRequest,
        RunReply,
        "application/json",
    ),
    Route(
        "runs.lineage",
        "GET",
        "/v1/projects/{project}/runs/{run}/lineage",
        None,
        RunLineageReply,
        None,
    ),
    Route("auth.status", "GET", "/v1/auth/status", None, AuthStatusReply, None),
    Route(
        "auth.logout",
        "POST",
        "/v1/auth/logout",
        EmptyRequest,
        LoggedOutReply,
        "application/json",
    ),
    Route("workspace.get", "GET", "/v1/workspace", None, ProjectListReply, None),
    Route(
        "organizations.list",
        "GET",
        "/v1/organizations",
        None,
        OrganizationListReply,
        None,
    ),
    Route(
        "organizations.create",
        "POST",
        "/v1/organizations",
        OrganizationCreateRequest,
        OrganizationReply,
        "application/json",
    ),
    Route("sso.get", "GET", "/v1/projects/{project}/sso", None, SsoReply, None),
    Route(
        "sso.configure",
        "POST",
        "/v1/projects/{project}/sso",
        SsoConfigureRequest,
        SsoReply,
        "application/json",
    ),
    Route(
        "audit.list",
        "GET",
        "/v1/projects/{project}/audit",
        None,
        AuditLogReply,
        None,
    ),
    Route(
        "device.start",
        "POST",
        "/v1/device-logins",
        DeviceLoginStart,
        DeviceLoginGrant,
        "application/json",
    ),
    Route(
        "device.token",
        "POST",
        "/v1/device-logins/token",
        DeviceLoginPoll,
        DeviceLoginToken,
        "application/json",
    ),
    Route("onboarding.get", "GET", "/v1/onboarding", None, OnboardingReply, None),
    Route("workspaces.list", "GET", "/v1/workspaces", None, WorkspaceListReply, None),
    Route(
        "workspaces.create",
        "POST",
        "/v1/workspaces",
        WorkspaceCreateRequest,
        WorkspaceCreatedReply,
        "application/json",
    ),
    Route(
        "workspaces.get",
        "GET",
        "/v1/workspaces/{workspace}",
        None,
        WorkspaceDetailsReply,
        None,
    ),
    Route(
        "workspaces.update",
        "POST",
        "/v1/workspaces/{workspace}",
        WorkspaceRenameRequest,
        WorkspaceSummaryReply,
        "application/json",
    ),
    Route(
        "workspaces.join",
        "POST",
        "/v1/workspaces/{workspace}/join",
        WorkspaceRefRequest,
        WorkspaceJoinReply,
        "application/json",
    ),
    Route(
        "workspaces.members",
        "GET",
        "/v1/workspaces/{workspace}/members",
        None,
        WorkspaceMemberListReply,
        None,
    ),
    Route(
        "members.update",
        "POST",
        "/v1/workspaces/{workspace}/members/role",
        MemberRoleChangeRequest,
        MemberRoleChangedReply,
        "application/json",
    ),
    Route(
        "members.remove",
        "POST",
        "/v1/workspaces/{workspace}/members/remove",
        MemberRemoveRequest,
        MemberRemovedReply,
        "application/json",
    ),
    Route(
        "invitations.list",
        "GET",
        "/v1/workspaces/{workspace}/invitations",
        None,
        InvitationListReply,
        None,
    ),
    Route(
        "invitations.create",
        "POST",
        "/v1/workspaces/{workspace}/invitations",
        InvitationCreateRequest,
        InvitationCreatedReply,
        "application/json",
    ),
    Route(
        "invitations.revoke",
        "POST",
        "/v1/workspaces/{workspace}/invitations/{id}/revoke",
        InvitationRevokeRequest,
        InvitationViewReply,
        "application/json",
    ),
    Route(
        "requests.list",
        "GET",
        "/v1/workspaces/{workspace}/join-requests",
        None,
        JoinRequestListReply,
        None,
    ),
    Route(
        "requests.approve",
        "POST",
        "/v1/workspaces/{workspace}/join-requests/{id}/approve",
        JoinRequestApproveRequest,
        JoinRequestApprovedReply,
        "application/json",
    ),
    Route(
        "requests.deny",
        "POST",
        "/v1/workspaces/{workspace}/join-requests/{id}/deny",
        JoinRequestDenyRequest,
        JoinRequestDeniedReply,
        "application/json",
    ),
    Route(
        "domains.list",
        "GET",
        "/v1/workspaces/{workspace}/domains",
        None,
        JoinDomainListReply,
        None,
    ),
    Route(
        "domains.configure",
        "POST",
        "/v1/workspaces/{workspace}/domains",
        JoinDomainConfigureRequest,
        JoinDomainConfiguredReply,
        "application/json",
    ),
    Route(
        "invitations.preview",
        "POST",
        "/v1/invitations/preview",
        InvitationTokenRequest,
        InvitationPreviewReply,
        "application/json",
    ),
    Route(
        "invitations.accept",
        "POST",
        "/v1/invitations/accept",
        InvitationTokenRequest,
        InvitationAcceptedReply,
        "application/json",
    ),
    Route(
        "requests.cancel",
        "POST",
        "/v1/join-requests/{id}/cancel",
        JoinRequestRefRequest,
        JoinRequestViewReply,
        "application/json",
    ),
    Route(
        "waitlist.join",
        "POST",
        "/v1/waitlist",
        WaitlistJoinRequest,
        WaitlistStatusReply,
        "application/json",
    ),
    Route(
        "waitlist.get",
        "GET",
        "/v1/waitlist/{id}",
        None,
        WaitlistEntryViewReply,
        None,
    ),
    Route(
        "waitlist.approve",
        "POST",
        "/v1/waitlist/{id}/approve",
        WaitlistRefRequest,
        WaitlistEntryViewReply,
        "application/json",
    ),
    Route(
        "device.preview",
        "POST",
        "/v1/device-logins/preview",
        DeviceLoginCodeRequest,
        DeviceLoginPreviewReply,
        "application/json",
    ),
    Route(
        "device.approve",
        "POST",
        "/v1/device-logins/approve",
        DeviceLoginApproveRequest,
        DeviceLoginApprovedReply,
        "application/json",
    ),
    Route(
        "device.deny",
        "POST",
        "/v1/device-logins/deny",
        DeviceLoginCodeRequest,
        DeviceLoginDeniedReply,
        "application/json",
    ),
)
