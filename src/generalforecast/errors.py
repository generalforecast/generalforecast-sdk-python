# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The errors the SDK raises, one class per category.

Every error has `message`, `hint`, `code`, `reason` and `request_id`, and
`recoverable`: true when the caller can fix the cause (bad input, a missing
resource, a conflict, a login, a payment), false for real failures
(`Internal`, `Unavailable`, a failed run, training or deploy). The CLI
colours by it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = [
    "AmbiguousReference",
    "CapacityExceeded",
    "Conflict",
    "Error",
    "Internal",
    "InvalidInput",
    "ModelNotReady",
    "NotFound",
    "PaymentRequired",
    "Timeout",
    "Unauthorized",
    "Unavailable",
    "Unsupported",
]

_ESCAPES = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])|\x9b[0-?]*[ -/]*[@-~]"
)


def printable(message: str) -> str:
    """Removes terminal escapes and non-printable characters from a message."""
    return "".join(
        " " if character in "\t\n\r" else character
        for character in _ESCAPES.sub("", message)
        if character in "\t\n\r" or character.isprintable()
    ).strip()


def reason_of(code: str) -> str:
    """A code's snake case `reason`, as the service spells it."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", code).lower()


class Error(Exception):
    """A General Forecast error.

    Attributes:
        message: One lowercase line naming the problem.
        hint: What to do about it, if there is a short answer.
        note: The cause, when the message alone does not say it.
        code: The stable code, such as `ValidationError` or `NotFound`.
        reason: The finer reason in snake case, such as `context_in_use`.
        request_id: The service's request id, when a request failed.
        status: The HTTP status, when a request failed.
        param: The field or column the error is about, if any.
        limit: The bound a capacity error enforces, if any.
        actual: The value past that bound, if any.
        recoverable: Whether the caller can fix it and try again.
    """

    code: str = "Error"
    recoverable: bool = True
    status: int | None = None
    _default_status: ClassVar[int | None] = None

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        note: str | None = None,
        code: str | None = None,
        reason: str | None = None,
        request_id: str | None = None,
        status: int | None = None,
        param: str | None = None,
        limit: Any = None,
        actual: Any = None,
        recoverable: bool | None = None,
    ) -> None:
        """Keeps every field; `code` and `recoverable` default to the class's."""
        super().__init__(message)
        self.message = printable(str(message))
        self.hint = hint
        self.note = note
        self.code = code or type(self).code
        self.reason = reason
        self.request_id = request_id
        self.status = status if status is not None else self._default_status
        self.param = param
        self.limit = limit
        self.actual = actual
        if recoverable is not None:
            self.recoverable = recoverable

    def __str__(self) -> str:
        """The message, then its note and hint on their own lines."""
        lines = [self.message]
        if self.note:
            lines.append(f"note: {self.note}")
        if self.hint:
            lines.append(f"help: {self.hint}")
        if self.request_id:
            lines.append(f"request: {self.request_id}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """The class and message."""
        return f"{type(self).__name__}({self.message!r})"

    def to_dict(self) -> dict[str, Any]:
        """The fields that are set, as `--json` prints an error."""
        fields = {
            "code": self.code,
            "message": self.message,
            "reason": self.reason,
            "hint": self.hint,
            "note": self.note,
            "param": self.param,
            "limit": self.limit,
            "actual": self.actual,
            "request_id": self.request_id,
            "recoverable": self.recoverable,
        }
        return {key: value for key, value in fields.items() if value is not None}

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles by class and fields."""
        state = dict(vars(self))
        message = state.pop("message")
        return (_restore, (type(self), message, state))


def _restore(cls: type[Error], message: str, state: dict[str, Any]) -> Error:
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    error.message = message
    error.__dict__.update(state)
    return error


class InvalidInput(Error):
    """An argument, a table or a spec the service cannot take as given."""

    code = "ValidationError"
    _default_status = 422


class NotFound(Error):
    """No resource by that name in the project."""

    code = "NotFound"
    _default_status = 404


class Unauthorized(Error):
    """Not logged in, or the key lacks the permission."""

    code = "Unauthorized"
    _default_status = 401


class CapacityExceeded(Error):
    """Past a size the service or this client holds, or a rate limit."""

    code = "CapacityExceeded"
    _default_status = 429


class ModelNotReady(Error):
    """A model or deployment is not ready yet; trying later works."""

    code = "ModelNotReady"
    _default_status = 503


class AmbiguousReference(Error):
    """A name that matches more than one thing, such as a short model name with no task."""

    code = "AmbiguousReference"
    _default_status = 422


class Conflict(Error):
    """The resource's state forbids the change, such as deleting a context a predictor serves."""

    code = "Conflict"
    _default_status = 409


class Unavailable(Error):
    """The service could not be reached or did not answer."""

    code = "Unavailable"
    recoverable = False
    _default_status = 503


class Unsupported(Error):
    """The service does not offer this here."""

    code = "Unsupported"
    _default_status = 501


class PaymentRequired(Error):
    """The workspace's plan or credits do not cover the call."""

    code = "PaymentRequired"
    _default_status = 402


class Internal(Error):
    """The service failed; the request id lets support find it."""

    code = "Internal"
    recoverable = False
    _default_status = 500


class Timeout(Error):
    """A wait ran past its timeout; the work carries on on the service."""

    code = "Timeout"


_BY_CODE: dict[str, type[Error]] = {
    "ValidationError": InvalidInput,
    "InvalidInput": InvalidInput,
    "NotFound": NotFound,
    "Unauthorized": Unauthorized,
    "CapacityExceeded": CapacityExceeded,
    "ModelNotReady": ModelNotReady,
    "AmbiguousReference": AmbiguousReference,
    "Conflict": Conflict,
    "AlreadyExists": Conflict,
    "ContextInUse": Conflict,
    "Unavailable": Unavailable,
    "Unsupported": Unsupported,
    "PaymentRequired": PaymentRequired,
    "Internal": Internal,
    "InternalError": Internal,
    "Timeout": Timeout,
}


def category(code: str | None, status: int | None = None) -> type[Error]:
    """The class an error with this wire code, or else this status, belongs to."""
    if code in _BY_CODE:
        return _BY_CODE[code]
    if status == 404 or (code or "").endswith("NotFound"):
        return NotFound
    if status in (401, 403):
        return Unauthorized
    if status == 402:
        return PaymentRequired
    if status == 409:
        return Conflict
    if status == 429:
        return CapacityExceeded
    if status == 501:
        return Unsupported
    if status in (502, 503, 504):
        return Unavailable
    if status is not None and status >= 500:
        return Internal
    return InvalidInput


def from_wire(
    body: Mapping[str, Any] | None,
    status: int | None,
    request_id: str | None = None,
) -> Error:
    """The error a service's error object describes."""
    body = body if isinstance(body, Mapping) else {}
    code = body.get("code") if isinstance(body.get("code"), str) else None
    message = body.get("message")
    if not isinstance(message, str) or not message:
        message = f"the service answered {status}" if status else "the request failed"
    cls = category(code, status)
    text = {
        key: body[key] if isinstance(body.get(key), str) else None
        for key in ("hint", "reason", "param")
    }
    note = body.get("note", body.get("because"))
    return cls(
        message[:512],
        hint=text["hint"],
        note=note if isinstance(note, str) else None,
        code=code or cls.code,
        reason=text["reason"],
        request_id=body.get("request_id")
        if isinstance(body.get("request_id"), str)
        else request_id,
        status=status,
        param=text["param"],
        limit=body.get("limit"),
        actual=body.get("actual"),
    )


def failed(
    what: str,
    error: Any,
    *,
    code: str,
    request_id: str | None = None,
    hint: str | None = None,
) -> Error:
    """A run, training or deploy that ended `failed`: not recoverable by the caller.

    `what` is the message. The resource's `error`, a message or an error
    object, is the note; an object's code picks the class, and its hint
    comes before `hint`.
    """
    cause = from_wire(error, None, request_id) if isinstance(error, Mapping) else None
    if cause is None:
        note = error if isinstance(error, str) and error else None
        return Error(
            what,
            note=note,
            hint=hint,
            code=code,
            request_id=request_id,
            recoverable=False,
        )
    cls = _BY_CODE.get(str(error.get("code")), Error)
    return cls(
        what,
        note=cause.message,
        hint="; ".join(h for h in (cause.hint, hint) if h) or None,
        code=cause.code if cls is not Error else code,
        reason=cause.reason,
        request_id=cause.request_id,
        recoverable=False,
    )
