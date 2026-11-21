# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The API contract, `api/openapi.yaml`: its routes, schemas and examples, for the fake service.

The fake answers with the spec's examples where the spec has them, and
checks every request the SDK sends and every reply it gives against the
route's schemas, so a test fails when the SDK, the fake and the contract
part ways.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SPEC_PATH = Path(__file__).resolve().parents[1] / "api" / "openapi.yaml"
BASE = "urn:gf-openapi"


# What the contract requires that the plane does not serve yet, so the fake
# leaves it out as the plane does: a model's pinned `model`, `limits` and
# `warm_shapes` (the plane sends `bounds` alone).
UNSERVED = {"Model": ("model", "limits", "warm_shapes")}


@cache
def spec() -> dict[str, Any]:
    """The contract, with what the plane does not serve yet (`UNSERVED`) not required."""
    found = yaml.safe_load(SPEC_PATH.read_text())
    for name, fields in UNSERVED.items():
        schema = found["components"]["schemas"][name]
        schema["required"] = [f for f in schema["required"] if f not in fields]
    return found


@dataclass(frozen=True)
class Operation:
    """One route of the contract."""

    method: str
    template: str
    pattern: re.Pattern[str]
    body: dict[str, Any]

    @property
    def id(self) -> str:
        return self.body["operationId"]

    def bound(self, path: str) -> dict[str, str]:
        """The path's parameters by name, as the template binds them."""
        found = self.pattern.match(path)
        assert found is not None, f"{path} is not {self.template}"
        return {key: unquote(value) for key, value in found.groupdict().items()}

    def parameters(self, where: str) -> dict[str, dict[str, Any]]:
        """The route's parameters `in` the query, a header or the path, by name."""
        found = (_resolved(p) for p in self.body.get("parameters", ()))
        return {p["name"]: p for p in found if p["in"] == where}

    def media(self) -> dict[str, Any]:
        """The request body's media types, none when the route takes no body."""
        request = self.body.get("requestBody")
        return _resolved(request).get("content", {}) if request else {}


@cache
def operations() -> list[Operation]:
    """Every route, the most literal paths first, so `/snapshots/synthetic` wins over `/snapshots/{snapshot}`."""
    found = []
    for template, item in spec()["paths"].items():
        pattern = re.compile(
            "^"
            + re.sub(r"\\\{([^}]+)\\\}", r"(?P<\1>[^/]+)", re.escape(template))
            + "$"
        )
        for method, body in item.items():
            if isinstance(body, dict) and "operationId" in body:
                found.append(Operation(method.upper(), template, pattern, body))
    return sorted(found, key=lambda op: -len(re.sub(r"\{[^}]+\}", "", op.template)))


def operation(method: str, path: str) -> Operation | None:
    """The route a request is for, or None when the contract has none."""
    return next(
        (op for op in operations() if op.method == method and op.pattern.match(path)),
        None,
    )


def _resolved(item: dict[str, Any]) -> dict[str, Any]:
    ref = item.get("$ref")
    if ref is None:
        return item
    found: Any = spec()
    for part in ref.removeprefix("#/").split("/"):
        found = found[part]
    return found


@cache
def _registry() -> Registry:
    resource = Resource.from_contents(spec(), default_specification=DRAFT202012)
    return Registry().with_resource(BASE, resource)


def _problems(schema: dict[str, Any], value: Any) -> list[str]:
    if "$ref" in schema:
        schema = {"$ref": BASE + schema["$ref"]}
    validator = Draft202012Validator(schema, registry=_registry())
    return [
        f"{'/'.join(map(str, error.absolute_path)) or 'the body'}: {error.message[:300]}"
        for error in validator.iter_errors(value)
    ]


def _typed(schema: dict[str, Any], value: str) -> Any:
    """A query or header value as the type its schema names, for the schema to check."""
    schema = _resolved(schema)
    kind = schema.get("type")
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            return float(value)
    except ValueError:
        return value
    if kind == "boolean" and value in ("true", "false"):
        return value == "true"
    return value


def parameter_problems(op: Operation, where: str, values: dict[str, str]) -> list[str]:
    """How a request's query, or headers, depart from the route's parameters.

    A query may hold only the parameters the route declares; headers may
    hold others. Each declared one is checked against its schema, and a
    required one must be there.
    """
    declared = op.parameters(where)
    if where == "header":
        values = {
            name: values[name.lower()] for name in declared if name.lower() in values
        }
    problems = [f"the {where} parameter `{name}` is not declared" for name in values if name not in declared]  # fmt: skip
    for name, parameter in declared.items():
        if name not in values:
            if parameter.get("required"):
                problems.append(f"the {where} parameter `{name}` is required")
            continue
        problems += [
            f"the {where} parameter `{name}`: {p}"
            for p in _problems(parameter["schema"], _typed(parameter["schema"], values[name]))
        ]  # fmt: skip
    return problems


def media_problems(op: Operation, content_type: str, body: bytes) -> list[str]:
    """How a request's body and its Content-Type depart from what the route takes."""
    request = op.body.get("requestBody")
    if not body:
        required = request is not None and _resolved(request).get("required")
        return ["the route requires a body"] if required else []
    kind = content_type.split(";")[0].strip()
    if kind not in op.media():
        return [f"the route takes {', '.join(op.media()) or 'no body'}, not `{kind}`"]
    return []


def request_problems(op: Operation, body: Any) -> list[str]:
    """How a JSON request body departs from the route's schema."""
    request = op.body.get("requestBody")
    if request is None:
        return []
    content = _resolved(request).get("content", {}).get("application/json")
    return _problems(content["schema"], body) if content else []


def reply_problems(op: Operation, status: int, body: Any) -> list[str]:
    """How a JSON reply departs from the route's schema for its status.

    An error the route declares no reply for is still an error reply: the
    envelope's, or a device login's outside it.
    """
    responses = op.body.get("responses", {})
    reply = responses.get(str(status)) or responses.get("default")
    if reply is None and status < 400:
        return [f"the contract has no {status} reply"]
    if reply is None:
        error = (
            "ErrorReply" if op.body.get("x-gf-envelope", True) else "LoginErrorReply"
        )
        return _problems({"$ref": f"#/components/schemas/{error}"}, body)
    content = _resolved(reply).get("content", {}).get("application/json")
    return _problems(content["schema"], body) if content else []


def example(operation_id: str, name: str | None = None, *, status: str = "200") -> Any:
    """The `result` of a reply example (the reply itself, outside the envelope), a copy to change freely."""
    op = next(op for op in operations() if op.id == operation_id)
    content = _resolved(op.body["responses"][status])["content"]["application/json"]
    examples = content["examples"]
    value = examples[name] if name is not None else next(iter(examples.values()))
    reply = value["value"]
    return copy.deepcopy(reply.get("result", reply))


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)
