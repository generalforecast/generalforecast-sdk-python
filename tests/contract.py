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

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SPEC_PATH = Path(__file__).resolve().parents[1] / "api" / "openapi.yaml"
BASE = "urn:gf-openapi"


@cache
def spec() -> dict[str, Any]:
    """The contract."""
    return yaml.safe_load(SPEC_PATH.read_text())


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


@cache
def operations() -> list[Operation]:
    """Every route, the most literal paths first, so `/snapshots/synthetic` wins over `/snapshots/{snapshot}`."""
    found = []
    for template, item in spec()["paths"].items():
        pattern = re.compile(
            "^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(template)) + "$"
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


def request_problems(op: Operation, body: Any) -> list[str]:
    """How a JSON request body departs from the route's schema."""
    request = op.body.get("requestBody")
    if request is None:
        return []
    content = _resolved(request).get("content", {}).get("application/json")
    return _problems(content["schema"], body) if content else []


# Where the contract's own examples break its schemas: a schema that adds
# fields with `allOf` to one that allows no others. Each is reported to the
# engine; replies on these routes are checked for everything else.
KNOWN_DEFECTS = {
    "snapshots.synthetic": "result: Additional properties are not allowed ('counts', 'options', 'seed', 'tables' were unexpected)",
    "workspaces.get": "result: Additional properties are not allowed ('created_at', 'organization' were unexpected)",
}


def reply_problems(op: Operation, status: int, body: Any) -> list[str]:
    """How a JSON reply departs from the route's schema for its status."""
    responses = op.body.get("responses", {})
    reply = responses.get(str(status)) or responses.get("default")
    if reply is None:
        if status >= 400:
            return []
        return [f"the contract has no {status} reply"]
    content = _resolved(reply).get("content", {}).get("application/json")
    found = _problems(content["schema"], body) if content else []
    return [p for p in found if p != KNOWN_DEFECTS.get(op.id)]


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
