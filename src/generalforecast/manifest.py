# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""A predictor directory: its manifest `gf.yaml` and its lock `gf.lock`.

The manifest holds intent. The service reads it whole: `gf check` sends its
text and writes the `gf.lock` the service answers with, and `gf push` sends
both. The SDK reads only what it needs to send them: the predictor's name
and the snapshot file its context names.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from generalforecast.errors import Error, InvalidInput, NotFound
from generalforecast.validation import NAME

MANIFEST = "gf.yaml"
LOCK = "gf.lock"
# The largest manifest the SDK reads.
MAX_MANIFEST_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Manifest:
    """A predictor directory's `gf.yaml`: its text as written and the document it holds."""

    root: Path
    text: str
    document: Mapping[str, Any]

    @property
    def name(self) -> str:
        """The predictor's name."""
        return self.document["name"]

    @property
    def model(self) -> str | None:
        """The model the manifest names, if any."""
        model = self.document.get("model")
        return model if isinstance(model, str) else None

    @property
    def context(self) -> Mapping[str, Any]:
        """The `context` section, empty when the manifest has none."""
        context = self.document.get("context")
        return context if isinstance(context, Mapping) else {}

    @property
    def context_kind(self) -> str:
        """snapshot, request or live."""
        for kind in ("snapshot", "request", "live"):
            if kind in self.context:
                return kind
        return "snapshot"

    @property
    def snapshot(self) -> str | None:
        """The snapshot file, relative to the directory, when the context is one."""
        value = self.context.get("snapshot")
        return value if isinstance(value, str) else None


def load(root: str | os.PathLike[str]) -> Manifest:
    """Reads `gf.yaml` in a predictor directory.

    Raises:
        NotFound: When the directory holds no `gf.yaml`.
        InvalidInput: When it is not a YAML mapping with a valid `name`.
    """
    root = Path(root).expanduser()
    path = root / MANIFEST
    if not path.is_file():
        raise NotFound(
            f"no {MANIFEST} in {root}",
            hint="run `gf init` to write one",
            param="manifest",
        )
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise InvalidInput(f"{MANIFEST} is larger than 1 MiB", param="manifest")
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise InvalidInput(
            f"{MANIFEST} is not valid YAML{where}", param="manifest"
        ) from None
    if not isinstance(document, dict):
        raise InvalidInput(f"{MANIFEST} must be a mapping", param="manifest")
    name = document.get("name")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise InvalidInput(
            f"{MANIFEST} needs a `name`: lowercase letters, digits and dashes",
            param="name",
        )
    return Manifest(root, text, document)


def dump(document: Mapping[str, Any]) -> str:
    """A manifest document as `gf.yaml` text, its keys in the order given."""
    return yaml.safe_dump(
        dict(document), sort_keys=False, default_flow_style=None, allow_unicode=True
    )


def write(
    document: Mapping[str, Any], root: str | os.PathLike[str], *, force: bool
) -> Path:
    """Writes `gf.yaml`; an existing one only with `force`."""
    path = Path(root).expanduser() / MANIFEST
    if path.exists() and not force:
        raise Error(
            "InvalidInput",
            f"{MANIFEST} already exists in {path.parent}",
            hint="pass --force to overwrite it",
        )
    _write_atomic(path, dump(document))
    return path


def read_lock(root: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The `gf.lock` beside a manifest, or None when there is none."""
    path = Path(root).expanduser() / LOCK
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InvalidInput(
            f"{LOCK} is not valid JSON",
            hint="run `gf check` to write it again",
            param="lock",
        ) from None
    if not isinstance(value, dict):
        raise InvalidInput(
            f"{LOCK} must be a JSON object",
            hint="run `gf check` to write it again",
            param="lock",
        )
    return value


def write_lock(root: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    """Writes `gf.lock` as the service answered it."""
    path = Path(root).expanduser() / LOCK
    _write_atomic(path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return path


def _write_atomic(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
