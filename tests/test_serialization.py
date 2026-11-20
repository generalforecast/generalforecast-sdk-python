# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""One canonical JSON encoding for content identity."""

import hashlib
import json
from types import MappingProxyType

import pytest

from generalforecast.serialization import canonical_json, content_digest
from generalforecast.validation import PredictorError, canonical, digest


@pytest.mark.parametrize("filename", ["model.safetensors", "échantillon.safetensors"])
def test_content_identity_is_one_encoding(filename):
    files = {filename: "a" * 64, "config.json": "b" * 64}
    assert digest(files) == content_digest(files)
    if filename.isascii():
        previous = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        assert digest(files) == "sha256:" + hashlib.sha256(previous).hexdigest()


def test_canonical_values_include_immutable_mappings_and_tuples():
    value = MappingProxyType({"z": (1, True, None), "a": "é"})
    assert (
        canonical_json(value)
        == canonical(value)
        == b'{"a":"\xc3\xa9","z":[1,true,null]}'
    )


@pytest.mark.parametrize(
    "value", [{1: "invalid"}, {"x": object()}, float("inf"), float("nan")]
)
def test_invalid_values_fail_in_the_shared_encoder_and_at_package_boundary(value):
    with pytest.raises(ValueError):
        canonical_json(value)
    with pytest.raises(PredictorError) as caught:
        canonical(value)
    assert caught.value.code == "InvalidPredictor"
