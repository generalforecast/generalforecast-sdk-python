# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""Data sources: the identity a source is granted, the pinned files, and the permission names keys carry."""

import pytest

from generalforecast._api import Key, PayloadError
from tests import test_hosted
from tests.test_hosted import client
from tests.test_hosted import run as gf

hosted = test_hosted.hosted

EMAIL = "gfws-3f9a1c2b4d5e6f70@gf-sources.iam.gserviceaccount.com"
PREPARED = {
    "kind": "bigquery",
    "identity": {
        "email": EMAIL,
        "unique_id": "104857600000000000001",
        "principal": f"serviceAccount:{EMAIL}",
    },
    "grants": [{"role": "roles/bigquery.dataViewer", "on": "dataset"}],
}
PIN = {
    "source_id": "source-1",
    "digest": "sha256:" + "a" * 64,
    "size_bytes": 10,
    "format": "jsonl",
    "mapping": {"key": "id", "features": {"x": "x"}, "labels": {}},
}


def test_prepare_prints_the_account_its_unique_id_and_each_role(
    wire, hosted, monkeypatch
):
    wire.expect("POST", "/sources:prepare", PREPARED, payload={"kind": "bigquery"})
    code, stdout, stderr = gf(
        "admin", "source", "prepare", "--kind", "bigquery",
        monkeypatch=monkeypatch, env=hosted,
    )  # fmt: skip
    assert code == 0, stderr
    assert EMAIL in stdout and "104857600000000000001" in stdout
    assert "roles/bigquery.dataViewer on the dataset" in stdout
    assert [path for path, _ in wire.requests] == ["/v1/projects/test/sources:prepare"]


def test_prepare_takes_only_a_kind_the_service_answers_for(hosted, monkeypatch):
    code, _, stderr = gf(
        "admin", "source", "prepare", "--kind", "redshift",
        monkeypatch=monkeypatch, env=hosted,
    )  # fmt: skip
    assert code == 2 and "redshift" in stderr


def test_the_client_prepares_a_source_and_pins_files_under_sources_pins(wire):
    snowflake = {
        **PREPARED,
        "kind": "snowflake",
        "grants": [],
        "oidc": {"issuer": "https://accounts.google.com", "subject": "1048"},
    }
    wire.expect("POST", "/sources:prepare", snowflake, payload={"kind": "snowflake"})
    spec = {"path": "rows.jsonl", "format": "jsonl", "mapping": PIN["mapping"]}
    wire.expect("POST", "/sources/pins", PIN, payload=spec)
    with client(wire) as sdk:
        assert sdk.sources.prepare("snowflake") == snowflake
        assert sdk.sources.pin(spec) == PIN
    assert [path for path, _ in wire.requests] == [
        "/v1/projects/test/sources:prepare",
        "/v1/projects/test/sources/pins",
    ]


def test_a_permission_name_this_sdk_does_not_know_is_kept_as_it_is():
    key = Key.from_payload(
        {
            "id": "key-0f1e2d3c4b5a69788796",
            "project": "acme",
            "name": "ci",
            "permissions": ["sources:manage", "forecasts:export"],
            "created_at": 1.0,
            "expires_at": None,
            "revoked_at": None,
        }
    )
    assert key.permissions == ("sources:manage", "forecasts:export")
    assert key.expires_at is None
    with pytest.raises(PayloadError):
        Key.from_payload({**key.to_payload(), "permissions": ["not a permission"]})
