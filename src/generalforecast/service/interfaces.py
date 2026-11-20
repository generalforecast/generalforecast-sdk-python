# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The typed shapes the hosted service's context and source routes take and return."""

from typing import Literal, NotRequired, TypedDict

from generalforecast.service.boundary import JsonObject

WriteOperation = Literal[
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
    "import_record",
]
LabelValue = bool | float | str


class ObservationInput(TypedDict):
    """Independently guarded components of one logical observation."""

    key: str
    features: NotRequired[dict[str, float]]
    labels: NotRequired[dict[str, LabelValue]]
    event_at: NotRequired[float]
    expected_feature_version: NotRequired[int]
    expected_label_versions: NotRequired[dict[str, int]]


class Mutation(TypedDict):
    """One bounded atomic ordinary request in the context write-ID namespace."""

    write_id: str
    operation: WriteOperation
    records: list[ObservationInput]


class ComponentVersions(TypedDict):
    """Independent current component versions, never a single row version."""

    key: str
    feature_version: int | None
    label_versions: dict[str, int]


class AcceptedWrite(TypedDict):
    """Original durable receipt; no serving readiness assertion."""

    project: str
    context: str
    write_id: str
    operation: WriteOperation
    sequence: int
    accepted_at: float
    durably_accepted: bool
    changes: list[ComponentVersions]
    retry_valid_until: float


class SnapshotRow(TypedDict):
    """Only the requested target and versions known at the requested instant."""

    key: str
    features: dict[str, float]
    label: LabelValue
    feature_version: int
    label_version: int
    feature_accepted_at: float
    label_accepted_at: float
    event_at: float | None


class ContextSnapshot(TypedDict):
    """A transactionally consistent bounded history view with explicit clocks."""

    project: str
    context: str
    target: str
    features: list[str]
    through_sequence: int
    knowledge_at: float
    event_before: float | None
    rows: list[SnapshotRow]
    source_wait_count: int


class ImportOutcome(TypedDict):
    """One source occurrence; infrastructure failures raise, not row-conflict."""

    ordinal: int
    status: Literal["accepted", "skipped", "conflict"]
    receipt: AcceptedWrite | None
    skipped_components: list[str]
    issue: JsonObject | None


class SourcePin(TypedDict):
    """Pinned bytes under the configured source root, not arbitrary URL access."""

    source_id: str
    digest: str
    size_bytes: int
    format: Literal["jsonl", "csv"]
    mapping: JsonObject


class SourceRecord(TypedDict):
    """Stable ordered occurrence; next_offset is the durable resume checkpoint."""

    ordinal: int
    offset: int
    next_offset: int
    record: ObservationInput | None
    issue: JsonObject | None
