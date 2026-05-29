"""Base types, protocols, and structured errors for Module 1: Storage.

This file is the format boundary. Everything the ingestion pipeline and
WindowStorage client need to speak to each other lives here. No other
module should import from outside this package.

Cross-module types (WindowKey, PoseRecord, PoseData, WindowManifest,
DatasetManifest) are defined in pipeline.interfaces.window and re-exported
here so existing internal imports within this module don't change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pipeline.interfaces.errors import (  # noqa: F401 — re-exported for callers
    StorageError,
    WindowStorageError,
)
from pipeline.interfaces.window import (  # noqa: F401 — re-exported for callers
    DatasetManifest,
    PoseData,
    PoseRecord,
    WindowKey,
    WindowManifest,
)


# ---------------------------------------------------------------------------
# Errors — loud by design, never silent
#
# StorageError (base) and WindowStorageError live in pipeline.interfaces.errors
# because they cross module boundaries; they are re-exported above so internal
# imports within this module keep working. The errors below are Storage-internal
# and never leave Module 1.
# ---------------------------------------------------------------------------

class SourceUnreachableError(StorageError):
    """Raised immediately when the source bucket / mount cannot be reached.

    Ingestion aborts. The caller must not retry without fixing connectivity.
    """
    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(
            f"[Storage] Source unreachable: {source!r}\n"
            f"  Detail: {detail}\n"
            f"  → Fix GCS/S3 credentials or network access before re-running."
        )
        _loud(self)


class SourceSchemaVersionError(StorageError):
    """Raised when the adapter's expected schema version doesn't match the source.

    Processing any segment under a mismatched schema would produce corrupted
    windows. Ingest aborts before touching any data.
    """
    def __init__(self, adapter: str, expected: str, found: str) -> None:
        self.adapter = adapter
        self.expected = expected
        self.found = found
        super().__init__(
            f"[Storage] Schema version mismatch in adapter {adapter!r}.\n"
            f"  Expected : {expected!r}\n"
            f"  Found    : {found!r}\n"
            f"  → Update the adapter or re-export the source data to schema {expected!r}."
        )
        _loud(self)


class SourceAdapterError(StorageError):
    """Wraps an uncategorized exception raised inside a SourceAdapter.

    The full traceback from the original exception is preserved as `cause`.
    """
    def __init__(self, adapter: str, segment_id: str, cause: BaseException) -> None:
        self.adapter = adapter
        self.segment_id = segment_id
        self.cause = cause
        super().__init__(
            f"[Storage] Adapter {adapter!r} failed on segment {segment_id!r}.\n"
            f"  Cause: {type(cause).__name__}: {cause}\n"
            f"  → Check adapter logs. Segment will be skipped."
        )
        _loud(self)


class IngestionError(StorageError):
    """Raised for fatal ingestion failures that stop the whole run."""
    def __init__(self, detail: str) -> None:
        super().__init__(f"[Storage] Ingestion fatal error: {detail}")
        _loud(self)


def _loud(exc: BaseException) -> None:
    """Print the error to stderr immediately so it's never silently swallowed."""
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"STORAGE ERROR: {type(exc).__name__}", file=sys.stderr)
    print(str(exc), file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data models — the format boundary
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """One camera frame: raw JPEG bytes + microsecond timestamp."""
    timestamp_us: int
    frame_index: int
    jpeg_bytes: bytes = field(repr=False)   # bytes — not repr'd to keep logs clean


@dataclass
class PoseArray:
    """Ordered sequence of PoseRecords for a segment or window."""
    records: list[PoseRecord] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0

    def slice_window(self, start_us: int, end_us: int) -> "PoseArray":
        """Return a PoseArray containing only records within [start_us, end_us].

        Assumption: timestamps are unique and non-overlapping across consecutive
        windows (microsecond resolution makes boundary collisions negligible in
        practice). If two windows share a boundary timestamp, the pose record
        appears in both — acceptable for VLM prompt context.
        """
        return PoseArray(
            records=[r for r in self.records if start_us <= r.timestamp_us <= end_us]
        )


@dataclass
class RawSegment:
    """Format-boundary object. Produced by a SourceAdapter; consumed by ingestion.

    Once a segment is represented as RawSegment, nothing downstream knows
    whether it came from TFRecord, Parquet, or any future source format.
    """
    segment_id: str
    source_format: str              # "waymo_parquet", "waymo_tfrecord", …
    source_schema_version: str      # tracks which schema produced this segment
    duration_seconds: float
    frame_rate_hz: float
    cameras: dict[str, list[Frame]]  # camera_name → frames sorted by timestamp
    pose: PoseArray
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of SourceAdapter.validate_segment()."""
    valid: bool
    segment_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def require_valid(self, adapter: str) -> None:
        """Raise SourceAdapterError if this result is not valid."""
        if not self.valid:
            raise SourceAdapterError(
                adapter, self.segment_id,
                ValueError(f"Validation failed: {'; '.join(self.errors)}")
            )


# ---------------------------------------------------------------------------
# WindowConfig and IngestionRequest
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """Parameters controlling how RawSegments are split into windows."""
    length_frames: int = 80          # 8 s at 10 Hz
    stride_frames: int = 80          # non-overlapping
    target_fps: int = 10
    cameras: tuple[str, ...] = (
        "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"
    )


@dataclass
class IngestionRequest:
    """Everything ingestion needs to process a list of segments."""
    segment_ids: list[str]
    bucket_uri: str                  # "gs://bucket/prefix" or "s3://bucket/prefix"
    window_config: WindowConfig
    source: "SourceAdapter"          # type: ignore[type-arg]
    force_reingest: bool = False


# ---------------------------------------------------------------------------
# SourceAdapter Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SourceAdapter(Protocol):
    """Interface every source format must implement.

    Adding a new source format means writing one class that satisfies this
    protocol — nothing else in the pipeline changes.
    """
    format_name: str        # "waymo_parquet", "waymo_tfrecord", …
    schema_version: str     # bump when the source schema changes breaking-ly

    def list_segments(self) -> list[str]:
        """Return all available segment IDs from this source."""
        ...

    def fetch_segment(self, segment_id: str) -> RawSegment:
        """Load one segment and return a RawSegment.

        Must raise:
          SourceUnreachableError  — if the source cannot be reached at all
          SourceAdapterError      — if this specific segment fails
        Never returns partial/degraded output silently.
        """
        ...

    def validate_segment(self, segment_id: str) -> ValidationResult:
        """Check that segment_id exists and has required structure.

        Does NOT fetch frame data — should be fast enough to run on all
        segment IDs before ingestion begins.
        """
        ...
