"""Shared window types — produced by Module 1: Storage, consumed by all downstream modules.

These are the canonical data shapes that cross the Storage module boundary.
Every module that reads from WindowStorage uses these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# WindowKey — universal window identifier used everywhere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowKey:
    """Identifies one windowed clip. Hashable, usable as dict key."""
    segment_id: str
    window_idx: int

    def __str__(self) -> str:
        return f"{self.segment_id}/{self.window_idx:04d}"

    def to_json(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "window_idx": self.window_idx}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "WindowKey":
        return cls(segment_id=str(d["segment_id"]), window_idx=int(d["window_idx"]))

    @classmethod
    def from_str(cls, s: str) -> "WindowKey":
        """Parse 'segment_id/0007' string form."""
        seg, _, idx_str = str(s).rpartition("/")
        return cls(segment_id=seg, window_idx=int(idx_str))


# ---------------------------------------------------------------------------
# PoseRecord — one ego-pose sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseRecord:
    """Ego vehicle pose at one timestamp."""
    timestamp_us: int
    x: float
    y: float
    z: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp_us": self.timestamp_us,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll_rad": self.roll_rad,
            "pitch_rad": self.pitch_rad,
            "yaw_rad": self.yaw_rad,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PoseRecord":
        return cls(
            timestamp_us=int(d["timestamp_us"]),
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            roll_rad=float(d.get("roll_rad", 0.0)),
            pitch_rad=float(d.get("pitch_rad", 0.0)),
            yaw_rad=float(d.get("yaw_rad", 0.0)),
        )


# ---------------------------------------------------------------------------
# PoseData — windowed pose sequence returned by WindowStorage
# ---------------------------------------------------------------------------

@dataclass
class PoseData:
    """Ordered ego-pose samples for one window."""
    segment_id: str
    window_idx: int
    records: list[PoseRecord]

    def to_json(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "window_idx": self.window_idx,
            "records": [r.to_json() for r in self.records],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PoseData":
        return cls(
            segment_id=str(d["segment_id"]),
            window_idx=int(d["window_idx"]),
            records=[PoseRecord.from_json(r) for r in d.get("records", [])],
        )


# ---------------------------------------------------------------------------
# WindowManifest — canonical metadata for one ingested window
# ---------------------------------------------------------------------------

@dataclass
class WindowManifest:
    """Written at ingest time; describes provenance, timing, and cameras."""
    segment_id: str
    window_idx: int
    source_format: str
    source_schema_version: str
    window_start_ts_us: int
    window_end_ts_us: int
    frame_count: int
    cameras: list[str]
    ingested_at: str                    # ISO-8601
    pose_summary: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def window_key(self) -> WindowKey:
        return WindowKey(segment_id=self.segment_id, window_idx=self.window_idx)

    def to_json(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "window_idx": self.window_idx,
            "source_format": self.source_format,
            "source_schema_version": self.source_schema_version,
            "window_start_ts_us": self.window_start_ts_us,
            "window_end_ts_us": self.window_end_ts_us,
            "frame_count": self.frame_count,
            "cameras": self.cameras,
            "ingested_at": self.ingested_at,
            "pose_summary": self.pose_summary,
            "extra": self.extra,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "WindowManifest":
        return cls(
            segment_id=str(d["segment_id"]),
            window_idx=int(d["window_idx"]),
            source_format=str(d["source_format"]),
            source_schema_version=str(d["source_schema_version"]),
            window_start_ts_us=int(d["window_start_ts_us"]),
            window_end_ts_us=int(d["window_end_ts_us"]),
            frame_count=int(d["frame_count"]),
            cameras=list(d["cameras"]),
            ingested_at=str(d["ingested_at"]),
            pose_summary=d.get("pose_summary"),
            extra=dict(d.get("extra", {})),
        )


# ---------------------------------------------------------------------------
# DatasetManifest — top-level dataset index
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """Dataset-level index written by IngestionPipeline."""
    bucket_uri: str
    window_count: int
    segment_count: int
    created_at: str
    updated_at: str
    windows: list[WindowKey] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "bucket_uri": self.bucket_uri,
            "window_count": self.window_count,
            "segment_count": self.segment_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "windows": [w.to_json() for w in self.windows],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DatasetManifest":
        return cls(
            bucket_uri=str(d["bucket_uri"]),
            window_count=int(d["window_count"]),
            segment_count=int(d["segment_count"]),
            created_at=str(d["created_at"]),
            updated_at=str(d["updated_at"]),
            windows=[WindowKey.from_json(w) for w in d.get("windows", [])],
        )
