"""Waymo Open Dataset v2 Parquet source adapter.

Reads camera_image Parquet files from a GCS bucket and returns RawSegment
objects. This is the only file in Module 1 that knows about GCS Parquet.

Standalone usage:
    from pipeline.modules.storage.adapters.parquet import WaymoParquetSource
    from google.auth import default as google_auth_default

    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    source = WaymoParquetSource(
        bucket="waymo_open_dataset_v_2_0_1",
        prefix="validation/camera_image",
        gcs_credentials=creds,
    )
    segments = source.list_segments()
    raw = source.fetch_segment(segments[0])
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.modules.storage.adapters.base import (
    Frame,
    PoseArray,
    PoseRecord,
    RawSegment,
    SourceAdapterError,
    SourceSchemaVersionError,
    SourceUnreachableError,
    ValidationResult,
    _loud,
)

# Schema version this adapter was written against.
# Bump here when the Waymo Parquet column layout changes.
_SCHEMA_VERSION = "waymo_v2_camera_image_v1"

# Required columns in camera_image Parquet. Both naming conventions are
# tried (Waymo v2 key-prefixed vs flat).
_REQUIRED_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "camera_name": ["key.camera_name", "camera_name"],
    "timestamp":   ["key.frame_timestamp_micros", "frame_timestamp_micros"],
    "image":       ["[CameraImageComponent].image", "image", "camera_image"],
}

CAMERA_NAMES: dict[int, str] = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


def _require_imports() -> tuple[Any, Any]:
    """Lazy-import heavy GCS deps so the module is importable without them.

    Raises ImportError with a clear installation message if missing.
    """
    try:
        import gcsfs  # noqa: PLC0415
    except ImportError:
        print(
            "\n[Storage/ParquetAdapter] MISSING DEPENDENCY: gcsfs\n"
            "  Install it with:  pip install gcsfs\n",
            file=sys.stderr,
        )
        raise

    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        print(
            "\n[Storage/ParquetAdapter] MISSING DEPENDENCY: pyarrow\n"
            "  Install it with:  pip install pyarrow\n",
            file=sys.stderr,
        )
        raise

    return gcsfs, pq


def _find_col(schema_names: list[str], candidates: list[str], context: str) -> str:
    """Return the first candidate column that exists in the schema.

    Raises KeyError with a detailed message if none are found — never
    returns a wrong column name silently.
    """
    for c in candidates:
        if c in schema_names:
            return c
    raise KeyError(
        f"[Storage/ParquetAdapter] Column not found in {context}.\n"
        f"  Tried   : {candidates}\n"
        f"  Found   : {schema_names[:30]}\n"
        f"  → The Parquet schema may have changed. Update _REQUIRED_COLUMN_CANDIDATES "
        f"or bump the schema version."
    )


class WaymoParquetSource:
    """SourceAdapter for Waymo Open Dataset v2 camera_image Parquet on GCS.

    Attributes
    ----------
    format_name      : "waymo_parquet"
    schema_version   : versioned against Waymo v2 column layout
    """

    format_name: str = "waymo_parquet"
    schema_version: str = _SCHEMA_VERSION

    def __init__(
        self,
        bucket: str,
        prefix: str,
        gcs_credentials: Any = None,
        pose_prefix: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        bucket          GCS bucket name (no gs:// prefix)
        prefix          Path prefix inside the bucket, e.g. "validation/camera_image"
        gcs_credentials google.auth credentials object; defaults to ADC
        pose_prefix     Optional GCS prefix for vehicle_pose Parquet (best-effort)
        """
        self._bucket = bucket.removeprefix("gs://").rstrip("/")
        self._prefix = prefix.strip("/")
        self._creds = gcs_credentials
        self._pose_prefix = pose_prefix
        self._fs: Any = None        # lazily constructed gcsfs.GCSFileSystem

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fs(self) -> Any:
        if self._fs is not None:
            return self._fs
        gcsfs, _ = _require_imports()
        try:
            self._fs = gcsfs.GCSFileSystem(token=self._creds or "google_default")
            # Probe connectivity with a cheap ls on the bucket root.
            self._fs.ls(self._bucket, detail=False)
        except Exception as exc:
            raise SourceUnreachableError(
                f"gs://{self._bucket}/{self._prefix}",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return self._fs

    def _parquet_path(self, segment_id: str) -> str:
        return f"{self._bucket}/{self._prefix}/{segment_id}.parquet"

    def _pose_path(self, segment_id: str) -> str | None:
        if not self._pose_prefix:
            return None
        prefix = self._pose_prefix.strip("/")
        return f"{self._bucket}/{prefix}/{segment_id}.parquet"

    # ------------------------------------------------------------------
    # SourceAdapter protocol
    # ------------------------------------------------------------------

    def list_segments(self) -> list[str]:
        """List all segment IDs available under bucket/prefix.

        Raises SourceUnreachableError if GCS cannot be reached.
        """
        fs = self._get_fs()
        try:
            files = fs.ls(f"{self._bucket}/{self._prefix}/", detail=False)
        except Exception as exc:
            raise SourceUnreachableError(
                f"gs://{self._bucket}/{self._prefix}",
                f"ls failed: {type(exc).__name__}: {exc}",
            ) from exc

        segment_ids = [
            Path(f).stem
            for f in files
            if str(f).endswith(".parquet")
        ]
        if not segment_ids:
            print(
                f"[Storage/ParquetAdapter] WARNING: No .parquet files found at "
                f"gs://{self._bucket}/{self._prefix}/",
                file=sys.stderr,
            )
        return sorted(segment_ids)

    def validate_segment(self, segment_id: str) -> ValidationResult:
        """Check that the segment Parquet exists and has required columns.

        Fast — only reads the schema, not the data rows.
        """
        _, pq = _require_imports()
        fs = self._get_fs()
        path = self._parquet_path(segment_id)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                schema_names: list[str] = pf.schema_arrow.names
        except FileNotFoundError:
            return ValidationResult(
                valid=False,
                segment_id=segment_id,
                errors=[f"Parquet not found: {path}"],
            )
        except Exception as exc:
            return ValidationResult(
                valid=False,
                segment_id=segment_id,
                errors=[f"Could not open Parquet: {type(exc).__name__}: {exc}"],
            )

        missing: list[str] = []
        for col_purpose, candidates in _REQUIRED_COLUMN_CANDIDATES.items():
            if not any(c in schema_names for c in candidates):
                missing.append(
                    f"  {col_purpose}: tried {candidates}, none found in schema"
                )
        if missing:
            errors.append(
                "Required columns absent:\n" + "\n".join(missing) +
                f"\n  Available columns: {schema_names[:30]}"
            )

        if errors:
            print(
                f"[Storage/ParquetAdapter] VALIDATION FAILED for {segment_id!r}:\n"
                + "\n".join(errors),
                file=sys.stderr,
            )

        return ValidationResult(
            valid=not errors,
            segment_id=segment_id,
            errors=errors,
            warnings=warnings,
        )

    def fetch_segment(self, segment_id: str) -> RawSegment:
        """Load all camera frames for one segment and return a RawSegment.

        Fails loudly:
          - SourceUnreachableError  : if GCS cannot be reached
          - SourceAdapterError      : if this specific segment is corrupt/missing
        Never returns partial data silently.
        """
        _, pq = _require_imports()
        fs = self._get_fs()
        path = self._parquet_path(segment_id)

        # ---- validate schema first ----
        result = self.validate_segment(segment_id)
        if not result.valid:
            raise SourceAdapterError(
                self.format_name, segment_id,
                ValueError(f"Segment failed validation: {'; '.join(result.errors)}")
            )

        # ---- read columns ----
        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                schema_names: list[str] = pf.schema_arrow.names

            col_image  = _find_col(schema_names, _REQUIRED_COLUMN_CANDIDATES["image"],       path)
            col_camera = _find_col(schema_names, _REQUIRED_COLUMN_CANDIDATES["camera_name"], path)
            col_ts     = _find_col(schema_names, _REQUIRED_COLUMN_CANDIDATES["timestamp"],   path)

            with fs.open(path, "rb") as f:
                import pyarrow.parquet as _pq  # noqa: PLC0415
                table = _pq.read_table(f, columns=[col_image, col_camera, col_ts])
        except SourceAdapterError:
            raise
        except Exception as exc:
            raise SourceAdapterError(self.format_name, segment_id, exc) from exc

        df = table.to_pandas().sort_values(col_ts)

        # ---- build per-camera Frame lists ----
        cameras: dict[str, list[Frame]] = {}
        for cam_int, cam_name in CAMERA_NAMES.items():
            cam_df = df[df[col_camera] == cam_int].reset_index(drop=True)
            if cam_df.empty:
                print(
                    f"[Storage/ParquetAdapter] WARNING: camera {cam_name!r} has "
                    f"no frames in segment {segment_id!r}.",
                    file=sys.stderr,
                )
                cameras[cam_name] = []
                continue
            cameras[cam_name] = [
                Frame(
                    timestamp_us=int(row[col_ts]),
                    frame_index=idx,
                    jpeg_bytes=bytes(row[col_image]),
                )
                for idx, row in cam_df.iterrows()
            ]

        if not cameras.get("FRONT"):
            raise SourceAdapterError(
                self.format_name, segment_id,
                ValueError("FRONT camera has no frames — cannot build windows.")
            )

        front_frames = cameras["FRONT"]
        duration_s = (
            (front_frames[-1].timestamp_us - front_frames[0].timestamp_us) / 1e6
            if len(front_frames) > 1 else 0.0
        )
        fps = len(front_frames) / duration_s if duration_s > 0 else 10.0

        # ---- best-effort pose ----
        pose = self._fetch_pose(fs, pq, segment_id)

        return RawSegment(
            segment_id=segment_id,
            source_format=self.format_name,
            source_schema_version=self.schema_version,
            duration_seconds=round(duration_s, 3),
            frame_rate_hz=round(fps, 2),
            cameras=cameras,
            pose=pose,
            source_metadata={
                "dataset": "waymo_open_dataset_v_2_0_1",
                "gcs_path": f"gs://{path}",
                "frame_count_front": len(front_frames),
            },
        )

    # ------------------------------------------------------------------
    # Best-effort pose extraction
    # ------------------------------------------------------------------

    def _fetch_pose(self, fs: Any, pq: Any, segment_id: str) -> PoseArray:
        """Try to load vehicle pose from the pose Parquet if configured.

        Returns an empty PoseArray (not an error) if unavailable — pose is
        best-effort for Phase 1. Downstream checks `pose.is_empty`.
        """
        pose_path = self._pose_path(segment_id)
        if pose_path is None:
            return PoseArray()

        try:
            with fs.open(pose_path, "rb") as f:
                table = pq.read_table(f)
            df = table.to_pandas()
        except Exception as exc:
            print(
                f"[Storage/ParquetAdapter] INFO: Could not load pose for "
                f"{segment_id!r}: {exc}. Continuing with empty pose.",
                file=sys.stderr,
            )
            return PoseArray()

        records: list[PoseRecord] = []
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        if ts_col is None:
            return PoseArray()

        for _, row in df.iterrows():
            records.append(PoseRecord(
                timestamp_us=int(row[ts_col]),
                x=float(row.get("x", row.get("translation_x", 0.0))),
                y=float(row.get("y", row.get("translation_y", 0.0))),
                z=float(row.get("z", row.get("translation_z", 0.0))),
                roll_rad=float(row.get("roll", 0.0)),
                pitch_rad=float(row.get("pitch", 0.0)),
                yaw_rad=float(row.get("yaw", 0.0)),
            ))
        return PoseArray(records=sorted(records, key=lambda r: r.timestamp_us))
