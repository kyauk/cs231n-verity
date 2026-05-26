"""Waymo Open Dataset TFRecord source adapter.

Handles TFRecord format via the `waymo-open-dataset` package.
Phase 1 primary format for seeded evaluation sets.

Installation:
    pip install waymo-open-dataset-tf-2-12-0  # or latest tf-compatible version

Standalone usage:
    from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource
    source = WaymoTFRecordSource(tfrecord_paths=["path/to/segment.tfrecord"])
    segments = source.list_segments()
    raw = source.fetch_segment(segments[0])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pipeline.modules.storage.adapters.base import (
    Frame,
    PoseArray,
    PoseRecord,
    RawSegment,
    SourceAdapterError,
    SourceUnreachableError,
    ValidationResult,
)

_SCHEMA_VERSION = "waymo_tfrecord_v1"


def _require_imports() -> Any:
    """Lazy-import waymo_open_dataset. Raises ImportError with clear message."""
    try:
        import waymo_open_dataset  # noqa: F401, PLC0415
        from waymo_open_dataset.utils import frame_utils  # noqa: F401, PLC0415
    except ImportError:
        print(
            "\n[Storage/TFRecordAdapter] MISSING DEPENDENCY: waymo-open-dataset\n"
            "  Install it with:\n"
            "    pip install waymo-open-dataset-tf-2-12-0\n"
            "  Requires TensorFlow and the matching CUDA environment.\n",
            file=sys.stderr,
        )
        raise
    return waymo_open_dataset


class WaymoTFRecordSource:
    """SourceAdapter for Waymo Open Dataset TFRecord files.

    Accepts either local paths or a directory of .tfrecord files.

    Attributes
    ----------
    format_name    : "waymo_tfrecord"
    schema_version : versioned against Waymo TFRecord v1 proto schema
    """

    format_name: str = "waymo_tfrecord"
    schema_version: str = _SCHEMA_VERSION

    def __init__(self, tfrecord_paths: list[str] | None = None, directory: str | None = None) -> None:
        """
        Parameters
        ----------
        tfrecord_paths  Explicit list of .tfrecord file paths.
        directory       Directory containing .tfrecord files (used if paths is None).

        Exactly one of these must be provided.
        """
        if tfrecord_paths is None and directory is None:
            raise IngestionError(  # type: ignore[name-defined]
                "WaymoTFRecordSource requires either tfrecord_paths or directory."
            )
        if tfrecord_paths is not None:
            self._paths: list[Path] = [Path(p) for p in tfrecord_paths]
        else:
            assert directory is not None
            d = Path(directory)
            self._paths = sorted(d.glob("*.tfrecord"))
            if not self._paths:
                print(
                    f"[Storage/TFRecordAdapter] WARNING: No .tfrecord files in {directory!r}",
                    file=sys.stderr,
                )

        # segment_id is the stem of the tfrecord filename
        self._id_to_path: dict[str, Path] = {p.stem: p for p in self._paths}

    def list_segments(self) -> list[str]:
        """Return segment IDs (filename stems) for all known TFRecord files."""
        for path in self._paths:
            if not path.exists():
                raise SourceUnreachableError(
                    str(path),
                    f"TFRecord file does not exist: {path}"
                )
        return sorted(self._id_to_path.keys())

    def validate_segment(self, segment_id: str) -> ValidationResult:
        """Check that the TFRecord file exists and is readable."""
        path = self._id_to_path.get(segment_id)
        if path is None:
            return ValidationResult(
                valid=False,
                segment_id=segment_id,
                errors=[f"Segment {segment_id!r} not in known tfrecord paths."],
            )
        if not path.exists():
            return ValidationResult(
                valid=False,
                segment_id=segment_id,
                errors=[f"TFRecord file not found: {path}"],
            )
        if path.stat().st_size == 0:
            return ValidationResult(
                valid=False,
                segment_id=segment_id,
                errors=[f"TFRecord file is empty: {path}"],
            )
        return ValidationResult(valid=True, segment_id=segment_id)

    def fetch_segment(self, segment_id: str) -> RawSegment:
        """Load all camera frames from a TFRecord and return a RawSegment.

        Raises:
          SourceUnreachableError  : file missing / unreadable
          SourceAdapterError      : corrupt TFRecord or parse failure
        """
        _require_imports()

        path = self._id_to_path.get(segment_id)
        if path is None or not path.exists():
            raise SourceUnreachableError(
                segment_id,
                f"TFRecord not found: {path or segment_id}"
            )

        try:
            return self._parse_tfrecord(segment_id, path)
        except (SourceUnreachableError, SourceAdapterError):
            raise
        except Exception as exc:
            raise SourceAdapterError(self.format_name, segment_id, exc) from exc

    def _parse_tfrecord(self, segment_id: str, path: Path) -> RawSegment:
        """Parse one TFRecord into a RawSegment using waymo-open-dataset protos."""
        import tensorflow as tf  # noqa: PLC0415

        CAMERA_ID_TO_NAME = {
            1: "FRONT",
            2: "FRONT_LEFT",
            3: "FRONT_RIGHT",
            4: "SIDE_LEFT",
            5: "SIDE_RIGHT",
        }

        cameras: dict[str, list[Frame]] = {name: [] for name in CAMERA_ID_TO_NAME.values()}
        pose_records: list[PoseRecord] = []
        frame_ts_list: list[int] = []

        dataset = tf.data.TFRecordDataset(str(path), compression_type="")
        for raw_record in dataset:
            from waymo_open_dataset import dataset_pb2  # noqa: PLC0415
            frame = dataset_pb2.Frame()
            frame.ParseFromString(raw_record.numpy())

            ts_us = frame.timestamp_micros
            frame_ts_list.append(ts_us)

            for cam_image in frame.images:
                cam_name = CAMERA_ID_TO_NAME.get(cam_image.name)
                if cam_name is None:
                    continue
                cameras[cam_name].append(Frame(
                    timestamp_us=ts_us,
                    frame_index=len(cameras[cam_name]),
                    jpeg_bytes=cam_image.image,
                ))

            # Ego pose from the frame context
            if frame.pose.transform:
                # transform is a 4x4 row-major matrix; extract translation
                t = frame.pose.transform
                pose_records.append(PoseRecord(
                    timestamp_us=ts_us,
                    x=float(t[3]),
                    y=float(t[7]),
                    z=float(t[11]),
                ))

        if not cameras.get("FRONT"):
            raise SourceAdapterError(
                self.format_name, segment_id,
                ValueError("FRONT camera has no frames in this TFRecord.")
            )

        front = cameras["FRONT"]
        duration_s = (
            (front[-1].timestamp_us - front[0].timestamp_us) / 1e6
            if len(front) > 1 else 0.0
        )
        fps = len(front) / duration_s if duration_s > 0 else 10.0

        return RawSegment(
            segment_id=segment_id,
            source_format=self.format_name,
            source_schema_version=self.schema_version,
            duration_seconds=round(duration_s, 3),
            frame_rate_hz=round(fps, 2),
            cameras=cameras,
            pose=PoseArray(records=pose_records),
            source_metadata={
                "dataset": "waymo_open_dataset",
                "tfrecord_path": str(path),
                "frame_count_front": len(front),
            },
        )
