"""Module 1: Storage — WindowStorage retrieval client.

Provides on-demand access to any window that has been ingested. This is the
interface every downstream module and the UI calls; nothing else in the system
reads directly from the bucket.

Standalone usage:
    from pipeline.modules.storage.client import WindowStorage
    from google.auth import default as google_auth_default

    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    storage = WindowStorage(
        bucket_uri="gs://your-bucket/verity",
        gcs_credentials=creds,
    )
    manifest = storage.get_window_manifest("segment_id", 0)
    url = storage.get_window_video_url("segment_id", 0, camera="FRONT")
    windows = storage.list_windows()
"""

from __future__ import annotations

import datetime
import json
import sys
from typing import Any

from pipeline.modules.storage.adapters.base import (
    DatasetManifest,
    PoseArray,
    PoseData,
    PoseRecord,
    StorageError,
    WindowKey,
    WindowManifest,
    WindowStorageError,
)
from pipeline.modules.storage.ingestion import _parse_bucket_uri


class WindowStorage:
    """Read-only interface over the canonical bucket window structure.

    All methods raise WindowStorageError when a window/segment is not found
    or cannot be read — never return None silently.

    Pre-signed URLs are generated per-call with the specified TTL. The URL
    points directly to the GCS blob; the browser streams video without
    touching the application server.
    """

    def __init__(
        self,
        bucket_uri: str,
        gcs_credentials: Any = None,
        gcs_project: str | None = None,
        sign_as: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        bucket_uri       "gs://bucket/prefix" written by IngestionPipeline
        gcs_credentials  google.auth credentials; defaults to ADC
        gcs_project      GCP project for the GCS client
        sign_as          Service account email for URL signing (local dev)
        """
        self._bucket_name, self._prefix = _parse_bucket_uri(bucket_uri)
        self._creds = gcs_credentials
        self._project = gcs_project
        self._sign_as = sign_as
        self._client: Any = None
        self._bucket_obj: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_bucket(self) -> Any:
        if self._bucket_obj is not None:
            return self._bucket_obj
        try:
            from google.cloud import storage  # noqa: PLC0415
        except ImportError:
            print(
                "\n[Storage/Client] MISSING DEPENDENCY: google-cloud-storage\n"
                "  Install it with:  pip install google-cloud-storage\n",
                file=sys.stderr,
            )
            raise
        try:
            self._client = storage.Client(
                credentials=self._creds, project=self._project
            )
            self._bucket_obj = self._client.bucket(self._bucket_name)
        except Exception as exc:
            raise WindowStorageError(
                f"gs://{self._bucket_name}/{self._prefix}",
                f"Could not connect to GCS: {type(exc).__name__}: {exc}",
            ) from exc
        return self._bucket_obj

    def _blob_path(self, *parts: str) -> str:
        return "/".join([self._prefix] + list(parts))

    def _read_json_blob(self, blob_name: str) -> dict[str, Any]:
        """Download and parse a JSON blob. Raises WindowStorageError on failure."""
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        try:
            data = blob.download_as_bytes()
        except Exception as exc:
            raise WindowStorageError(
                blob_name,
                f"Download failed: {type(exc).__name__}: {exc}",
            ) from exc
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise WindowStorageError(
                blob_name,
                f"JSON parse failed: {exc}",
            ) from exc

    def _read_bytes_blob(self, blob_name: str) -> bytes:
        """Download a blob as raw bytes. Raises WindowStorageError on failure."""
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        try:
            return blob.download_as_bytes()
        except Exception as exc:
            raise WindowStorageError(
                blob_name,
                f"Download failed: {type(exc).__name__}: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_window_manifest(self, segment_id: str, window_idx: int) -> WindowManifest:
        """Return the WindowManifest for a specific window.

        Raises WindowStorageError if the manifest does not exist.
        """
        blob_name = self._blob_path(
            "windows", segment_id, f"{window_idx:04d}", "manifest.json"
        )
        data = self._read_json_blob(blob_name)

        try:
            return WindowManifest(
                segment_id=data["segment_id"],
                window_idx=data["window_idx"],
                source_format=data["source_format"],
                source_schema_version=data["source_schema_version"],
                window_start_ts_us=data["window_start_ts_us"],
                window_end_ts_us=data["window_end_ts_us"],
                frame_count=data["frame_count"],
                cameras=data["cameras"],
                ingested_at=data["ingested_at"],
                pose_summary=data.get("pose_summary"),
                extra=data.get("extra", {}),
            )
        except KeyError as exc:
            raise WindowStorageError(
                blob_name,
                f"Manifest missing required field: {exc}. "
                f"This window may have been ingested by an older schema version.",
            ) from exc

    def get_window_video_url(
        self,
        segment_id: str,
        window_idx: int,
        camera: str = "FRONT",
        ttl_seconds: int = 3600,
    ) -> str:
        """Return a pre-signed URL for the camera MP4 of a specific window.

        The URL is valid for `ttl_seconds` (default 1 hour). The caller
        (e.g. the Judge UI) sets this as <video src="..."> and the browser
        streams directly from GCS.

        Raises WindowStorageError if the blob does not exist.
        """
        blob_name = self._blob_path(
            "windows", segment_id, f"{window_idx:04d}", f"camera_{camera}.mp4"
        )
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        try:
            if not blob.exists():
                raise WindowStorageError(
                    blob_name,
                    f"Camera {camera!r} MP4 not found for window "
                    f"{segment_id}/{window_idx:04d}. "
                    f"Available cameras can be found in the window manifest.",
                )
        except WindowStorageError:
            raise
        except Exception as exc:
            raise WindowStorageError(
                blob_name,
                f"Could not check blob existence: {exc}",
            ) from exc

        expiration = datetime.timedelta(seconds=ttl_seconds)

        try:
            if self._sign_as:
                from google.auth import impersonated_credentials  # noqa: PLC0415
                from google.auth import default as _adc  # noqa: PLC0415
                source_creds, _ = _adc(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                target_creds = impersonated_credentials.Credentials(
                    source_credentials=source_creds,
                    target_principal=self._sign_as,
                    target_scopes=[
                        "https://www.googleapis.com/auth/devstorage.read_only"
                    ],
                )
                return blob.generate_signed_url(
                    version="v4", expiration=expiration, method="GET",
                    credentials=target_creds,
                )
            return blob.generate_signed_url(
                version="v4", expiration=expiration, method="GET",
            )
        except Exception as exc:
            raise WindowStorageError(
                blob_name,
                f"Signed URL generation failed: {exc}. "
                f"Ensure the service account has roles/iam.serviceAccountTokenCreator.",
            ) from exc

    def get_window_pose(self, segment_id: str, window_idx: int) -> PoseData:
        """Return the pose data for a specific window.

        Returns an empty PoseData if pose.parquet was not written
        (best-effort at ingest time).
        """
        blob_name = self._blob_path(
            "windows", segment_id, f"{window_idx:04d}", "pose.parquet"
        )
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        try:
            if not blob.exists():
                return PoseData(
                    segment_id=segment_id,
                    window_idx=window_idx,
                    records=[],
                )
        except Exception:
            return PoseData(segment_id=segment_id, window_idx=window_idx, records=[])

        raw_bytes = self._read_bytes_blob(blob_name)

        try:
            import io  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
            table = pq.read_table(io.BytesIO(raw_bytes))
            df = table.to_pandas()
            records = [
                PoseRecord(
                    timestamp_us=int(row["timestamp_us"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    roll_rad=float(row.get("roll_rad", 0.0)),
                    pitch_rad=float(row.get("pitch_rad", 0.0)),
                    yaw_rad=float(row.get("yaw_rad", 0.0)),
                )
                for _, row in df.iterrows()
            ]
            return PoseData(
                segment_id=segment_id,
                window_idx=window_idx,
                records=records,
            )
        except Exception as exc:
            print(
                f"[Storage/Client] WARNING: Could not parse pose.parquet for "
                f"{segment_id}/{window_idx:04d}: {exc}. Returning empty pose.",
                file=sys.stderr,
            )
            return PoseData(segment_id=segment_id, window_idx=window_idx, records=[])

    def list_windows(self, segment_id: str | None = None) -> list[WindowKey]:
        """Return all ingested WindowKeys, optionally filtered by segment.

        Reads from segment index files rather than listing all blobs — O(segments)
        not O(windows * cameras).
        """
        bucket = self._get_bucket()
        keys: list[WindowKey] = []

        if segment_id is not None:
            seg_index_blob = self._blob_path("segments", segment_id, "index.json")
            try:
                data = self._read_json_blob(seg_index_blob)
            except WindowStorageError as exc:
                raise WindowStorageError(
                    seg_index_blob,
                    f"Segment {segment_id!r} not found or not yet ingested. "
                    f"Run IngestionPipeline.ingest() first.",
                ) from exc

            for entry in data.get("windows", []):
                if entry.get("status") == "ok":
                    keys.append(WindowKey(segment_id=segment_id, window_idx=entry["window_idx"]))
            return keys

        # All segments: walk segments/*/index.json
        seg_prefix = self._blob_path("segments") + "/"
        try:
            blobs = bucket.list_blobs(prefix=seg_prefix)
            for blob in blobs:
                if not blob.name.endswith("/index.json"):
                    continue
                try:
                    data = json.loads(blob.download_as_bytes().decode("utf-8"))
                    seg_id = data.get("segment_id", "")
                    for entry in data.get("windows", []):
                        if entry.get("status") == "ok":
                            keys.append(
                                WindowKey(segment_id=seg_id, window_idx=entry["window_idx"])
                            )
                except Exception as exc:
                    print(
                        f"[Storage/Client] WARNING: Could not parse index "
                        f"{blob.name!r}: {exc}",
                        file=sys.stderr,
                    )
        except Exception as exc:
            raise WindowStorageError(
                seg_prefix,
                f"Could not list segment indexes: {exc}",
            ) from exc

        return sorted(keys, key=lambda k: (k.segment_id, k.window_idx))

    def get_dataset_manifest(self) -> DatasetManifest:
        """Return the dataset-level manifest (total window count, segment list).

        Raises WindowStorageError if the manifest has not been written yet.
        """
        blob_name = self._blob_path("index", "manifest.json")
        data = self._read_json_blob(blob_name)

        # Reconstruct window keys from segment count (manifest lists segments, not windows)
        windows: list[WindowKey] = []
        try:
            for seg_id in data.get("segments", []):
                seg_idx_blob = self._blob_path("segments", seg_id, "index.json")
                try:
                    seg_data = self._read_json_blob(seg_idx_blob)
                    for entry in seg_data.get("windows", []):
                        if entry.get("status") == "ok":
                            windows.append(
                                WindowKey(segment_id=seg_id, window_idx=entry["window_idx"])
                            )
                except WindowStorageError:
                    pass
        except Exception:
            pass

        return DatasetManifest(
            bucket_uri=data.get("bucket_uri", f"gs://{self._bucket_name}/{self._prefix}"),
            window_count=len(windows),
            segment_count=data.get("segment_count", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            windows=windows,
        )
