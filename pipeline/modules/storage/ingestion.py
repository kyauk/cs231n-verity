"""Module 1: Storage — ingestion pipeline.

Takes an IngestionRequest, windows each RawSegment, encodes MP4s via ffmpeg,
uploads to GCS/S3, and writes the canonical manifest structure.

Bucket layout written by this module:
  {bucket}/windows/{segment_id}/{window_idx:04d}/
      manifest.json
      camera_FRONT.mp4
      camera_FRONT_LEFT.mp4
      camera_FRONT_RIGHT.mp4
      camera_SIDE_LEFT.mp4
      camera_SIDE_RIGHT.mp4
      pose.parquet
      pose_summary.json
  {bucket}/segments/{segment_id}/index.json
  {bucket}/index/manifest.json
  {bucket}/errors/ingestion/{YYYY-MM-DD}/{segment_id}.json

Standalone usage:
    from pipeline.modules.storage.ingestion import IngestionPipeline
    from pipeline.modules.storage.adapters.base import IngestionRequest, WindowConfig
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
    request = IngestionRequest(
        segment_ids=source.list_segments()[:5],
        bucket_uri="gs://your-dest-bucket/verity",
        window_config=WindowConfig(),
        source=source,
    )
    pipeline = IngestionPipeline(gcs_credentials=creds)
    result = pipeline.ingest(request)
    print(result)
"""

from __future__ import annotations

import datetime
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.modules.storage.adapters.base import (
    Frame,
    IngestionError,
    IngestionRequest,
    PoseArray,
    RawSegment,
    SourceAdapterError,
    SourceUnreachableError,
    StorageError,
    WindowConfig,
    WindowManifest,
)

_MAX_UPLOAD_RETRIES = 5
_RETRY_BACKOFF_BASE_S = 2.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WindowIngestionResult:
    segment_id: str
    window_idx: int
    success: bool
    error: str | None = None


@dataclass
class SegmentIngestionResult:
    segment_id: str
    windows_succeeded: int
    windows_failed: int
    skipped: bool = False
    error: str | None = None


@dataclass
class IngestionResult:
    """Summary returned by IngestionPipeline.ingest()."""
    bucket_uri: str
    segments_succeeded: int
    segments_failed: int
    segments_skipped: int
    windows_total: int
    window_results: list[WindowIngestionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"IngestionResult(bucket={self.bucket_uri!r}, "
            f"segments_ok={self.segments_succeeded}, "
            f"segments_failed={self.segments_failed}, "
            f"segments_skipped={self.segments_skipped}, "
            f"windows={self.windows_total})"
        )


# ---------------------------------------------------------------------------
# GCS storage backend
# ---------------------------------------------------------------------------

def _parse_bucket_uri(uri: str) -> tuple[str, str]:
    """Split 'gs://bucket/prefix' into ('bucket', 'prefix').

    Raises IngestionError for malformed URIs — wrong URI = wrong data destination.
    """
    uri = uri.strip().rstrip("/")
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise IngestionError(f"GCS URI has no bucket: {uri!r}")
        return bucket, prefix
    if uri.startswith("s3://"):
        raise IngestionError(
            f"S3 support not yet implemented. Got: {uri!r}\n"
            "  → Use a gs:// URI for now."
        )
    raise IngestionError(
        f"Unsupported bucket URI scheme: {uri!r}\n"
        "  → Must start with gs:// (GCS) or s3:// (S3, Phase 2)."
    )


class _GCSBackend:
    """Thin wrapper around google-cloud-storage for blob operations."""

    def __init__(self, gcs_credentials: Any, project: str | None = None) -> None:
        try:
            from google.cloud import storage  # noqa: PLC0415
        except ImportError:
            print(
                "\n[Storage/Ingestion] MISSING DEPENDENCY: google-cloud-storage\n"
                "  Install it with:  pip install google-cloud-storage\n",
                file=sys.stderr,
            )
            raise

        self._storage = storage
        self._creds = gcs_credentials
        self._project = project
        self._clients: dict[str, Any] = {}  # one client per bucket name
        self._clients_lock = threading.Lock()

    def _client_for(self, bucket_name: str) -> Any:
        with self._clients_lock:
            if bucket_name not in self._clients:
                self._clients[bucket_name] = self._storage.Client(
                    credentials=self._creds,
                    project=self._project,
                )
            return self._clients[bucket_name]

    def blob_exists(self, bucket_name: str, blob_name: str) -> bool:
        client = self._client_for(bucket_name)
        return client.bucket(bucket_name).blob(blob_name).exists()

    def upload_bytes(
        self,
        bucket_name: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload bytes with exponential-backoff retry."""
        client = self._client_for(bucket_name)
        blob = client.bucket(bucket_name).blob(blob_name)
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
            try:
                blob.upload_from_string(data, content_type=content_type)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_UPLOAD_RETRIES:
                    sleep = _RETRY_BACKOFF_BASE_S ** attempt
                    print(
                        f"[Storage/Ingestion] Upload attempt {attempt}/{_MAX_UPLOAD_RETRIES} "
                        f"failed for {blob_name!r}: {exc}. Retrying in {sleep:.0f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(sleep)
        # After all retries exhausted — loud failure
        assert last_exc is not None
        print(
            f"\n[Storage/Ingestion] UPLOAD FAILED after {_MAX_UPLOAD_RETRIES} retries "
            f"for blob {blob_name!r}.\n"
            f"  Last error: {last_exc}\n"
            f"  → Segment will be marked incomplete.",
            file=sys.stderr,
        )
        raise StorageError(
            f"Upload failed after {_MAX_UPLOAD_RETRIES} retries: {blob_name}"
        ) from last_exc

    def signed_url(
        self,
        bucket_name: str,
        blob_name: str,
        ttl_seconds: int = 3600,
        sign_as: str | None = None,
    ) -> str:
        """Generate a v4 signed GET URL for a blob."""
        client = self._client_for(bucket_name)
        expiration = datetime.timedelta(seconds=ttl_seconds)
        blob = client.bucket(bucket_name).blob(blob_name)

        if sign_as:
            from google.auth import impersonated_credentials  # noqa: PLC0415
            from google.auth import default as _adc  # noqa: PLC0415
            source_creds, _ = _adc(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            target_creds = impersonated_credentials.Credentials(
                source_credentials=source_creds,
                target_principal=sign_as,
                target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
            )
            return blob.generate_signed_url(
                version="v4", expiration=expiration, method="GET",
                credentials=target_creds,
            )
        return blob.generate_signed_url(
            version="v4", expiration=expiration, method="GET",
        )


# ---------------------------------------------------------------------------
# MP4 encoding (ffmpeg)
# ---------------------------------------------------------------------------

def _encode_and_upload_camera(
    cam_name: str,
    frames: list[Frame],
    fps: int,
    bucket_name: str,
    blob_path: str,
    backend: "_GCSBackend",
) -> None:
    """Encode one camera's frames to MP4 and upload to GCS. Raises on failure."""
    mp4_bytes = _encode_mp4(frames, fps)
    backend.upload_bytes(bucket_name, blob_path, mp4_bytes, content_type="video/mp4")


def _encode_mp4(frames: list[Frame], fps: int) -> bytes:
    """Encode a list of JPEG frames into an H.264 MP4 using ffmpeg.

    Returns raw MP4 bytes.
    Raises RuntimeError (loud) if ffmpeg fails or returns a non-zero exit code.
    """
    if not frames:
        raise RuntimeError("Cannot encode MP4: empty frame list.")

    # Decode first frame to get resolution for ffmpeg input spec.
    import numpy as np  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    first_arr = np.frombuffer(frames[0].jpeg_bytes, dtype=np.uint8)
    first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(
            "ffmpeg encode aborted: could not decode first JPEG frame. "
            "The frame bytes may be corrupt."
        )
    h, w = first_img.shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name

    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-crf", "23",
                out_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None

        for frame in frames:
            arr = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print(
                    f"[Storage/Ingestion] WARNING: could not decode frame at "
                    f"ts={frame.timestamp_us}; substituting black frame.",
                    file=sys.stderr,
                )
                img = np.zeros((h, w, 3), dtype=np.uint8)
            proc.stdin.write(img.tobytes())

        proc.stdin.close()
        # NB: do not call proc.communicate() here — stdin is already closed and
        # communicate() would re-flush it, raising "ValueError: flush of closed
        # file" on Python 3.10+. Read stderr directly and wait for exit instead.
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        proc.wait()
        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}.\n"
                f"ffmpeg stderr:\n{stderr_text[-2000:]}"
            )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        Path(out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Ingests RawSegments into the canonical windowed bucket structure.

    This class is the only one that writes to the destination bucket.
    """

    def __init__(
        self,
        gcs_credentials: Any = None,
        gcs_project: str | None = None,
    ) -> None:
        self._creds = gcs_credentials
        self._project = gcs_project
        self._backend: _GCSBackend | None = None

    def _get_backend(self) -> _GCSBackend:
        if self._backend is None:
            self._backend = _GCSBackend(self._creds, self._project)
        return self._backend

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ingest(self, request: IngestionRequest) -> IngestionResult:
        """Run end-to-end ingestion for all segment_ids in the request.

        Failure modes:
          - SourceUnreachableError  : propagated immediately (fatal)
          - SourceSchemaVersionError: propagated immediately (fatal)
          - Corrupt segment         : logged to errors/, skipped, continue
          - ffmpeg failure          : logged to errors/, window skipped, continue
          - Upload failure          : logged to errors/, segment marked incomplete, continue
        """
        bucket_name, prefix = _parse_bucket_uri(request.bucket_uri)
        backend = self._get_backend()
        now_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        result = IngestionResult(
            bucket_uri=request.bucket_uri,
            segments_succeeded=0,
            segments_failed=0,
            segments_skipped=0,
            windows_total=0,
        )

        for segment_id in request.segment_ids:
            seg_result = self._ingest_segment(
                segment_id=segment_id,
                request=request,
                bucket_name=bucket_name,
                prefix=prefix,
                backend=backend,
                now_date=now_date,
            )
            result.windows_total += seg_result.windows_succeeded + seg_result.windows_failed

            if seg_result.skipped:
                result.segments_skipped += 1
            elif seg_result.error and seg_result.windows_succeeded == 0:
                result.segments_failed += 1
                result.errors.append(f"{segment_id}: {seg_result.error}")
            else:
                result.segments_succeeded += 1
                if seg_result.error:
                    result.errors.append(f"{segment_id}: {seg_result.error}")

        # Write/update dataset-level index
        try:
            self._update_dataset_index(bucket_name, prefix, backend)
        except Exception as exc:
            print(
                f"[Storage/Ingestion] WARNING: Failed to update dataset index: {exc}",
                file=sys.stderr,
            )

        print(f"\n[Storage/Ingestion] Done. {result}", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # Per-segment
    # ------------------------------------------------------------------

    def _ingest_segment(
        self,
        segment_id: str,
        request: IngestionRequest,
        bucket_name: str,
        prefix: str,
        backend: _GCSBackend,
        now_date: str,
    ) -> SegmentIngestionResult:
        # --- Skip check ---
        if not request.force_reingest:
            seg_index_blob = f"{prefix}/segments/{segment_id}/index.json"
            if backend.blob_exists(bucket_name, seg_index_blob):
                print(
                    f"[Storage/Ingestion] Skipping {segment_id!r} (already ingested). "
                    f"Use force_reingest=True to override.",
                    file=sys.stderr,
                )
                return SegmentIngestionResult(
                    segment_id=segment_id,
                    windows_succeeded=0,
                    windows_failed=0,
                    skipped=True,
                )

        # --- Fetch segment ---
        try:
            raw = request.source.fetch_segment(segment_id)
        except (SourceUnreachableError, SourceAdapterError, StorageError) as exc:
            error_msg = str(exc)
            self._write_error_log(
                bucket_name, prefix, now_date, segment_id,
                error_type=type(exc).__name__,
                detail=error_msg,
                backend=backend,
            )
            return SegmentIngestionResult(
                segment_id=segment_id,
                windows_succeeded=0,
                windows_failed=0,
                error=error_msg,
            )
        except Exception as exc:
            wrapped = SourceAdapterError(request.source.format_name, segment_id, exc)
            self._write_error_log(
                bucket_name, prefix, now_date, segment_id,
                error_type="SourceAdapterError",
                detail=str(wrapped),
                backend=backend,
            )
            return SegmentIngestionResult(
                segment_id=segment_id,
                windows_succeeded=0,
                windows_failed=0,
                error=str(wrapped),
            )

        # --- Window and ingest ---
        windows = _build_windows(raw, request.window_config)
        windows_ok = 0
        windows_fail = 0
        window_index: list[dict[str, Any]] = []

        for window_idx, window_frames in enumerate(windows):
            ok = self._ingest_window(
                raw=raw,
                window_idx=window_idx,
                window_frames=window_frames,
                config=request.window_config,
                bucket_name=bucket_name,
                prefix=prefix,
                backend=backend,
                now_date=now_date,
            )
            if ok:
                windows_ok += 1
                window_index.append({"window_idx": window_idx, "status": "ok"})
            else:
                windows_fail += 1
                window_index.append({"window_idx": window_idx, "status": "failed"})

        # Write segment-level index
        seg_index = {
            "segment_id": segment_id,
            "source_format": raw.source_format,
            "source_schema_version": raw.source_schema_version,
            "windows_ok": windows_ok,
            "windows_failed": windows_fail,
            "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
            "windows": window_index,
        }
        try:
            backend.upload_bytes(
                bucket_name,
                f"{prefix}/segments/{segment_id}/index.json",
                json.dumps(seg_index, indent=2).encode(),
                content_type="application/json",
            )
        except StorageError as exc:
            print(
                f"[Storage/Ingestion] WARNING: Could not write segment index for "
                f"{segment_id!r}: {exc}",
                file=sys.stderr,
            )

        error_msg = (
            f"{windows_fail} of {windows_ok + windows_fail} windows failed"
            if windows_fail > 0 else None
        )
        return SegmentIngestionResult(
            segment_id=segment_id,
            windows_succeeded=windows_ok,
            windows_failed=windows_fail,
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Per-window
    # ------------------------------------------------------------------

    def _ingest_window(
        self,
        raw: RawSegment,
        window_idx: int,
        window_frames: dict[str, list[Frame]],
        config: WindowConfig,
        bucket_name: str,
        prefix: str,
        backend: _GCSBackend,
        now_date: str,
    ) -> bool:
        """Encode + upload one window. Returns True on success, False on any failure."""
        segment_id = raw.segment_id
        window_base = f"{prefix}/windows/{segment_id}/{window_idx:04d}"

        front = window_frames.get("FRONT", [])
        if not front:
            self._write_error_log(
                bucket_name, prefix, now_date, segment_id,
                error_type="EmptyWindowError",
                detail=f"Window {window_idx} has no FRONT frames.",
                backend=backend,
            )
            return False

        start_ts = front[0].timestamp_us
        end_ts = front[-1].timestamp_us

        # --- Encode + upload MP4 per camera (concurrent) ---
        # Each camera is independent: submit all, then collect results.
        # Manifest is written only after all cameras succeed.
        # Orphaned blobs from succeeded cameras on partial failure are acceptable
        # (manifest is never written, so list_windows() never surfaces this window).
        camera_futures: dict[Future[None], str] = {}
        with ThreadPoolExecutor(max_workers=len(config.cameras)) as cam_pool:
            for cam_name in config.cameras:
                frames = window_frames.get(cam_name, [])
                if not frames:
                    print(
                        f"[Storage/Ingestion] WARNING: {segment_id}/{window_idx:04d} "
                        f"camera {cam_name!r} has no frames — skipping camera.",
                        file=sys.stderr,
                    )
                    continue
                fut = cam_pool.submit(
                    _encode_and_upload_camera,
                    cam_name,
                    frames,
                    config.target_fps,
                    bucket_name,
                    f"{window_base}/camera_{cam_name}.mp4",
                    backend,
                )
                camera_futures[fut] = cam_name

        failed = False
        for fut, cam_name in camera_futures.items():
            try:
                fut.result()
            except StorageError:
                # upload_bytes already logged the detail
                failed = True
            except Exception as exc:
                print(
                    f"\n[Storage/Ingestion] ffmpeg FAILED for "
                    f"{segment_id}/{window_idx:04d}/{cam_name}: {exc}",
                    file=sys.stderr,
                )
                self._write_error_log(
                    bucket_name, prefix, now_date, segment_id,
                    error_type="FFmpegError",
                    detail=f"window={window_idx} camera={cam_name}: {exc}",
                    backend=backend,
                )
                failed = True

        if failed:
            return False

        # --- Write pose.parquet ---
        pose_slice = raw.pose.slice_window(start_ts, end_ts)
        if not pose_slice.is_empty:
            try:
                pose_bytes = _pose_to_parquet(pose_slice)
                backend.upload_bytes(
                    bucket_name,
                    f"{window_base}/pose.parquet",
                    pose_bytes,
                    content_type="application/octet-stream",
                )
            except Exception as exc:
                print(
                    f"[Storage/Ingestion] WARNING: Could not write pose.parquet for "
                    f"{segment_id}/{window_idx:04d}: {exc}",
                    file=sys.stderr,
                )

        # --- Write pose_summary.json ---
        pose_summary = _pose_summary_text(pose_slice, segment_id, window_idx)
        try:
            backend.upload_bytes(
                bucket_name,
                f"{window_base}/pose_summary.json",
                json.dumps({"summary": pose_summary}).encode(),
                content_type="application/json",
            )
        except Exception as exc:
            print(
                f"[Storage/Ingestion] WARNING: Could not write pose_summary.json "
                f"for {segment_id}/{window_idx:04d}: {exc}",
                file=sys.stderr,
            )

        # --- Write manifest.json ---
        manifest = WindowManifest(
            segment_id=segment_id,
            window_idx=window_idx,
            source_format=raw.source_format,
            source_schema_version=raw.source_schema_version,
            window_start_ts_us=start_ts,
            window_end_ts_us=end_ts,
            frame_count=len(front),
            cameras=list(config.cameras),
            ingested_at=datetime.datetime.utcnow().isoformat() + "Z",
            pose_summary=pose_summary,
        )
        try:
            backend.upload_bytes(
                bucket_name,
                f"{window_base}/manifest.json",
                json.dumps(manifest.to_json(), indent=2).encode(),
                content_type="application/json",
            )
        except StorageError:
            return False

        return True

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------

    def _write_error_log(
        self,
        bucket_name: str,
        prefix: str,
        date_str: str,
        segment_id: str,
        error_type: str,
        detail: str,
        backend: _GCSBackend,
    ) -> None:
        entry = {
            "segment_id": segment_id,
            "error_type": error_type,
            "detail": detail,
            "logged_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        blob_name = f"{prefix}/errors/ingestion/{date_str}/{segment_id}.json"
        try:
            backend.upload_bytes(
                bucket_name,
                blob_name,
                json.dumps(entry, indent=2).encode(),
                content_type="application/json",
            )
        except Exception as exc:
            # If we can't even write the error log, print loudly to stderr
            print(
                f"[Storage/Ingestion] CRITICAL: Could not write error log "
                f"for {segment_id!r} to {blob_name!r}: {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Dataset-level index
    # ------------------------------------------------------------------

    def _update_dataset_index(
        self, bucket_name: str, prefix: str, backend: _GCSBackend
    ) -> None:
        """Rebuild the top-level index/manifest.json from segment index files."""
        # This is best-effort; failure is warned not raised.
        from google.cloud import storage as gcs  # noqa: PLC0415
        client = gcs.Client(credentials=self._creds, project=self._project)
        bucket = client.bucket(bucket_name)

        seg_blobs = [
            b for b in bucket.list_blobs(prefix=f"{prefix}/segments/")
            if b.name.endswith("/index.json")
        ]
        segment_ids = [
            b.name.split("/segments/")[1].split("/index.json")[0]
            for b in seg_blobs
        ]

        manifest = {
            "bucket_uri": f"gs://{bucket_name}/{prefix}",
            "segment_count": len(segment_ids),
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "segments": sorted(segment_ids),
        }
        backend.upload_bytes(
            bucket_name,
            f"{prefix}/index/manifest.json",
            json.dumps(manifest, indent=2).encode(),
            content_type="application/json",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_windows(
    raw: RawSegment, config: WindowConfig
) -> list[dict[str, list[Frame]]]:
    """Split a RawSegment's frames into windows using the FRONT camera as anchor.

    Returns a list of dicts {camera_name: [Frame, ...]} — one dict per window.
    Each window contains exactly config.length_frames FRONT frames (or fewer
    for the last window if it doesn't divide evenly).
    """
    anchor = raw.cameras.get("FRONT", [])
    if not anchor:
        return []

    windows: list[dict[str, list[Frame]]] = []
    start = 0
    while start < len(anchor):
        end = start + config.length_frames
        anchor_slice = anchor[start:end]

        window: dict[str, list[Frame]] = {"FRONT": anchor_slice}

        # Resolve nearest frame per other camera using timestamp proximity
        anchor_start_ts = anchor_slice[0].timestamp_us
        anchor_end_ts = anchor_slice[-1].timestamp_us

        for cam_name in config.cameras:
            if cam_name == "FRONT":
                continue
            cam_frames = raw.cameras.get(cam_name, [])
            sliced = [
                f for f in cam_frames
                if anchor_start_ts <= f.timestamp_us <= anchor_end_ts
            ]
            window[cam_name] = sliced

        windows.append(window)
        start += config.stride_frames

    return windows


def _pose_to_parquet(pose: PoseArray) -> bytes:
    """Serialize a PoseArray to Parquet bytes via pyarrow."""
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        return b""  # pose.parquet is best-effort

    rows = [
        {
            "timestamp_us": r.timestamp_us,
            "x": r.x, "y": r.y, "z": r.z,
            "roll_rad": r.roll_rad,
            "pitch_rad": r.pitch_rad,
            "yaw_rad": r.yaw_rad,
        }
        for r in pose.records
    ]
    table = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _pose_summary_text(
    pose: PoseArray, segment_id: str, window_idx: int
) -> str | None:
    """Generate a short natural-language pose summary for VLM prompts.

    Returns None if pose is empty.
    """
    if pose.is_empty:
        return None
    records = pose.records
    start = records[0]
    end = records[-1]

    dx = end.x - start.x
    dy = end.y - start.y
    dist = (dx**2 + dy**2) ** 0.5
    direction = "forward" if dx > 0 else "backward"

    return (
        f"Segment {segment_id}, window {window_idx}. "
        f"Vehicle traveled approximately {dist:.1f} meters {direction} "
        f"over {len(records)} pose samples. "
        f"Start position: ({start.x:.1f}, {start.y:.1f}). "
        f"End position: ({end.x:.1f}, {end.y:.1f})."
    )
