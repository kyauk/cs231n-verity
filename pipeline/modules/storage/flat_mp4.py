"""FlatMP4Storage — read a flat bucket of MP4 files as Verity windows.

The customer use case this serves: an AV engineer has a GCS bucket of
~10–100 MP4 segment videos (filenames as IDs) and wants quick Verity analysis
without setting up Waymo Parquet/TFRecord ingestion.

Filename conventions (chosen by `cameras` length at construction):

  cameras = ["FRONT"]                       (single-camera)
    gs://my-bucket/prefix/drive_001.mp4
    gs://my-bucket/prefix/drive_002.mp4
    → segment_id = filename stem

  cameras = ["FRONT", "FRONT_LEFT", ...]    (multi-camera)
    gs://my-bucket/prefix/drive_001_FRONT.mp4
    gs://my-bucket/prefix/drive_001_FRONT_LEFT.mp4
    → segment_id = filename minus _<camera>.mp4 suffix

Every MP4 maps to one window (`window_idx = 0`). There is no auto-slicing —
if windowing is needed, use the canonical `IngestionPipeline` instead.

Validation + safety
-------------------
- **Truncated / zero-byte MP4s are rejected at retrieval time.** Cheap GCS
  metadata gives us `blob.size` for free in the same round-trip that
  confirms existence; anything below `_MIN_MP4_BYTES` (1024) is treated as
  corruption and raises WindowStorageError before the VLM is invoked.
  Deep validation (decoding `ftyp` boxes etc.) is deliberately not done —
  the size floor catches the common corruption modes cheaply.
- **GCS client init is thread-safe** via double-checked locking on
  `_bucket_lock`; concurrent first-callers will not double-construct.

Documented assumption (still load-bearing)
------------------------------------------
- **Multi-camera filename parsing is greedy by camera-suffix match.** If a
  configured camera name happens to appear as a substring of a segment ID
  (e.g. cameras=["FRONT", "REAR"] and file `drive_REAR_001_FRONT.mp4`), the
  parser splits on the first `_<camera>` suffix it finds — here, `_FRONT` —
  yielding segment_id="drive_REAR_001". Customers should pick segment IDs
  that do not contain configured camera names as substrings.

Standalone usage:
    from pipeline.modules.storage import FlatMP4Storage

    storage = FlatMP4Storage(
        bucket_uri="gs://my-mp4-bucket/drives",
        cameras=["FRONT"],
    )
    for window in storage.list_windows():
        url = storage.get_window_video_url(
            window.segment_id, window.window_idx, camera="FRONT",
        )
        # → signed URL to drives/<segment_id>.mp4
"""

from __future__ import annotations

import datetime
import sys
import threading
from typing import Any

from pipeline.interfaces.errors import WindowStorageError
from pipeline.interfaces.window import WindowKey, WindowManifest
from pipeline.modules.storage.ingestion import _parse_bucket_uri


_SOURCE_FORMAT = "flat_mp4"
_SOURCE_SCHEMA_VERSION = "1.0"

# Floor for MP4 blob size, in bytes. Anything smaller is treated as
# corrupted (truncated upload, zero-byte placeholder, mis-renamed text file)
# and rejected at retrieval time before the VLM tries to fetch it.
# A real MP4 with one I-frame is comfortably above this.
_MIN_MP4_BYTES = 1024


class FlatMP4Storage:
    """Flat-bucket MP4 implementation of the WindowStorageBase Protocol.

    Read-only. Stateless. Treats each MP4 in the bucket as one window
    (window_idx=0). For multi-window-per-segment workflows, use the
    canonical IngestionPipeline / WindowStorage path instead.

    Parameters
    ----------
    bucket_uri       "gs://bucket/prefix" containing the MP4 files
    cameras          REQUIRED, non-empty list of camera names. Determines
                     the filename convention (see module docstring) and the
                     visual-arm embedding dimensionality
                     (len(cameras) * 256-d per Cosmos-Embed1).
    gcs_credentials  google.auth credentials; defaults to ADC
    gcs_project      GCP project for the GCS client
    sign_as          Service account email for URL signing (when ADC is a
                     user refresh-token; see README "GCS signed URLs")
    """

    def __init__(
        self,
        bucket_uri: str,
        cameras: list[str],
        gcs_credentials: Any = None,
        gcs_project: str | None = None,
        sign_as: str | None = None,
    ) -> None:
        if not cameras:
            raise ValueError(
                "FlatMP4Storage requires a non-empty `cameras` list. "
                "Declare what your MP4s contain (e.g. cameras=['FRONT'])."
            )
        self._bucket_name, self._prefix = _parse_bucket_uri(bucket_uri)
        self._cameras: tuple[str, ...] = tuple(cameras)
        self._creds = gcs_credentials
        self._project = gcs_project
        self._sign_as = sign_as
        self._client: Any = None
        self._bucket_obj: Any = None
        # Guards lazy GCS client init. See test_thread_safety.py.
        self._bucket_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> tuple[str, ...]:
        """The camera names this storage was configured with."""
        return self._cameras

    def list_windows(self, segment_id: str | None = None) -> list[WindowKey]:
        """List one WindowKey per unique segment in the bucket.

        Walks the bucket prefix once, parses each *.mp4 filename per the
        configured convention, and deduplicates by segment_id. If `segment_id`
        is given, filter to just that segment (returns at most one key).
        """
        bucket = self._get_bucket()
        seen: set[str] = set()
        for blob in bucket.list_blobs(prefix=self._prefix):
            name = blob.name
            if not name.endswith(".mp4"):
                continue
            parsed = self._parse_blob_name(name)
            if parsed is None:
                continue
            sid, _cam = parsed
            if segment_id is not None and sid != segment_id:
                continue
            seen.add(sid)
        return sorted(
            (WindowKey(segment_id=sid, window_idx=0) for sid in seen),
            key=lambda k: k.segment_id,
        )

    def get_window_video_url(
        self,
        segment_id: str,
        window_idx: int,
        camera: str = "FRONT",
        ttl_seconds: int = 3600,
    ) -> str:
        """Return a signed v4 URL for `segment_id`'s `camera` MP4.

        Raises WindowStorageError for any of:
          - window_idx != 0 (flat mode is single-window-per-segment)
          - camera not in the configured `cameras` list
          - MP4 blob does not exist (or its metadata is unreadable)
          - MP4 is empty or truncated (size < _MIN_MP4_BYTES = 1024)
          - URL signing fails (typically: ADC has no private key)

        The size check is free — GCS returns blob.size in the same metadata
        round-trip that confirms existence (blob.reload()). It catches the
        most common corruption modes (zero-byte uploads, interrupted
        transfers, mis-renamed text files) before the VLM tries to fetch
        the URL and fails opaquely.
        """
        if window_idx != 0:
            raise WindowStorageError(
                f"{segment_id}/{window_idx:04d}",
                f"FlatMP4Storage is single-window-per-segment. "
                f"Requested window_idx={window_idx}, only 0 is valid.",
            )
        if camera not in self._cameras:
            raise WindowStorageError(
                f"{segment_id}/{window_idx:04d}/{camera}",
                f"Camera {camera!r} is not configured. "
                f"This storage was constructed with cameras={list(self._cameras)}.",
            )

        blob_name = self._blob_name_for(segment_id, camera)
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        # reload() makes one HTTP HEAD-equivalent call that BOTH confirms
        # the blob exists AND populates .size / .content_type. The previous
        # exists() form gave us no metadata for free.
        try:
            blob.reload()
        except Exception as exc:
            raise WindowStorageError(
                blob_name,
                f"MP4 not found or unreadable at gs://{self._bucket_name}/{blob_name}. "
                f"Check your filename convention matches `cameras={list(self._cameras)}`. "
                f"(Underlying error: {type(exc).__name__}: {exc})",
            ) from exc

        if blob.size is None or blob.size < _MIN_MP4_BYTES:
            raise WindowStorageError(
                blob_name,
                f"MP4 at gs://{self._bucket_name}/{blob_name} is empty or "
                f"truncated (size={blob.size}, minimum={_MIN_MP4_BYTES}). "
                f"Re-upload or remove the corrupted blob.",
            )

        expiration = datetime.timedelta(seconds=ttl_seconds)
        try:
            if self._sign_as:
                from google.auth import default as _adc  # noqa: PLC0415
                from google.auth import impersonated_credentials  # noqa: PLC0415
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
                f"See README → 'GCS signed URLs' for the three working setups.",
            ) from exc

    def get_window_manifest(
        self, segment_id: str, window_idx: int
    ) -> WindowManifest:
        """Synthesize a WindowManifest for a flat-MP4 window.

        Carries `source_format='flat_mp4'`, `cameras=<configured list>`, and
        `pose_summary=None`. `frame_count` is 0 (unknown without decoding).
        """
        if window_idx != 0:
            raise WindowStorageError(
                f"{segment_id}/{window_idx:04d}",
                f"FlatMP4Storage is single-window-per-segment. "
                f"Requested window_idx={window_idx}, only 0 is valid.",
            )
        return WindowManifest(
            segment_id=segment_id,
            window_idx=0,
            source_format=_SOURCE_FORMAT,
            source_schema_version=_SOURCE_SCHEMA_VERSION,
            window_start_ts_us=0,
            window_end_ts_us=0,
            frame_count=0,
            cameras=list(self._cameras),
            ingested_at=datetime.datetime.utcnow().isoformat() + "Z",
            pose_summary=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blob_name_for(self, segment_id: str, camera: str) -> str:
        """Build the blob name per the configured filename convention."""
        if len(self._cameras) == 1:
            filename = f"{segment_id}.mp4"
        else:
            filename = f"{segment_id}_{camera}.mp4"
        return f"{self._prefix}/{filename}" if self._prefix else filename

    def _parse_blob_name(self, blob_name: str) -> tuple[str, str] | None:
        """Reverse `_blob_name_for`: blob_name → (segment_id, camera) or None.

        Returns None when the blob doesn't match the configured convention.
        Skipped blobs are silently ignored — list_windows just won't surface
        them. Stray non-conforming MP4s in the bucket are tolerated.
        """
        # Strip the prefix if present
        name = blob_name
        if self._prefix and name.startswith(self._prefix + "/"):
            name = name[len(self._prefix) + 1:]
        if "/" in name:  # nested folder — flatten via stem (use filename only)
            name = name.rsplit("/", 1)[-1]
        if not name.endswith(".mp4"):
            return None
        stem = name[:-4]

        if len(self._cameras) == 1:
            return (stem, self._cameras[0])

        # Multi-camera: filename must end with _<camera> for some configured camera.
        for cam in self._cameras:
            suffix = f"_{cam}"
            if stem.endswith(suffix):
                segment_id = stem[: -len(suffix)]
                if segment_id:
                    return (segment_id, cam)
        return None

    def _get_bucket(self) -> Any:
        # Double-checked locking: hot path is lock-free after first init.
        if self._bucket_obj is not None:
            return self._bucket_obj
        with self._bucket_lock:
            if self._bucket_obj is not None:  # another thread won the race
                return self._bucket_obj
            try:
                from google.cloud import storage  # noqa: PLC0415
            except ImportError:
                print(
                    "\n[Storage/FlatMP4Storage] MISSING DEPENDENCY: google-cloud-storage\n"
                    "  Install it with:  pip install google-cloud-storage\n",
                    file=sys.stderr,
                )
                raise
            self._client = storage.Client(
                credentials=self._creds, project=self._project
            )
            self._bucket_obj = self._client.bucket(self._bucket_name)
            return self._bucket_obj
