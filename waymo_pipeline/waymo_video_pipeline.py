"""Waymo v2 Parquet -> MP4 pipeline.

This mirrors the legacy ``pipeline/waymo_video_pipeline.py`` and is treated as
golden. Steps:
  1. Discover segment IDs from gs://waymo_open_dataset_v_2_0_1/validation/camera_image/
  2. Download Parquet per segment, reconstruct per-camera MP4s at 10 Hz
  3. Upload MP4s to gs://nvidia-adr-waymo-segment-videos/
  4. Generate segment_index.json with signed URLs (7-day expiry)
  5. Verify first segment by displaying first frame

Usage (on BREV instance with waymo-video-pipeline SA attached):
  python -m waymo_pipeline.waymo_video_pipeline --num-segments 20

Usage (local with user ADC + Token Creator on SA):
  python -m waymo_pipeline.waymo_video_pipeline --num-segments 20 \
    --sign-as waymo-video-pipeline@nvidia-adr.iam.gserviceaccount.com
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gcsfs
import numpy as np
import pyarrow.parquet as pq
from google.auth import default as google_auth_default
from google.cloud import storage
from tqdm import tqdm

# -- Constants ----------------------------------------------------------------
# All bucket/prefix values can be overridden at runtime via env vars so that
# any GCS-hosted driving dataset can be pointed at without code changes.

SOURCE_BUCKET = os.environ.get("WAYMO_SOURCE_BUCKET", "waymo_open_dataset_v_2_0_1")
SOURCE_PREFIX = os.environ.get("WAYMO_SOURCE_PREFIX", "validation/camera_image")
DEST_BUCKET = os.environ.get("WAYMO_DEST_BUCKET", "nvidia-adr-waymo-segment-videos")
DEST_PREFIX = "segments"
INDEX_FILE = "segment_index.json"
FPS = 10
SIGNED_URL_EXPIRY_DAYS = 7

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


# -- Step 1: Discover segments ------------------------------------------------

def discover_segments(
    fs: gcsfs.GCSFileSystem,
    bucket: str = SOURCE_BUCKET,
    prefix: str = SOURCE_PREFIX,
) -> list[str]:
    """Enumerate segment ids (Parquet files) under gs://<bucket>/<prefix>/."""
    print(f"\n[Step 1] Discovering segments in gs://{bucket}/{prefix}/")
    files = fs.ls(f"{bucket}/{prefix}/")
    segment_ids: list[str] = []
    for f in files:
        name = Path(f).name
        if name.endswith(".parquet"):
            segment_ids.append(name[: -len(".parquet")])
    print(f"  Found {len(segment_ids)} segments total.")
    return sorted(segment_ids)


# -- Step 2: Reconstruct MP4s for one segment ---------------------------------

def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode one JPEG-encoded camera frame into a BGR array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG frame")
    return img


def encode_mp4_ffmpeg(frames: list[np.ndarray], out_path: str, fps: int = FPS) -> None:
    """Encode a list of BGR frames into an H.264 MP4 via ffmpeg."""
    if not frames:
        raise ValueError("No frames to encode")
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
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
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def process_segment(
    segment_id: str,
    fs: gcsfs.GCSFileSystem,
    out_dir: Path,
    bucket: str = SOURCE_BUCKET,
    prefix: str = SOURCE_PREFIX,
) -> dict[str, str]:
    """Reconstruct per-camera MP4s for one segment. Returns {camera: mp4_path}."""
    parquet_path = f"{bucket}/{prefix}/{segment_id}.parquet"
    print(f"  Reading Parquet: {parquet_path}")

    with fs.open(parquet_path, "rb") as f:
        pf = pq.ParquetFile(f)
        schema_names = pf.schema_arrow.names

    def find_col(candidates: list[str]) -> str:
        for c in candidates:
            if c in schema_names:
                return c
        raise KeyError(f"None of {candidates} found. Available: {schema_names[:20]}")

    col_image = find_col(["[CameraImageComponent].image", "image", "camera_image"])
    col_camera = find_col(["key.camera_name", "camera_name"])
    col_ts = find_col(["key.frame_timestamp_micros", "frame_timestamp_micros"])

    with fs.open(parquet_path, "rb") as f:
        table = pq.read_table(f, columns=[col_image, col_camera, col_ts])

    df = table.to_pandas().sort_values(col_ts)

    camera_paths: dict[str, str] = {}
    for cam_int, cam_name in CAMERA_NAMES.items():
        cam_df = df[df[col_camera] == cam_int].reset_index(drop=True)
        if cam_df.empty:
            print(f"    [{cam_name}] No frames found, skipping.")
            continue
        frames = [decode_jpeg(bytes(row[col_image])) for _, row in cam_df.iterrows()]
        out_path = out_dir / f"{segment_id}_{cam_name}.mp4"
        encode_mp4_ffmpeg(frames, str(out_path))
        camera_paths[cam_name] = str(out_path)
        print(f"    [{cam_name}] {len(frames)} frames -> {out_path.name}")

    return camera_paths


# -- Step 3: Upload to GCS ----------------------------------------------------

def upload_mp4s(
    camera_paths: dict[str, str],
    segment_id: str,
    gcs_client: storage.Client,
) -> dict[str, str]:
    """Upload reconstructed MP4s. Returns {camera: gcs_blob_name}."""
    bucket = gcs_client.bucket(DEST_BUCKET)
    blob_names: dict[str, str] = {}
    for cam_name, local_path in camera_paths.items():
        blob_name = f"{DEST_PREFIX}/{segment_id}/{segment_id}_{cam_name}.mp4"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path, content_type="video/mp4")
        blob_names[cam_name] = blob_name
        print(f"    Uploaded gs://{DEST_BUCKET}/{blob_name}")
    return blob_names


# -- Step 4: Generate signed URLs ---------------------------------------------

def sign_blob(
    blob: storage.Blob,
    sign_as: str | None,
    expiry_days: int = SIGNED_URL_EXPIRY_DAYS,
) -> str:
    """Generate a v4 signed GET URL for one blob (impersonating an SA if asked)."""
    expiration = datetime.timedelta(days=expiry_days)
    if sign_as:
        from google.auth import impersonated_credentials
        source_creds, _ = google_auth_default(
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


def generate_signed_urls(
    blob_names: dict[str, str],
    gcs_client: storage.Client,
    sign_as: str | None,
) -> dict[str, str]:
    """Generate signed URLs for every camera blob of a segment."""
    bucket = gcs_client.bucket(DEST_BUCKET)
    return {
        cam: sign_blob(bucket.blob(blob_name), sign_as)
        for cam, blob_name in blob_names.items()
    }


# -- Step 5: Verify -----------------------------------------------------------

def verify_first_frame(mp4_path: str) -> None:
    """Decode the first frame of a reconstructed MP4 as a sanity check."""
    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"  [Verify] Could not read first frame from {mp4_path}")
        return
    h, w = frame.shape[:2]
    print(f"  [Verify] First frame decoded OK -- resolution: {w}x{h}")
    thumb_path = mp4_path.replace(".mp4", "_thumb.jpg")
    cv2.imwrite(thumb_path, frame)
    print(f"  [Verify] Thumbnail saved to {thumb_path}")


# -- Resumability -------------------------------------------------------------

def segment_already_done(segment_id: str, gcs_client: storage.Client) -> bool:
    """True when every camera MP4 for the segment already exists in GCS."""
    bucket = gcs_client.bucket(DEST_BUCKET)
    for cam_name in CAMERA_NAMES.values():
        blob_name = f"{DEST_PREFIX}/{segment_id}/{segment_id}_{cam_name}.mp4"
        if not bucket.blob(blob_name).exists():
            return False
    return True


# -- Parallel worker ----------------------------------------------------------

def _process_one(
    seg_id: str,
    seg_index: int,
    total: int,
    out_dir: Path,
    creds: object,
    project: str,
    src_bucket: str,
    src_prefix: str,
    sign_as: str | None,
    skip_upload: bool,
    verify_lock: threading.Lock,
    verified: list[bool],
) -> tuple[str, dict[str, str] | None]:
    """Process one segment end-to-end in a thread. Returns (seg_id, url_map | None)."""
    # Each thread needs its own GCS clients — neither gcsfs nor storage.Client is thread-safe.
    fs = gcsfs.GCSFileSystem(token=creds)
    gcs_client = storage.Client(credentials=creds, project=project or "nvidia-adr")

    print(f"[{seg_index}/{total}] Segment: {seg_id}")

    if not skip_upload and segment_already_done(seg_id, gcs_client):
        print(f"  [{seg_id}] Already processed -- skipping.")
        blob_names = {
            cam: f"{DEST_PREFIX}/{seg_id}/{seg_id}_{cam}.mp4"
            for cam in CAMERA_NAMES.values()
        }
        return seg_id, generate_signed_urls(blob_names, gcs_client, sign_as)

    seg_out_dir = out_dir / seg_id
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        camera_paths = process_segment(seg_id, fs, seg_out_dir, bucket=src_bucket, prefix=src_prefix)
    except Exception as e:  # noqa: BLE001
        print(f"  [{seg_id}] ERROR: {e}")
        return seg_id, None

    with verify_lock:
        if not verified[0] and camera_paths:
            front_path = camera_paths.get("FRONT") or next(iter(camera_paths.values()))
            print(f"\n[Step 5] Verifying {seg_id}...")
            verify_first_frame(front_path)
            verified[0] = True

    if skip_upload:
        return seg_id, {cam: str(p) for cam, p in camera_paths.items()}

    blob_names = upload_mp4s(camera_paths, seg_id, gcs_client)
    urls = generate_signed_urls(blob_names, gcs_client, sign_as)

    for p in camera_paths.values():
        try:
            os.remove(p)
        except OSError:
            pass

    return seg_id, urls


# -- Main ---------------------------------------------------------------------

def main() -> None:
    """Run end-to-end Waymo Parquet -> MP4 -> GCS reconstruction and indexing."""
    parser = argparse.ArgumentParser(description="Waymo v2 -> MP4 pipeline")
    parser.add_argument("--num-segments", type=int, default=20,
                        help="Number of segments to process (0 = all)")
    parser.add_argument("--out-dir", type=str, default="/tmp/waymo_mp4s")
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent segments to process")
    parser.add_argument("--sign-as", type=str, default=None,
                        help="Service account to impersonate for signing (local dev only).")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--index-out", type=str, default="segment_index.json")
    parser.add_argument(
        "--data-source-uri", default=os.environ.get("DATA_SOURCE_URI", ""),
        help="GCS URI of dataset root, e.g. gs://bucket/validation/camera_image",
    )
    args = parser.parse_args()

    src_bucket, src_prefix = SOURCE_BUCKET, SOURCE_PREFIX
    uri = (args.data_source_uri or "").strip().rstrip("/")
    if uri.startswith("gs://"):
        without_scheme = uri[len("gs://"):]
        _b, _, _p = without_scheme.partition("/")
        src_bucket = _b
        src_prefix = _p or src_prefix

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    creds, project = google_auth_default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    fs = gcsfs.GCSFileSystem(token=creds)

    all_segments = discover_segments(fs, bucket=src_bucket, prefix=src_prefix)
    segments = all_segments if args.num_segments == 0 else all_segments[: args.num_segments]
    print(f"\nProcessing {len(segments)} of {len(all_segments)} segments with {args.workers} workers.")

    index: dict[str, dict[str, str]] = {}
    verify_lock = threading.Lock()
    verified: list[bool] = [False]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_one,
                seg_id, i + 1, len(segments),
                out_dir, creds, project,
                src_bucket, src_prefix,
                args.sign_as, args.skip_upload,
                verify_lock, verified,
            ): seg_id
            for i, seg_id in enumerate(segments)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Segments", unit="seg"):
            seg_id, urls = future.result()
            if urls is not None:
                index[seg_id] = urls

    index_path = Path(args.index_out)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n[Step 4] segment_index.json written -> {index_path.resolve()}")
    print(f"         {len(index)} segments indexed.")
    print("\nDone.")


if __name__ == "__main__":
    main()
