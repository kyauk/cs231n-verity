"""Waymo scene-window JSONL -> embedding-vector JSONL transformer.

Mirrors ``pipeline/embed_scenes.py``. Each scene window is sliced into temporal
windows; every camera clip is embedded via the Cosmos Embed1 NIM and the
per-camera 256-d vectors are concatenated. Waymo has a five-camera rig, so the
concatenated embedding is 5 * 256 = 1280-d.

Usage:
  python -m waymo_pipeline.waymo_embed_scenes \
      --input-jsonl outputs/waymo_scene_windows.jsonl \
      --output-jsonl outputs/waymo_window_embeddings.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import threading

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

from waymo_pipeline.models.scene_window import (
    EXPECTED_CHANNELS,
    SceneWindow,
    WindowEmbeddingRecord,
)

COSMOS_MODEL = "nvidia/cosmos-embed1"
EMBED_DIM = 256  # per-camera; concatenated output is 256 * len(EXPECTED_CHANNELS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Waymo embedding stage."""
    p = argparse.ArgumentParser(description="Embed Waymo scene windows into vectors.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument(
        "--cosmos-url",
        default=os.getenv("COSMOS_EMBED1_URL", "http://localhost:8080"),
    )
    p.add_argument("--clip-fps", type=int, default=10)
    p.add_argument("--window-size-ticks", type=int, default=0, help="0 = whole scene.")
    p.add_argument("--window-stride-ticks", type=int, default=0, help="0 = non-overlapping.")
    p.add_argument("--resume-from", default=None, help="Existing JSONL; skip embedded ids.")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel embedding threads (each does GCS+ffmpeg+Cosmos IO).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene / window reading
# ---------------------------------------------------------------------------

def iter_scenes(path: str):
    """Stream-parse SceneWindow records from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield SceneWindow.model_validate(json.loads(line))


def load_done_ids(path: str | None) -> set[str]:
    """Load window_id values already embedded, for resume support."""
    if not path or not os.path.isfile(path):
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                wid = json.loads(line).get("window_id")
                if wid is not None:
                    done.add(str(wid))
            except (json.JSONDecodeError, TypeError):
                pass
    return done


def make_windows(scene: SceneWindow, window_size: int, window_stride: int):
    """Yield (window_id, start_tick_idx, end_tick_idx) tuples for a scene."""
    ticks = scene.ticks
    if not ticks:
        return
    size = window_size or len(ticks)
    stride = window_stride or size
    start = 0
    idx = 0
    while start < len(ticks):
        end = min(start + size, len(ticks))
        if end <= start:
            break
        yield (f"{scene.scene_token_hex}_w{idx:03d}", start, end)
        idx += 1
        start += stride


# ---------------------------------------------------------------------------
# Camera-clip rendering (Waymo: trim per-camera segment MP4 to the tick window)
# ---------------------------------------------------------------------------

def _parse_camera_uri(uri: str) -> tuple[str, int]:
    """Parse a ``gs://.../<seg>_<CAM>.mp4#t=<idx>`` URI into (gs_path, frame_idx)."""
    base, _, frag = uri.partition("#t=")
    return base, int(frag) if frag.isdigit() else 0


def render_clip_for_window(
    fs,
    scene: SceneWindow,
    start: int,
    end: int,
    channel: str,
    out_dir: str,
    clip_fps: int,
) -> str | None:
    """Trim one camera's segment MP4 to the tick window [start, end).

    Downloads the segment MP4 from GCS once, then extracts the frame range with
    ffmpeg. Returns the local clip path, or None if the channel is unavailable.
    """
    ticks = scene.ticks[start:end]
    if not ticks:
        return None
    first_uri = ticks[0].frames_by_channel.get(channel)
    last_uri = ticks[-1].frames_by_channel.get(channel)
    if not first_uri or not last_uri:
        return None

    gs_path, first_idx = _parse_camera_uri(first_uri)
    _, last_idx = _parse_camera_uri(last_uri)

    local_segment = os.path.join(out_dir, f"{channel}_segment.mp4")
    if not os.path.exists(local_segment):
        with fs.open(gs_path.replace("gs://", ""), "rb") as src, open(
            local_segment, "wb"
        ) as dst:
            dst.write(src.read())

    clip_path = os.path.join(out_dir, f"{channel}_clip.mp4")
    n_frames = max(1, last_idx - first_idx + 1)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", local_segment,
            "-vf", f"select=gte(n\\,{first_idx})*lt(n\\,{first_idx + n_frames})",
            "-vsync", "0",
            "-r", str(clip_fps),
            "-pix_fmt", "yuv420p",
            clip_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return clip_path


# ---------------------------------------------------------------------------
# Cosmos Embed1
# ---------------------------------------------------------------------------

def verify_cosmos(url: str) -> None:
    """Fail fast if the Cosmos Embed1 NIM is not reachable."""
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/health/ready", timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Cosmos NIM at {url}. Ensure the container is running."
        ) from exc
    if r.status_code != 200:
        raise RuntimeError(f"Cosmos health check failed: {r.status_code}")
    print(f"Cosmos Embed1 healthy at {url}")


def embed_clip(clip_path: str, cosmos_url: str) -> np.ndarray:
    """Send a video clip to Cosmos Embed1 NIM and return its 256-d embedding."""
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "input": [f"data:video/mp4;base64,{b64}"],
        "request_type": "query",
        "encoding_format": "float",
        "model": COSMOS_MODEL,
    }
    r = requests.post(
        f"{cosmos_url.rstrip('/')}/v1/embeddings", json=payload, timeout=120
    )
    if r.status_code != 200:
        raise RuntimeError(f"Cosmos returned {r.status_code}: {r.text[:500]}")
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Window embedding
# ---------------------------------------------------------------------------

def embed_window(
    fs,
    scene: SceneWindow,
    window_id: str,
    start: int,
    end: int,
    cosmos_url: str,
    clip_fps: int,
) -> WindowEmbeddingRecord:
    """Embed one temporal window across the Waymo five-camera rig (5 x 256d)."""
    tmp = tempfile.mkdtemp(prefix="cosmos_waymo_")
    try:
        per_camera: list[np.ndarray] = []
        for channel in EXPECTED_CHANNELS:
            clip_path = render_clip_for_window(
                fs, scene, start, end, channel, tmp, clip_fps
            )
            if not clip_path:
                per_camera.append(np.zeros(EMBED_DIM, dtype=np.float32))
                continue
            per_camera.append(embed_clip(clip_path, cosmos_url))

        concat = np.concatenate(per_camera)
        vec = concat / (np.linalg.norm(concat) + 1e-12)

        ticks = scene.ticks
        return WindowEmbeddingRecord(
            window_id=window_id,
            scene_token_hex=scene.scene_token_hex,
            log_id=scene.log_id,
            scenario_tags=scene.scenario_tags,
            window_start_ts=ticks[start].tick_timestamp,
            window_end_ts=ticks[end - 1].tick_timestamp,
            camera_set=list(EXPECTED_CHANNELS),
            embedding=vec.astype(float).tolist(),
            quality=scene.quality.model_dump(),
            metadata={"tick_count": end - start, "encoder": "cosmos_embed1", "dataset": "waymo"},
        )
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Slice scenes into windows, embed each, and write WindowEmbeddingRecord JSONL."""
    load_dotenv()
    args = parse_args()

    import gcsfs
    from google.auth import default as google_auth_default

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    verify_cosmos(args.cosmos_url)
    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    fs = gcsfs.GCSFileSystem(token=creds)

    done_ids = load_done_ids(args.resume_from)
    if done_ids:
        print(f"Resuming: skipping {len(done_ids)} already-embedded windows.")

    work: list[tuple[SceneWindow, str, int, int]] = []
    for scene in iter_scenes(args.input_jsonl):
        for window_id, start, end in make_windows(
            scene, args.window_size_ticks, args.window_stride_ticks
        ):
            if window_id not in done_ids:
                work.append((scene, window_id, start, end))

    print(f"{len(work)} windows to embed  (max_workers={args.max_workers}).")

    same_file = (
        args.resume_from
        and os.path.normpath(args.resume_from) == os.path.normpath(args.output_jsonl)
    )
    write_mode = "a" if same_file else "w"

    write_lock = threading.Lock()
    completed = 0
    failed = 0

    def _embed_one(item: tuple) -> WindowEmbeddingRecord:
        scene, window_id, start, end = item
        return embed_window(fs, scene, window_id, start, end, args.cosmos_url, args.clip_fps)

    with open(args.output_jsonl, write_mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_embed_one, item): item[1] for item in work}
            pbar = tqdm(as_completed(futures), total=len(futures), unit="window")
            for future in pbar:
                window_id = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        out.write(json.dumps(record.model_dump()) + "\n")
                        out.flush()
                    completed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"\n[warn] {window_id} failed: {exc}")
                pbar.set_postfix(ok=completed, fail=failed)

    print(f"Done. {completed} embedded, {failed} failed -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
