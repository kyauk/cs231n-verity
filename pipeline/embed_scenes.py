"""Scene-window JSONL to embedding-vector JSONL transformer.

Uses render_window_clip to assemble per-camera clips from S3 frames,
then embeds each clip via Cosmos Embed1 and concatenates the vectors.

Usage:
  python -m pipeline.embed_scenes \
      --input-jsonl outputs/scene_windows.jsonl \
      --output-jsonl outputs/window_embeddings.jsonl
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import shutil
import tempfile
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from pipeline.models.scene_window import (
    EXPECTED_CHANNELS,
    SceneWindow,
    WindowEmbeddingRecord,
)
from pipeline.render_window_clip import render_clips_for_window
from pipeline.s3_retrieval import make_s3_client


# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

COSMOS_MODEL = "nvidia/cosmos-embed1"
EMBED_DIM = 256  # per-camera; concatenated output is 256 * len(EXPECTED_CHANNELS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    '''
    Purpose: Parse CLI arguments for the embedding stage.
    Parameters: None.
    Returns: argparse.Namespace with input/output paths, Cosmos URL, and
        window parameters.
    Called by: main().
    Calls: argparse.ArgumentParser().
    '''
    p = argparse.ArgumentParser(
        description="Embed scene windows into vectors.",
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument(
        "--cosmos-url",
        default=os.getenv("COSMOS_EMBED1_URL", "http://localhost:8080"),
    )
    p.add_argument(
        "--frame-resize", default="224x224",
        help="WxH for assembled clips.",
    )
    p.add_argument("--clip-fps", type=int, default=10)
    p.add_argument(
        "--window-size-ticks", type=int, default=0,
        help="0 = whole scene.",
    )
    p.add_argument(
        "--window-stride-ticks", type=int, default=0,
        help="0 = non-overlapping.",
    )
    p.add_argument(
        "--resume-from", default=None,
        help="Existing JSONL; skip embedded window_ids.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene / window reading
# ---------------------------------------------------------------------------

def iter_scenes(path: str):
    '''
    Purpose: Stream parse SceneWindow records from a JSONL file.
    Parameters:
        path (str): JSONL file path.
    Returns: Generator[SceneWindow].
    Called by: main(), render_window_clip CLI.
    Calls: json.loads(), SceneWindow.model_validate().
    '''
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield SceneWindow.model_validate(json.loads(line))


def load_done_ids(path: str | None) -> set[str]:
    '''
    Purpose: Load window_id values already embedded for resume support.
    Parameters:
        path (str | None): Existing embedding JSONL, or None.
    Returns:
        set[str]: Window IDs to skip.
    Called by: main().
    Calls: open(), json.loads().
    '''
    if not path or not os.path.isfile(path):
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                wid = row.get("window_id")
                if wid is not None:
                    done.add(str(wid))
            except (json.JSONDecodeError, TypeError):
                pass
    return done


def make_windows(scene: SceneWindow, window_size: int, window_stride: int):
    '''
    Purpose: Yield (window_id, start_tick_idx, end_tick_idx) tuples for
        a scene, deterministically sliced into temporal windows.
    Parameters:
        scene (SceneWindow): Source scene with ordered ticks.
        window_size (int): Ticks per window (0 = whole scene).
        window_stride (int): Stride between windows (0 = non-overlapping).
    Returns: Generator of (str, int, int) tuples.
    Called by: main(), render_window_clip CLI.
    Calls: None.
    '''
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
# Cosmos Embed1
# ---------------------------------------------------------------------------

def verify_cosmos(url: str) -> None:
    '''
    Purpose: Fail fast if Cosmos Embed1 NIM is not reachable.
    Parameters:
        url (str): Base URL of the Cosmos Embed1 NIM.
    Returns: None (raises on failure).
    Called by: main().
    Calls: requests.get().
    '''
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/health/ready", timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Cosmos NIM at {url}. "
            "Ensure the container is running."
        ) from exc
    if r.status_code != 200:
        raise RuntimeError(
            f"Cosmos health check failed: {r.status_code}"
        )
    print(f"Cosmos Embed1 healthy at {url}")


def embed_clip(clip_path: str, cosmos_url: str) -> np.ndarray:
    '''
    Purpose: Send a video clip to Cosmos Embed1 NIM and return its
        256-d embedding vector.
    Parameters:
        clip_path (str): Local path to an mp4 video clip.
        cosmos_url (str): Base URL of the Cosmos Embed1 NIM service.
    Returns:
        np.ndarray: 256-dimensional float32 embedding.
    Called by: embed_window().
    Calls: requests.post().
    '''
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "input": [f"data:video/mp4;base64,{b64}"],
        "request_type": "query",
        "encoding_format": "float",
        "model": COSMOS_MODEL,
    }
    r = requests.post(
        f"{cosmos_url.rstrip('/')}/v1/embeddings",
        json=payload, timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Cosmos returned {r.status_code}: {r.text[:500]}"
        )
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Window embedding: render clips -> embed -> concatenate
# ---------------------------------------------------------------------------

def embed_window(
    s3,
    scene: SceneWindow,
    window_id: str,
    start: int,
    end: int,
    cosmos_url: str,
    frame_size: tuple[int, int],
    clip_fps: int,
) -> WindowEmbeddingRecord:
    '''
    Purpose: Embed one temporal window via Cosmos Embed1 with
        8-camera concatenation (8 x 256d = 2048d).  Delegates clip
        rendering to render_window_clip.render_clips_for_window().
    Parameters:
        s3: boto3 S3 client.
        scene (SceneWindow): Parent scene.
        window_id (str): Unique window identifier.
        start (int): Start tick index (inclusive).
        end (int): End tick index (exclusive).
        cosmos_url (str): Cosmos NIM base URL.
        frame_size (tuple[int, int]): (width, height) for clips.
        clip_fps (int): FPS for assembled clips.
    Returns:
        WindowEmbeddingRecord: Embedding record with 2048d vector.
    Called by: main().
    Calls: render_clips_for_window(), embed_clip().
    '''
    tmp = tempfile.mkdtemp(prefix="cosmos_")
    try:
        clips = render_clips_for_window(
            s3, scene, start, end,
            EXPECTED_CHANNELS, tmp, frame_size, clip_fps,
        )

        per_camera: list[np.ndarray] = []
        for channel in EXPECTED_CHANNELS:
            clip_path = clips.get(channel)
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
            window_start_ts=ticks[start].lidar_timestamp,
            window_end_ts=ticks[end - 1].lidar_timestamp,
            camera_set=list(EXPECTED_CHANNELS),
            embedding=vec.astype(float).tolist(),
            quality=scene.quality.model_dump(),
            metadata={"tick_count": end - start, "encoder": "cosmos_embed1"},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    '''
    Purpose: Slice scenes into temporal windows, encode each window,
        and write WindowEmbeddingRecord JSONL output.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: iter_scenes(), make_windows(), load_done_ids(),
        verify_cosmos(), embed_window().
    '''
    load_dotenv()
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    frame_size = tuple(int(x) for x in args.frame_resize.split("x"))

    verify_cosmos(args.cosmos_url)
    s3 = make_s3_client()

    done_ids = load_done_ids(args.resume_from)
    if done_ids:
        print(f"Resuming: skipping {len(done_ids)} already-embedded windows.")

    work: list[tuple[SceneWindow, str, int, int]] = []
    for scene in iter_scenes(args.input_jsonl):
        for window_id, start, end in make_windows(
            scene, args.window_size_ticks, args.window_stride_ticks,
        ):
            if window_id not in done_ids:
                work.append((scene, window_id, start, end))

    print(f"{len(work)} windows to embed.")

    same_file = (
        args.resume_from
        and os.path.normpath(args.resume_from)
        == os.path.normpath(args.output_jsonl)
    )
    write_mode = "a" if same_file else "w"

    with open(args.output_jsonl, write_mode, encoding="utf-8") as out:
        for scene, window_id, start, end in tqdm(work, unit="window"):
            record = embed_window(
                s3, scene, window_id, start, end,
                args.cosmos_url, frame_size, args.clip_fps,
            )
            out.write(json.dumps(record.model_dump()) + "\n")
            out.flush()

    print(f"Done. {len(work)} windows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
