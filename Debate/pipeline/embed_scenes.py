"""Scene-window JSONL to embedding-vector JSONL transformer.

Supports temporal windowing and multiple encoder backends:
  - fake: deterministic placeholder for pipeline validation
  - cosmos_embed1: NVIDIA Cosmos Embed1 NIM for video embeddings
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
from typing import Iterable
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from pipeline.models.scene_window import (
    EXPECTED_CHANNELS,
    SceneWindow,
    WindowEmbeddingRecord,
    WindowSpec,
)


def parse_args() -> argparse.Namespace:
    '''
    Purpose: Parse CLI arguments for the window-based embedding stage.
    Parameters: None.
    Returns: argparse.Namespace with input/output, encoder, and window
        options.
    Called by: main().
    Calls: argparse.ArgumentParser().
    '''
    p = argparse.ArgumentParser(
        description="Embed scene-window artifacts into vectors.",
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument(
        "--encoder",
        default="fake",
        choices=["fake", "cosmos_embed1"],
    )
    p.add_argument("--embedding-dim", type=int, default=2048)

    # Temporal window parameters
    p.add_argument(
        "--window-size-ticks", type=int, default=16,
        help="Ticks per window. 0 = entire scene as one window.",
    )
    p.add_argument(
        "--window-stride-ticks", type=int, default=16,
        help="Stride between windows. Equals window-size for "
             "non-overlapping.",
    )
    p.add_argument(
        "--max-windows-per-scene", type=int, default=0,
        help="Cap windows per scene. 0 = unlimited.",
    )

    # Cosmos Embed1 parameters
    p.add_argument(
        "--cosmos-url",
        default=os.getenv(
            "COSMOS_EMBED1_URL", "http://localhost:8080",
        ),
    )
    p.add_argument(
        "--frame-resize", default="224x224",
        help="WxH pixel size for assembled video frames.",
    )
    p.add_argument(
        "--clip-fps", type=int, default=10,
        help="Frames-per-second for assembled video clips.",
    )
    return p.parse_args()


# -------------------------------------------------------------------
# Scene JSONL reader
# -------------------------------------------------------------------

def iter_scene_windows(path: str) -> Iterable[SceneWindow]:
    '''
    Purpose: Stream parse SceneWindow records from JSONL.
    Parameters:
        path (str): JSONL file path containing scene artifacts.
    Returns:
        Iterable[SceneWindow]: Parsed scene-window objects.
    Called by: main().
    Calls: json.loads(), SceneWindow.model_validate().
    '''
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield SceneWindow.model_validate(json.loads(line))


# -------------------------------------------------------------------
# Window slicing
# -------------------------------------------------------------------

def slice_windows(
    scene: SceneWindow,
    window_size: int,
    window_stride: int,
    max_windows: int,
) -> list[WindowSpec]:
    '''
    Purpose: Deterministically slice a scene's ticks into temporal
        windows.
    Parameters:
        scene (SceneWindow): Source scene with ordered ticks.
        window_size (int): Ticks per window (0 = whole scene).
        window_stride (int): Stride between window starts.
        max_windows (int): Cap on windows per scene (0 = unlimited).
    Returns:
        list[WindowSpec]: Ordered window specs with stable IDs.
    Called by: main().
    Calls: None.
    '''
    ticks = scene.ticks
    if not ticks:
        return []

    effective_size = window_size if window_size > 0 else len(ticks)
    effective_stride = (
        window_stride if window_stride > 0 else effective_size
    )

    windows: list[WindowSpec] = []
    start = 0
    while start < len(ticks):
        end = min(start + effective_size, len(ticks))
        if end <= start:
            break
        idx = len(windows)
        windows.append(WindowSpec(
            window_id=f"{scene.scene_token_hex}_w{idx:03d}",
            scene_token_hex=scene.scene_token_hex,
            log_id=scene.log_id,
            window_index=idx,
            start_tick_idx=start,
            end_tick_idx=end,
            start_ts=ticks[start].lidar_timestamp,
            end_ts=ticks[end - 1].lidar_timestamp,
            camera_set=list(EXPECTED_CHANNELS),
            tick_count=end - start,
        ))
        if max_windows > 0 and len(windows) >= max_windows:
            break
        start += effective_stride
    return windows


# -------------------------------------------------------------------
# Frame materialization (S3 download)
# -------------------------------------------------------------------

def materialize_window_frames(
    s3_client: object,
    scene: SceneWindow,
    window_spec: WindowSpec,
    tmp_dir: str,
) -> dict[str, list[str]]:
    '''
    Purpose: Download camera frames from S3 for one temporal window.
    Parameters:
        s3_client (object): boto3 S3 client.
        scene (SceneWindow): Parent scene holding tick frame URIs.
        window_spec (WindowSpec): Window defining tick range.
        tmp_dir (str): Local temp directory for downloaded frames.
    Returns:
        dict[str, list[str]]: Channel -> ordered list of local file
            paths.
    Called by: encode_window_cosmos().
    Calls: s3_client.download_file().
    '''
    frames: dict[str, list[str]] = {
        ch: [] for ch in EXPECTED_CHANNELS
    }
    start = window_spec.start_tick_idx
    end = window_spec.end_tick_idx

    for rel_idx, tick_idx in enumerate(range(start, end)):
        tick = scene.ticks[tick_idx]
        for channel in EXPECTED_CHANNELS:
            uri = tick.frames_by_channel.get(channel)
            if not uri:
                continue
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            ext = os.path.splitext(key)[1] or ".jpg"
            local_path = os.path.join(
                tmp_dir, channel, f"{rel_idx:06d}{ext}",
            )
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3_client.download_file(bucket, key, local_path)
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise
                # Fallback: layout BUCKET/<prefix>/<log_id>/<CAM_*>/<filename>
                # e.g. nuplan-v1.1/sensor_blobs/camera_0/nuplan-v1.1_mini_camera_0/<file_name>/CAM_B0/...
                prefix_before_log = os.getenv(
                    "S3_NUPLAN_EMBED_CAMERA_PREFIX",
                    "nuplan-v1.1/sensor_blobs/camera_0/nuplan-v1.1_mini_camera_0",
                )
                alt_key = f"{prefix_before_log.rstrip('/')}/{scene.log_id}/{channel}/{os.path.basename(key)}"
                s3_client.download_file(bucket, alt_key, local_path)
            frames[channel].append(local_path)
    return frames


# -------------------------------------------------------------------
# Video assembly (frames -> mp4)
# -------------------------------------------------------------------

def assemble_clip(
    frame_paths: list[str],
    output_path: str,
    fps: int = 10,
    size: tuple[int, int] = (224, 224),
) -> str:
    '''
    Purpose: Stitch ordered image frames into an mp4 video clip.
    Parameters:
        frame_paths (list[str]): Ordered local image file paths.
        output_path (str): Destination mp4 path.
        fps (int): Video frames per second.
        size (tuple[int, int]): (width, height) for output frames.
    Returns:
        str: Path to the assembled clip.
    Called by: encode_window_cosmos().
    Calls: cv2.VideoWriter(), cv2.imread(), cv2.resize().
    '''
    # Use mp4v (MPEG-4); avc1/H.264 can fail on headless/CI (h264_v4l2m2m device)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    frames_written = 0
    try:
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, size)
            writer.write(img)
            frames_written += 1
        if frames_written == 0 and frame_paths:
            # Avoid 0-frame clip (causes Cosmos "All inputs failed")
            black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            writer.write(black)
            frames_written = 1
    finally:
        writer.release()
    return output_path


def _reencode_mp4_to_h264_if_available(clip_path: str) -> str:
    '''
    Purpose: Re-encode MP4 to H.264 with ffmpeg so Cosmos receives
        recommended format; return path to use (H.264 file or original).
    Parameters:
        clip_path (str): Path to assembled mp4v clip.
    Returns:
        str: Path to clip to send (H.264 temp file or clip_path).
    Called by: encode_window_cosmos().
    Calls: subprocess.run(), shutil.copy().
    '''
    import subprocess
    out_path = clip_path + ".h264.mp4"
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", clip_path,
                "-c:v", "libx264", "-preset", "fast",
                "-f", "mp4", "-movflags", "+faststart",
                out_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0 and os.path.isfile(out_path):
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return clip_path


# -------------------------------------------------------------------
# Cosmos Embed1 client (NVIDIA recommended payload format)
# -------------------------------------------------------------------

COSMOS_REQUEST_TYPE_QUERY = "query"
COSMOS_ENCODING_FORMAT = "float"
COSMOS_MODEL = "nvidia/cosmos-embed1"


def build_cosmos_embeddings_payload(
    input_items: list[str],
    request_type: str = COSMOS_REQUEST_TYPE_QUERY,
) -> dict:
    '''
    Purpose: Build the Cosmos Embed1 /v1/embeddings request body in
        NVIDIA-recommended structure (input, request_type, encoding_format,
        model).
    Parameters:
        input_items (list[str]): List of input strings, e.g. one
            "data:video/mp4;base64,..." for query, or multiple
            "data:video/mp4;presigned_url,<url>" for bulk_video.
        request_type (str): "query" or "bulk_video". Default query.
    Returns:
        dict: Payload for POST /v1/embeddings.
    Called by: cosmos_embed_clip().
    Calls: None.
    '''
    return {
        "input": input_items,
        "request_type": request_type,
        "encoding_format": COSMOS_ENCODING_FORMAT,
        "model": COSMOS_MODEL,
    }


def cosmos_embed_clip(
    clip_path: str,
    cosmos_url: str,
) -> np.ndarray:
    '''
    Purpose: Send a video clip to Cosmos Embed1 NIM and return its
        256-d embedding vector.
    Parameters:
        clip_path (str): Local path to an mp4 video clip.
        cosmos_url (str): Base URL of the Cosmos Embed1 NIM service.
    Returns:
        np.ndarray: 256-dimensional float32 embedding.
    Called by: encode_window_cosmos().
    Calls: build_cosmos_embeddings_payload(), requests.post().
    '''
    with open(clip_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    input_item = f"data:video/mp4;base64,{video_b64}"
    payload = build_cosmos_embeddings_payload(
        [input_item],
        request_type=COSMOS_REQUEST_TYPE_QUERY,
    )
    url = f"{cosmos_url.rstrip('/')}/v1/embeddings"
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Cosmos Embed1 returned HTTP {resp.status_code}: "
            f"{resp.text[:500]}"
        )
    data = resp.json()
    vec = data["data"][0]["embedding"]
    return np.array(vec, dtype=np.float32)


def verify_cosmos_health(cosmos_url: str) -> None:
    '''
    Purpose: Fail fast if Cosmos Embed1 NIM is not reachable.
    Parameters:
        cosmos_url (str): Base URL of the Cosmos Embed1 NIM.
    Returns: None (raises on failure).
    Called by: main().
    Calls: requests.get().
    '''
    url = f"{cosmos_url.rstrip('/')}/v1/health/ready"
    try:
        resp = requests.get(url, timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Cosmos Embed1 NIM at {url}. "
            "Ensure the container is running: "
            "docker compose --profile gpu up -d cosmos-embed1"
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"Cosmos Embed1 health check failed ({resp.status_code})"
            f": {resp.text[:300]}"
        )
    print(f"Cosmos Embed1 NIM healthy at {cosmos_url}")


# -------------------------------------------------------------------
# Encoder dispatch
# -------------------------------------------------------------------

def encode_window_cosmos(
    s3_client: object,
    scene: SceneWindow,
    window_spec: WindowSpec,
    cosmos_url: str,
    frame_size: tuple[int, int],
    clip_fps: int,
) -> np.ndarray:
    '''
    Purpose: Embed one temporal window via Cosmos Embed1 with
        8-camera concatenation (8 x 256d = 2048d).
    Parameters:
        s3_client (object): boto3 S3 client.
        scene (SceneWindow): Parent scene.
        window_spec (WindowSpec): Window to embed.
        cosmos_url (str): Cosmos NIM base URL.
        frame_size (tuple[int, int]): (width, height) for clips.
        clip_fps (int): FPS for assembled clips.
    Returns:
        np.ndarray: L2-normalized 2048d float32 vector.
    Called by: main().
    Calls: materialize_window_frames(), assemble_clip(),
        cosmos_embed_clip().
    '''
    tmp_dir = tempfile.mkdtemp(prefix="cosmos_frames_")
    try:
        camera_frames = materialize_window_frames(
            s3_client, scene, window_spec, tmp_dir,
        )
        camera_embeddings: list[np.ndarray] = []
        for channel in EXPECTED_CHANNELS:
            frames = camera_frames.get(channel, [])
            if not frames:
                camera_embeddings.append(
                    np.zeros(256, dtype=np.float32),
                )
                continue
            clip_path = os.path.join(tmp_dir, f"{channel}.mp4")
            assemble_clip(
                frames, clip_path, fps=clip_fps, size=frame_size,
            )
            send_path = _reencode_mp4_to_h264_if_available(clip_path)
            emb = cosmos_embed_clip(send_path, cosmos_url)
            camera_embeddings.append(emb)

        concatenated = np.concatenate(camera_embeddings)
        norm = np.linalg.norm(concatenated)
        return concatenated / (norm + 1e-12)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fake_encode_window(
    window_spec: WindowSpec,
    embedding_dim: int,
) -> np.ndarray:
    '''
    Purpose: Deterministic placeholder encoder keyed on window_id.
    Parameters:
        window_spec (WindowSpec): Window to encode.
        embedding_dim (int): Target vector dimension.
    Returns:
        np.ndarray: Deterministic L2-normalized float32 vector.
    Called by: main().
    Calls: hashlib.sha256(), numpy.random.default_rng().
    '''
    seed_material = (
        f"{window_spec.window_id}|"
        f"{window_spec.scene_token_hex}|"
        f"{window_spec.tick_count}"
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(
        size=(embedding_dim,), dtype=np.float32,
    )
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


# -------------------------------------------------------------------
# Legacy single-scene encoder (preserved for backward compat)
# -------------------------------------------------------------------

def fake_encode_scene(
    scene: SceneWindow,
    embedding_dim: int,
) -> np.ndarray:
    '''
    Purpose: Deterministic placeholder encoder for pipeline bring-up.
    Parameters:
        scene (SceneWindow): Extracted scene artifact.
        embedding_dim (int): Target vector dimension.
    Returns:
        np.ndarray: Deterministic float32 embedding vector.
    Called by: encode_scene().
    Calls: hashlib.sha256(), numpy.random.default_rng().
    '''
    seed_material = (
        f"{scene.scene_token_hex}|{scene.log_id}|"
        f"{len(scene.ticks)}|{','.join(scene.scenario_tags)}"
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(
        size=(embedding_dim,), dtype=np.float32,
    )
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


def encode_scene(
    scene: SceneWindow,
    encoder: str,
    embedding_dim: int,
) -> np.ndarray:
    '''
    Purpose: Legacy dispatch for single-scene encoding.
    Parameters:
        scene (SceneWindow): Scene artifact.
        encoder (str): Encoder backend id.
        embedding_dim (int): Target vector dimension.
    Returns:
        np.ndarray: Scene embedding vector.
    Called by: External callers using the old API.
    Calls: fake_encode_scene().
    '''
    if encoder == "fake":
        return fake_encode_scene(scene, embedding_dim)
    raise NotImplementedError(
        f"Encoder '{encoder}' not implemented for legacy "
        "single-scene path. Use the window-based pipeline."
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    '''
    Purpose: Slice scenes into temporal windows, encode each window,
        and write WindowEmbeddingRecord JSONL output.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: iter_scene_windows(), slice_windows(),
        fake_encode_window(), encode_window_cosmos().
    '''
    load_dotenv()
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    frame_w, frame_h = (int(x) for x in args.frame_resize.split("x"))
    frame_size = (frame_w, frame_h)

    s3_client = None
    if args.encoder == "cosmos_embed1":
        verify_cosmos_health(args.cosmos_url)
        from pipeline.s3_retrieval import make_s3_client
        s3_client = make_s3_client()

    count = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for scene in iter_scene_windows(args.input_jsonl):
            windows = slice_windows(
                scene,
                args.window_size_ticks,
                args.window_stride_ticks,
                args.max_windows_per_scene,
            )
            for win in windows:
                if args.encoder == "fake":
                    emb = fake_encode_window(
                        win, args.embedding_dim,
                    )
                elif args.encoder == "cosmos_embed1":
                    emb = encode_window_cosmos(
                        s3_client, scene, win,
                        args.cosmos_url, frame_size,
                        args.clip_fps,
                    )
                else:
                    raise NotImplementedError(
                        f"Encoder '{args.encoder}' not supported."
                    )

                record = WindowEmbeddingRecord(
                    window_id=win.window_id,
                    scene_token_hex=scene.scene_token_hex,
                    log_id=scene.log_id,
                    scenario_tags=scene.scenario_tags,
                    window_start_ts=win.start_ts,
                    window_end_ts=win.end_ts,
                    camera_set=win.camera_set,
                    embedding=emb.astype(float).tolist(),
                    quality=scene.quality.model_dump(),
                    metadata={
                        "window_index": win.window_index,
                        "tick_count": win.tick_count,
                        "encoder": args.encoder,
                    },
                )
                out.write(
                    json.dumps(record.model_dump()) + "\n",
                )
                count += 1

            if count % 50 == 0 and count > 0:
                print(f"  ... {count} windows embedded so far")

    print(f"Embedded {count} windows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
