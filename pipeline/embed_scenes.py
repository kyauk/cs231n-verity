"""Scene-window JSONL to embedding-vector JSONL transformer.

Supports temporal windowing and multiple encoder backends:
  - fake: deterministic placeholder for pipeline validation
  - cosmos_embed1: NVIDIA Cosmos Embed1 NIM for video embeddings

Usage:
  python -m pipeline.embed_scenes \\
      --input-jsonl outputs/scene_windows.jsonl \\
      --output-jsonl outputs/window_embeddings.jsonl \\
      --encoder cosmos_embed1
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
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


# -------------------------------------------------------------------
# Tmux progress bar
# -------------------------------------------------------------------

def _in_tmux() -> bool:
    '''
    Purpose: Detect whether the process is running inside a tmux session.
    Parameters: None.
    Returns: bool — True if TMUX env var is set and tmux binary exists.
    Called by: TmuxProgress.__init__().
    Calls: shutil.which().
    '''
    return bool(os.environ.get("TMUX")) and shutil.which("tmux") is not None


class TmuxProgress:
    '''Lightweight tmux status-right progress bar for long-running jobs.

    Updates the tmux status bar with a visual progress indicator.
    Gracefully no-ops when not running inside tmux.
    '''

    BAR_WIDTH = 20

    def __init__(self, total: int, label: str = "Embedding"):
        '''
        Purpose: Initialize progress tracker and save original tmux
            status-right so it can be restored on cleanup.
        Parameters:
            total (int): Total number of work items.
            label (str): Short prefix label shown in the bar.
        Returns: None.
        Called by: main().
        Calls: _in_tmux(), subprocess.run().
        '''
        self.total = max(total, 1)
        self.label = label
        self.current = 0
        self.active = _in_tmux()
        self._original_status: str | None = None
        self._start_time = time.monotonic()

        if self.active:
            try:
                result = subprocess.run(
                    ["tmux", "show-option", "-gv", "status-right"],
                    capture_output=True, text=True, timeout=5,
                )
                self._original_status = result.stdout.strip()
                subprocess.run(
                    ["tmux", "set", "-g", "status-interval", "1"],
                    capture_output=True, timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.active = False

    def update(self, current: int | None = None) -> None:
        '''
        Purpose: Push a progress update to the tmux status bar.
        Parameters:
            current (int | None): Absolute progress count. If None,
                increments by 1.
        Returns: None.
        Called by: main() inner loop.
        Calls: subprocess.run().
        '''
        if not self.active:
            return
        self.current = current if current is not None else self.current + 1
        pct = self.current / self.total
        filled = int(self.BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)

        elapsed = time.monotonic() - self._start_time
        if self.current > 0:
            eta_s = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA {int(eta_s // 60)}m{int(eta_s % 60):02d}s"
        else:
            eta_str = "ETA --"

        status = (
            f" {self.label}: {self.current}/{self.total} "
            f"({pct:.0%}) [{bar}] {eta_str} "
        )
        try:
            subprocess.run(
                ["tmux", "set", "-g", "status-right", status],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.active = False

    def finish(self) -> None:
        '''
        Purpose: Restore the original tmux status-right value.
        Parameters: None.
        Returns: None.
        Called by: main() on completion.
        Calls: subprocess.run().
        '''
        if not self.active:
            return
        elapsed = time.monotonic() - self._start_time
        restore = self._original_status or ""
        try:
            subprocess.run(
                ["tmux", "set", "-g", "status-right", restore],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        mins, secs = divmod(int(elapsed), 60)
        print(f"Total embedding time: {mins}m{secs:02d}s")


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
    p.add_argument(
        "--resume-from",
        default=None,
        help="Path to existing embedding JSONL; skip any window_id already in it.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Rotate to a new output file every N scenes (e.g. 50). 0 = single file.",
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


def load_resume_window_ids(path: str) -> set[str]:
    '''
    Purpose: Load all window_id values from an embedding JSONL for resume.
    Parameters:
        path (str): Path to existing embedding JSONL.
    Returns:
        set[str]: Window IDs that should be skipped when resuming.
    Called by: main().
    Calls: open(), json.loads().
    '''
    out: set[str] = set()
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                wid = row.get("window_id")
                if wid is not None:
                    out.add(str(wid))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def output_stem_for_checkpoints(output_jsonl: str) -> str:
    '''
    Purpose: Get the stem for checkpoint filenames (e.g. embedding_001.jsonl).
    Parameters:
        output_jsonl (str): Base output path (e.g. outputs/embedding.jsonl).
    Returns:
        str: Stem without .jsonl (e.g. outputs/embedding).
    Called by: main().
    Calls: os.path.splitext(), os.path.join().
    '''
    base = os.path.basename(output_jsonl)
    stem_name = base.replace(".jsonl", "").rstrip("_")
    dirname = os.path.dirname(output_jsonl)
    return os.path.join(dirname, stem_name) if dirname else stem_name


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

def get_embed_camera_prefix_list() -> list[str]:
    '''
    Purpose: Return list of S3 path prefixes (no bucket) to try when
        resolving camera frame keys on 404. Supports multiple prefixes
        and optional {a-b} range expansion.
    Parameters: None.
    Returns:
        list[str]: Ordered list of prefixes to try (e.g. camera_2 .. camera_8).
    Called by: materialize_window_frames().
    Calls: os.getenv(), re.search().
    '''
    def expand_range(s: str) -> list[str]:
        # Replace single {a-b} with a, a+1, ..., b to produce multiple prefixes.
        match = re.search(r"\{(\d+)-(\d+)\}", s)
        if not match:
            return [s.strip()] if s.strip() else []
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo > hi:
            return [s.strip()] if s.strip() else []
        return [
            (s[: match.start()] + str(i) + s[match.end() :]).strip()
            for i in range(lo, hi + 1)
        ]

    prefixes_env = os.getenv("S3_NUPLAN_EMBED_CAMERA_PREFIXES", "").strip()
    if prefixes_env:
        out: list[str] = []
        for part in prefixes_env.split(","):
            part = part.strip()
            if not part:
                continue
            out.extend(expand_range(part))
        return out if out else [prefixes_env]

    single = os.getenv(
        "S3_NUPLAN_EMBED_CAMERA_PREFIX",
        "nuplan-v1.1/sensor_blobs/camera_0/nuplan-v1.1_mini_camera_0",
    ).strip()
    return expand_range(single)


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
                # Fallback 1: key may have duplicate log_id (e.g. .../log_id/log_id/CAM_B0/...).
                # Try key with one log_id segment removed so existing scene_windows.jsonl still works.
                double_segment = f"/{scene.log_id}/{scene.log_id}/"
                if double_segment in key:
                    alt_key = key.replace(double_segment, f"/{scene.log_id}/", 1)
                    try:
                        s3_client.download_file(bucket, alt_key, local_path)
                        last_err = None
                    except ClientError as err:
                        last_err = err
                else:
                    last_err = e
                # Fallback 2: try other camera_* prefixes, preserving path suffix
                # (log_id/CAM_ANGLE/filename or log_id/filename) so frames under
                # camera_1..8 are found when the written URI used camera_0.
                if last_err is not None and last_err.response["Error"]["Code"] == "404":
                    prefix_list = get_embed_camera_prefix_list()
                    last_err = None
                    suffix = None
                    match = re.search(r"/(nuplan-v1\.1_mini_camera_\d+)(/.*)$", key)
                    if match:
                        suffix = match.group(2).lstrip("/")
                    if not suffix:
                        suffix = f"{scene.log_id}/{os.path.basename(key)}"
                    for prefix in prefix_list:
                        alt_key = f"{prefix.rstrip('/')}/{suffix}"
                        try:
                            s3_client.download_file(bucket, alt_key, local_path)
                            last_err = None
                            break
                        except ClientError as err:
                            if err.response["Error"]["Code"] != "404":
                                raise
                            last_err = err
                    if last_err is not None:
                        raise RuntimeError(
                            f"S3 key not found: bucket={bucket} key={key}; "
                            f"tried deduplicated key and {len(prefix_list)} fallback prefix(es), all 404."
                        ) from last_err
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

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    '''
    Purpose: Slice scenes into temporal windows, encode each window,
        and write WindowEmbeddingRecord JSONL output. Supports resume-from
        (skip already-embedded window_ids) and checkpoint-every (rotate to
        a new file every N scenes).
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: iter_scene_windows(), slice_windows(), load_resume_window_ids(),
        output_stem_for_checkpoints(), fake_encode_window(), encode_window_cosmos().
    '''
    load_dotenv()
    args = parse_args()
    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frame_w, frame_h = (int(x) for x in args.frame_resize.split("x"))
    frame_size = (frame_w, frame_h)

    s3_client = None
    if args.encoder == "cosmos_embed1":
        verify_cosmos_health(args.cosmos_url)
        from pipeline.s3_retrieval import make_s3_client
        s3_client = make_s3_client()

    scenes = list(iter_scene_windows(args.input_jsonl))
    work_plan: list[tuple[SceneWindow, list[WindowSpec]]] = []
    total_windows = 0
    for scene in scenes:
        windows = slice_windows(
            scene,
            args.window_size_ticks,
            args.window_stride_ticks,
            args.max_windows_per_scene,
        )
        work_plan.append((scene, windows))
        total_windows += len(windows)

    done_window_ids: set[str] = set()
    if args.resume_from:
        done_window_ids = load_resume_window_ids(args.resume_from)
        if not done_window_ids and os.path.isfile(args.resume_from):
            pass
        elif not os.path.isfile(args.resume_from):
            print(f"Warning: resume-from file not found, skipping no windows: {args.resume_from}")
        else:
            print(f"Resuming from {args.resume_from} ({len(done_window_ids)} window_ids to skip).")

    total_to_do = sum(
        1 for _scene, wins in work_plan for w in wins if w.window_id not in done_window_ids
    )
    print(
        f"Loaded {len(scenes)} scenes -> "
        f"{total_windows} windows to embed ({total_to_do} new after resume)"
    )

    progress = TmuxProgress(total_to_do, label="Embed")
    count = 0
    checkpoint_every = getattr(args, "checkpoint_every", 0) or 0
    stem = output_stem_for_checkpoints(args.output_jsonl) if checkpoint_every > 0 else None
    current_checkpoint = 0
    out_file = None

    try:
        for scene_idx, (scene, windows) in enumerate(work_plan):
            if checkpoint_every > 0:
                desired = (scene_idx // checkpoint_every) + 1
                if out_file is None or desired != current_checkpoint:
                    if out_file is not None:
                        out_file.close()
                    path = f"{stem}_{desired:03d}.jsonl"
                    out_file = open(path, "w", encoding="utf-8")
                    current_checkpoint = desired
                    print(f"Checkpoint: writing to {path}")
            elif out_file is None:
                out_file = open(args.output_jsonl, "w", encoding="utf-8")

            for win in windows:
                if win.window_id in done_window_ids:
                    continue
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
                out_file.write(
                    json.dumps(record.model_dump()) + "\n",
                )
                count += 1
                progress.update(count)
    finally:
        if out_file is not None:
            out_file.close()

    progress.finish()
    if count == 0:
        print("Embedded 0 new windows (all skipped by resume).")
    elif checkpoint_every > 0:
        print(f"Embedded {count} windows -> {stem}_*.jsonl (checkpoints)")
    else:
        print(f"Embedded {count} windows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
