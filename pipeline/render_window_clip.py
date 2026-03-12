"""Render per-camera MP4 clips from scene-window S3 frames.

Library usage (called by embed_scenes):
    clips = render_clips_for_window(s3, scene, 0, 400, channels, tmp, ...)
    # clips == {"CAM_F0": "/tmp/.../CAM_F0.mp4", "CAM_B0": ...}

CLI usage (single-camera preview):
    python -m pipeline.render_window_clip \
        --input-jsonl outputs/scene_windows.jsonl \
        --window-id fdab35e465d1528f_w000 \
        --channel CAM_F0 \
        --output outputs/fdab35e465d1528f_w000_CAM_F0.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import cv2
import numpy as np
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from pipeline.models.scene_window import EXPECTED_CHANNELS, SceneWindow
from pipeline.s3_retrieval import make_s3_client


# ---------------------------------------------------------------------------
# S3 fallback prefix resolution
# ---------------------------------------------------------------------------

def get_embed_camera_prefix_list() -> list[str]:
    '''
    Purpose: Return list of S3 path prefixes to try when resolving camera
        frame keys on 404.  Reads S3_NUPLAN_EMBED_CAMERA_PREFIXES (comma-
        separated) or S3_NUPLAN_EMBED_CAMERA_PREFIX (single, with optional
        {a-b} range expansion).
    Parameters: None.
    Returns:
        list[str]: Ordered fallback prefixes.
    Called by: _download_frame().
    Calls: os.getenv(), re.search().
    '''
    def expand_range(s: str) -> list[str]:
        match = re.search(r"\{(\d+)-(\d+)\}", s)
        if not match:
            return [s.strip()] if s.strip() else []
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo > hi:
            return [s.strip()] if s.strip() else []
        return [
            (s[: match.start()] + str(i) + s[match.end():]).strip()
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


# ---------------------------------------------------------------------------
# S3 frame download (single frame with 404 fallback)
# ---------------------------------------------------------------------------

def _download_frame(s3, scene: SceneWindow, tick_idx: int,
                    channel: str, dest: str) -> None:
    '''
    Purpose: Download one camera frame from S3, with 404 fallback logic
        (deduplicated log_id segment, then env-var-driven prefix list).
    Parameters:
        s3: boto3 S3 client.
        scene (SceneWindow): Parent scene.
        tick_idx (int): Index into scene.ticks.
        channel (str): Camera channel name.
        dest (str): Local destination file path.
    Returns: None (raises on failure).
    Called by: render_clip_for_channel().
    Calls: s3.download_file(), get_embed_camera_prefix_list().
    '''
    uri = scene.ticks[tick_idx].frames_by_channel.get(channel)
    if not uri:
        raise ValueError(f"No URI for {channel} at tick {tick_idx}")

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        s3.download_file(bucket, key, dest)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

    double = f"/{scene.log_id}/{scene.log_id}/"
    if double in key:
        alt = key.replace(double, f"/{scene.log_id}/", 1)
        try:
            s3.download_file(bucket, alt, dest)
            return
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

    suffix = None
    match = re.search(r"/(nuplan-v1\.1_mini_camera_\d+)(/.*)$", key)
    if match:
        suffix = match.group(2).lstrip("/")
    if not suffix:
        suffix = f"{scene.log_id}/{os.path.basename(key)}"

    prefix_list = get_embed_camera_prefix_list()
    for prefix in prefix_list:
        alt = f"{prefix.rstrip('/')}/{suffix}"
        try:
            s3.download_file(bucket, alt, dest)
            return
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

    raise RuntimeError(
        f"Frame not found on S3: bucket={bucket} key={key}; "
        f"tried {len(prefix_list)} fallback prefix(es), all 404."
    )


# ---------------------------------------------------------------------------
# Clip assembly (frames -> mp4v -> H.264)
# ---------------------------------------------------------------------------

def _assemble_and_encode(frame_paths: list[str], out_path: str,
                         fps: int, size: tuple[int, int]) -> str:
    '''
    Purpose: Stitch ordered image frames into an mp4 video clip, then
        re-encode to H.264 if ffmpeg is available.
    Parameters:
        frame_paths (list[str]): Ordered local image file paths.
        out_path (str): Destination mp4 path.
        fps (int): Video frames per second.
        size (tuple[int, int]): (width, height) for output frames.
    Returns:
        str: Path to the final clip (H.264 if possible, else mp4v).
    Called by: render_clip_for_channel().
    Calls: cv2.VideoWriter(), cv2.imread(), cv2.resize(), subprocess.run().
    '''
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
    )
    last_good: np.ndarray | None = None
    try:
        for path in frame_paths:
            img = cv2.imread(path)
            if img is not None:
                last_good = cv2.resize(img, size)
            if last_good is not None:
                writer.write(last_good)
        if last_good is None and frame_paths:
            black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            writer.write(black)
    finally:
        writer.release()

    h264_path = out_path + ".h264.mp4"
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-c:v", "libx264", "-preset", "fast",
                "-f", "mp4", "-movflags", "+faststart",
                h264_path,
            ],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and os.path.isfile(h264_path):
            os.replace(h264_path, out_path)
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if os.path.isfile(h264_path):
        os.remove(h264_path)
    return out_path


# ---------------------------------------------------------------------------
# Public API: per-channel and per-window clip rendering
# ---------------------------------------------------------------------------

def render_clip_for_channel(
    s3,
    scene: SceneWindow,
    start: int,
    end: int,
    channel: str,
    tmp_dir: str,
    frame_size: tuple[int, int] = (224, 224),
    fps: int = 10,
) -> str:
    '''
    Purpose: Download frames for one camera channel from S3, assemble
        into an mp4 clip, and H.264 re-encode.
    Parameters:
        s3: boto3 S3 client.
        scene (SceneWindow): Parent scene with tick frame URIs.
        start (int): Start tick index (inclusive).
        end (int): End tick index (exclusive).
        channel (str): Camera channel name (e.g. "CAM_F0").
        tmp_dir (str): Temp directory for downloaded frames and clip.
        frame_size (tuple[int, int]): (width, height) for output frames.
        fps (int): Video frames per second.
    Returns:
        str: Path to the assembled clip.
    Called by: render_clips_for_window(), CLI main().
    Calls: _download_frame(), _assemble_and_encode().
    '''
    frame_paths: list[str] = []
    for rel_idx, tick_idx in enumerate(range(start, end)):
        if not scene.ticks[tick_idx].frames_by_channel.get(channel):
            continue
        ext = os.path.splitext(
            urlparse(scene.ticks[tick_idx].frames_by_channel[channel]).path
        )[1] or ".jpg"
        dest = os.path.join(tmp_dir, channel, f"{rel_idx:06d}{ext}")
        _download_frame(s3, scene, tick_idx, channel, dest)
        frame_paths.append(dest)

    clip_path = os.path.join(tmp_dir, f"{channel}.mp4")
    return _assemble_and_encode(frame_paths, clip_path, fps, frame_size)


def render_clips_for_window(
    s3,
    scene: SceneWindow,
    start: int,
    end: int,
    channels: tuple[str, ...] | list[str],
    tmp_dir: str,
    frame_size: tuple[int, int] = (224, 224),
    fps: int = 10,
) -> dict[str, str]:
    '''
    Purpose: Render clips for all requested camera channels in parallel.
    Parameters:
        s3: boto3 S3 client.
        scene (SceneWindow): Parent scene.
        start (int): Start tick index (inclusive).
        end (int): End tick index (exclusive).
        channels (tuple[str, ...] | list[str]): Camera channels to render.
        tmp_dir (str): Temp directory for frames and clips.
        frame_size (tuple[int, int]): (width, height) for output frames.
        fps (int): Video frames per second.
    Returns:
        dict[str, str]: Channel name -> path to assembled clip.
    Called by: embed_scenes.embed_window().
    Calls: render_clip_for_channel() via ThreadPoolExecutor.
    '''
    clips: dict[str, str] = {}
    max_workers = min(len(channels), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                render_clip_for_channel,
                s3, scene, start, end, ch, tmp_dir, frame_size, fps,
            ): ch
            for ch in channels
        }
        for fut in as_completed(futures):
            ch = futures[fut]
            clips[ch] = fut.result()
    return clips


# ---------------------------------------------------------------------------
# CLI: render a single-camera clip for preview / inspection
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    '''
    Purpose: Parse CLI arguments for single-camera clip rendering.
    Parameters: None.
    Returns: argparse.Namespace.
    Called by: main().
    Calls: argparse.ArgumentParser().
    '''
    p = argparse.ArgumentParser(
        description="Render a single-camera MP4 for one scene window.",
    )
    p.add_argument("--input-jsonl", required=True,
                   help="Path to scene_windows JSONL.")
    p.add_argument("--window-id", required=True,
                   help="Target window_id (e.g. fdab35e465d1528f_w000).")
    p.add_argument("--channel", default="CAM_F0",
                   choices=list(EXPECTED_CHANNELS),
                   help="Camera channel to render (default CAM_F0).")
    p.add_argument("--output", default=None,
                   help="Output MP4 path. Default: outputs/<window_id>_<channel>.mp4")
    p.add_argument("--frame-resize", default="224x224",
                   help="WxH pixel size (default 224x224).")
    p.add_argument("--clip-fps", type=int, default=10,
                   help="Frames per second (default 10).")
    p.add_argument("--window-size-ticks", type=int, default=0,
                   help="Ticks per window (0 = whole scene).")
    p.add_argument("--window-stride-ticks", type=int, default=0,
                   help="Stride between windows (0 = non-overlapping).")
    return p.parse_args()


def main() -> int:
    '''
    Purpose: Find the target window in the scene JSONL, download frames
        for one camera channel from S3, and render a single-camera MP4.
    Parameters: None (uses parse_args()).
    Returns: 0 on success, 1 on error.
    Called by: CLI invocation.
    Calls: iter_scenes(), make_windows(), render_clip_for_channel().
    '''
    load_dotenv()
    args = parse_args()

    if not os.path.isfile(args.input_jsonl):
        print(f"Error: input not found: {args.input_jsonl}", file=sys.stderr)
        return 1

    from pipeline.embed_scenes import iter_scenes, make_windows

    target_scene = None
    target_start = None
    target_end = None
    for scene in iter_scenes(args.input_jsonl):
        for window_id, start, end in make_windows(
            scene, args.window_size_ticks, args.window_stride_ticks,
        ):
            if window_id == args.window_id:
                target_scene = scene
                target_start = start
                target_end = end
                break
        if target_scene is not None:
            break

    if target_scene is None:
        print(
            f"Error: window_id '{args.window_id}' not found in "
            f"{args.input_jsonl}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Found window {args.window_id}: "
        f"scene={target_scene.scene_token_hex}, "
        f"log={target_scene.log_id}, "
        f"ticks={target_end - target_start} "
        f"({target_start}:{target_end})"
    )

    frame_w, frame_h = (int(x) for x in args.frame_resize.split("x"))
    frame_size = (frame_w, frame_h)
    output_path = (
        args.output
        or f"outputs/{args.window_id}_{args.channel}.mp4"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    s3 = make_s3_client()
    tmp_dir = tempfile.mkdtemp(prefix="render_clip_")
    try:
        print(f"Downloading {args.channel} frames from S3...")
        clip_path = render_clip_for_channel(
            s3, target_scene, target_start, target_end,
            args.channel, tmp_dir, frame_size, args.clip_fps,
        )
        shutil.copy2(clip_path, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Saved: {output_path} ({size_kb:.0f} KB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
