"""Create visual verification artifacts (frame grids + mp4) for flagged windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from pipeline.models.scene_window import SceneWindow


def parse_args() -> argparse.Namespace:
    """
    Purpose: Parse CLI args for flagged-window visualization artifact generation.
    Parameters:
        None
    Returns:
        argparse.Namespace: Parsed command-line args.
    Called by: main()
    Calls: argparse.ArgumentParser.parse_args()
    """

    parser = argparse.ArgumentParser(
        description="Create frame grids and MP4 clips for top flagged anomaly windows.",
    )
    parser.add_argument(
        "--flagged-jsonl",
        default="outputs/flagged_windows.jsonl",
        help="Anomaly output JSONL from pipeline.anomaly_detect.",
    )
    parser.add_argument(
        "--scene-windows-jsonl",
        default="outputs/scene_windows.jsonl",
        help="Scene window source JSONL with tick/frame references.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/flagged_visuals",
        help="Directory for generated frame grids, mp4 clips, and manifest.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of highest-ranked flagged windows to visualize.",
    )
    parser.add_argument(
        "--camera-channel",
        default="CAM_F0",
        help="Camera channel to visualize (e.g., CAM_F0).",
    )
    parser.add_argument(
        "--window-size-ticks",
        type=int,
        default=16,
        help="Tick window size used during embedding stage.",
    )
    parser.add_argument(
        "--window-stride-ticks",
        type=int,
        default=16,
        help="Tick stride used during embedding stage.",
    )
    parser.add_argument(
        "--max-grid-frames",
        type=int,
        default=9,
        help="Max number of frames to show in grid image per flagged window.",
    )
    parser.add_argument(
        "--clip-fps",
        type=int,
        default=8,
        help="FPS for generated mp4 clips.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Width for generated output frames.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=360,
        help="Height for generated output frames.",
    )
    parser.add_argument(
        "--materialized-images-dir",
        default="",
        help="Optional local root where scene images were downloaded.",
    )
    parser.add_argument(
        "--only-noise",
        action="store_true",
        help="If set, only visualize windows where is_noise=true.",
    )
    return parser.parse_args()


def _parse_window_id(window_id: str) -> tuple[str, int]:
    """
    Purpose: Split window_id into scene token and integer window index.
    Parameters:
        window_id (str): Window identifier like '<scene_token>_w003'.
    Returns:
        tuple[str, int]: (scene_token_hex, window_index)
    Called by: ticks_for_window(), build_visual_artifact()
    Calls: re.match(), int()
    """

    match = re.match(r"^(?P<scene>[0-9a-fA-F]+)_w(?P<idx>\d+)$", window_id)
    if match is None:
        raise ValueError(f"Invalid window_id format: {window_id}")
    return match.group("scene"), int(match.group("idx"))


def _safe_read_jsonl(path: str) -> list[dict[str, Any]]:
    """
    Purpose: Read JSONL records into memory.
    Parameters:
        path (str): JSONL path.
    Returns:
        list[dict[str, Any]]: Parsed rows.
    Called by: main()
    Calls: open(), json.loads()
    """

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_scene_map(path: str) -> dict[str, SceneWindow]:
    """
    Purpose: Load scene windows into a map keyed by scene_token_hex.
    Parameters:
        path (str): Scene windows JSONL path.
    Returns:
        dict[str, SceneWindow]: Parsed scene map.
    Called by: main()
    Calls: SceneWindow.model_validate()
    """

    scenes: dict[str, SceneWindow] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            scene = SceneWindow.model_validate(json.loads(line))
            scenes[scene.scene_token_hex] = scene
    return scenes


def _resolve_local_image(
    uri: str,
    materialized_root: str,
    log_id: str,
    scene_token_hex: str,
    channel: str,
) -> str | None:
    """
    Purpose: Resolve a frame URI to an existing local image path when available.
    Parameters:
        uri (str): Frame URI (s3://, file://, or local path).
        materialized_root (str): Optional downloaded-image root directory.
        log_id (str): Log id for candidate path construction.
        scene_token_hex (str): Scene token for candidate path construction.
        channel (str): Camera channel.
    Returns:
        str | None: Existing local file path if found, else None.
    Called by: load_window_frames()
    Calls: os.path.isfile(), urlparse()
    """

    if not uri:
        return None

    if os.path.isfile(uri):
        return uri
    if uri.startswith("file://"):
        maybe = uri.replace("file://", "", 1)
        if os.path.isfile(maybe):
            return maybe

    if not materialized_root:
        return None

    parsed = urlparse(uri)
    key_path = parsed.path.lstrip("/")
    basename = os.path.basename(key_path)

    candidates = [
        os.path.join(materialized_root, log_id, scene_token_hex, channel, basename),
        os.path.join(materialized_root, log_id, channel, basename),
        os.path.join(materialized_root, channel, basename),
        os.path.join(materialized_root, key_path),
        os.path.join(materialized_root, basename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _make_placeholder_frame(text: str, width: int, height: int) -> np.ndarray:
    """
    Purpose: Create a placeholder frame when source image is missing.
    Parameters:
        text (str): Overlay label text.
        width (int): Output width in pixels.
        height (int): Output height in pixels.
    Returns:
        np.ndarray: BGR image for OpenCV writing.
    Called by: load_window_frames()
    Calls: cv2.putText(), numpy.zeros()
    """

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 2)
    cv2.putText(
        canvas,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def ticks_for_window(
    scene: SceneWindow,
    window_index: int,
    window_size_ticks: int,
    window_stride_ticks: int,
) -> list[Any]:
    """
    Purpose: Reconstruct tick span for a window index using embedding-time slicing params.
    Parameters:
        scene (SceneWindow): Scene artifact with ordered ticks.
        window_index (int): Window index parsed from window_id.
        window_size_ticks (int): Tick count per window.
        window_stride_ticks (int): Tick stride between windows.
    Returns:
        list[Any]: Ticks covered by requested window.
    Called by: build_visual_artifact()
    Calls: list slicing
    """

    ticks = scene.ticks
    if window_size_ticks <= 0:
        return ticks
    start = window_index * max(window_stride_ticks, 1)
    end = min(start + window_size_ticks, len(ticks))
    if start >= len(ticks) or end <= start:
        return []
    return ticks[start:end]


def load_window_frames(
    scene: SceneWindow,
    window_id: str,
    ticks: list[Any],
    channel: str,
    frame_width: int,
    frame_height: int,
    materialized_images_dir: str,
) -> tuple[list[np.ndarray], int]:
    """
    Purpose: Load (or synthesize) ordered frame images for one flagged window.
    Parameters:
        scene (SceneWindow): Scene object holding URI metadata.
        window_id (str): Window identifier used for overlays.
        ticks (list[Any]): Tick objects for the selected window.
        channel (str): Target camera channel.
        frame_width (int): Frame width for output.
        frame_height (int): Frame height for output.
        materialized_images_dir (str): Optional local image root.
    Returns:
        tuple[list[np.ndarray], int]: List of output frames and missing-frame count.
    Called by: build_visual_artifact()
    Calls: _resolve_local_image(), cv2.imread(), cv2.resize(), _make_placeholder_frame()
    """

    frames: list[np.ndarray] = []
    missing = 0
    for tick_index, tick in enumerate(ticks):
        uri = tick.frames_by_channel.get(channel, "")
        local_path = _resolve_local_image(
            uri=uri,
            materialized_root=materialized_images_dir,
            log_id=scene.log_id,
            scene_token_hex=scene.scene_token_hex,
            channel=channel,
        )

        image: np.ndarray | None = None
        if local_path is not None:
            image = cv2.imread(local_path)
        if image is None:
            missing += 1
            image = _make_placeholder_frame(
                text=f"{window_id} | {channel} | missing frame {tick_index}",
                width=frame_width,
                height=frame_height,
            )
        else:
            image = cv2.resize(image, (frame_width, frame_height))
            cv2.putText(
                image,
                f"{window_id} | {channel} | tick {tick_index}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        frames.append(image)
    return frames, missing


def _pick_grid_indices(total: int, max_frames: int) -> list[int]:
    """
    Purpose: Select evenly spaced frame indices for contact sheet generation.
    Parameters:
        total (int): Total available frame count.
        max_frames (int): Maximum frames to include.
    Returns:
        list[int]: Selected indices in ascending order.
    Called by: save_frame_grid()
    Calls: set(), sorted()
    """

    if total <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    indices = np.linspace(0, total - 1, num=max_frames, dtype=int).tolist()
    return sorted(set(indices))


def save_frame_grid(
    frames: list[np.ndarray],
    output_path: str,
    max_frames: int,
) -> None:
    """
    Purpose: Save a contact sheet image for quick visual triage.
    Parameters:
        frames (list[np.ndarray]): Ordered frame images.
        output_path (str): Output JPG/PNG path.
        max_frames (int): Max sampled frames for grid.
    Returns:
        None
    Called by: build_visual_artifact()
    Calls: _pick_grid_indices(), cv2.imwrite()
    """

    if len(frames) == 0:
        raise ValueError("Cannot build grid from empty frame list.")

    indices = _pick_grid_indices(len(frames), max_frames)
    sampled = [frames[i] for i in indices]

    tile_h, tile_w = sampled[0].shape[:2]
    cols = min(3, len(sampled))
    rows = int(math.ceil(len(sampled) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(sampled):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        x0 = c * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = frame

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)


def save_mp4(
    frames: list[np.ndarray],
    output_path: str,
    fps: int,
) -> None:
    """
    Purpose: Persist ordered frames into an mp4 clip for anomaly review.
    Parameters:
        frames (list[np.ndarray]): Ordered frame images.
        output_path (str): Destination mp4 path.
        fps (int): Frames per second.
    Returns:
        None
    Called by: build_visual_artifact()
    Calls: cv2.VideoWriter()
    """

    if len(frames) == 0:
        raise ValueError("Cannot create mp4 from empty frame list.")
    height, width = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def build_visual_artifact(
    flagged_row: dict[str, Any],
    scene_map: dict[str, SceneWindow],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Purpose: Generate frame grid + mp4 for one flagged anomaly row.
    Parameters:
        flagged_row (dict[str, Any]): One anomaly row from flagged JSONL.
        scene_map (dict[str, SceneWindow]): Scene metadata map.
        args (argparse.Namespace): CLI args.
    Returns:
        dict[str, Any]: Manifest row describing generated artifact paths and quality.
    Called by: main()
    Calls: _parse_window_id(), ticks_for_window(), load_window_frames(), save_frame_grid(), save_mp4()
    """

    window_id = str(flagged_row.get("window_id", ""))
    scene_token_hex, window_index = _parse_window_id(window_id)
    scene = scene_map.get(scene_token_hex)
    if scene is None:
        raise ValueError(f"Scene token not found for window: {window_id}")

    ticks = ticks_for_window(
        scene=scene,
        window_index=window_index,
        window_size_ticks=args.window_size_ticks,
        window_stride_ticks=args.window_stride_ticks,
    )
    if len(ticks) == 0:
        raise ValueError(f"No ticks resolved for window: {window_id}")

    frames, missing_count = load_window_frames(
        scene=scene,
        window_id=window_id,
        ticks=ticks,
        channel=args.camera_channel,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        materialized_images_dir=args.materialized_images_dir,
    )

    window_dir = Path(args.output_dir) / window_id
    grid_path = str(window_dir / f"{window_id}_grid.jpg")
    mp4_path = str(window_dir / f"{window_id}.mp4")
    save_frame_grid(frames=frames, output_path=grid_path, max_frames=args.max_grid_frames)
    save_mp4(frames=frames, output_path=mp4_path, fps=args.clip_fps)

    artifact = {
        "window_id": window_id,
        "scene_token_hex": scene_token_hex,
        "log_id": flagged_row.get("log_id"),
        "cluster_label": flagged_row.get("cluster_label"),
        "is_noise": flagged_row.get("is_noise"),
        "outlier_score": flagged_row.get("outlier_score"),
        "anomaly_rank": flagged_row.get("anomaly_rank"),
        "camera_channel": args.camera_channel,
        "tick_count": len(ticks),
        "missing_frame_count": missing_count,
        "grid_path": grid_path,
        "mp4_path": mp4_path,
    }
    return artifact


def main() -> int:
    """
    Purpose: Generate visual QA artifacts for top flagged anomaly windows.
    Parameters:
        None
    Returns:
        int: Exit code (0 success, 1 error).
    Called by: CLI entrypoint
    Calls: parse_args(), _safe_read_jsonl(), _load_scene_map(), build_visual_artifact()
    """

    args = parse_args()
    try:
        flagged_rows = _safe_read_jsonl(args.flagged_jsonl)
        if args.only_noise:
            flagged_rows = [row for row in flagged_rows if bool(row.get("is_noise"))]
        flagged_rows = sorted(
            flagged_rows,
            key=lambda row: int(row.get("anomaly_rank", 10**9)),
        )[: args.top_k]
        if len(flagged_rows) == 0:
            raise ValueError("No flagged rows selected. Check --only-noise and input artifacts.")

        scene_map = _load_scene_map(args.scene_windows_jsonl)

        os.makedirs(args.output_dir, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for row in flagged_rows:
            window_id = str(row.get("window_id", ""))
            try:
                artifact = build_visual_artifact(
                    flagged_row=row,
                    scene_map=scene_map,
                    args=args,
                )
                manifest_rows.append(artifact)
                print(f"created {window_id}")
            except Exception as error:  # noqa: BLE001
                errors.append({"window_id": window_id, "error": str(error)})
                print(f"failed {window_id}: {error}")

        manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row) + "\n")

        summary_path = os.path.join(args.output_dir, "summary.json")
        summary = {
            "selected_count": len(flagged_rows),
            "created_count": len(manifest_rows),
            "error_count": len(errors),
            "manifest_path": manifest_path,
            "camera_channel": args.camera_channel,
            "only_noise": args.only_noise,
            "top_k": args.top_k,
            "errors": errors,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"saved manifest: {manifest_path}")
        print(f"saved summary: {summary_path}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
