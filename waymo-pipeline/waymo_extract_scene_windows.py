"""Scene-window extraction from Waymo Open Dataset v2 camera_image Parquet.

Mirrors ``pipeline/extract_scene_windows.py`` (which reads nuPlan sqlite logs)
but reads Waymo segment Parquet files. The Waymo schema is timestamp-keyed per
camera; one "tick" is a frame timestamp for which every camera in the rig has a
frame within the sync tolerance.

Usage:
  python -m waymo_pipeline.waymo_extract_scene_windows \
      --output-jsonl outputs/waymo_scene_windows.jsonl \
      --max-segments 5
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import gcsfs
import pyarrow.parquet as pq
from dotenv import load_dotenv
from google.auth import default as google_auth_default

from waymo_pipeline.models.scene_window import (
    EXPECTED_CHANNELS,
    SceneQuality,
    SceneWindow,
    TickFrames,
)
from waymo_pipeline.waymo_video_pipeline import (
    CAMERA_NAMES,
    SOURCE_BUCKET,
    SOURCE_PREFIX,
    discover_segments,
)

# Waymo cameras run at ~10 Hz; ticks within a scene are grouped into windows of
# this many frames. A nuPlan "scene" is ~20 s of log; a Waymo segment is ~20 s,
# so one segment maps to one scene window by default.
DEFAULT_SCENE_TICKS = 0  # 0 = whole segment is one scene


@dataclass(frozen=True)
class CameraFrame:
    """One decoded camera frame reference from the Waymo Parquet."""

    channel: str
    timestamp: int
    frame_index: int


def _p95(values: list[int]) -> int:
    """Integer p95 summary for offset diagnostics."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return int(ordered[idx])


def load_camera_frames(
    fs: gcsfs.GCSFileSystem, segment_id: str
) -> dict[str, list[CameraFrame]]:
    """Load per-camera frame timestamps for one Waymo segment.

    Returns a mapping {camera_name: [CameraFrame, ...]} sorted by timestamp.
    """
    parquet_path = f"{SOURCE_BUCKET}/{SOURCE_PREFIX}/{segment_id}.parquet"
    with fs.open(parquet_path, "rb") as f:
        pf = pq.ParquetFile(f)
        schema_names = pf.schema_arrow.names

    def find_col(candidates: list[str]) -> str:
        for c in candidates:
            if c in schema_names:
                return c
        raise KeyError(f"None of {candidates} found. Available: {schema_names[:20]}")

    col_camera = find_col(["key.camera_name", "camera_name"])
    col_ts = find_col(["key.frame_timestamp_micros", "frame_timestamp_micros"])

    with fs.open(parquet_path, "rb") as f:
        table = pq.read_table(f, columns=[col_camera, col_ts])
    df = table.to_pandas().sort_values(col_ts)

    by_channel: dict[str, list[CameraFrame]] = defaultdict(list)
    for cam_int, cam_name in CAMERA_NAMES.items():
        cam_df = df[df[col_camera] == cam_int].reset_index(drop=True)
        for frame_index, ts in enumerate(cam_df[col_ts].tolist()):
            by_channel[cam_name].append(
                CameraFrame(channel=cam_name, timestamp=int(ts), frame_index=frame_index)
            )
    return dict(by_channel)


def build_channel_time_index(
    frames_by_channel: dict[str, list[CameraFrame]]
) -> dict[str, tuple[list[int], list[CameraFrame]]]:
    """Build per-channel sorted timestamp index for nearest-frame lookups."""
    index: dict[str, tuple[list[int], list[CameraFrame]]] = {}
    for channel, frames in frames_by_channel.items():
        ordered = sorted(frames, key=lambda f: f.timestamp)
        index[channel] = ([f.timestamp for f in ordered], ordered)
    return index


def resolve_tick_frames(
    anchor_timestamp: int,
    channel_index: dict[str, tuple[list[int], list[CameraFrame]]],
    segment_id: str,
    tolerance_us: int,
) -> tuple[dict[str, str], dict[str, int], str | None]:
    """Resolve the nearest camera frame per channel for one anchor timestamp.

    Frame references are GCS video URIs of the form
    ``gs://nvidia-adr-waymo-segment-videos/segments/<seg>/<seg>_<CAM>.mp4#t=<idx>``
    so downstream stages can locate the camera clip and the frame index.
    """
    frames_by_channel: dict[str, str] = {}
    offset_by_channel: dict[str, int] = {}
    for channel in EXPECTED_CHANNELS:
        if channel not in channel_index:
            return {}, {}, f"missing_channel_index:{channel}"
        times, frames = channel_index[channel]
        if not times:
            return {}, {}, f"missing_channel_data:{channel}"
        ins = bisect.bisect_left(times, anchor_timestamp)
        candidates: list[int] = []
        if ins < len(times):
            candidates.append(ins)
        if ins - 1 >= 0:
            candidates.append(ins - 1)
        best_idx = min(candidates, key=lambda i: abs(times[i] - anchor_timestamp))
        offset = abs(times[best_idx] - anchor_timestamp)
        if offset > tolerance_us:
            return {}, {}, f"offset_exceeds_tolerance:{channel}"
        frame = frames[best_idx]
        uri = (
            f"gs://nvidia-adr-waymo-segment-videos/segments/"
            f"{segment_id}/{segment_id}_{channel}.mp4#t={frame.frame_index}"
        )
        frames_by_channel[channel] = uri
        offset_by_channel[channel] = int(offset)
    return frames_by_channel, offset_by_channel, None


def extract_scene_windows(
    fs: gcsfs.GCSFileSystem,
    segment_id: str,
    tolerance_us: int = 50_000,
    min_complete_tick_rate: float = 0.9,
    scene_ticks: int = DEFAULT_SCENE_TICKS,
) -> list[SceneWindow]:
    """Build scene-window artifacts from one Waymo segment.

    The FRONT camera is the anchor; every other camera is synchronized to it.
    """
    frames_by_channel = load_camera_frames(fs, segment_id)
    channel_index = build_channel_time_index(frames_by_channel)

    anchor_frames = frames_by_channel.get("FRONT", [])
    if not anchor_frames:
        return []

    # Slice the anchor timeline into scenes.
    size = scene_ticks or len(anchor_frames)
    scenes: list[SceneWindow] = []
    scene_idx = 0
    start = 0
    while start < len(anchor_frames):
        end = min(start + size, len(anchor_frames))
        anchor_slice = anchor_frames[start:end]

        valid: list[TickFrames] = []
        drop_reasons: dict[str, int] = defaultdict(int)
        offsets_by_channel: dict[str, list[int]] = defaultdict(list)

        for anchor in anchor_slice:
            frame_map, offset_map, drop_reason = resolve_tick_frames(
                anchor.timestamp, channel_index, segment_id, tolerance_us
            )
            if drop_reason is not None:
                drop_reasons[drop_reason] += 1
                continue
            for channel, offset in offset_map.items():
                offsets_by_channel[channel].append(offset)
            valid.append(
                TickFrames(
                    tick_token_hex=f"{segment_id}_t{anchor.frame_index:05d}",
                    tick_timestamp=anchor.timestamp,
                    frames_by_channel=frame_map,
                    offset_us_by_channel=offset_map,
                )
            )

        total_ticks = len(anchor_slice)
        valid_ticks = len(valid)
        complete_rate = valid_ticks / total_ticks if total_ticks else 0.0
        if complete_rate >= min_complete_tick_rate and valid:
            quality = SceneQuality(
                total_ticks=total_ticks,
                valid_ticks=valid_ticks,
                complete_tick_rate=complete_rate,
                dropped_ticks=total_ticks - valid_ticks,
                drop_reasons=dict(drop_reasons),
                p95_offset_us_by_channel={
                    ch: _p95(vals) for ch, vals in offsets_by_channel.items()
                },
            )
            scenes.append(
                SceneWindow(
                    scene_token_hex=f"{segment_id}_s{scene_idx:03d}",
                    log_id=segment_id,
                    scenario_tags=[],
                    ticks=valid,
                    quality=quality,
                    metadata={
                        "sync_tolerance_us": tolerance_us,
                        "dataset": "waymo_open_dataset_v_2_0_1",
                        "tick_count_median_timestamp": valid[len(valid) // 2].tick_timestamp,
                    },
                )
            )
            scene_idx += 1
        start += size

    return scenes


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/prefix`` into ``(bucket, prefix)``.

    Falls back to the module-level defaults when the URI is absent or not a
    recognised GCS URI so the script remains runnable without arguments.
    """
    uri = uri.strip().rstrip("/")
    if uri.startswith("gs://"):
        without_scheme = uri[len("gs://"):]
        bucket, _, prefix = without_scheme.partition("/")
        return bucket, prefix or SOURCE_PREFIX
    # Not a GCS URI — honour env-var overrides then hardcoded defaults.
    return SOURCE_BUCKET, SOURCE_PREFIX


def main() -> None:
    """Run end-to-end Waymo retrieval + scene-window extraction to JSONL."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extract Waymo scene windows.")
    parser.add_argument("--output-jsonl", default="outputs/waymo_scene_windows.jsonl")
    parser.add_argument("--max-segments", type=int, default=5)
    parser.add_argument("--sync-tolerance-us", type=int, default=50_000)
    parser.add_argument("--min-complete-tick-rate", type=float, default=0.9)
    parser.add_argument("--scene-ticks", type=int, default=DEFAULT_SCENE_TICKS)
    parser.add_argument(
        "--data-source-uri",
        default=os.environ.get("DATA_SOURCE_URI", ""),
        help="GCS URI of the dataset root, e.g. gs://my-bucket/validation/camera_image",
    )
    args = parser.parse_args()

    bucket, prefix = _parse_gcs_uri(args.data_source_uri)
    print(f"[Config] source bucket={bucket}  prefix={prefix}")

    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    fs = gcsfs.GCSFileSystem(token=creds)

    all_segments = discover_segments(fs, bucket=bucket, prefix=prefix)
    segments = all_segments[: args.max_segments] if args.max_segments else all_segments

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    scene_count = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for seg_id in segments:
            scenes = extract_scene_windows(
                fs=fs,
                segment_id=seg_id,
                tolerance_us=args.sync_tolerance_us,
                min_complete_tick_rate=args.min_complete_tick_rate,
                scene_ticks=args.scene_ticks,
            )
            print(f"segment={seg_id} scenes_extracted={len(scenes)}")
            for scene in scenes:
                out.write(json.dumps(scene.model_dump(), sort_keys=True) + "\n")
            scene_count += len(scenes)

    print(f"Completed. scene_count={scene_count}, output_jsonl={args.output_jsonl}")


if __name__ == "__main__":
    main()
