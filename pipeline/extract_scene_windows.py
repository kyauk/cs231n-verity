"""Scene-window extraction from a nuPlan sqlite log file."""

from __future__ import annotations

import bisect
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable

from pipeline.models.scene_window import EXPECTED_CHANNELS, SceneQuality, SceneWindow, TickFrames


@dataclass(frozen=True)
class LidarTick:
    """One lidar tick record from sqlite."""

    token: bytes
    next_token: bytes | None
    prev_token: bytes | None
    scene_token: bytes
    timestamp: int


@dataclass(frozen=True)
class ImageFrame:
    """One camera image record from sqlite."""

    token: bytes
    channel: str
    filename_jpg: str
    timestamp: int


def _to_hex(blob: bytes | None) -> str:
    """Purpose: Convert binary sqlite token into hex string.
    Parameters:
        blob (bytes | None): Binary token from sqlite row.
    Returns:
        str: Hex string, or empty string for None.
    Called by: Scene/window serialization helpers.
    Calls: bytes.hex().
    """
    if blob is None:
        return ""
    return blob.hex()


def load_lidar_ticks(conn: sqlite3.Connection) -> list[LidarTick]:
    """Purpose: Load all lidar ticks required for scene traversal.
    Parameters:
        conn (sqlite3.Connection): Open sqlite connection to one log db.
    Returns:
        list[LidarTick]: Tick records for traversal and timestamp lookups.
    Called by: extract_scene_windows().
    Calls: sqlite cursor execute/fetchall.
    """
    rows = conn.execute(
        """
        SELECT token, next_token, prev_token, scene_token, timestamp
        FROM lidar_pc
        """
    ).fetchall()
    return [
        LidarTick(
            token=row[0],
            next_token=row[1],
            prev_token=row[2],
            scene_token=row[3],
            timestamp=row[4],
        )
        for row in rows
    ]


def load_image_frames(conn: sqlite3.Connection) -> list[ImageFrame]:
    """Purpose: Load image rows with resolved channel names.
    Parameters:
        conn (sqlite3.Connection): Open sqlite connection to one log db.
    Returns:
        list[ImageFrame]: Image rows joined with camera channels.
    Called by: extract_scene_windows().
    Calls: sqlite cursor execute/fetchall.
    """
    rows = conn.execute(
        """
        SELECT i.token, c.channel, i.filename_jpg, i.timestamp
        FROM image i
        JOIN camera c ON i.camera_token = c.token
        """
    ).fetchall()
    return [
        ImageFrame(
            token=row[0],
            channel=row[1],
            filename_jpg=row[2],
            timestamp=row[3],
        )
        for row in rows
    ]


def load_scene_tokens(conn: sqlite3.Connection) -> list[bytes]:
    """Purpose: Enumerate scene boundaries from sqlite.
    Parameters:
        conn (sqlite3.Connection): Open sqlite connection to one log db.
    Returns:
        list[bytes]: Scene tokens present in scene table.
    Called by: extract_scene_windows().
    Calls: sqlite cursor execute/fetchall.
    """
    rows = conn.execute("SELECT token FROM scene").fetchall()
    return [row[0] for row in rows]


def load_scenario_tags_by_scene(
    conn: sqlite3.Connection, lidar_scene_map: dict[bytes, bytes]
) -> dict[bytes, set[str]]:
    """Purpose: Map scenario labels to scene tokens through lidar anchors.
    Parameters:
        conn (sqlite3.Connection): Open sqlite connection to one log db.
        lidar_scene_map (dict[bytes, bytes]): lidar_token -> scene_token mapping.
    Returns:
        dict[bytes, set[str]]: Scene token to scenario label set.
    Called by: extract_scene_windows().
    Calls: sqlite cursor execute/fetchall.
    """
    rows = conn.execute("SELECT lidar_pc_token, type FROM scenario_tag").fetchall()
    by_scene: dict[bytes, set[str]] = defaultdict(set)
    for lidar_token, label in rows:
        scene_token = lidar_scene_map.get(lidar_token)
        if scene_token is not None and label:
            by_scene[scene_token].add(label)
    return by_scene


def build_scene_tick_order(lidar_ticks: Iterable[LidarTick]) -> dict[bytes, list[LidarTick]]:
    """Purpose: Build deterministic ordered lidar ticks for each scene.
    Parameters:
        lidar_ticks (Iterable[LidarTick]): Raw lidar tick rows.
    Returns:
        dict[bytes, list[LidarTick]]: Scene token -> ordered tick list.
    Called by: extract_scene_windows().
    Calls: In-memory linked-list traversal.
    """
    scene_to_ticks: dict[bytes, list[LidarTick]] = defaultdict(list)
    tick_by_token: dict[bytes, LidarTick] = {}
    for tick in lidar_ticks:
        scene_to_ticks[tick.scene_token].append(tick)
        tick_by_token[tick.token] = tick

    ordered: dict[bytes, list[LidarTick]] = {}
    for scene_token, ticks in scene_to_ticks.items():
        local = {t.token: t for t in ticks}
        # Prefer linked-list head within scene; fallback to timestamp sort.
        heads = [t for t in ticks if t.prev_token not in local]
        if not heads:
            ordered[scene_token] = sorted(ticks, key=lambda x: x.timestamp)
            continue
        head = min(heads, key=lambda x: x.timestamp)
        current = head
        visited: set[bytes] = set()
        chain: list[LidarTick] = []
        while current.token not in visited:
            visited.add(current.token)
            chain.append(current)
            if current.next_token is None or current.next_token not in local:
                break
            current = local[current.next_token]
        if len(chain) != len(ticks):
            # Recover safely if chain is broken.
            ordered[scene_token] = sorted(ticks, key=lambda x: x.timestamp)
        else:
            ordered[scene_token] = chain
    return ordered


def _p95(values: list[int]) -> int:
    """Purpose: Compute an integer p95 summary for offset diagnostics.
    Parameters:
        values (list[int]): Offset values in microseconds.
    Returns:
        int: Rounded p95 value, or 0 for empty inputs.
    Called by: _scene_quality().
    Calls: sorted().
    """
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return int(ordered[idx])


def build_channel_time_index(image_frames: list[ImageFrame]) -> dict[str, tuple[list[int], list[ImageFrame]]]:
    """Purpose: Build per-channel sorted timestamp index for nearest lookups.
    Parameters:
        image_frames (list[ImageFrame]): Raw image records with channels.
    Returns:
        dict[str, tuple[list[int], list[ImageFrame]]]: Channel -> (timestamps, frames).
    Called by: extract_scene_windows().
    Calls: sorted().
    """
    grouped: dict[str, list[ImageFrame]] = defaultdict(list)
    for frame in image_frames:
        grouped[frame.channel].append(frame)
    index: dict[str, tuple[list[int], list[ImageFrame]]] = {}
    for channel, frames in grouped.items():
        ordered = sorted(frames, key=lambda f: f.timestamp)
        index[channel] = ([f.timestamp for f in ordered], ordered)
    return index


def resolve_tick_frames(
    lidar_timestamp: int,
    channel_index: dict[str, tuple[list[int], list[ImageFrame]]],
    image_s3_prefix: str,
    tolerance_us: int,
) -> tuple[dict[str, str], dict[str, int], str | None]:
    """Purpose: Resolve nearest camera frame per channel for one lidar tick.
    Parameters:
        lidar_timestamp (int): Tick timestamp in microseconds.
        channel_index (dict[str, tuple[list[int], list[ImageFrame]]]): Per-channel timestamp index.
        image_s3_prefix (str): Prefix used to materialize S3 URIs.
        tolerance_us (int): Max absolute timestamp offset accepted.
    Returns:
        tuple[dict[str, str], dict[str, int], str | None]:
            - channel -> S3 URI frame map
            - channel -> offset_us map
            - drop reason if incomplete
    Called by: extract_scene_windows().
    Calls: bisect_left().
    """
    frames_by_channel: dict[str, str] = {}
    offset_by_channel: dict[str, int] = {}
    for channel in EXPECTED_CHANNELS:
        if channel not in channel_index:
            return {}, {}, f"missing_channel_index:{channel}"
        times, frames = channel_index[channel]
        if not times:
            return {}, {}, f"missing_channel_data:{channel}"
        ins = bisect.bisect_left(times, lidar_timestamp)
        candidates: list[int] = []
        if ins < len(times):
            candidates.append(ins)
        if ins - 1 >= 0:
            candidates.append(ins - 1)
        best_idx = min(candidates, key=lambda i: abs(times[i] - lidar_timestamp))
        offset = abs(times[best_idx] - lidar_timestamp)
        if offset > tolerance_us:
            return {}, {}, f"offset_exceeds_tolerance:{channel}"
        frame = frames[best_idx]
        frames_by_channel[channel] = f"{image_s3_prefix}/{frame.filename_jpg}"
        offset_by_channel[channel] = int(offset)
    return frames_by_channel, offset_by_channel, None


def extract_scene_windows(
    db_path: str,
    log_id: str,
    image_s3_prefix: str,
    tolerance_us: int = 50_000,
    min_complete_tick_rate: float = 0.9,
) -> list[SceneWindow]:
    """Purpose: Build scene-window artifacts from one nuPlan sqlite file.
    Parameters:
        db_path (str): Local sqlite path for one log segment.
        log_id (str): Log id used in output metadata.
        image_s3_prefix (str): S3 prefix root for this log's camera images.
        tolerance_us (int): Max absolute timestamp offset for frame sync.
        min_complete_tick_rate (float): Scene acceptance threshold.
    Returns:
        list[SceneWindow]: Valid scene artifacts ready for downstream embedding.
    Called by: pipeline/retrieve_scene_windows_s3.py.
    Calls: sqlite3.connect(), load_* helpers, resolve_tick_frames().
    """
    conn = sqlite3.connect(db_path)
    try:
        lidar_ticks = load_lidar_ticks(conn)
        image_frames = load_image_frames(conn)
        scene_tokens = load_scene_tokens(conn)
        scene_tick_order = build_scene_tick_order(lidar_ticks)
        channel_index = build_channel_time_index(image_frames)

        lidar_scene_map = {tick.token: tick.scene_token for tick in lidar_ticks}
        scene_tags = load_scenario_tags_by_scene(conn, lidar_scene_map)

        artifacts: list[SceneWindow] = []
        for scene_token in scene_tokens:
            ticks = scene_tick_order.get(scene_token, [])
            if not ticks:
                continue
            valid: list[TickFrames] = []
            drop_reasons: dict[str, int] = defaultdict(int)
            offsets_by_channel: dict[str, list[int]] = defaultdict(list)
            for tick in ticks:
                frame_map, offset_map, drop_reason = resolve_tick_frames(
                    tick.timestamp,
                    channel_index,
                    image_s3_prefix=image_s3_prefix,
                    tolerance_us=tolerance_us,
                )
                if drop_reason is not None:
                    drop_reasons[drop_reason] += 1
                    continue
                for channel, offset in offset_map.items():
                    offsets_by_channel[channel].append(offset)
                valid.append(
                    TickFrames(
                        lidar_token_hex=_to_hex(tick.token),
                        lidar_timestamp=tick.timestamp,
                        frames_by_channel=frame_map,
                        offset_us_by_channel=offset_map,
                    )
                )

            total_ticks = len(ticks)
            valid_ticks = len(valid)
            complete_rate = valid_ticks / total_ticks if total_ticks > 0 else 0.0
            if complete_rate < min_complete_tick_rate:
                continue

            quality = SceneQuality(
                total_ticks=total_ticks,
                valid_ticks=valid_ticks,
                complete_tick_rate=complete_rate,
                dropped_ticks=total_ticks - valid_ticks,
                drop_reasons=dict(drop_reasons),
                p95_offset_us_by_channel={
                    channel: _p95(vals) for channel, vals in offsets_by_channel.items()
                },
            )
            artifacts.append(
                SceneWindow(
                    scene_token_hex=_to_hex(scene_token),
                    log_id=log_id,
                    scenario_tags=sorted(scene_tags.get(scene_token, set())),
                    ticks=valid,
                    quality=quality,
                    metadata={
                        "sync_tolerance_us": tolerance_us,
                        "tick_count_median_timestamp": (
                            int(median([tick.lidar_timestamp for tick in valid])) if valid else 0
                        ),
                    },
                )
            )
        return artifacts
    finally:
        conn.close()

