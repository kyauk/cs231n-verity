"""Data models for Waymo scene-window extraction artifacts.

Mirrors ``pipeline/models/scene_window.py`` but uses the Waymo Open Dataset
five-camera rig instead of the nuPlan eight-camera channel set.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# Waymo Open Dataset v2 camera rig. Integer ids map to these names in
# waymo_video_pipeline.CAMERA_NAMES.
EXPECTED_CHANNELS: tuple[str, ...] = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
)


class TickFrames(BaseModel):
    """One synchronized capture tick with per-camera frame references."""

    tick_token_hex: str
    tick_timestamp: int
    frames_by_channel: dict[str, str]
    offset_us_by_channel: dict[str, int]


class SceneQuality(BaseModel):
    """Per-scene extraction quality summary."""

    total_ticks: int
    valid_ticks: int
    complete_tick_rate: float
    dropped_ticks: int
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    p95_offset_us_by_channel: dict[str, int] = Field(default_factory=dict)


class SceneWindow(BaseModel):
    """Encoder-agnostic scene artifact for downstream embedding."""

    scene_token_hex: str
    log_id: str  # Waymo segment id
    scenario_tags: list[str] = Field(default_factory=list)
    ticks: list[TickFrames] = Field(default_factory=list)
    quality: SceneQuality
    metadata: dict[str, Any] = Field(default_factory=dict)


class WindowEmbeddingRecord(BaseModel):
    """Output contract for one embedded temporal window."""

    window_id: str
    scene_token_hex: str
    log_id: str
    scenario_tags: list[str] = Field(default_factory=list)
    window_start_ts: int
    window_end_ts: int
    camera_set: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
