"""Data models for scene-window extraction artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


EXPECTED_CHANNELS: tuple[str, ...] = (
    "CAM_B0",
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
)


class TickFrames(BaseModel):
    """One lidar tick with synchronized camera frame references."""

    lidar_token_hex: str
    lidar_timestamp: int
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
    log_id: str
    scenario_tags: list[str] = Field(default_factory=list)
    ticks: list[TickFrames] = Field(default_factory=list)
    quality: SceneQuality
    metadata: dict[str, Any] = Field(default_factory=dict)


class WindowSpec(BaseModel):
    """Identifies one temporal window within a scene."""

    window_id: str
    scene_token_hex: str
    log_id: str
    window_index: int
    start_tick_idx: int
    end_tick_idx: int
    start_ts: int
    end_ts: int
    camera_set: list[str] = Field(default_factory=list)
    tick_count: int


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

