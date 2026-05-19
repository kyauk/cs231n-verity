"""Typed handoff contracts for the Waymo anomaly / description / debate stages.

Mirrors ``relevant_video_debate_files/pipeline/models/handoff_contracts.py`` so
the Waymo runner emits exactly the response shapes the frontend already
expects (WorkspaceFlaggedItem, WorkspaceReasoningItem, RegressionCaseProposal).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnomalyResultRecord(BaseModel):
    """One anomaly output row consumed by the description / debate stages."""

    window_id: str
    scene_token_hex: str
    log_id: str
    scenario_tags: list[str] = Field(default_factory=list)
    window_start_ts: int | None = None
    window_end_ts: int | None = None
    cluster_label: int
    is_noise: bool
    cluster_probability: float = 0.0
    outlier_score: float
    anomaly_rank: int
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneDescriptionInputRecord(BaseModel):
    """One scene-description request generated from anomaly outputs."""

    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    scenario_tags: list[str] = Field(default_factory=list)
    cluster_label: int
    is_noise: bool
    outlier_score: float
    anomaly_rank: int
    media_refs: list[str] = Field(default_factory=list)
    prompt_context: dict[str, Any] = Field(default_factory=dict)


class DebateInputRecord(BaseModel):
    """One debate request generated from description + suite context."""

    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    scene_description: str
    anomaly_rationale: str
    severity_hint: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    regression_suite: list[str] = Field(default_factory=list)
    recommendation_question: str = (
        "Should this scenario be added to the regression suite?"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneDescriptionOutputRecord(BaseModel):
    """One scene-description output row."""

    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    scene_description: str
    anomaly_rationale: str
    confidence: Literal["low", "medium", "high", "unknown"] = "unknown"
    model_source: str
    media_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebateOutputRecord(BaseModel):
    """One debate output row."""

    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    decision: Literal["yes", "no"]
    recommendation: Literal["add_immediately", "already_covered", "not_critical"]
    priority_score: float
    rationale: str
    model_source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionCaseProposal(BaseModel):
    """Structured regression-case proposal -- rich output artifact."""

    case_id: str
    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    generated_at: str

    failure_mode: str
    why_anomalous: str
    evidence_summary: str

    risk_level: Literal["critical", "high", "medium", "low"]
    affected_capability: str
    affected_odds: list[str] = Field(default_factory=list)

    counterarguments: list[str] = Field(default_factory=list)
    rebuttal_summary: str = ""

    decision: Literal["add_to_suite", "monitor", "dismiss"]
    recommended_test_spec: str
    scenario_variants: list[str] = Field(default_factory=list)
    confidence: float
    uncertainty_factors: list[str] = Field(default_factory=list)

    debate_transcript: list[str] = Field(default_factory=list)
    model_source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
