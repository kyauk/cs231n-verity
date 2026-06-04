"""Shared debate types — produced by Module 9: Debate (multi-agent analysis).

The debate arm consumes a flagged/anomalous window (typically surfaced by
Module 8: Clustering) and runs a multi-agent, tool-augmented Risk-vs-Coverage
rebuttal loop, producing a structured regression-case proposal. These are the
only types that cross its module boundary — dataclasses with to_json/from_json,
pinned by a round-trip test, like every other interface here.

(Ported from relevant_video_debate_files/pipeline/models/handoff_contracts.py,
the canonical copy; pydantic -> dataclass to match this package's convention.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebateInput:
    """One debate request: a described, anomalous scene + suite context."""
    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    scene_description: str
    anomaly_rationale: str
    severity_hint: str = "unknown"          # low|medium|high|critical|unknown
    regression_suite: list[str] = field(default_factory=list)
    media_refs: list[str] = field(default_factory=list)   # video URLs/paths for VLM followups
    recommendation_question: str = "Should this scenario be added to the regression suite?"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "window_id": self.window_id,
            "scene_token_hex": self.scene_token_hex, "log_id": self.log_id,
            "scene_description": self.scene_description,
            "anomaly_rationale": self.anomaly_rationale,
            "severity_hint": self.severity_hint,
            "regression_suite": list(self.regression_suite),
            "media_refs": list(self.media_refs),
            "recommendation_question": self.recommendation_question,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DebateInput":
        return cls(
            run_id=str(d["run_id"]), window_id=str(d["window_id"]),
            scene_token_hex=str(d.get("scene_token_hex", "")), log_id=str(d.get("log_id", "")),
            scene_description=str(d.get("scene_description", "")),
            anomaly_rationale=str(d.get("anomaly_rationale", "")),
            severity_hint=str(d.get("severity_hint", "unknown")),
            regression_suite=list(d.get("regression_suite", [])),
            media_refs=list(d.get("media_refs", [])),
            recommendation_question=str(d.get("recommendation_question", "")),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class SceneDescription:
    """VLM scene-description output (debate's describe stage)."""
    run_id: str
    window_id: str
    scene_description: str
    anomaly_rationale: str
    confidence: str = "unknown"             # low|medium|high|unknown
    model_source: str = ""
    media_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "window_id": self.window_id,
            "scene_description": self.scene_description,
            "anomaly_rationale": self.anomaly_rationale,
            "confidence": self.confidence, "model_source": self.model_source,
            "media_refs": list(self.media_refs), "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SceneDescription":
        return cls(
            run_id=str(d.get("run_id", "")), window_id=str(d["window_id"]),
            scene_description=str(d.get("scene_description", "")),
            anomaly_rationale=str(d.get("anomaly_rationale", "")),
            confidence=str(d.get("confidence", "unknown")),
            model_source=str(d.get("model_source", "")),
            media_refs=list(d.get("media_refs", [])), metadata=dict(d.get("metadata", {})),
        )


@dataclass
class RegressionCaseProposal:
    """The rich debate artifact: a structured regression-case proposal."""
    case_id: str
    run_id: str
    window_id: str
    scene_token_hex: str
    log_id: str
    generated_at: str
    failure_mode: str
    why_anomalous: str
    evidence_summary: str
    risk_level: str                         # critical|high|medium|low
    affected_capability: str
    affected_odds: list[str] = field(default_factory=list)
    counterarguments: list[str] = field(default_factory=list)
    rebuttal_summary: str = ""
    decision: str = "monitor"               # add_to_suite|monitor|dismiss
    recommended_test_spec: str = ""
    scenario_variants: list[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty_factors: list[str] = field(default_factory=list)
    debate_transcript: list[str] = field(default_factory=list)
    model_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "run_id": self.run_id, "window_id": self.window_id,
            "scene_token_hex": self.scene_token_hex, "log_id": self.log_id,
            "generated_at": self.generated_at, "failure_mode": self.failure_mode,
            "why_anomalous": self.why_anomalous, "evidence_summary": self.evidence_summary,
            "risk_level": self.risk_level, "affected_capability": self.affected_capability,
            "affected_odds": list(self.affected_odds),
            "counterarguments": list(self.counterarguments),
            "rebuttal_summary": self.rebuttal_summary, "decision": self.decision,
            "recommended_test_spec": self.recommended_test_spec,
            "scenario_variants": list(self.scenario_variants), "confidence": self.confidence,
            "uncertainty_factors": list(self.uncertainty_factors),
            "debate_transcript": list(self.debate_transcript),
            "model_source": self.model_source, "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RegressionCaseProposal":
        return cls(
            case_id=str(d["case_id"]), run_id=str(d.get("run_id", "")),
            window_id=str(d.get("window_id", "")), scene_token_hex=str(d.get("scene_token_hex", "")),
            log_id=str(d.get("log_id", "")), generated_at=str(d.get("generated_at", "")),
            failure_mode=str(d.get("failure_mode", "")), why_anomalous=str(d.get("why_anomalous", "")),
            evidence_summary=str(d.get("evidence_summary", "")),
            risk_level=str(d.get("risk_level", "low")),
            affected_capability=str(d.get("affected_capability", "")),
            affected_odds=list(d.get("affected_odds", [])),
            counterarguments=list(d.get("counterarguments", [])),
            rebuttal_summary=str(d.get("rebuttal_summary", "")),
            decision=str(d.get("decision", "monitor")),
            recommended_test_spec=str(d.get("recommended_test_spec", "")),
            scenario_variants=list(d.get("scenario_variants", [])),
            confidence=float(d.get("confidence", 0.0)),
            uncertainty_factors=list(d.get("uncertainty_factors", [])),
            debate_transcript=list(d.get("debate_transcript", [])),
            model_source=str(d.get("model_source", "")), metadata=dict(d.get("metadata", {})),
        )


@dataclass
class DebateResult:
    """The debate module's one-call output: decision + proposal + description."""
    window_id: str
    decision: str                           # yes|no  (add to suite?)
    recommendation: str                     # add_immediately|already_covered|not_critical
    priority_score: float
    rationale: str
    proposal: RegressionCaseProposal
    description: SceneDescription
    model_source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id, "decision": self.decision,
            "recommendation": self.recommendation, "priority_score": self.priority_score,
            "rationale": self.rationale, "proposal": self.proposal.to_json(),
            "description": self.description.to_json(), "model_source": self.model_source,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DebateResult":
        return cls(
            window_id=str(d["window_id"]), decision=str(d.get("decision", "no")),
            recommendation=str(d.get("recommendation", "not_critical")),
            priority_score=float(d.get("priority_score", 0.0)),
            rationale=str(d.get("rationale", "")),
            proposal=RegressionCaseProposal.from_json(d["proposal"]),
            description=SceneDescription.from_json(d["description"]),
            model_source=str(d.get("model_source", "")),
        )
