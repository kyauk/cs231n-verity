"""Build RegressionCaseProposal artifacts from debate output metadata.

Step 1 of the tool-augmented debate engine stashes rich per-actor structured
output under namespaced ``proposal_*`` keys inside
``DebateOutputRecord.metadata``. This module consumes those keys and assembles
a first-class :class:`RegressionCaseProposal` model that downstream consumers
(dashboard, video lab, remote runner) can render directly without digging
through metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from waymo_pipeline.models.handoff_contracts import (
    DebateOutputRecord,
    RegressionCaseProposal,
)


_ALLOWED_RISK_LEVELS: set[str] = {"critical", "high", "medium", "low"}
_ALLOWED_DECISIONS: set[str] = {"add_to_suite", "monitor", "dismiss"}


def _as_string(value: Any, fallback: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif item is not None:
            out.append(str(item))
    return out


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_risk_level(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOWED_RISK_LEVELS:
            return normalized
    return "low"


def _coerce_decision(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in _ALLOWED_DECISIONS:
            return normalized
    return "monitor"


def build_proposal_from_debate_output(
    debate_output: DebateOutputRecord,
    run_id: str,
) -> RegressionCaseProposal:
    """Assemble a RegressionCaseProposal from a debate output row.

    Parameters:
        debate_output (DebateOutputRecord): Debate output row whose ``metadata``
            was enriched by the tool-augmented debate engine with ``proposal_*``
            keys.
        run_id (str): Current pipeline run id.

    Returns:
        RegressionCaseProposal: Structured proposal artifact.
    """

    metadata = debate_output.metadata or {}

    case_id = f"case_{uuid4().hex[:8]}"
    generated_at = datetime.now(timezone.utc).isoformat()

    failure_mode = _as_string(
        metadata.get("proposal_failure_mode"),
        fallback=debate_output.rationale or "Failure mode not captured.",
    )
    why_anomalous = _as_string(
        metadata.get("proposal_why_anomalous"),
        fallback="Why-anomalous rationale not captured.",
    )
    evidence_summary = _as_string(
        metadata.get("proposal_evidence_summary"),
        fallback="Evidence summary not captured.",
    )

    risk_level = _coerce_risk_level(metadata.get("proposal_risk_level"))
    affected_capability = _as_string(
        metadata.get("proposal_affected_capability")
        or metadata.get("capability_tag"),
        fallback="unspecified_capability",
    )
    affected_odds = _as_string_list(metadata.get("proposal_affected_odds"))

    counterarguments = _as_string_list(metadata.get("proposal_counterarguments"))
    rebuttal_summary = _as_string(metadata.get("proposal_rebuttal_summary"))

    decision = _coerce_decision(metadata.get("proposal_decision"))
    recommended_test_spec = _as_string(
        metadata.get("proposal_recommended_test_spec"),
        fallback="No concrete test specification produced.",
    )
    scenario_variants = _as_string_list(metadata.get("proposal_scenario_variants"))
    confidence = max(
        0.0,
        min(
            1.0,
            _as_float(
                metadata.get("proposal_confidence"),
                fallback=_as_float(debate_output.priority_score, 0.0),
            ),
        ),
    )
    uncertainty_factors = _as_string_list(
        metadata.get("proposal_uncertainty_factors")
    )

    debate_transcript = _as_string_list(metadata.get("debate_history"))

    proposal_metadata = {
        "debate_mode": metadata.get("debate_mode", ""),
        "debate_recommendation": debate_output.recommendation,
        "debate_decision": debate_output.decision,
        "priority_score": debate_output.priority_score,
        "rationale": debate_output.rationale,
    }

    return RegressionCaseProposal(
        case_id=case_id,
        run_id=run_id,
        window_id=debate_output.window_id,
        scene_token_hex=debate_output.scene_token_hex,
        log_id=debate_output.log_id,
        generated_at=generated_at,
        failure_mode=failure_mode,
        why_anomalous=why_anomalous,
        evidence_summary=evidence_summary,
        risk_level=risk_level,  # type: ignore[arg-type]
        affected_capability=affected_capability,
        affected_odds=affected_odds,
        counterarguments=counterarguments,
        rebuttal_summary=rebuttal_summary,
        decision=decision,  # type: ignore[arg-type]
        recommended_test_spec=recommended_test_spec,
        scenario_variants=scenario_variants,
        confidence=confidence,
        uncertainty_factors=uncertainty_factors,
        debate_transcript=debate_transcript,
        model_source=debate_output.model_source,
        metadata=proposal_metadata,
    )
