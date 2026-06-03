"""Derive priority scores and evidence-adjusted confidence from debate signals."""

from __future__ import annotations

import re
from typing import Any

from waymo_pipeline.react_loop import ActorContribution

_RISK_WEIGHT: dict[str, float] = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}
_DECISION_WEIGHT: dict[str, float] = {
    "add_to_suite": 1.0,
    "monitor": 0.55,
    "dismiss": 0.2,
}
_SEVERITY_WEIGHT: dict[str, float] = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
    "unknown": 0.45,
}

_OVERLAP_RE = re.compile(r"highest overlap is\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
_JACCARD_RE = re.compile(r"jaccard=([0-9]*\.?[0-9]+)", re.IGNORECASE)


def observation_indicates_tool_error(observation: str) -> bool:
    """Return True when a tool observation looks like a failed tool call."""

    lowered = observation.lower().strip()
    if not lowered:
        return False
    if "cuda out of memory" in lowered:
        return True
    if " error:" in lowered or lowered.endswith(" error"):
        return True
    if lowered.startswith("error:"):
        return True
    if " tool error" in lowered or "vlm_followup error" in lowered:
        return True
    return False


def summarize_tool_evidence(contributions: list[ActorContribution]) -> dict[str, Any]:
    """Count tool calls vs failed observations across actor steps."""

    tool_calls = 0
    tool_errors = 0
    for contribution in contributions:
        for step in contribution.steps:
            if step.tool_call is None:
                continue
            tool_calls += 1
            if observation_indicates_tool_error(step.observation or ""):
                tool_errors += 1

    ratio = (tool_errors / tool_calls) if tool_calls else 0.0
    return {
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "tool_failure_ratio": ratio,
    }


def extract_max_suite_overlap(contributions: list[ActorContribution]) -> float | None:
    """Parse suite_similarity observations for the highest jaccard overlap seen."""

    best: float | None = None
    for contribution in contributions:
        for step in contribution.steps:
            observation = step.observation or ""
            overlap_match = _OVERLAP_RE.search(observation)
            if overlap_match:
                value = float(overlap_match.group(1))
                best = value if best is None else max(best, value)
            for jaccard_text in _JACCARD_RE.findall(observation):
                value = float(jaccard_text)
                best = value if best is None else max(best, value)
    return best


def scene_analyst_had_vlm_failures(scene_contribution: ActorContribution) -> bool:
    """True when Scene Analyst vlm_followup calls returned errors (not text-only fallback)."""

    for step in scene_contribution.steps:
        if step.tool_call is None or step.tool_call.tool_name != "vlm_followup":
            continue
        observation = step.observation or ""
        if observation.startswith("[text-only]"):
            continue
        if observation_indicates_tool_error(observation):
            return True
    return False


def apply_evidence_penalty(
    confidence: float,
    *,
    tool_failure_ratio: float,
    had_vlm_failures: bool,
) -> float:
    """Reduce arbiter confidence when tools failed or visual evidence is weak."""

    adjusted = confidence
    if tool_failure_ratio > 0:
        adjusted *= max(0.35, 1.0 - 0.65 * tool_failure_ratio)
    if had_vlm_failures:
        adjusted = min(adjusted, 0.65)
        if tool_failure_ratio >= 0.5:
            adjusted = min(adjusted, 0.5)
    return max(0.05, min(1.0, round(adjusted, 3)))


def compute_priority_score(
    *,
    proposal_decision: str,
    risk_level: str,
    severity_hint: str,
    max_suite_overlap: float | None,
    tool_failure_ratio: float,
) -> float:
    """Combine structured debate signals into a triage priority score in [0, 1]."""

    risk_key = str(risk_level).strip().lower()
    decision_key = proposal_decision.strip().lower()
    severity_key = str(severity_hint).strip().lower()

    risk_w = _RISK_WEIGHT.get(risk_key, 0.5)
    decision_w = _DECISION_WEIGHT.get(decision_key, 0.5)
    severity_w = _SEVERITY_WEIGHT.get(severity_key, 0.45)

    base = 0.45 * risk_w + 0.35 * decision_w + 0.20 * severity_w

    if max_suite_overlap is not None:
        novelty = max(0.0, min(1.0, 1.0 - max_suite_overlap))
        base = 0.82 * base + 0.18 * novelty

    reliability = max(0.4, 1.0 - 0.55 * tool_failure_ratio)
    score = base * reliability
    return max(0.0, min(1.0, round(score, 3)))
