"""Module 9: Debate — the one-call orchestrator.

:class:`Debater` ties together the three porting stages of the debate arm:

  1. describe (optional)  — if ``DebateInput.scene_description`` is empty, run a
     VLM describe pass to populate :class:`SceneDescription`;
  2. debate               — run the four-actor tool-augmented debate
     (:func:`pipeline.modules.debate.actors.run_tool_augmented_debate`),
     threading the injected clients down through the ReAct loop and tools;
  3. propose              — build a :class:`RegressionCaseProposal` from the
     debate output (:mod:`pipeline.modules.debate.proposals`).

It then returns a :class:`DebateResult`. All model access is via the injected
``TextLLMClient`` / ``VLMClient`` protocols (lego rule).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pipeline.interfaces.debate import (
    DebateInput,
    DebateResult,
    SceneDescription,
)
from pipeline.modules.debate.actors import run_tool_augmented_debate
from pipeline.modules.debate.config import (
    DebateConfig,
    TextLLMClient,
    VLMClient,
)
from pipeline.modules.debate.proposals import build_proposal_from_debate_output


def _extract_first_json_object(candidate: str) -> str | None:
    """Return the first balanced ``{...}`` block from ``candidate``."""

    start = candidate.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(candidate)):
        char = candidate[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    return None


def _parse_description_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of a VLM describe response into the description schema.

    Mirrors the description-stage contract: a JSON object with
    ``scene_description``, ``anomaly_rationale``, and ``confidence``. Tolerates
    fenced-code wrappers and surrounding prose; missing keys fall back to
    empty / ``unknown``.
    """

    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return {
            "scene_description": text,
            "anomaly_rationale": "",
            "confidence": "unknown",
        }

    confidence = str(parsed.get("confidence", "unknown")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "unknown"

    return {
        "scene_description": str(parsed.get("scene_description", "")).strip(),
        "anomaly_rationale": str(parsed.get("anomaly_rationale", "")).strip(),
        "confidence": confidence,
    }


class Debater:
    """Multi-agent debate orchestrator for a single flagged window."""

    def __init__(
        self,
        text_client: TextLLMClient,
        vlm_client: VLMClient,
        config: DebateConfig = DebateConfig(),
    ) -> None:
        self.text_client = text_client
        self.vlm_client = vlm_client
        self.config = config

    # ------------------------------------------------------------------
    # Stage 1: describe
    # ------------------------------------------------------------------

    def _describe(self, input: DebateInput) -> SceneDescription:
        """Produce a SceneDescription, calling the VLM only when needed."""

        existing = (input.scene_description or "").strip()
        if existing:
            return SceneDescription(
                run_id=input.run_id,
                window_id=input.window_id,
                scene_description=existing,
                anomaly_rationale=input.anomaly_rationale,
                confidence="unknown",
                model_source="",
                media_refs=list(input.media_refs),
                metadata=dict(input.metadata),
            )

        anomaly_priors = {
            "window_id": input.window_id,
            "scene_token_hex": input.scene_token_hex,
            "log_id": input.log_id,
            "severity_hint": input.severity_hint,
            "anomaly_rationale": input.anomaly_rationale,
        }
        video_ref = input.media_refs[0] if input.media_refs else ""
        raw = self.vlm_client.describe(video_ref, anomaly_priors)
        parsed = _parse_description_json(raw)

        scene_description = parsed["scene_description"]
        anomaly_rationale = parsed["anomaly_rationale"] or input.anomaly_rationale
        return SceneDescription(
            run_id=input.run_id,
            window_id=input.window_id,
            scene_description=scene_description,
            anomaly_rationale=anomaly_rationale,
            confidence=parsed["confidence"],
            model_source=getattr(self.vlm_client, "model_id", ""),
            media_refs=list(input.media_refs),
            metadata=dict(input.metadata),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, input: DebateInput) -> DebateResult:
        """Describe (if needed), debate, and build a structured proposal."""

        description = self._describe(input)

        # The debate consumes the (possibly freshly described) scene + rationale.
        debate_record = DebateInput(
            run_id=input.run_id,
            window_id=input.window_id,
            scene_token_hex=input.scene_token_hex,
            log_id=input.log_id,
            scene_description=description.scene_description,
            anomaly_rationale=description.anomaly_rationale,
            severity_hint=input.severity_hint,
            regression_suite=list(input.regression_suite),
            media_refs=list(input.media_refs),
            recommendation_question=input.recommendation_question,
            metadata=dict(input.metadata),
        )

        debate_output, _proposal_metadata = run_tool_augmented_debate(
            debate_record,
            media_refs=list(input.media_refs),
            text_client=self.text_client,
            vlm_client=self.vlm_client,
            config=self.config,
        )

        proposal = build_proposal_from_debate_output(debate_output, input.run_id)

        return DebateResult(
            window_id=debate_output.window_id,
            decision=debate_output.decision,
            recommendation=debate_output.recommendation,
            priority_score=debate_output.priority_score,
            rationale=debate_output.rationale,
            proposal=proposal,
            description=description,
            model_source=debate_output.model_source,
        )
