"""Tool-augmented ReAct debate orchestrator.

Wires four specialist actors (Scene Analyst, Risk Assessor, Coverage Analyst,
and Synthesis Arbiter) into a sequential debate using
:func:`pipeline.react_loop.run_actor`. Produces a backward-compatible
:class:`DebateOutputRecord` whose rich per-actor data is stashed in
``metadata`` under namespaced keys so existing consumers (frontend, remote
runner) keep working unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pipeline.actor_tools import DebateContext
from pipeline.models.handoff_contracts import DebateInputRecord, DebateOutputRecord
from pipeline.react_loop import ActorContribution, run_actor


# ---------------------------------------------------------------------------
# Actor system prompts
# ---------------------------------------------------------------------------


SCENE_ANALYST_SYSTEM_PROMPT = (
    "You are a scene analysis specialist for autonomous driving safety. Your "
    "job is to deeply understand what is happening in a flagged driving scene. "
    "You have access to a vision-language model that can answer targeted "
    "visual questions about the video. Use it to clarify ambiguous details - "
    "ask about specific objects, positions, trajectories, visibility "
    "conditions, or traffic elements. Prefer 1-3 well-targeted VLM queries "
    "over lengthy speculation. When finished, produce a JSON with keys: "
    "`failure_mode` (string: what failure or near-failure occurred), "
    "`why_anomalous` (string: why this deviates from normal driving), "
    "`evidence_summary` (string: key visual evidence supporting your finding)."
)

RISK_ASSESSOR_SYSTEM_PROMPT = (
    "You are an AV safety risk assessment specialist. Given a scene "
    "description and analysis findings, assess the safety risk level and "
    "identify which AV capabilities are affected. Use the safety taxonomy "
    "lookup tool to ground your assessment in a formal capability taxonomy. "
    "When finished, produce a JSON with keys: `risk_level` (one of: critical, "
    "high, medium, low), `affected_capability` (string: primary capability "
    "tag from taxonomy), `affected_odds` (list of strings: relevant "
    "Operational Design Domain conditions like weather, lighting, road type)."
)

COVERAGE_ANALYST_SYSTEM_PROMPT = (
    "You are a regression suite coverage analyst. Determine whether this "
    "scenario is genuinely novel or already covered by existing tests. Use "
    "the suite similarity tool to get quantitative overlap scores against the "
    "current suite. Play devil's advocate: produce strong counterarguments "
    "for why this scenario might NOT warrant addition. Be rigorous and "
    "skeptical. When finished, produce a JSON with keys: `counterarguments` "
    "(list of strings: reasons this might not need adding), `rebuttal_summary` "
    "(string: honest assessment of whether the counterarguments hold up or not)."
)

SYNTHESIS_ARBITER_SYSTEM_PROMPT = (
    "You are the final synthesis arbiter. You have received analyses from "
    "three specialists: a Scene Analyst (what happened), a Risk Assessor "
    "(how dangerous), and a Coverage Analyst (counterarguments for "
    "redundancy). Weigh all evidence. Produce a JSON with keys: `decision` "
    "(one of: add_to_suite, monitor, dismiss), `recommended_test_spec` "
    "(string: a concrete, actionable regression test specification an "
    "engineer could implement), `scenario_variants` (list of strings: "
    "perturbation variants to also test, e.g. different weather/lighting/"
    "actor behavior), `confidence` (float 0-1: your confidence in the "
    "decision), `uncertainty_factors` (list of strings: what you're unsure "
    "about that a human reviewer should verify)."
)


# ---------------------------------------------------------------------------
# User-message builders
# ---------------------------------------------------------------------------


def _format_suite(regression_suite: list[str]) -> str:
    if not regression_suite:
        return "(regression suite is currently empty)"
    return "\n".join(
        f"{index + 1}. {entry}" for index, entry in enumerate(regression_suite)
    )


def _build_scene_analyst_message(context: DebateContext) -> str:
    return (
        f"Window ID: {context.window_id}\n"
        f"Severity hint: {context.severity_hint}\n\n"
        f"Scene description (from prior VLM pass):\n{context.scene_description}\n\n"
        f"Anomaly rationale (from prior VLM pass):\n{context.anomaly_rationale}\n\n"
        "Your task: use vlm_followup sparingly (1-3 questions) to resolve the "
        "most important ambiguities, then finish with the required JSON."
    )


def _build_risk_assessor_message(
    context: DebateContext,
    scene_output: dict[str, Any],
) -> str:
    return (
        f"Scene description:\n{context.scene_description}\n\n"
        f"Anomaly rationale:\n{context.anomaly_rationale}\n\n"
        f"Severity hint from anomaly stage: {context.severity_hint}\n\n"
        f"Scene Analyst findings (JSON):\n{json.dumps(scene_output, indent=2)}\n\n"
        "Your task: call safety_taxonomy_lookup at least once to ground your "
        "analysis in the capability taxonomy, then finish with the required JSON."
    )


def _build_coverage_analyst_message(
    context: DebateContext,
    scene_output: dict[str, Any],
) -> str:
    return (
        f"Scene description:\n{context.scene_description}\n\n"
        f"Scene Analyst findings (JSON):\n{json.dumps(scene_output, indent=2)}\n\n"
        f"Existing regression suite:\n{_format_suite(context.regression_suite)}\n\n"
        "Your task: call suite_similarity with a concise scenario description, "
        "then argue skeptically against adding this case. Finish with the "
        "required JSON."
    )


def _build_arbiter_message(
    context: DebateContext,
    scene_output: dict[str, Any],
    risk_output: dict[str, Any],
    coverage_output: dict[str, Any],
) -> str:
    return (
        f"Window ID: {context.window_id}\n"
        f"Severity hint from anomaly stage: {context.severity_hint}\n\n"
        f"Scene description:\n{context.scene_description}\n\n"
        f"Anomaly rationale:\n{context.anomaly_rationale}\n\n"
        f"Existing regression suite:\n{_format_suite(context.regression_suite)}\n\n"
        f"Scene Analyst output (JSON):\n{json.dumps(scene_output, indent=2)}\n\n"
        f"Risk Assessor output (JSON):\n{json.dumps(risk_output, indent=2)}\n\n"
        f"Coverage Analyst output (JSON):\n{json.dumps(coverage_output, indent=2)}\n\n"
        "Your task: no tools are needed; reason over the three contributions "
        "above and finish with the required JSON."
    )


# ---------------------------------------------------------------------------
# Transcript / rationale helpers
# ---------------------------------------------------------------------------


def _actor_transcript_lines(contribution: ActorContribution) -> list[str]:
    """Flatten one actor's steps into human-readable transcript lines."""

    lines: list[str] = []
    actor_prefix = f"[{contribution.actor_name}]"
    for step_index, step in enumerate(contribution.steps, start=1):
        thought_preview = (step.thought or "").strip().replace("\n", " ")
        if len(thought_preview) > 400:
            thought_preview = thought_preview[:397] + "..."

        if step.tool_call is None:
            lines.append(
                f"{actor_prefix} Step {step_index} Thought: {thought_preview}"
            )
            continue

        tool_name = step.tool_call.tool_name
        try:
            tool_input_rendered = json.dumps(step.tool_call.tool_input, ensure_ascii=False)
        except (TypeError, ValueError):
            tool_input_rendered = str(step.tool_call.tool_input)
        if len(tool_input_rendered) > 300:
            tool_input_rendered = tool_input_rendered[:297] + "..."

        line = (
            f"{actor_prefix} Step {step_index} "
            f"Thought: {thought_preview} | "
            f"Action: {tool_name} | "
            f"Action Input: {tool_input_rendered}"
        )

        if step.observation is not None:
            observation_preview = step.observation.strip().replace("\n", " ")
            if len(observation_preview) > 400:
                observation_preview = observation_preview[:397] + "..."
            line += f" | Observation: {observation_preview}"
        lines.append(line)

    if contribution.final_output:
        try:
            final_rendered = json.dumps(contribution.final_output, ensure_ascii=False)
        except (TypeError, ValueError):
            final_rendered = str(contribution.final_output)
        if len(final_rendered) > 600:
            final_rendered = final_rendered[:597] + "..."
        lines.append(f"{actor_prefix} Final: {final_rendered}")

    return lines


_ARBITER_DECISION_MAP: dict[str, tuple[str, str]] = {
    "add_to_suite": ("yes", "add_immediately"),
    "monitor": ("yes", "not_critical"),
    "dismiss": ("no", "already_covered"),
}


def _coerce_decision(raw_decision: Any) -> tuple[str, str, str]:
    """Map arbiter decision string to (proposal_decision, decision, recommendation)."""

    if isinstance(raw_decision, str):
        normalized = raw_decision.strip().lower().replace(" ", "_").replace("-", "_")
    else:
        normalized = ""

    if normalized in _ARBITER_DECISION_MAP:
        decision, recommendation = _ARBITER_DECISION_MAP[normalized]
        return normalized, decision, recommendation

    # Fallback: unknown decision string. Default to conservative "monitor"
    # mapping so a DebateOutputRecord is always produced.
    return "monitor", "yes", "not_critical"


def _coerce_confidence(raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _compose_rationale(
    arbiter_output: dict[str, Any],
    risk_output: dict[str, Any],
    coverage_output: dict[str, Any],
) -> str:
    """Assemble a one-paragraph rationale for the DebateOutputRecord."""

    proposal_decision = str(arbiter_output.get("decision", "")).strip() or "unspecified"
    test_spec = str(arbiter_output.get("recommended_test_spec", "")).strip()
    risk_level = str(risk_output.get("risk_level", "")).strip() or "unspecified"
    capability = str(risk_output.get("affected_capability", "")).strip() or "unspecified"
    rebuttal = str(coverage_output.get("rebuttal_summary", "")).strip()

    pieces = [
        f"Arbiter decision: {proposal_decision}.",
        f"Risk level: {risk_level}; primary capability: {capability}.",
    ]
    if rebuttal:
        pieces.append(f"Coverage rebuttal: {rebuttal}")
    if test_spec:
        pieces.append(f"Recommended test spec: {test_spec}")

    rationale = " ".join(pieces).strip()
    if not rationale:
        rationale = (
            "Tool-augmented debate completed but actors returned insufficient "
            "structured output to compose a rationale."
        )
    return rationale


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------


def _make_actor_llm_call():
    """Return an llm_call callable that honors REACT_ACTOR_TEMPERATURE.

    The existing :func:`_nim_text_chat_completion` reads ``DEBATE_TEMPERATURE``
    from the environment. To avoid modifying it we temporarily override that
    env var for the duration of each call so actors run at their own
    temperature without disturbing other code paths.
    """

    # Local import to avoid circular import at module load time.
    from pipeline.stage_describe_and_debate import _nim_text_chat_completion

    def _call(messages: list[dict[str, Any]]) -> str:
        actor_temperature = os.getenv("REACT_ACTOR_TEMPERATURE")
        if actor_temperature is None:
            return _nim_text_chat_completion(messages)

        previous = os.environ.get("DEBATE_TEMPERATURE")
        os.environ["DEBATE_TEMPERATURE"] = actor_temperature
        try:
            return _nim_text_chat_completion(messages)
        finally:
            if previous is None:
                os.environ.pop("DEBATE_TEMPERATURE", None)
            else:
                os.environ["DEBATE_TEMPERATURE"] = previous

    return _call


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_tool_augmented_debate(
    record: DebateInputRecord,
    media_refs: list[str],
) -> tuple[DebateOutputRecord, dict[str, Any]]:
    """Run the four-actor tool-augmented debate for one window.

    Returns a backward-compatible :class:`DebateOutputRecord` alongside the
    raw ``proposal_metadata`` dict so Step 2 can adopt the richer per-actor
    fields directly without re-parsing JSONL rows.
    """

    # Local import to avoid circular import at module load time.
    from pipeline.stage_describe_and_debate import _emit_pipeline_progress

    context = DebateContext(
        media_refs=list(media_refs or []),
        scene_description=record.scene_description,
        anomaly_rationale=record.anomaly_rationale,
        severity_hint=str(record.severity_hint),
        regression_suite=list(record.regression_suite),
        window_id=record.window_id,
        video_fps=float(os.getenv("COSMOS_HF_VIDEO_FPS", "8")),
    )

    max_steps = int(os.getenv("REACT_MAX_STEPS_PER_ACTOR", "5"))
    llm_call = _make_actor_llm_call()

    _emit_pipeline_progress(
        "debate",
        "Tool-augmented debate",
        "Running Scene Analyst, Risk Assessor, Coverage Analyst, Synthesis Arbiter.",
    )

    _emit_pipeline_progress(
        "debate_round",
        "Scene Analyst",
        "Interrogating the video with targeted visual follow-ups...",
    )
    scene_contribution = run_actor(
        actor_name="Scene Analyst",
        system_prompt=SCENE_ANALYST_SYSTEM_PROMPT,
        available_tools=["vlm_followup"],
        user_message=_build_scene_analyst_message(context),
        context=context,
        max_steps=max_steps,
        llm_call=llm_call,
    )

    _emit_pipeline_progress(
        "debate_round",
        "Risk Assessor",
        "Grounding risk in the AV safety capability taxonomy...",
    )
    risk_contribution = run_actor(
        actor_name="Risk Assessor",
        system_prompt=RISK_ASSESSOR_SYSTEM_PROMPT,
        available_tools=["safety_taxonomy_lookup"],
        user_message=_build_risk_assessor_message(
            context, scene_contribution.final_output
        ),
        context=context,
        max_steps=max_steps,
        llm_call=llm_call,
    )

    _emit_pipeline_progress(
        "debate_round",
        "Coverage Analyst",
        "Comparing scenario against existing regression suite...",
    )
    coverage_contribution = run_actor(
        actor_name="Coverage Analyst",
        system_prompt=COVERAGE_ANALYST_SYSTEM_PROMPT,
        available_tools=["suite_similarity"],
        user_message=_build_coverage_analyst_message(
            context, scene_contribution.final_output
        ),
        context=context,
        max_steps=max_steps,
        llm_call=llm_call,
    )

    _emit_pipeline_progress(
        "debate_judge",
        "Synthesis Arbiter",
        "Weighing all three analyses and emitting final verdict...",
    )
    arbiter_contribution = run_actor(
        actor_name="Synthesis Arbiter",
        system_prompt=SYNTHESIS_ARBITER_SYSTEM_PROMPT,
        available_tools=[],
        user_message=_build_arbiter_message(
            context,
            scene_contribution.final_output,
            risk_contribution.final_output,
            coverage_contribution.final_output,
        ),
        context=context,
        max_steps=max_steps,
        llm_call=llm_call,
    )

    scene_output = scene_contribution.final_output or {}
    risk_output = risk_contribution.final_output or {}
    coverage_output = coverage_contribution.final_output or {}
    arbiter_output = arbiter_contribution.final_output or {}

    all_contributions = [
        scene_contribution,
        risk_contribution,
        coverage_contribution,
        arbiter_contribution,
    ]

    debate_history: list[str] = []
    for contribution in all_contributions:
        debate_history.extend(_actor_transcript_lines(contribution))

    proposal_decision, decision, recommendation = _coerce_decision(
        arbiter_output.get("decision")
    )
    priority_score = _coerce_confidence(arbiter_output.get("confidence"))

    capability_tag = str(risk_output.get("affected_capability", "")).strip()
    if not capability_tag:
        capability_tag = "unspecified_capability"

    rationale = _compose_rationale(arbiter_output, risk_output, coverage_output)

    proposal_metadata: dict[str, Any] = {
        "debate_mode": "react_tool_augmented",
        "debate_history": debate_history,
        "capability_tag": capability_tag,
        "judge_raw_output": arbiter_contribution.raw_llm_output,
        "actor_contributions": [c.model_dump() for c in all_contributions],
        "proposal_failure_mode": scene_output.get("failure_mode", ""),
        "proposal_why_anomalous": scene_output.get("why_anomalous", ""),
        "proposal_evidence_summary": scene_output.get("evidence_summary", ""),
        "proposal_risk_level": risk_output.get("risk_level", ""),
        "proposal_affected_capability": risk_output.get("affected_capability", ""),
        "proposal_affected_odds": risk_output.get("affected_odds", []),
        "proposal_counterarguments": coverage_output.get("counterarguments", []),
        "proposal_rebuttal_summary": coverage_output.get("rebuttal_summary", ""),
        "proposal_decision": proposal_decision,
        "proposal_recommended_test_spec": arbiter_output.get(
            "recommended_test_spec", ""
        ),
        "proposal_scenario_variants": arbiter_output.get("scenario_variants", []),
        "proposal_confidence": priority_score,
        "proposal_uncertainty_factors": arbiter_output.get(
            "uncertainty_factors", []
        ),
    }

    enriched_metadata = dict(record.metadata)
    enriched_metadata.update(proposal_metadata)

    debate_output = DebateOutputRecord(
        run_id=record.run_id,
        window_id=record.window_id,
        scene_token_hex=record.scene_token_hex,
        log_id=record.log_id,
        decision=decision,
        recommendation=recommendation,
        priority_score=priority_score,
        rationale=rationale,
        model_source="nim_text_react_debate",
        metadata=enriched_metadata,
    )

    return debate_output, proposal_metadata
