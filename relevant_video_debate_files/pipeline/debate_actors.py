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
from pipeline.debate_scoring import (
    apply_evidence_penalty,
    compute_priority_score,
    extract_max_suite_overlap,
    scene_analyst_had_vlm_failures,
    summarize_tool_evidence,
)
from pipeline.models.handoff_contracts import DebateInputRecord, DebateOutputRecord
from pipeline.react_loop import ActorContribution, run_actor


# ---------------------------------------------------------------------------
# Actor system prompts
# ---------------------------------------------------------------------------


SCENE_ANALYST_SYSTEM_PROMPT = (
    "You are a scene analysis specialist for autonomous driving safety. "
    "IMPORTANT FRAMING: every video you review is a REAL recording from a "
    "vehicle data-collection fleet. It is never synthetic, adversarial, or "
    "fabricated. Do NOT speculate about whether the footage is fake, edited, "
    "or staged - assume ground-truth reality in every frame. "
    "This pipeline flags clips because they may be valuable edge cases for "
    "autonomous vehicle regression testing: rare real-world situations an AV "
    "might handle poorly, worth a test engineer's review. That is unrelated to "
    "whether the recording is trustworthy. "
    "Never explain what you see as a glitch, unrealistic physics, rendering "
    "error, or 'inconsistent scene' - if something looks abrupt or unexplained "
    "in one view, treat it as incomplete causal information and use the VLM "
    "tool to ask what happened before or off-frame (debris source, prior "
    "impact, occluded object, etc.). "
    "Your job is to explain what actually happened in the world: identify the "
    "agents involved (ego vehicle, other vehicles, pedestrians, cyclists, "
    "infrastructure), reconstruct the causal chain (e.g. 'a lead vehicle "
    "struck a construction sign sitting in the travel lane and was deflected "
    "into the ego lane'), and pinpoint why this situation is a valuable AV "
    "regression edge case. "
    "You may use vlm_followup sparingly (0-2 calls) to clarify critical causal "
    "ambiguities; answers are drawn from the prior scene description unless "
    "live GPU VLM is enabled. Prefer finishing over repeated tool calls. "
    "When finished, produce a JSON with keys: "
    "`failure_mode` (string: the real-world event / causal chain, e.g. "
    "'lead vehicle struck a misplaced construction sign on the highway and "
    "rotated into the ego lane, forcing an emergency lateral maneuver' - "
    "describe what WORLD EVENT the AV would need to handle, not a data "
    "quality concern), "
    "`why_valuable_for_regression` (string: why this scenario is a valuable "
    "edge case for AV regression - what makes planner / perception / prediction "
    "behavior hard to specify or under-tested, e.g. 'ego must react to a "
    "vehicle arriving laterally from outside the normal longitudinal lead "
    "set, with no prior cue'), "
    "`evidence_summary` (string: key visual evidence from the video that "
    "supports the causal reconstruction)."
)

RISK_ASSESSOR_SYSTEM_PROMPT = (
    "You are an AV safety risk assessment specialist. "
    "IMPORTANT FRAMING: the scene you are assessing is a REAL recorded "
    "driving situation, not synthetic data. It was surfaced as a candidate "
    "valuable edge case for autonomous vehicle regression testing - a "
    "rare real-world situation an AV might handle poorly. "
    "Your question is always: 'if an AV encountered THIS scenario, which "
    "safety-critical capabilities would be stressed and how badly could it "
    "fail?' - never 'is this data trustworthy?' and never 'is the physics "
    "realistic?' or 'is this a glitch?'. "
    "Given the scene description and the Scene Analyst's causal findings, "
    "assess the safety risk level, identify which AV capabilities are "
    "stressed, and capture the Operational Design Domain (ODD) conditions "
    "under which the risk manifests. Use the safety taxonomy lookup tool to "
    "ground your assessment in the formal capability taxonomy (call it at "
    "least once with keywords drawn from the actual event). "
    "When finished, produce a JSON with keys: "
    "`risk_level` (one of: critical, high, medium, low - reflect the "
    "plausible worst-case AV outcome in this scenario), "
    "`affected_capability` (string: primary capability tag from the "
    "taxonomy, e.g. `pedestrian_detection`, `occluded_object_handling`), "
    "`affected_odds` (list of strings: relevant ODD conditions such as "
    "weather, lighting, road type, traffic density, speed regime)."
)

COVERAGE_ANALYST_SYSTEM_PROMPT = (
    "You are a regression-suite coverage analyst for an autonomous driving "
    "test program. "
    "IMPORTANT FRAMING: the scenario under review is a REAL real-world "
    "driving edge case, not synthetic or suspicious data. Your scepticism is "
    "directed only at whether this scenario is genuinely NEW for the test "
    "suite - never at whether the video is authentic. "
    "Your job is to stress-test the proposal to add this scenario: determine "
    "whether existing regression tests already exercise the same AV "
    "capability under similar ODD conditions, and whether the marginal "
    "coverage gain justifies the cost. Use the suite_similarity tool with a "
    "concise description of the real-world event to get quantitative overlap "
    "scores against the current suite. "
    "Play devil's advocate: produce strong counterarguments for why this "
    "scenario might NOT warrant addition (e.g. 'a near-duplicate already "
    "exists', 'the event sequence is too specific to be a useful test case', "
    "'the capability is already stressed by entry N'). Be rigorous and "
    "skeptical, but ALWAYS about coverage - not about data authenticity. "
    "When finished, produce a JSON with keys: "
    "`counterarguments` (list of strings: reasons this might not need "
    "adding), "
    "`rebuttal_summary` (string: honest assessment of whether the "
    "counterarguments hold up or not, and on balance whether the scenario "
    "fills a real coverage gap). "
    "When you are done, you MUST use Action: finish — never Action: None."
)

SYNTHESIS_ARBITER_SYSTEM_PROMPT = (
    "You are the final synthesis arbiter for an autonomous-driving "
    "regression-suite triage. "
    "IMPORTANT FRAMING: the scene is a REAL real-world recording surfaced as "
    "a candidate valuable edge case for autonomous vehicle regression "
    "testing. Your decision is about test value, not data validity. "
    "Do not treat sudden debris, abrupt maneuvers, or hard-to-parse motion as "
    "evidence of fake footage - treat them as real events to triage into tests. "
    "You have received analyses from three specialists: a Scene Analyst "
    "(what actually happened in the world and why it is an AV edge case), a "
    "Risk Assessor (which capabilities are stressed and how badly), and a "
    "Coverage Analyst (skeptical counterarguments against adding it to the "
    "suite). Weigh all three contributions. "
    "Your `recommended_test_spec` must describe a concrete regression test "
    "an AV engineer could implement in simulation or on a replay rig - "
    "reference the real-world event, the stressed capability, the ODD, and "
    "the success criteria (e.g. 'Ego on highway at 30 m/s; a vehicle "
    "intrudes laterally into the ego lane from the adjacent lane following "
    "impact with a static obstacle; SUCCESS = ego decelerates and performs "
    "a safe lateral avoidance while maintaining lane discipline'). "
    "`scenario_variants` should list perturbations that increase coverage "
    "breadth (different weather, lighting, ego speed, intrusion angle, etc.). "
    "When finished, produce a JSON with keys: "
    "`decision` (one of: add_to_suite, monitor, dismiss), "
    "`recommended_test_spec` (string: concrete, actionable regression test "
    "specification), "
    "`scenario_variants` (list of strings: perturbation variants to also "
    "test), "
    "`confidence` (float 0-1: how certain you are in the decision given the "
    "evidence quality — lower if specialists disagreed or tools failed), "
    "`uncertainty_factors` (list of strings: what a human reviewer should "
    "verify)."
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
        "The following is a REAL recorded driving clip from a data-collection "
        "fleet. It was surfaced as a candidate valuable edge case for "
        "autonomous vehicle regression testing (rare real-world situation an "
        "AV might mishandle) - not because the data is suspicious.\n\n"
        f"Window ID: {context.window_id}\n"
        "Severity hint (upstream statistical score only, not a verdict on "
        f"authenticity): {context.severity_hint}\n\n"
        f"Scene description (from prior VLM pass):\n{context.scene_description}\n\n"
        "Regression-value rationale (from prior VLM pass; why this clip may "
        f"matter for AV testing):\n{context.anomaly_rationale}\n\n"
        "Your task: reconstruct what actually happened in the world and why "
        "an AV would find this hard. You may use vlm_followup at most twice "
        "for critical ambiguities, then finish with the required JSON. Do NOT "
        "question whether the footage is real, and do NOT attribute events "
        "to glitches or unrealistic physics."
    )


def _build_risk_assessor_message(
    context: DebateContext,
    scene_output: dict[str, Any],
) -> str:
    return (
        f"Scene description:\n{context.scene_description}\n\n"
        f"Regression-value rationale (prior stage):\n{context.anomaly_rationale}\n\n"
        f"Severity hint (upstream statistical score only): {context.severity_hint}\n\n"
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
        f"Severity hint (upstream statistical score only): {context.severity_hint}\n\n"
        f"Scene description:\n{context.scene_description}\n\n"
        f"Regression-value rationale (prior stage):\n{context.anomaly_rationale}\n\n"
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

_SCENE_OUTPUT_KEYS = ("failure_mode", "why_valuable_for_regression", "evidence_summary")
_RISK_OUTPUT_KEYS = ("risk_level", "affected_capability")
_COVERAGE_OUTPUT_KEYS = ("counterarguments", "rebuttal_summary")


def _actor_llm_failed(contribution: ActorContribution) -> bool:
    for step in contribution.steps:
        if step.thought and "llm_call raised exception" in step.thought:
            return True
    return False


def _merge_actor_output(
    contribution: ActorContribution,
    required_keys: tuple[str, ...],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if _actor_llm_failed(contribution):
        return dict(fallback)

    output = dict(contribution.final_output or {})
    if not all(key in output and output[key] not in (None, "", []) for key in required_keys):
        merged = dict(fallback)
        merged.update({key: value for key, value in output.items() if value not in (None, "")})
        return merged
    return output


def _fallback_scene_output(record: DebateInputRecord) -> dict[str, Any]:
    return {
        "failure_mode": (record.anomaly_rationale or record.scene_description)[:600],
        "why_valuable_for_regression": (record.anomaly_rationale or "Edge case for AV regression.")[
            :600
        ],
        "evidence_summary": (record.scene_description or "See prior scene description.")[:600],
    }


def _fallback_risk_output(
    context: DebateContext,
    scene_output: dict[str, Any],
) -> dict[str, Any]:
    from pipeline.actor_tools import safety_taxonomy_lookup

    query = str(
        scene_output.get("failure_mode") or context.scene_description or context.anomaly_rationale
    )[:240]
    taxonomy_text = safety_taxonomy_lookup({"query": query}, context)
    capability = "unspecified_capability"
    for line in taxonomy_text.splitlines():
        if "capability_tag=" in line:
            capability = line.split("capability_tag=", 1)[1].split()[0].strip()
            break

    severity = str(context.severity_hint).strip().lower()
    risk_level = {
        "critical": "high",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "unknown": "medium",
    }.get(severity, "medium")

    return {
        "risk_level": risk_level,
        "affected_capability": capability,
        "affected_odds": [],
    }


def _fallback_coverage_output() -> dict[str, Any]:
    return {
        "counterarguments": [
            "Coverage analysis did not complete; novelty versus the existing suite is uncertain."
        ],
        "rebuttal_summary": (
            "Automated fallback: treat marginal coverage gain as unverified until "
            "a human reviewer confirms suite overlap."
        ),
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

    scene_output = _merge_actor_output(
        scene_contribution,
        _SCENE_OUTPUT_KEYS,
        _fallback_scene_output(record),
    )
    risk_output = _merge_actor_output(
        risk_contribution,
        _RISK_OUTPUT_KEYS,
        _fallback_risk_output(context, scene_output),
    )
    coverage_output = _merge_actor_output(
        coverage_contribution,
        _COVERAGE_OUTPUT_KEYS,
        _fallback_coverage_output(),
    )
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
    arbiter_confidence = _coerce_confidence(arbiter_output.get("confidence"))
    risk_level = str(risk_output.get("risk_level", "")).strip().lower() or "medium"

    tool_evidence = summarize_tool_evidence(all_contributions)
    max_suite_overlap = extract_max_suite_overlap([coverage_contribution])
    had_vlm_failures = scene_analyst_had_vlm_failures(scene_contribution)

    proposal_confidence = apply_evidence_penalty(
        arbiter_confidence,
        tool_failure_ratio=float(tool_evidence["tool_failure_ratio"]),
        had_vlm_failures=had_vlm_failures,
    )
    priority_score = compute_priority_score(
        proposal_decision=proposal_decision,
        risk_level=risk_level,
        severity_hint=str(record.severity_hint),
        max_suite_overlap=max_suite_overlap,
        tool_failure_ratio=float(tool_evidence["tool_failure_ratio"]),
    )

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
        "proposal_why_anomalous": (
            scene_output.get("why_valuable_for_regression")
            or scene_output.get("why_anomalous", "")
        ),
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
        "proposal_confidence": proposal_confidence,
        "proposal_uncertainty_factors": arbiter_output.get(
            "uncertainty_factors", []
        ),
        "scoring": {
            "arbiter_confidence_raw": arbiter_confidence,
            "proposal_confidence": proposal_confidence,
            "priority_score": priority_score,
            "risk_level": risk_level,
            "severity_hint": str(record.severity_hint),
            "max_suite_overlap": max_suite_overlap,
            "had_vlm_failures": had_vlm_failures,
            **tool_evidence,
        },
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