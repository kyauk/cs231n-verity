"""Tool definitions and dispatch for ReAct-style debate actors.

This module provides the small set of tools that debate actors can invoke
during their ReAct loops, plus a shared :class:`DebateContext` carrying
scenario state and a :func:`execute_tool` dispatcher.

Nothing here touches the existing ``cosmos_multi_agent_debate`` code path;
these primitives are consumed only by the new :mod:`pipeline.debate_actors`
orchestrator (Step 1 of the tool-using ReAct rework).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared state passed to tools
# ---------------------------------------------------------------------------


@dataclass
class DebateContext:
    """Shared state for a single window's debate.

    Attributes:
        media_refs: Absolute/relative paths to the video and/or image files
            that describe the flagged window.
        scene_description: Scene summary from the description stage.
        anomaly_rationale: Regression-value rationale from the description stage
            (API field name; content describes edge-case value for AV testing).
        severity_hint: ``low|medium|high|critical`` coarse hint from upstream
            statistical scoring (not a judgment of recording authenticity).
        regression_suite: Existing regression-suite scenario strings.
        window_id: Identifier for the flagged window (used for logging).
        video_fps: Sampling FPS for video frames when querying the VLM.
    """

    media_refs: list[str] = field(default_factory=list)
    scene_description: str = ""
    anomaly_rationale: str = ""
    severity_hint: str = "unknown"
    regression_suite: list[str] = field(default_factory=list)
    window_id: str = ""
    video_fps: float = field(
        default_factory=lambda: float(os.getenv("COSMOS_HF_VIDEO_FPS", "8"))
    )


# ---------------------------------------------------------------------------
# Safety taxonomy used by the Risk Assessor
# ---------------------------------------------------------------------------


SAFETY_TAXONOMY: dict[str, dict[str, Any]] = {
    "pedestrian_detection": {
        "description": (
            "Detection, tracking, and intent prediction of pedestrians in and near "
            "the drivable region, including partially occluded, jaywalking, and "
            "child pedestrians."
        ),
        "risk_factors": [
            "low light or glare",
            "partial occlusion by parked vehicles",
            "unusual poses (kneeling, running, crouching)",
            "jaywalking outside marked crosswalks",
        ],
        "typical_failure_modes": [
            "missed detection of pedestrian stepping off curb",
            "late intent prediction for jaywalker",
            "confusing pedestrian with static object after occlusion",
        ],
        "severity_guidance": (
            "Critical when ego speed combined with distance gives <2s time-to-collision; "
            "high when pedestrian enters lane within 3s; medium when on shoulder."
        ),
    },
    "vehicle_tracking": {
        "description": (
            "Persistent tracking and state estimation of surrounding vehicles, "
            "including heavy vehicles, motorcycles, and vehicles undergoing "
            "aggressive maneuvers."
        ),
        "risk_factors": [
            "track ID switches through occlusion",
            "cut-ins from adjacent lanes",
            "erratic braking by lead vehicle",
            "motorcycles filtering between lanes",
        ],
        "typical_failure_modes": [
            "dropped track during brief occlusion",
            "velocity estimate lag on sudden deceleration",
            "identity swap between two similar vehicles",
        ],
        "severity_guidance": (
            "Critical on lead-vehicle hard-brake misses; high on cut-in misdetection; "
            "medium on identity swaps that do not affect planning."
        ),
    },
    "traffic_signal_compliance": {
        "description": (
            "Recognition of traffic lights, stop signs, yield signs, and associated "
            "right-of-way reasoning at intersections and controlled crossings."
        ),
        "risk_factors": [
            "sun glare on signal heads",
            "signal head partially occluded by foliage",
            "flashing or malfunctioning signals",
            "ambiguous overhead gantries",
        ],
        "typical_failure_modes": [
            "missing red light detection",
            "misreading protected-vs-permitted left-turn signal",
            "ignoring stop sign at unmarked intersection",
        ],
        "severity_guidance": (
            "Critical when ego proceeds through red; high when late detection "
            "triggers hard brake; medium when behavior is conservative-but-legal."
        ),
    },
    "occluded_object_handling": {
        "description": (
            "Reasoning about partially or fully occluded road users and static "
            "objects, including dooring risk, emerging pedestrians from behind "
            "buses, and debris in lane."
        ),
        "risk_factors": [
            "dense parked-vehicle rows",
            "large trucks blocking adjacent-lane visibility",
            "construction barrels along curb",
            "children behind parked cars",
        ],
        "typical_failure_modes": [
            "no hazard-margin for potential pedestrian emerging from occlusion",
            "sudden-appearance detection with no reaction budget",
            "debris classified as drivable surface",
        ],
        "severity_guidance": (
            "Critical when ego lacks reaction time on reveal; high when soft braking "
            "required; medium when only lateral nudge is needed."
        ),
    },
    "adverse_weather_perception": {
        "description": (
            "Perception robustness in rain, snow, fog, spray, and low-light "
            "conditions, including sensor degradation and false positives from "
            "precipitation."
        ),
        "risk_factors": [
            "heavy rain occluding camera",
            "snow accumulation on sensors",
            "dense fog reducing effective range",
            "tire spray from lead vehicle",
        ],
        "typical_failure_modes": [
            "range collapse in fog without conservative planner fallback",
            "rain-drop artifacts treated as obstacles",
            "lane-line detection loss under snow cover",
        ],
        "severity_guidance": (
            "Critical when planner does not downgrade ODD; high on sustained "
            "detection loss; medium on transient sensor noise."
        ),
    },
    "lane_keeping": {
        "description": (
            "Lane-line detection, lane-assignment estimation, and stable lateral "
            "control on curves, in construction zones, and over worn markings."
        ),
        "risk_factors": [
            "faded or missing lane markings",
            "temporary construction striping overlapping old paint",
            "sharp curves with limited preview",
            "lane merges and splits",
        ],
        "typical_failure_modes": [
            "centering onto wrong lane in merge",
            "oscillation near faded paint",
            "following old markings through construction zone",
        ],
        "severity_guidance": (
            "Critical on wrong-way lane pickup; high on lateral drift toward adjacent "
            "lane; medium on minor centering oscillation."
        ),
    },
    "intersection_navigation": {
        "description": (
            "Behavior at intersections including unprotected lefts, four-way stops, "
            "roundabouts, and interactions with cross-traffic."
        ),
        "risk_factors": [
            "unprotected left across fast oncoming traffic",
            "multi-way stops with ambiguous priority",
            "yield reasoning in roundabouts",
            "cross-traffic running red",
        ],
        "typical_failure_modes": [
            "aggressive gap acceptance on unprotected left",
            "stalling indefinitely at four-way stop",
            "failing to yield to roundabout occupant",
        ],
        "severity_guidance": (
            "Critical on gap-acceptance errors with closing traffic; high on "
            "right-of-way reversals; medium on excessive conservatism."
        ),
    },
    "emergency_vehicle_response": {
        "description": (
            "Recognition of emergency vehicles via lights/sirens/visual cues and "
            "lawful yielding behavior."
        ),
        "risk_factors": [
            "siren audible before visual detection",
            "approach from behind in dense traffic",
            "unconventional emergency vehicles (e.g., unmarked)",
        ],
        "typical_failure_modes": [
            "no pull-over response to approaching ambulance",
            "blocking intersection during emergency pass-through",
            "delayed reaction in multi-lane merge",
        ],
        "severity_guidance": (
            "Critical when failure blocks an active emergency response; high on "
            "delayed yield; medium on ambiguous cue handling."
        ),
    },
    "construction_zone_handling": {
        "description": (
            "Navigation of active construction zones including cones, barrels, "
            "flaggers, temporary signage, and altered lane geometry."
        ),
        "risk_factors": [
            "flagger hand-signal interpretation",
            "cones that shift lane boundaries",
            "temporary reduced speed limits",
            "workers within the drivable area",
        ],
        "typical_failure_modes": [
            "ignoring flagger direction",
            "clipping cone line on lane shift",
            "maintaining posted speed instead of temporary limit",
        ],
        "severity_guidance": (
            "Critical on worker-proximity errors; high on cone violations; medium on "
            "speed-limit overshoot without other risk."
        ),
    },
    "cyclist_interaction": {
        "description": (
            "Detection, tracking, and safe clearance of cyclists, including "
            "those sharing the lane, in bike lanes, and performing unsignaled "
            "maneuvers."
        ),
        "risk_factors": [
            "unsignaled cyclist lane changes",
            "dooring hazards from parked cars",
            "cyclists in driver blind spot during right turns",
            "shared narrow lanes",
        ],
        "typical_failure_modes": [
            "unsafe passing clearance",
            "right-hook turn into cyclist",
            "late detection of cyclist entering from bike lane",
        ],
        "severity_guidance": (
            "Critical on right-hook or passing-clearance conflicts; high on late "
            "detection with braking required; medium on overly conservative holds."
        ),
    },
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def vlm_followup(tool_input: dict[str, Any], context: DebateContext) -> str:
    """Ask the Cosmos VLM a targeted follow-up question about the media.

    Rebuilds the media/video message structure used by the description stage
    and appends the actor's question as a final text block, then delegates to
    the shared ``_hf_chat_completion`` entry point.
    """

    question = str(tool_input.get("question", "")).strip()
    if not question:
        return "vlm_followup error: missing non-empty 'question' in tool input."

    if not context.media_refs:
        return (
            "vlm_followup error: no media attached to this window; cannot "
            "answer a visual question."
        )

    # Local import to avoid circular import at module load time.
    from pipeline.stage_describe_and_debate import _hf_chat_completion

    media_blocks: list[dict[str, Any]] = []
    for media_ref in context.media_refs:
        abs_path = os.path.abspath(media_ref)
        lowered = media_ref.lower()
        if lowered.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
            media_blocks.append(
                {"type": "video", "video": abs_path, "fps": context.video_fps}
            )
        elif lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
            media_blocks.append({"type": "image", "image": abs_path})

    if not media_blocks:
        return (
            "vlm_followup error: no video or image files in media_refs; "
            "supported extensions are mp4/mov/mkv/avi/webm/png/jpg/jpeg/webp."
        )

    media_blocks.append({"type": "text", "text": question})

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a vision-language assistant answering targeted "
                "follow-up questions about a REAL recorded autonomous-driving "
                "scene. The video is genuine fleet footage - do NOT speculate "
                "about whether it is fake, edited, or staged. Do NOT explain "
                "sudden motion, debris, or collisions as glitches, unrealistic "
                "physics, or rendering errors; infer real-world causes or say "
                "the precursor is not visible. Describe what you actually see: "
                "agents (vehicles, pedestrians, cyclists, infrastructure), "
                "their positions and trajectories, the likely cause of any "
                "incident, and environmental conditions. "
                "Answer concisely (<=80 words) grounded strictly in what is "
                "visible. If a specific detail is not visually evident, say "
                "so explicitly rather than guessing."
            ),
        },
        {"role": "user", "content": media_blocks},
    ]

    try:
        return _hf_chat_completion(messages=messages).strip()
    except Exception as error:  # noqa: BLE001
        return f"vlm_followup error: {error}"


def safety_taxonomy_lookup(
    tool_input: dict[str, Any],
    context: DebateContext,  # noqa: ARG001 (kept for uniform signature)
) -> str:
    """Return the top taxonomy entries matching the query keywords."""

    query = str(tool_input.get("query", "")).strip().lower()
    if not query:
        return "safety_taxonomy_lookup error: missing non-empty 'query' in tool input."

    query_tokens = {token for token in _tokenize(query) if token}
    if not query_tokens:
        return "safety_taxonomy_lookup error: query produced no usable tokens."

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for key, entry in SAFETY_TAXONOMY.items():
        haystack_tokens: set[str] = set()
        haystack_tokens.update(_tokenize(key))
        haystack_tokens.update(_tokenize(entry.get("description", "")))
        for failure_mode in entry.get("typical_failure_modes", []):
            haystack_tokens.update(_tokenize(str(failure_mode)))
        overlap = len(query_tokens & haystack_tokens)
        scored.append((overlap, key, entry))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top = [item for item in scored if item[0] > 0][:3]
    if not top:
        top = scored[:3]

    lines: list[str] = []
    for rank, (score, key, entry) in enumerate(top, start=1):
        risk_factors = "; ".join(entry.get("risk_factors", []))
        failure_modes = "; ".join(entry.get("typical_failure_modes", []))
        lines.append(
            f"{rank}. capability_tag={key} (keyword_overlap={score})\n"
            f"   description: {entry.get('description', '')}\n"
            f"   risk_factors: {risk_factors}\n"
            f"   typical_failure_modes: {failure_modes}\n"
            f"   severity_guidance: {entry.get('severity_guidance', '')}"
        )
    return "\n".join(lines)


def suite_similarity(tool_input: dict[str, Any], context: DebateContext) -> str:
    """Rank existing regression-suite entries by Jaccard token overlap."""

    scenario = str(tool_input.get("scenario_description", "")).strip()
    if not scenario:
        return (
            "suite_similarity error: missing non-empty 'scenario_description' "
            "in tool input."
        )

    suite = context.regression_suite
    if not suite:
        return (
            "suite_similarity: regression suite is empty; this scenario is "
            "trivially novel relative to current coverage (novelty=1.00)."
        )

    scenario_tokens = set(_tokenize(scenario))

    ranked: list[tuple[float, int, str]] = []
    for index, entry in enumerate(suite, start=1):
        entry_tokens = set(_tokenize(entry))
        if not scenario_tokens and not entry_tokens:
            jaccard = 0.0
        else:
            union = scenario_tokens | entry_tokens
            intersection = scenario_tokens & entry_tokens
            jaccard = (len(intersection) / len(union)) if union else 0.0
        ranked.append((jaccard, index, entry))

    ranked.sort(key=lambda item: (-item[0], item[1]))

    lines: list[str] = ["Ranked regression-suite overlap (jaccard):"]
    for jaccard, index, entry in ranked:
        snippet = entry if len(entry) <= 160 else entry[:157] + "..."
        lines.append(f"  entry {index}: jaccard={jaccard:.2f} :: {snippet}")

    top_score, top_index, _top_entry = ranked[0]
    if top_score >= 0.60:
        novelty_line = (
            f"Novelty assessment: high overlap of {top_score:.2f} with entry "
            f"{top_index}, suggesting significant redundancy."
        )
    elif top_score >= 0.35:
        novelty_line = (
            f"Novelty assessment: moderate overlap of {top_score:.2f} with entry "
            f"{top_index}; partial coverage, some novel aspects likely remain."
        )
    else:
        novelty_line = (
            f"Novelty assessment: highest overlap is {top_score:.2f} with entry "
            f"{top_index}, suggesting this scenario is largely novel."
        )
    lines.append(novelty_line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch + tool descriptions
# ---------------------------------------------------------------------------


_TOOL_FUNCTIONS = {
    "vlm_followup": vlm_followup,
    "safety_taxonomy_lookup": safety_taxonomy_lookup,
    "suite_similarity": suite_similarity,
}


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    context: DebateContext,
) -> str:
    """Dispatch a tool call by name.

    Returns a human-readable string observation. Unknown tools and malformed
    inputs are handled by returning an error string rather than raising, so the
    ReAct loop can surface the problem back to the model as an Observation.
    """

    tool = _TOOL_FUNCTIONS.get(tool_name)
    if tool is None:
        return f"Unknown tool: {tool_name}"

    if not isinstance(tool_input, dict):
        return (
            f"Tool '{tool_name}' expected a JSON object as Action Input, "
            f"got {type(tool_input).__name__}."
        )

    try:
        return tool(tool_input, context)
    except Exception as error:  # noqa: BLE001
        return f"Tool '{tool_name}' raised an exception: {error}"


TOOL_DESCRIPTIONS: dict[str, str] = {
    "vlm_followup": (
        "Ask a targeted visual question about the video/image. "
        'Input: {"question": "your question"}'
    ),
    "safety_taxonomy_lookup": (
        "Look up AV safety taxonomy entries. "
        'Input: {"query": "capability or failure keyword"}'
    ),
    "suite_similarity": (
        "Compare a scenario against existing regression suite. "
        'Input: {"scenario_description": "description text"}'
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-word tokenizer used for simple overlap scoring."""

    if not text:
        return []
    out: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            if current:
                out.append("".join(current))
                current = []
    if current:
        out.append("".join(current))
    return out
