"""Generic ReAct loop for tool-using debate actors.

This module is intentionally actor-agnostic: it parses the LLM's
Thought/Action/Action Input responses, dispatches tool calls through
:func:`pipeline.actor_tools.execute_tool`, and returns a structured
:class:`ActorContribution` describing what the actor thought, did, and
concluded.

Actor prompts, tool whitelists, and user-message construction live in
:mod:`pipeline.debate_actors` so this loop can stay reusable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from waymo_pipeline.actor_tools import (
    DebateContext,
    TOOL_DESCRIPTIONS,
    execute_tool,
)


# ---------------------------------------------------------------------------
# Structured step / contribution records
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation emitted by an actor."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ActorStep(BaseModel):
    """One Thought/Action/Observation cycle inside an actor's ReAct loop."""

    thought: str = ""
    tool_call: ToolCall | None = None
    observation: str | None = None


class ActorContribution(BaseModel):
    """Full record of one actor's ReAct execution."""

    actor_name: str
    actor_role: str = ""
    steps: list[ActorStep] = Field(default_factory=list)
    final_output: dict[str, Any] = Field(default_factory=dict)
    raw_llm_output: str = ""


# ---------------------------------------------------------------------------
# ReAct format plumbing
# ---------------------------------------------------------------------------


_REACT_INSTRUCTION_TEMPLATE = """
You have access to the following tools:
{tool_block}

On each turn, respond in EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: <tool_name OR "finish">
Action Input: <JSON object - tool input dict OR your final structured output>

When you have enough information, use Action: finish and put your final answer as a JSON object in Action Input.

Action: finish is ALWAYS valid (even when you also have other tools). Do not put markdown separators or extra text after the action name.

Never use Action: None, Action: null, or a blank action. When you are done, you MUST use Action: finish with a non-empty JSON object.

Do NOT include any text outside of this Thought/Action/Action Input format.
""".strip()


_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?=\nAction Input:)", re.DOTALL)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL)

_INVALID_ACTION_NAMES = frozenset({"none", "null", "n/a", "na", ""})

# Action Input keys that belong to tool calls, not actor final answers.
_TOOL_INPUT_ONLY_KEYS = frozenset({"question", "query", "scenario_description"})


def _normalize_action_name(action: str) -> str:
    """Return the canonical action token (e.g. ``finish``, ``vlm_followup``).

    Small models often append markdown or transcript separators after the
    action name (``finish ---``, ``finish | Action Input:``). Strip those so
    ``finish`` is recognized even when the raw ``Action:`` line is messy.
    """

    text = (action or "").strip()
    if not text:
        return ""
    # Keep only the first alphanumeric/underscore token.
    match = re.match(r"([A-Za-z_][\w-]*)", text)
    if match:
        return match.group(1).lower().replace("-", "_")
    return text.split()[0].lower()


def _payload_has_content(payload: dict[str, Any]) -> bool:
    """True when ``payload`` contains at least one non-empty value."""

    for value in payload.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return True
    return False


def _payload_matches_required_keys(
    payload: dict[str, Any],
    required_keys: tuple[str, ...],
) -> bool:
    if not required_keys:
        return _payload_has_content(payload)
    return all(
        key in payload and payload[key] not in (None, "", [])
        for key in required_keys
    )


def _looks_like_tool_input(payload: dict[str, Any]) -> bool:
    """Heuristic: reject salvage of ``{question: ...}``-style tool payloads."""

    if not payload:
        return False
    keys = set(payload.keys())
    if keys and keys <= _TOOL_INPUT_ONLY_KEYS:
        return True
    if len(keys) == 1 and "question" in keys:
        return True
    return False


def _build_tool_block(available_tools: list[str]) -> str:
    """Render the tool-description bullet list for the system prompt."""

    lines: list[str] = []
    for name in available_tools:
        description = TOOL_DESCRIPTIONS.get(name, f"(no description for {name})")
        lines.append(f"- {name}: {description}")
    if not lines:
        lines.append("- (no tools available; go straight to Action: finish)")
    return "\n".join(lines)


def _construct_system_prompt(system_prompt: str, available_tools: list[str]) -> str:
    """Append the ReAct instruction block to the caller-provided system prompt."""

    instruction = _REACT_INSTRUCTION_TEMPLATE.format(
        tool_block=_build_tool_block(available_tools)
    )
    return f"{system_prompt.strip()}\n\n{instruction}"


def _parse_react_response(raw: str) -> tuple[str | None, str | None, str | None]:
    """Extract (thought, action, action_input_raw) from the LLM response.

    Returns ``(None, None, None)`` if the response does not contain both an
    ``Action:`` and an ``Action Input:`` marker, since without those the loop
    cannot decide what to do next.
    """

    if not raw or "Action:" not in raw or "Action Input:" not in raw:
        return None, None, None

    thought_match = _THOUGHT_RE.search(raw)
    action_match = _ACTION_RE.search(raw)
    action_input_match = _ACTION_INPUT_RE.search(raw)

    if action_match is None or action_input_match is None:
        return None, None, None

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip()
    action_input = action_input_match.group(1).strip()
    return thought, action, action_input


def _parse_action_input_json(action_input_raw: str) -> tuple[dict[str, Any] | None, str]:
    """Best-effort JSON parse of an Action Input payload.

    Returns ``(parsed_dict_or_None, error_message_or_empty_string)``. Strips
    fenced-code wrappers so the LLM is free to wrap the JSON if it likes.
    """

    if action_input_raw is None:
        return None, "empty Action Input"

    text = action_input_raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            return None, f"Action Input was not valid JSON: {error}"
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as nested_error:
            return None, f"Action Input was not valid JSON: {nested_error}"

    if not isinstance(parsed, dict):
        return None, "Action Input must be a JSON object."
    return parsed, ""


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _salvage_final_output(
    steps: list[ActorStep],
    last_raw_output: str,
    required_output_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Best-effort structured output when max steps hit without a valid finish."""

    for step in reversed(steps):
        if step.tool_call is None or step.tool_call.tool_name != "finish":
            continue
        payload = step.tool_call.tool_input
        if not isinstance(payload, dict) or not _payload_has_content(payload):
            continue
        if _looks_like_tool_input(payload):
            continue
        if _payload_matches_required_keys(payload, required_output_keys):
            return dict(payload)

    salvage_candidate = last_raw_output
    _thought, action, action_input_raw = _parse_react_response(last_raw_output)
    if action_input_raw:
        salvage_candidate = action_input_raw
    extracted = _extract_first_json_object(salvage_candidate)
    if extracted is None:
        return {}
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    if _looks_like_tool_input(parsed):
        return {}
    if required_output_keys and not _payload_matches_required_keys(
        parsed, required_output_keys
    ):
        return {}
    return parsed


def run_actor(
    actor_name: str,
    system_prompt: str,
    available_tools: list[str],
    user_message: str,
    context: DebateContext,
    max_steps: int,
    llm_call: Callable[[list[dict[str, Any]]], str],
    required_output_keys: tuple[str, ...] = (),
) -> ActorContribution:
    """Run an actor's ReAct loop and return the structured contribution.

    The loop alternates LLM calls and tool invocations until the LLM emits
    ``Action: finish`` or ``max_steps`` is reached. Each LLM turn counts as
    one step regardless of whether a tool was successfully invoked.
    """

    max_steps = max(1, int(max_steps))

    constructed_system_prompt = _construct_system_prompt(system_prompt, available_tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": constructed_system_prompt},
        {"role": "user", "content": user_message},
    ]

    steps: list[ActorStep] = []
    raw_chunks: list[str] = []
    final_output: dict[str, Any] = {}
    last_raw_output = ""

    for _step_index in range(max_steps):
        raw_output = ""
        llm_error: Exception | None = None
        for llm_attempt in range(2):
            try:
                raw_output = llm_call(messages)
                llm_error = None
                break
            except Exception as error:  # noqa: BLE001
                llm_error = error
        if llm_error is not None:
            steps.append(
                ActorStep(
                    thought=f"[llm_call raised exception: {llm_error}]",
                    tool_call=None,
                    observation=None,
                )
            )
            break

        raw_output = (raw_output or "").strip()
        last_raw_output = raw_output
        raw_chunks.append(raw_output)

        thought, action, action_input_raw = _parse_react_response(raw_output)

        if action is None:
            steps.append(
                ActorStep(
                    thought=raw_output,
                    tool_call=None,
                    observation=None,
                )
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please respond in the required Thought/Action/Action "
                        "Input format."
                    ),
                }
            )
            continue

        action_normalized = _normalize_action_name(action)
        if action_normalized in _INVALID_ACTION_NAMES:
            observation = (
                "Invalid action. When you are done, use Action: finish with your "
                f"final JSON object. Otherwise choose one of: {available_tools}"
            )
            steps.append(
                ActorStep(
                    thought=thought or "",
                    tool_call=None,
                    observation=observation,
                )
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue with your next Thought.",
                }
            )
            continue

        if action_normalized == "finish":
            parsed_input, parse_error = _parse_action_input_json(action_input_raw or "")
            if parsed_input is None:
                steps.append(
                    ActorStep(
                        thought=thought or "",
                        tool_call=ToolCall(tool_name="finish", tool_input={}),
                        observation=f"Invalid finish payload: {parse_error}",
                    )
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your finish payload was invalid: {parse_error}. "
                            "Return Action: finish with a valid JSON object as "
                            "Action Input."
                        ),
                    }
                )
                continue

            if not _payload_has_content(parsed_input):
                keys_hint = (
                    f" Required keys: {list(required_output_keys)}."
                    if required_output_keys
                    else ""
                )
                observation = (
                    "Finish payload must be a non-empty JSON object with your "
                    f"final answer.{keys_hint}"
                )
                steps.append(
                    ActorStep(
                        thought=thought or "",
                        tool_call=ToolCall(tool_name="finish", tool_input={}),
                        observation=observation,
                    )
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Observation: {observation}\n\n"
                            "Return Action: finish with the complete JSON object "
                            "in Action Input (not an empty {{}})."
                        ),
                    }
                )
                continue

            if required_output_keys and not _payload_matches_required_keys(
                parsed_input, required_output_keys
            ):
                missing = [
                    key
                    for key in required_output_keys
                    if key not in parsed_input
                    or parsed_input[key] in (None, "", [])
                ]
                observation = (
                    "Finish payload is missing required keys or has empty values: "
                    f"{missing}. Required: {list(required_output_keys)}."
                )
                steps.append(
                    ActorStep(
                        thought=thought or "",
                        tool_call=ToolCall(tool_name="finish", tool_input=parsed_input),
                        observation=observation,
                    )
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Observation: {observation}\n\n"
                            "Return Action: finish with the full JSON object."
                        ),
                    }
                )
                continue

            final_output = parsed_input
            steps.append(
                ActorStep(
                    thought=thought or "",
                    tool_call=ToolCall(tool_name="finish", tool_input=parsed_input),
                    observation=None,
                )
            )
            break

        if action_normalized not in {_normalize_action_name(t) for t in available_tools}:
            tool_list = list(available_tools)
            if "finish" not in tool_list:
                tool_list = [*tool_list, "finish"]
            observation = (
                f"Unknown tool: {action}. Available tools: {tool_list}"
            )
            steps.append(
                ActorStep(
                    thought=thought or "",
                    tool_call=ToolCall(tool_name=action, tool_input={}),
                    observation=observation,
                )
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue with your next Thought.",
                }
            )
            continue

        tool_name = action_normalized
        parsed_input, parse_error = _parse_action_input_json(action_input_raw or "")
        if parsed_input is None:
            observation = (
                f"Could not parse Action Input for tool '{tool_name}': {parse_error}"
            )
            steps.append(
                ActorStep(
                    thought=thought or "",
                    tool_call=ToolCall(tool_name=tool_name, tool_input={}),
                    observation=observation,
                )
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue with your next Thought.",
                }
            )
            continue

        observation = execute_tool(tool_name, parsed_input, context)
        steps.append(
            ActorStep(
                thought=thought or "",
                tool_call=ToolCall(tool_name=tool_name, tool_input=parsed_input),
                observation=observation,
            )
        )
        messages.append({"role": "assistant", "content": raw_output})
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue with your next Thought.",
            }
        )

    if not final_output and last_raw_output:
        final_output = _salvage_final_output(
            steps, last_raw_output, required_output_keys
        )

    return ActorContribution(
        actor_name=actor_name,
        actor_role=system_prompt.strip()[:100],
        steps=steps,
        final_output=final_output,
        raw_llm_output="\n\n---\n\n".join(raw_chunks),
    )
