"""
Prompt templates for failure capsule generation.
"""

import json
from datetime import datetime
from typing import Any

CAPSULE_SYSTEM_PROMPT = """
You are an autonomy debugging triage assistant.
Return only valid JSON with this exact schema:
{
  "summary": "string",
  "scenario_type": "string or null",
  "failure_mode_hints": ["string", "..."],
  "likely_subsystem": "string or null",
  "severity_cue": "critical | high | medium | low | unknown",
  "tags": ["string", "..."]
}

Rules:
- Do not include markdown or prose outside JSON.
- Ground output in the provided ticket text only.
- If evidence is weak, use "unknown" severity.
- Use empty arrays instead of invented values.
""".strip()


def build_capsule_user_prompt(
    title: str,
    raw_text: str,
    metadata: dict[str, Any],
) -> str:
    '''
    Purpose: Build a deterministic user prompt payload for capsule-generation LLM calls.
    Parameters:
    title (str): Failure ticket title used for context.
    raw_text (str): Raw ticket narrative and evidence details from ingestion.
    metadata (dict[str, Any]): Structured metadata such as source, timestamp, and IDs.
    Returns:
    str: Formatted prompt string containing all ticket context for the model.
    Called by: backend/services/llm_client.py -> generate_triage_summary()
    Calls: json.dumps()
    '''
    prompt_payload = {
        "title": title,
        "raw_text": raw_text,
        "metadata": {
            "source_type": metadata.get("source_type"),
            "source_ref": metadata.get("source_ref"),
            "event_timestamp": (
                metadata.get("event_timestamp").isoformat()
                if isinstance(metadata.get("event_timestamp"), datetime)
                else metadata.get("event_timestamp")
            ),
            "agent_id": metadata.get("agent_id"),
            "scenario_id": metadata.get("scenario_id"),
            "artifacts_ref": metadata.get("artifacts_ref"),
        },
    }
    return (
        "Analyze this failure ticket and produce a normalized capsule JSON.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
    )
