"""Module 9: Debate — configuration, client protocols, and error types.

The debate arm drives a multi-agent loop with two model dependencies:
  * a text LLM for the debate turns (Scene Analyst / Risk / Coverage / Arbiter), and
  * a VLM for scene description + the Scene Analyst's video follow-ups.
Both are injected Protocols (NIM impls for production, Stubs for offline/tests),
mirroring the Scorer's TextClient and the Encoder's VLMClient. The debate engine
must call only these — never a hardcoded NIM endpoint (lego rule).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Client protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class TextLLMClient(Protocol):
    """Chat LLM for debate turns. `messages` is OpenAI-style [{role, content}]."""
    model_id: str

    def complete(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        """Return the assistant's text response."""
        ...


@runtime_checkable
class VLMClient(Protocol):
    """Video describer for the description stage + Scene Analyst follow-ups."""
    model_id: str

    def describe(self, video_ref: str, anomaly_priors: dict) -> str:
        """Return JSON: {scene_description, anomaly_rationale, confidence}."""
        ...

    def followup(self, video_ref: str, prompt: str) -> str:
        """Return free-text answer to a targeted question about the clip."""
        ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    """Debate hyperparameters. Defaults mirror the project .env (DEBATE_*)."""
    debate_rounds: int = 2          # Risk<->Coverage rebuttal rounds
    max_actor_steps: int = 5        # ReAct steps per actor
    temperature: float = 0.2
    top_k: int = 1                  # number of top-ranked anomalies to debate

    @classmethod
    def from_env(cls) -> "DebateConfig":
        def _i(name: str, d: int) -> int:
            try:
                return int(os.environ.get(name, d))
            except (TypeError, ValueError):
                return d
        def _f(name: str, d: float) -> float:
            try:
                return float(os.environ.get(name, d))
            except (TypeError, ValueError):
                return d
        return cls(
            debate_rounds=_i("DEBATE_ROUNDS", 2),
            max_actor_steps=_i("DEBATE_MAX_ACTOR_STEPS", 5),
            temperature=_f("DEBATE_TEMPERATURE", 0.2),
            top_k=_i("DEBATE_TOP_K", 1),
        )


# ---------------------------------------------------------------------------
# Stubs (offline / tests — no NIM/GPU)
# ---------------------------------------------------------------------------

class StubTextLLMClient:
    """Deterministic debate-turn LLM. Returns well-formed actor JSON/text."""
    model_id: str = "stub/debate-llm"

    def complete(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        # A final-answer JSON the arbiter/actors can parse; no tool calls.
        return (
            'Thought: the scenario is rare and safety-relevant.\n'
            'Final Answer: {"decision": "add_to_suite", '
            '"recommendation": "add_immediately", "risk_level": "high", '
            '"affected_capability": "VRU detection", "failure_mode": "late detection", '
            '"why_anomalous": "stub", "evidence_summary": "stub", '
            '"recommended_test_spec": "stub spec", "confidence": 0.8, '
            '"scenario_variants": ["v1"], "counterarguments": [], '
            '"rebuttal_summary": ""}'
        )


class StubVLMClient:
    """Deterministic describer for offline runs/tests."""
    model_id: str = "stub/debate-vlm"

    def describe(self, video_ref: str, anomaly_priors: dict) -> str:
        return (
            '{"scene_description": "a stub scene", '
            '"anomaly_rationale": "stub rationale", "confidence": "high"}'
        )

    def followup(self, video_ref: str, prompt: str) -> str:
        return "Stub follow-up observation."


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DebateError(Exception):
    """Base error for the debate module."""


class DebateModelUnavailableError(DebateError):
    """A debate model endpoint (text LLM or VLM) could not be reached."""
