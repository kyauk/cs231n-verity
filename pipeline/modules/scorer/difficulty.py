"""Module 4: Scorer — frontier difficulty arm.

Simulates what a deployed AV reasoning model would do with the composition.
Runs 3 calls with different constituent orderings. From the 3 responses,
computes difficulty signals:

  mean_confidence       : average of 3 confidence scores
  action_variance       : fraction of runs where action differs from the mode
  reasoning_mismatch    : fraction of runs where reasoning_consistent_with_action=False

frontier_difficulty_score = (
    action_variance     * signal_weights[0]   (default 0.5 — most reliable)
    + (1 - mean_confidence) * signal_weights[1]  (default 0.3)
    + reasoning_mismatch    * signal_weights[2]  (default 0.2 — noisiest)
)

If all 3 runs fail to parse: score=None, signals={} — logged, not raised.
This matches the failure mode: "proxy reasoner unavailable → use only
novelty + plausibility for ranking."

Seed for constituent orderings uses hashlib.sha256 (not built-in hash()) for
determinism across Python processes.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any

from pipeline.modules.scorer.plausibility import (
    _three_orderings,
    describe_composition,
)
from pipeline.modules.scorer.config import TextClient

_PROMPT_DIR = pathlib.Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_difficulty_json(text: str) -> dict[str, Any]:
    """Extract {"action": str, "confidence": float, "reasoning_consistent_with_action": bool}.

    Raises ValueError if no valid JSON with required fields is found.
    """
    candidates = [text.strip()]

    fence = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

    bare = re.search(r"```\s*([\s\S]*?)```", text, re.DOTALL)
    if bare:
        candidates.append(bare.group(1).strip())

    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "action" in obj and "confidence" in obj:
                confidence = float(obj["confidence"])
                confidence = max(0.0, min(1.0, confidence))
                return {
                    "action": str(obj["action"]),
                    "confidence": confidence,
                    "reasoning_consistent_with_action": bool(
                        obj.get("reasoning_consistent_with_action", True)
                    ),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    raise ValueError(
        f"No valid difficulty JSON found.\nResponse preview: {text[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_difficulty_signals(
    results: list[dict[str, Any]],
    signal_weights: tuple[float, float, float],
) -> tuple[float, dict[str, float]]:
    """Compute difficulty score and signals from 3 parsed difficulty responses.

    Parameters
    ----------
    results
        List of 1–3 parsed dicts with "action", "confidence",
        "reasoning_consistent_with_action".
    signal_weights
        (action_variance_w, inverse_confidence_w, reasoning_mismatch_w).

    Returns
    -------
    (frontier_difficulty_score, signals_dict)
    """
    confidences = [r["confidence"] for r in results]
    actions = [r["action"] for r in results]
    consistencies = [r["reasoning_consistent_with_action"] for r in results]

    mean_confidence = sum(confidences) / len(confidences)

    # Action variance: fraction of runs where action differs from the mode.
    mode_action = Counter(actions).most_common(1)[0][0]
    action_variance = sum(1 for a in actions if a != mode_action) / len(actions)

    reasoning_mismatch = sum(1 for c in consistencies if not c) / len(consistencies)

    w0, w1, w2 = signal_weights
    score = (
        action_variance       * w0
        + (1 - mean_confidence) * w1
        + reasoning_mismatch    * w2
    )
    score = max(0.0, min(1.0, score))

    signals: dict[str, float] = {
        "mean_confidence": mean_confidence,
        "action_variance": action_variance,
        "reasoning_action_mismatch": reasoning_mismatch,
    }
    return score, signals


# ---------------------------------------------------------------------------
# Difficulty arm
# ---------------------------------------------------------------------------

def _load_prompt_template(version: str) -> str:
    path = _PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"[Scorer/Difficulty] Prompt template {version!r} not found at {path}"
        )
    return path.read_text(encoding="utf-8")


class DifficultyArm:
    """Scores one CompositionProposal for frontier difficulty.

    Runs 3 calls with different constituent orderings.
    On partial parse failure, uses available results.
    On total parse failure, returns (None, {}) — not raised.
    """

    def __init__(
        self,
        client: TextClient,
        prompt_version: str = "v1_difficulty",
        signal_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self._client = client
        self._template = _load_prompt_template(prompt_version)
        self._signal_weights = signal_weights

        if "{{COMPOSITION}}" not in self._template:
            raise ValueError(
                f"[Scorer/Difficulty] Prompt template {prompt_version!r} is missing "
                f"{{{{COMPOSITION}}}} placeholder."
            )

    def score(
        self,
        composition_id: str,
        constituents: list[str],
        marginal_frequencies: dict[str, float],
        expected_joint: float,
        observed_joint: float,
    ) -> tuple[float | None, dict[str, float]]:
        """Run difficulty scoring. Returns (score | None, signals_dict).

        Returns (None, {}) if all 3 runs fail — caller treats as unavailable.
        """
        results: list[dict[str, Any]] = []

        for i in range(3):
            description = describe_composition(
                composition_id=composition_id,
                constituents=constituents,
                marginal_frequencies=marginal_frequencies,
                expected_joint=expected_joint,
                observed_joint=observed_joint,
                ordering_seed_offset=i,
            )
            prompt = self._template.replace("{{COMPOSITION}}", description)

            try:
                raw = self._client.complete(prompt)
                parsed = _extract_difficulty_json(raw)
                results.append(parsed)
            except Exception as exc:
                print(
                    f"[Scorer/Difficulty] Run {i+1}/3 failed for {composition_id!r}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        if not results:
            print(
                f"[Scorer/Difficulty] All runs failed for {composition_id!r}. "
                f"Setting frontier_difficulty_score=None.",
                file=sys.stderr,
            )
            return None, {}

        score, signals = compute_difficulty_signals(results, self._signal_weights)
        return score, signals


# ---------------------------------------------------------------------------
# Stub clients
# ---------------------------------------------------------------------------

class StubDifficultyClient:
    """Deterministic stub for tests and offline runs."""
    model_id: str = "stub/difficulty"

    def complete(self, prompt: str) -> str:
        return '{"action": "slow_down", "confidence": 0.55, "reasoning_consistent_with_action": true}'


class FailingDifficultyClient:
    """Stub that always returns unparseable output."""
    model_id: str = "stub/difficulty-failing"

    def complete(self, prompt: str) -> str:
        return "The scenario is complex and I cannot determine the right action."
