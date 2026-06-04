"""Selection module configuration + prompts.

The "expressive salience" layer: rank real scenes by model-judged difficulty +
behavior-novelty, and synthesize novel generatable scenarios from a composition.
Pure config; the module imports only pipeline.interfaces + external SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    # objective: score = difficulty*w_difficulty + behavior_novelty*w_novelty
    w_difficulty: float = 0.7
    w_novelty: float = 0.3
    # behavior-novelty is computed over THESE axes only — never agent attributes
    # (car colour etc.), which kept polluting the ranking.
    novelty_axes: frozenset[str] = frozenset({"interactions", "conditions", "ego_maneuver"})
    candidates: int = 40       # cheap pre-rank -> VLM-verify this many
    topk: int = 20             # final shortlist
    synth_temperature: float = 0.7


# Independent difficulty cross-check (separate VLM look — the confabulation guard).
DIFFICULTY_PROMPT = (
    "You are an AV safety analyst. Rate how operationally DIFFICULT this clip is for "
    "a driver by scoring FOUR independent factors, each from 0 to 25. Judge each one "
    "on its own, then they will be summed into a 0-100 total — like a human glancing "
    "at a road and feeling '7/10' because several things add up.\n\n"
    "  visibility (0-25): lighting, weather, glare, occlusion, sensor-degrading "
    "conditions. 0 = perfect clear day; 25 = near-blind (dense fog / heavy rain / "
    "direct glare / unlit night).\n"
    "  agents (0-25): how many other road users, how close, how unpredictable. "
    "0 = empty road; 25 = dense pedestrians / cyclists / vehicles moving erratically.\n"
    "  maneuver (0-25): complexity of what the ego must DO. 0 = straight, empty lane; "
    "25 = unprotected turn across conflicting traffic, tight merge, ambiguous right-of-way.\n"
    "  hazard (0-25): active danger. 0 = none; 25 = imminent conflict / near-miss / "
    "sudden emergence / loss of traction.\n\n"
    "Give each factor the EXACT value it deserves (e.g. 6, 13, 19, 22) — most clips "
    "score low on most factors, so totals are usually modest and rarely round. "
    "Respond in ENGLISH ONLY. Answer ONLY: "
    '{"visibility": <0-25>, "agents": <0-25>, "maneuver": <0-25>, "hazard": <0-25>, '
    '"reason": "<one short reason in English>"}'
)

# The four rubric factors the parser sums into a 0-100 difficulty total.
DIFFICULTY_FACTORS = ("visibility", "agents", "maneuver", "hazard")

# Synthesize a NOVEL generatable scene from a composition's atoms (text-only).
SYNTHESIS_PROMPT = (
    "You are designing a NOVEL driving scenario for an autonomous-vehicle simulator.\n\n"
    "The following conditions and behaviors are known to co-occur in real fleet data:\n"
    "  {atoms}\n"
    "{grounding}\n"
    "Write ONE vivid, plausible, simulator-buildable scenario that EMBODIES this combination: "
    "the road type and lane layout, intersections / signals, weather, lighting and time of day, "
    "the surrounding setting, and the agents (vehicles, pedestrians, cyclists) — with positions, "
    "motions, and intent — that would PRODUCE these conditions and behaviors. Finish with one "
    "sentence on what makes it operationally challenging for the automated driver.\n\n"
    "IMPORTANT: do NOT describe a specific observed clip and do NOT say 'the video shows'. "
    "SYNTHESIZE a new scene from the components; it need not match any single real clip. "
    "Write 5-8 concrete, specific sentences. Respond in ENGLISH ONLY — no other language."
)
