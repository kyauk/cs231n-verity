"""Synthesize a NOVEL, generatable scenario from a composition's atoms.

NOT a re-watch/reconstruction of one observed clip. A proposal/composition often
has no single motivating scene (or several identical ones), and the goal is to
GENERATE a new scene a simulator could build — one that EMBODIES the
conditions+behaviors — grounded by (but not copied from) real evidence.

Pure text (a NIM text call); temperature>0 so the same atoms generalize into a
plausible novel scene rather than a verbatim caption.
"""
from __future__ import annotations

import os

_SYNTH_PROMPT = (
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
    "Write 5-8 concrete, specific sentences."
)


def synthesize_scenario(
    atoms: list[str],
    grounding: list[str] | None = None,
    model_id: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 600,
) -> str:
    """atoms: ['interactions:occluded_pedestrian', 'conditions:heavy_rain', ...].
    grounding: a few real evidence sentences (inspiration, not to be copied)."""
    from openai import OpenAI  # noqa: PLC0415

    model = model_id or os.environ.get(
        "STRUCTURE_NIM_MODEL_ID", os.environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason1-7b"))
    base = os.environ.get("NVIDIA_BASE_URL", "http://localhost:8081/v1")
    key = os.environ.get("NVIDIA_API_KEY", "local")

    atom_str = ", ".join(a.split(":", 1)[-1].replace("_", " ") for a in atoms) or "a routine driving scene"
    ground = ""
    if grounding:
        snippets = " ".join(g.strip() for g in grounding[:5] if g.strip())
        if snippets:
            ground = (f"For grounding only (real observations — take inspiration, do NOT copy "
                      f"verbatim): {snippets}\n")
    content = _SYNTH_PROMPT.format(atoms=atom_str, grounding=ground)
    try:
        client = OpenAI(api_key=key, base_url=base)
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": content}])
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"(scenario synthesis unavailable: {exc})"


if __name__ == "__main__":  # quick manual smoke
    import sys
    demo = ["conditions:reduced_visibility", "interactions:occluded_pedestrian_emergence",
            "weather:heavy_rain", "ego_maneuver:slowing_to_stop"]
    print(synthesize_scenario(demo, grounding=["A pedestrian steps off the curb from behind a stopped bus."]))
