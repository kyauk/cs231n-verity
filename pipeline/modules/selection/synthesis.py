"""Synthesize a NOVEL, generatable scenario from a composition's atoms.

NOT a re-watch/reconstruction of one observed clip. A composition often has no
single motivating scene; the goal is to GENERATE a new scene a simulator could
build that EMBODIES the conditions+behaviors, grounded by (not copied from) real
evidence. Pure text (a NIM text call); temperature>0 so the atoms generalize.
"""

from __future__ import annotations

import os

from pipeline.modules.selection.config import SYNTHESIS_PROMPT


def synthesize_scenario(
    atoms: list[str],
    grounding: list[str] | None = None,
    model_id: str | None = None,
    temperature: float = 0.5,
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
    content = SYNTHESIS_PROMPT.format(atoms=atom_str, grounding=ground)
    try:
        client = OpenAI(api_key=key, base_url=base)
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": content}])
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"(scenario synthesis unavailable: {exc})"
