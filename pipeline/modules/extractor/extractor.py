"""Stage 1: Reason-first extraction with a hard evidence/interpretation split.

    Extractor(reason, structure, embed).extract(scene_id, video_ref) -> list[RawDescriptor]

Two passes, by design:
  1. The VLM produces a FREE-FORM scene analysis — unconstrained, so it never has
     to fight a schema (which is what made the old encoder drop its own boxes).
  2. A structuring pass extracts typed descriptors from that text, and each one
     stores a SPAN POINTER back to the sentence that justified it. That pointer is
     the fool-proofing: a bad atom is later auditable to its source — you can read
     the reasoning and tell whether the model misperceived the scene or the
     structuring pass misparsed it.

Output is immutable evidence (RawDescriptor). The extractor decides nothing about
labels — that's the curator's job, firewalled away.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline.interfaces.taxonomy import RawDescriptor
from pipeline.modules.extractor.clients import Embedder, ReasonClient, StructureClient
from pipeline.modules.extractor.config import (
    ExtractorConfig,
    ReasoningUnavailableError,
    StructuringError,
)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# The AXES are fixed (entity boundaries), but a 7B structuring model won't reliably
# echo the exact strings — it says "vehicles" for agents, "lane markings" for road,
# etc. Normalize those synonyms to the canonical axis instead of rejecting the
# descriptor (which was discarding good evidence and failing whole clips). A
# descriptor whose axis maps to nothing is dropped individually; the clip still
# succeeds on the rest.
_AXIS_ALIASES: dict[str, str] = {
    # agents
    "agent": "agents", "vehicle": "agents", "vehicles": "agents", "object": "agents",
    "objects": "agents", "actor": "agents", "actors": "agents", "road_user": "agents",
    "road_users": "agents", "pedestrian": "agents", "pedestrians": "agents", "cyclist": "agents",
    # ego_maneuver
    "ego": "ego_maneuver", "ego_vehicle": "ego_maneuver", "ego_action": "ego_maneuver",
    "maneuver": "ego_maneuver", "ego_behavior": "ego_maneuver", "ego_maneuvers": "ego_maneuver",
    # interactions
    "interaction": "interactions", "behavior": "interactions", "behaviors": "interactions",
    "relation": "interactions", "relations": "interactions", "event": "interactions",
    "events": "interactions",
    # conditions
    "condition": "conditions", "environment": "conditions", "scene": "conditions",
    "traffic_signal": "conditions", "traffic_signals": "conditions", "signals": "conditions",
    "traffic_control": "conditions", "traffic_light": "conditions", "traffic_lights": "conditions",
    "hazard": "conditions", "hazards": "conditions", "safety": "conditions",
    # road
    "road_geometry": "road", "geometry": "road", "layout": "road", "road_layout": "road",
    "lane": "road", "lanes": "road", "lane_marking": "road", "lane_markings": "road",
    "road_marking": "road", "road_markings": "road", "intersection": "road",
    # weather / time
    "weather_conditions": "weather", "sky": "weather",
    "time_of_day": "time", "lighting": "time", "daylight": "time", "daytime": "time",
}


def _normalize_axis(axis: str, valid: frozenset[str]) -> str | None:
    """Map the model's axis phrasing to a canonical axis, or None if unmappable."""
    key = axis.strip().lower().replace(" ", "_").replace("-", "_")
    if key in valid:
        return key
    canon = _AXIS_ALIASES.get(key)
    return canon if (canon in valid) else None


def _load_prompt(prompt_id: str) -> str:
    path = _PROMPT_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"[Extractor] prompt {prompt_id!r} not found at {path}")
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON object from a model response (fences, tags, first {...})."""
    text = text.strip()
    for pat in (r"```json\s*(.*?)```", r"```\s*([\s\S]*?)```", r"<json>\s*(.*?)\s*</json>"):
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise StructuringError(f"no parseable JSON in structuring output: {text[:300]!r}")


class Extractor:
    """Produces immutable RawDescriptor evidence for one scene."""

    def __init__(
        self,
        reason_client: ReasonClient,
        structure_client: StructureClient,
        embedder: Embedder,
        config: ExtractorConfig = ExtractorConfig(),
    ) -> None:
        self._reason = reason_client
        self._structure = structure_client
        self._embed = embedder
        self._config = config

    def extract(self, scene_id: str, video_ref: str) -> list[RawDescriptor]:
        cfg = self._config

        # --- pass 1: free-form reasoning (unconstrained) ----------------------
        try:
            reasoning = self._reason.describe(video_ref, _load_prompt(cfg.reason_prompt_id))
        except Exception as exc:  # noqa: BLE001
            raise ReasoningUnavailableError(f"reasoning failed for {scene_id}: {exc}") from exc
        if not reasoning.strip():
            raise ReasoningUnavailableError(f"empty reasoning for {scene_id}")

        # --- pass 2: structuring into typed descriptors + span pointers -------
        raw = self._structure.structure(reasoning, _load_prompt(cfg.structure_prompt_id))
        items = _extract_json(raw).get("descriptors", [])
        if not isinstance(items, list):
            raise StructuringError(f"'descriptors' is not a list for {scene_id}")

        cleaned: list[tuple[str, str, str, float]] = []   # (axis, text, span, salience)
        for it in items[: cfg.max_descriptors_per_scene]:
            if not isinstance(it, dict):
                continue
            axis = _normalize_axis(str(it.get("axis", "")), cfg.axes)
            text = str(it.get("text", "")).strip()
            span = str(it.get("span", "")).strip()
            if axis is None or not text:
                continue
            # model-judged criticality (learned, not whitelisted). Default to routine
            # rather than missing, so an unscored descriptor never reads as "hard".
            try:
                sal = max(0.0, min(1.0, float(it.get("salience", 0.1))))
            except (TypeError, ValueError):
                sal = 0.1
            # span pointer must actually point into the reasoning (auditability)
            if cfg.require_span and span and span not in reasoning:
                span = _closest_span(span, reasoning)
            cleaned.append((axis, text, span or reasoning[:160], sal))

        if not cleaned:
            raise StructuringError(f"no valid typed descriptors for {scene_id}")

        # --- embed the descriptor phrases (curator clusters on these) ---------
        embeddings = self._embed.embed([t for _, t, _, _ in cleaned])

        return [
            RawDescriptor(
                scene_id=scene_id, axis=axis, text=text,
                reasoning_span=span, embedding=tuple(emb), salience=sal,
            )
            for (axis, text, span, sal), emb in zip(cleaned, embeddings)
        ]


def _closest_span(span: str, reasoning: str) -> str:
    """If the model's span isn't verbatim, snap to the sentence with most word overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    want = set(span.lower().split())
    if not want or not sentences:
        return span
    best = max(sentences, key=lambda s: len(want & set(s.lower().split())))
    return best.strip()
