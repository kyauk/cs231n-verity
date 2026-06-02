"""Module 2: Encoder — locked vocabulary (v1.0).

Every tag in every field lives here. This file is the single source of truth.
To add a new tag: update the set, bump VOCABULARY_VERSION, and reprocess.
To remove a tag: same — do not silently drop it from downstream records.

Validation returns a list of violations (strings). An empty list means clean.
Callers decide whether to raise or record the violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VOCABULARY_VERSION = "1.0"


def _not_in_vocab(value: Any, vocab: "frozenset[str]") -> bool:
    """True if ``value`` should count as a vocabulary violation.

    Guards against non-string scalars (dict/list/int) the VLM occasionally
    emits where a scalar tag is expected. Doing a bare ``value in frozenset``
    on an unhashable dict raises ``TypeError: unhashable type: 'dict'`` and
    escapes the retry loop as an "unknown" failure; treating it as a normal
    violation instead lets the stricter-prompt retry recover the window.
    """
    return not isinstance(value, str) or value not in vocab


# ---------------------------------------------------------------------------
# Tag sets
# ---------------------------------------------------------------------------

AGENTS: frozenset[str] = frozenset({
    "pedestrian",
    "cyclist",
    "motorcycle",
    "bus",
    "truck",
    "car",
    "emergency_vehicle",
    "construction_worker",
    "parked_vehicle",
    "animal",
    "debris",
    "unknown_agent",
})

WEATHER: frozenset[str] = frozenset({
    "clear",
    "rain",
    "fog",
    "snow",
    "overcast",
})

TIME_OF_DAY: frozenset[str] = frozenset({
    "day",
    "dusk_dawn",
    "night",
})

LIGHTING_CONDITION: frozenset[str] = frozenset({
    "well_lit",
    "dim",
    "glare",
    "unlit",
})

ROAD_GEOMETRY: frozenset[str] = frozenset({
    "straight",
    "curve",
    "intersection",
    "roundabout",
    "merge",
    "ramp",
})

TRAFFIC_CONTROL: frozenset[str] = frozenset({
    "none",
    "traffic_light",
    "stop_sign",
    "yield_sign",
    "officer_directed",
    "construction_signals",
})

EGO_TASK: frozenset[str] = frozenset({
    "cruising",
    "turning_left",
    "turning_right",
    "lane_change",
    "stopping",
    "reversing",
})

# The 10 compositional condition tags — the long-tail targets for the
# Hypothesizer. These are what Module 3 operates on.
CONDITIONS: frozenset[str] = frozenset({
    "night_driving",
    "rain",
    "fog",
    "construction_zone",
    "school_zone",
    "emergency_scene",
    "heavy_traffic",
    "unusual_road_markings",
    "animal_crossing",
    "road_debris",
})

LANE_COUNT_MIN = 1
LANE_COUNT_MAX = 8


# ---------------------------------------------------------------------------
# Vocabulary container
# ---------------------------------------------------------------------------

@dataclass
class Vocabulary:
    version: str = VOCABULARY_VERSION

    agents: frozenset[str] = field(default_factory=lambda: AGENTS)
    weather: frozenset[str] = field(default_factory=lambda: WEATHER)
    time_of_day: frozenset[str] = field(default_factory=lambda: TIME_OF_DAY)
    lighting_condition: frozenset[str] = field(default_factory=lambda: LIGHTING_CONDITION)
    road_geometry: frozenset[str] = field(default_factory=lambda: ROAD_GEOMETRY)
    traffic_control: frozenset[str] = field(default_factory=lambda: TRAFFIC_CONTROL)
    ego_task: frozenset[str] = field(default_factory=lambda: EGO_TASK)
    conditions: frozenset[str] = field(default_factory=lambda: CONDITIONS)

    def validate_fields(self, fields: dict[str, Any]) -> list[str]:
        """Return a list of vocabulary violations. Empty list = clean."""
        violations: list[str] = []

        # agents
        agents = fields.get("agents")
        if agents is not None:
            if not isinstance(agents, list):
                violations.append(f"agents must be a list, got {type(agents).__name__}")
            else:
                bad = [a for a in agents if _not_in_vocab(a, self.agents)]
                if bad:
                    violations.append(
                        f"agents contains unknown tags: {bad}. "
                        f"Valid: {sorted(self.agents)}"
                    )

        # environment
        env = fields.get("environment") or {}
        if env:
            w = env.get("weather")
            if w is not None and _not_in_vocab(w, self.weather):
                violations.append(
                    f"environment.weather={w!r} not in vocabulary. "
                    f"Valid: {sorted(self.weather)}"
                )
            tod = env.get("time_of_day")
            if tod is not None and _not_in_vocab(tod, self.time_of_day):
                violations.append(
                    f"environment.time_of_day={tod!r} not in vocabulary. "
                    f"Valid: {sorted(self.time_of_day)}"
                )
            lc = env.get("lighting_condition")
            if lc is not None and _not_in_vocab(lc, self.lighting_condition):
                violations.append(
                    f"environment.lighting_condition={lc!r} not in vocabulary. "
                    f"Valid: {sorted(self.lighting_condition)}"
                )

        # road
        road = fields.get("road") or {}
        if road:
            geo = road.get("geometry")
            if geo is not None and _not_in_vocab(geo, self.road_geometry):
                violations.append(
                    f"road.geometry={geo!r} not in vocabulary. "
                    f"Valid: {sorted(self.road_geometry)}"
                )
            lc_count = road.get("lane_count")
            if lc_count is not None:
                if not isinstance(lc_count, int):
                    violations.append(
                        f"road.lane_count must be int, got {type(lc_count).__name__}"
                    )
                elif not (LANE_COUNT_MIN <= lc_count <= LANE_COUNT_MAX):
                    violations.append(
                        f"road.lane_count={lc_count} out of range "
                        f"[{LANE_COUNT_MIN}, {LANE_COUNT_MAX}]"
                    )

        # traffic_control
        tc = fields.get("traffic_control")
        if tc is not None and _not_in_vocab(tc, self.traffic_control):
            violations.append(
                f"traffic_control={tc!r} not in vocabulary. "
                f"Valid: {sorted(self.traffic_control)}"
            )

        # ego_task
        et = fields.get("ego_task")
        if et is not None and _not_in_vocab(et, self.ego_task):
            violations.append(
                f"ego_task={et!r} not in vocabulary. "
                f"Valid: {sorted(self.ego_task)}"
            )

        # conditions
        conds = fields.get("conditions")
        if conds is not None:
            if not isinstance(conds, list):
                violations.append(
                    f"conditions must be a list, got {type(conds).__name__}"
                )
            else:
                bad = [c for c in conds if _not_in_vocab(c, self.conditions)]
                if bad:
                    violations.append(
                        f"conditions contains unknown tags: {bad}. "
                        f"Valid: {sorted(self.conditions)}"
                    )

        return violations

    def fill_fraction(self, fields: dict[str, Any]) -> float:
        """What fraction of schema fields are non-null (0.0–1.0).

        Used for calibration checks (spec: ≥80% filled to pass calibration).
        """
        checks = [
            fields.get("agents") is not None,
            (fields.get("environment") or {}).get("weather") is not None,
            (fields.get("environment") or {}).get("time_of_day") is not None,
            (fields.get("environment") or {}).get("lighting_condition") is not None,
            (fields.get("road") or {}).get("geometry") is not None,
            (fields.get("road") or {}).get("lane_count") is not None,
            fields.get("traffic_control") is not None,
            fields.get("ego_task") is not None,
            fields.get("conditions") is not None,
        ]
        return sum(checks) / len(checks)

    def prompt_context(self) -> str:
        """Return a compact vocabulary reference for inclusion in VLM prompts."""
        return (
            f"agents ({len(self.agents)} tags): {', '.join(sorted(self.agents))}\n"
            f"weather: {', '.join(sorted(self.weather))}\n"
            f"time_of_day: {', '.join(sorted(self.time_of_day))}\n"
            f"lighting_condition: {', '.join(sorted(self.lighting_condition))}\n"
            f"road.geometry: {', '.join(sorted(self.road_geometry))}\n"
            f"road.lane_count: integer {LANE_COUNT_MIN}–{LANE_COUNT_MAX}\n"
            f"traffic_control: {', '.join(sorted(self.traffic_control))}\n"
            f"ego_task: {', '.join(sorted(self.ego_task))}\n"
            f"conditions ({len(self.conditions)} tags): {', '.join(sorted(self.conditions))}"
        )


# Singleton default — import this instead of constructing each time.
DEFAULT_VOCABULARY = Vocabulary()
