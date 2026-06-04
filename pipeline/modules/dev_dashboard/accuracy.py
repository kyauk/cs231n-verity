"""Pure-function diff: hand-labeled gold set vs encoder schema records.

Per-window comparison across the locked v1.0 schema fields. Scalar fields
use exact equality. Multi-value fields (agents, conditions) report
precision / recall / F1 plus a strict-equality match boolean. No statistics
are computed here — the report is raw counters, and the operator does
aggregation offline (or via the frontend's rendered totals).

This module is intentionally not in `pipeline.interfaces`: the AccuracyReport
shape is consumed by the dev_dashboard frontend over HTTP only, never by
another Python module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey


# Field paths (dotted) in the locked v1.0 schema. Multi-value fields are
# the ones that carry lists; everything else is scalar.
_SCALAR_FIELDS: tuple[str, ...] = (
    "environment.weather",
    "environment.time_of_day",
    "environment.lighting_condition",
    "road.geometry",
    "road.lane_count",
    "traffic_control",
    "ego_task",
)
_MULTI_VALUE_FIELDS: tuple[str, ...] = ("agents", "conditions")
ALL_FIELDS: tuple[str, ...] = _MULTI_VALUE_FIELDS + _SCALAR_FIELDS


class AccuracyDiffError(Exception):
    """Raised when the gold set itself is malformed (not when matches fail)."""


@dataclass
class FieldDiff:
    """One field comparison for one window."""
    field_path: str
    gold: Any
    vlm: Any
    match: bool                       # strict equality
    precision: float | None = None    # multi-value only
    recall: float | None = None       # multi-value only
    f1: float | None = None           # multi-value only

    def to_json(self) -> dict[str, Any]:
        return {
            "field_path": self.field_path,
            "gold": self.gold,
            "vlm": self.vlm,
            "match": self.match,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class WindowDiff:
    """All field comparisons for one window."""
    window_id: WindowKey
    fields: list[FieldDiff]

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id.to_json(),
            "fields": [f.to_json() for f in self.fields],
        }


@dataclass
class MissingEntry:
    """A window appears in one source but not the other."""
    window_id: WindowKey
    direction: str   # "missing_in_records" | "missing_in_gold"

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id.to_json(),
            "direction": self.direction,
        }


@dataclass
class AccuracyReport:
    """Full diff of one gold set against one analyze run."""
    windows: list[WindowDiff]
    # field_path → (matches, total_compared). Excludes windows where either
    # side was missing for that field.
    field_aggregates: dict[str, tuple[int, int]]
    missing_entries: list[MissingEntry]
    schema_version: str = "1.0"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "windows": [w.to_json() for w in self.windows],
            "field_aggregates": {
                k: list(v) for k, v in self.field_aggregates.items()
            },
            "missing_entries": [m.to_json() for m in self.missing_entries],
        }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compute_diff(
    gold: dict[str, Any],
    records: list[SchemaRecord],
) -> AccuracyReport:
    """Compute the per-field diff between a gold set and analyze output.

    Parameters
    ----------
    gold
        Parsed gold-set JSON of the shape documented in the module
        docstring. Must contain {"schema_version": str, "labels": list[...]}.
    records
        SchemaRecords from a `pipeline.run analyze` run. Failed records
        (failure_mode is not None) are still compared — their fields are
        likely null, which surfaces as mismatches.

    Raises
    ------
    AccuracyDiffError
        If the gold set is structurally invalid (missing keys, bad types).
    """
    schema_version, labels = _parse_gold(gold)

    # Index both sides by WindowKey for joining.
    gold_by_key: dict[WindowKey, dict[str, Any]] = {}
    for entry in labels:
        key = _parse_window_key(entry.get("window_id"))
        gold_fields = entry.get("fields")
        if not isinstance(gold_fields, dict):
            raise AccuracyDiffError(
                f"gold entry {key} is missing 'fields' or it is not a dict."
            )
        gold_by_key[key] = gold_fields

    records_by_key: dict[WindowKey, SchemaRecord] = {
        r.window_id: r for r in records
    }

    # Build the per-window diffs over the intersection.
    window_diffs: list[WindowDiff] = []
    aggregates: dict[str, list[int]] = {p: [0, 0] for p in ALL_FIELDS}
    common_keys = sorted(
        gold_by_key.keys() & records_by_key.keys(),
        key=lambda k: (k.segment_id, k.window_idx),
    )
    for key in common_keys:
        diffs = _diff_one_window(gold_by_key[key], records_by_key[key].fields)
        window_diffs.append(WindowDiff(window_id=key, fields=diffs))
        for d in diffs:
            if d.gold is None or d.vlm is None:
                continue  # missing on one side → skip aggregate
            agg = aggregates[d.field_path]
            agg[1] += 1
            if d.match:
                agg[0] += 1

    # Missing entries on either side.
    missing: list[MissingEntry] = []
    for key in sorted(gold_by_key.keys() - records_by_key.keys(),
                      key=lambda k: (k.segment_id, k.window_idx)):
        missing.append(MissingEntry(window_id=key, direction="missing_in_records"))
    for key in sorted(records_by_key.keys() - gold_by_key.keys(),
                      key=lambda k: (k.segment_id, k.window_idx)):
        missing.append(MissingEntry(window_id=key, direction="missing_in_gold"))

    return AccuracyReport(
        windows=window_diffs,
        field_aggregates={k: (v[0], v[1]) for k, v in aggregates.items()},
        missing_entries=missing,
        schema_version=schema_version,
    )


def gold_template() -> dict[str, Any]:
    """Return the copy-paste JSON template the frontend offers."""
    return {
        "schema_version": "1.0",
        "labels": [
            {
                "window_id": {"segment_id": "drive_001", "window_idx": 0},
                "fields": {
                    "agents": ["car", "pedestrian"],
                    "environment": {
                        "weather": "clear",
                        "time_of_day": "day",
                        "lighting_condition": "well_lit",
                    },
                    "road": {"geometry": "straight", "lane_count": 2},
                    "traffic_control": "none",
                    "ego_task": "cruising",
                    "conditions": [],
                },
                "label_source": "human:your_name",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _parse_gold(gold: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(gold, dict):
        raise AccuracyDiffError("gold must be a JSON object, got " + type(gold).__name__)
    schema_version = str(gold.get("schema_version", "1.0"))
    labels = gold.get("labels")
    if not isinstance(labels, list):
        raise AccuracyDiffError("gold['labels'] must be a list.")
    if not labels:
        raise AccuracyDiffError("gold['labels'] is empty — nothing to diff.")
    return schema_version, labels


def _parse_window_key(raw: Any) -> WindowKey:
    if isinstance(raw, dict):
        return WindowKey.from_json(raw)
    if isinstance(raw, str):
        return WindowKey.from_str(raw)
    raise AccuracyDiffError(
        f"window_id must be a dict or string, got {type(raw).__name__}: {raw!r}"
    )


def _diff_one_window(gold_fields: dict[str, Any],
                       vlm_fields: dict[str, Any]) -> list[FieldDiff]:
    diffs: list[FieldDiff] = []
    for path in ALL_FIELDS:
        gold_val = _resolve_path(gold_fields, path)
        vlm_val = _resolve_path(vlm_fields, path)
        if path in _MULTI_VALUE_FIELDS:
            diffs.append(_diff_multi_value(path, gold_val, vlm_val))
        else:
            diffs.append(_diff_scalar(path, gold_val, vlm_val))
    return diffs


def _resolve_path(fields: dict[str, Any], path: str) -> Any:
    """Walk a dotted path through a nested dict, returning None on miss."""
    cur: Any = fields
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _diff_scalar(path: str, gold: Any, vlm: Any) -> FieldDiff:
    if gold is None or vlm is None:
        return FieldDiff(field_path=path, gold=gold, vlm=vlm, match=False)
    return FieldDiff(field_path=path, gold=gold, vlm=vlm, match=gold == vlm)


def _diff_multi_value(path: str, gold: Any, vlm: Any) -> FieldDiff:
    if gold is None or vlm is None:
        return FieldDiff(field_path=path, gold=gold, vlm=vlm, match=False)
    gold_set = set(gold) if isinstance(gold, list) else set()
    vlm_set = set(vlm) if isinstance(vlm, list) else set()
    tp = len(gold_set & vlm_set)
    precision = tp / len(vlm_set) if vlm_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return FieldDiff(
        field_path=path, gold=list(gold), vlm=list(vlm),
        match=gold_set == vlm_set,
        precision=precision, recall=recall, f1=f1,
    )
