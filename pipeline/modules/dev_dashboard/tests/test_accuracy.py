"""Unit tests for the gold-vs-VLM diff."""

from __future__ import annotations

import pytest

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.dev_dashboard import (
    ALL_FIELDS,
    AccuracyDiffError,
    AccuracyReport,
    compute_diff,
    gold_template,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _gold(window_id: dict, fields: dict) -> dict:
    return {
        "schema_version": "1.0",
        "labels": [{"window_id": window_id, "fields": fields,
                    "label_source": "human:test"}],
    }


def _vlm_record(key: WindowKey, fields: dict, succeeded: bool = True) -> SchemaRecord:
    return SchemaRecord(
        window_id=key, arm="reasoning", schema_version="1.0",
        prompt_template_id="v1_describe", fields=fields,
        failure_mode=None if succeeded else "invalid_json",
    )


_BASE_FIELDS = {
    "agents": ["car"],
    "environment": {"weather": "clear", "time_of_day": "day",
                    "lighting_condition": "well_lit"},
    "road": {"geometry": "straight", "lane_count": 2},
    "traffic_control": "none", "ego_task": "cruising", "conditions": [],
}


# ---------------------------------------------------------------------------
# Happy path: perfect match
# ---------------------------------------------------------------------------

def test_perfect_match_yields_all_field_matches() -> None:
    key = WindowKey(segment_id="s1", window_idx=0)
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    records = [_vlm_record(key, _BASE_FIELDS)]
    report = compute_diff(gold, records)

    assert isinstance(report, AccuracyReport)
    assert len(report.windows) == 1
    assert len(report.missing_entries) == 0
    for fd in report.windows[0].fields:
        assert fd.match, f"field {fd.field_path} unexpectedly mismatched"
    # Aggregates: every field 1/1
    for path in ALL_FIELDS:
        assert report.field_aggregates[path] == (1, 1), path


# ---------------------------------------------------------------------------
# Scalar mismatch
# ---------------------------------------------------------------------------

def test_scalar_mismatch_recorded() -> None:
    key = WindowKey(segment_id="s1", window_idx=0)
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    vlm_fields = dict(_BASE_FIELDS)
    vlm_fields["environment"] = dict(_BASE_FIELDS["environment"])
    vlm_fields["environment"]["weather"] = "fog"   # mismatch
    records = [_vlm_record(key, vlm_fields)]
    report = compute_diff(gold, records)
    by_path = {fd.field_path: fd for fd in report.windows[0].fields}
    assert by_path["environment.weather"].match is False
    assert by_path["environment.weather"].gold == "clear"
    assert by_path["environment.weather"].vlm == "fog"
    # Other fields still match
    assert by_path["agents"].match is True
    # Aggregate: weather 0/1, agents 1/1
    assert report.field_aggregates["environment.weather"] == (0, 1)
    assert report.field_aggregates["agents"] == (1, 1)


# ---------------------------------------------------------------------------
# Multi-value precision / recall / F1
# ---------------------------------------------------------------------------

def test_multi_value_partial_overlap_computes_prf() -> None:
    """Gold=[car,pedestrian] VLM=[car,truck] → precision=0.5, recall=0.5, F1=0.5."""
    key = WindowKey(segment_id="s1", window_idx=0)
    gold_fields = dict(_BASE_FIELDS); gold_fields["agents"] = ["car", "pedestrian"]
    vlm_fields = dict(_BASE_FIELDS); vlm_fields["agents"] = ["car", "truck"]
    gold = _gold({"segment_id": "s1", "window_idx": 0}, gold_fields)
    records = [_vlm_record(key, vlm_fields)]
    report = compute_diff(gold, records)
    by_path = {fd.field_path: fd for fd in report.windows[0].fields}
    agents = by_path["agents"]
    assert agents.match is False  # not strict-equal sets
    assert agents.precision == pytest.approx(0.5)
    assert agents.recall == pytest.approx(0.5)
    assert agents.f1 == pytest.approx(0.5)


def test_multi_value_exact_match_has_f1_one() -> None:
    key = WindowKey(segment_id="s1", window_idx=0)
    gold_fields = dict(_BASE_FIELDS); gold_fields["agents"] = ["car", "pedestrian"]
    vlm_fields = dict(_BASE_FIELDS); vlm_fields["agents"] = ["pedestrian", "car"]
    gold = _gold({"segment_id": "s1", "window_idx": 0}, gold_fields)
    records = [_vlm_record(key, vlm_fields)]
    report = compute_diff(gold, records)
    agents = next(fd for fd in report.windows[0].fields if fd.field_path == "agents")
    assert agents.match is True
    assert agents.f1 == pytest.approx(1.0)


def test_multi_value_empty_sets_match() -> None:
    """conditions=[] vs conditions=[] is a match, F1=0 (no positives)."""
    key = WindowKey(segment_id="s1", window_idx=0)
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    records = [_vlm_record(key, _BASE_FIELDS)]
    report = compute_diff(gold, records)
    conditions = next(fd for fd in report.windows[0].fields
                      if fd.field_path == "conditions")
    assert conditions.match is True


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------

def test_missing_field_on_vlm_side_not_counted_in_aggregate() -> None:
    """When VLM record's field is None (typical failure mode), skip the
    field for that window in the aggregate so denominators are honest."""
    key = WindowKey(segment_id="s1", window_idx=0)
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    bad_vlm = {"agents": None, "environment": None, "road": None,
               "traffic_control": None, "ego_task": None, "conditions": None}
    records = [_vlm_record(key, bad_vlm, succeeded=False)]
    report = compute_diff(gold, records)
    # Every aggregate denominator should be 0 — VLM gave nothing.
    for path in ALL_FIELDS:
        matches, total = report.field_aggregates[path]
        assert total == 0, f"field {path} should not count: got {matches}/{total}"


# ---------------------------------------------------------------------------
# Missing windows
# ---------------------------------------------------------------------------

def test_window_in_gold_but_not_records_reported_as_missing() -> None:
    gold = _gold({"segment_id": "missing_seg", "window_idx": 0}, _BASE_FIELDS)
    records: list[SchemaRecord] = []
    report = compute_diff(gold, records)
    assert len(report.windows) == 0
    assert len(report.missing_entries) == 1
    assert report.missing_entries[0].direction == "missing_in_records"


def test_window_in_records_but_not_gold_reported_as_missing() -> None:
    """Gold covers a subset of analyze output — the rest go in missing."""
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    records = [
        _vlm_record(WindowKey(segment_id="s1", window_idx=0), _BASE_FIELDS),
        _vlm_record(WindowKey(segment_id="s2", window_idx=0), _BASE_FIELDS),
    ]
    report = compute_diff(gold, records)
    assert len(report.windows) == 1
    assert len(report.missing_entries) == 1
    assert report.missing_entries[0].direction == "missing_in_gold"
    assert report.missing_entries[0].window_id.segment_id == "s2"


# ---------------------------------------------------------------------------
# Window-id parsing — both shapes
# ---------------------------------------------------------------------------

def test_window_id_accepts_string_form() -> None:
    """Gold-set files may use 'seg/0000' shorthand."""
    gold = {"schema_version": "1.0",
            "labels": [{"window_id": "s1/0000", "fields": _BASE_FIELDS,
                        "label_source": "human:test"}]}
    key = WindowKey(segment_id="s1", window_idx=0)
    records = [_vlm_record(key, _BASE_FIELDS)]
    report = compute_diff(gold, records)
    assert len(report.windows) == 1
    assert report.windows[0].window_id == key


# ---------------------------------------------------------------------------
# Malformed gold set
# ---------------------------------------------------------------------------

def test_malformed_gold_no_labels_key_raises() -> None:
    with pytest.raises(AccuracyDiffError, match="labels"):
        compute_diff({"schema_version": "1.0"}, [])


def test_empty_labels_list_raises() -> None:
    with pytest.raises(AccuracyDiffError, match="empty"):
        compute_diff({"schema_version": "1.0", "labels": []}, [])


def test_malformed_entry_missing_fields_raises() -> None:
    bad = {"schema_version": "1.0",
           "labels": [{"window_id": "s/0"}]}  # no "fields"
    with pytest.raises(AccuracyDiffError, match="fields"):
        compute_diff(bad, [])


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def test_report_to_json_is_serializable() -> None:
    import json
    key = WindowKey(segment_id="s1", window_idx=0)
    gold = _gold({"segment_id": "s1", "window_idx": 0}, _BASE_FIELDS)
    records = [_vlm_record(key, _BASE_FIELDS)]
    report = compute_diff(gold, records)
    raw = json.dumps(report.to_json())  # must not raise
    assert "s1" in raw


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def test_gold_template_round_trips_through_compute_diff() -> None:
    """The bundled template MUST be a valid gold-set — if it crashes
    compute_diff, the copy-paste UX is broken."""
    template = gold_template()
    # Template references segment "drive_001" — supply matching record
    key = WindowKey(segment_id="drive_001", window_idx=0)
    fields = template["labels"][0]["fields"]
    records = [_vlm_record(key, fields)]
    report = compute_diff(template, records)
    assert len(report.windows) == 1
    # All fields should match (template ⇄ identical VLM output)
    for fd in report.windows[0].fields:
        assert fd.match, f"{fd.field_path} mismatched on identical input"
