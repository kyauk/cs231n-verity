"""Round-trip serialization tests for every interface type.

Every type in pipeline/interfaces/ must pass a to_json() → from_json() round-trip.
These tests are the load-bearing contract tests — if they pass, downstream modules
can rely on consistent serialization at every boundary.

Run:
    python -m pytest pipeline/interfaces/tests/test_roundtrip.py -v
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# window.py types
# ---------------------------------------------------------------------------

def test_window_key_roundtrip() -> None:
    from pipeline.interfaces.window import WindowKey
    original = WindowKey(segment_id="seg_abc123", window_idx=7)
    restored = WindowKey.from_json(original.to_json())
    assert restored == original
    assert str(restored) == "seg_abc123/0007"


def test_window_key_str_roundtrip() -> None:
    from pipeline.interfaces.window import WindowKey
    original = WindowKey(segment_id="seg_abc123", window_idx=42)
    restored = WindowKey.from_str(str(original))
    assert restored == original


def test_window_key_json_serializable() -> None:
    from pipeline.interfaces.window import WindowKey
    key = WindowKey(segment_id="seg_001", window_idx=0)
    assert json.dumps(key.to_json())  # must not raise


def test_window_key_hashable() -> None:
    from pipeline.interfaces.window import WindowKey
    k1 = WindowKey(segment_id="seg_001", window_idx=0)
    k2 = WindowKey(segment_id="seg_001", window_idx=0)
    k3 = WindowKey(segment_id="seg_002", window_idx=0)
    assert k1 == k2
    assert k1 != k3
    assert len({k1, k2, k3}) == 2


def test_pose_record_roundtrip() -> None:
    from pipeline.interfaces.window import PoseRecord
    original = PoseRecord(
        timestamp_us=1_000_000,
        x=10.5, y=-3.2, z=0.1,
        roll_rad=0.01, pitch_rad=0.02, yaw_rad=1.57,
    )
    restored = PoseRecord.from_json(original.to_json())
    assert restored == original


def test_pose_record_defaults_roundtrip() -> None:
    from pipeline.interfaces.window import PoseRecord
    original = PoseRecord(timestamp_us=0, x=0.0, y=0.0, z=0.0)
    restored = PoseRecord.from_json(original.to_json())
    assert restored == original
    assert restored.roll_rad == 0.0


def test_pose_data_roundtrip() -> None:
    from pipeline.interfaces.window import PoseData, PoseRecord
    original = PoseData(
        segment_id="seg_001",
        window_idx=2,
        records=[
            PoseRecord(timestamp_us=0, x=0.0, y=0.0, z=0.0),
            PoseRecord(timestamp_us=1_000_000, x=5.0, y=1.0, z=0.0),
        ],
    )
    restored = PoseData.from_json(original.to_json())
    assert restored.segment_id == original.segment_id
    assert restored.window_idx == original.window_idx
    assert len(restored.records) == 2
    assert restored.records[0] == original.records[0]
    assert restored.records[1] == original.records[1]


def test_pose_data_empty_records_roundtrip() -> None:
    from pipeline.interfaces.window import PoseData
    original = PoseData(segment_id="seg_001", window_idx=0, records=[])
    restored = PoseData.from_json(original.to_json())
    assert restored.records == []


def test_window_manifest_roundtrip() -> None:
    from pipeline.interfaces.window import WindowManifest
    original = WindowManifest(
        segment_id="seg_001",
        window_idx=3,
        source_format="waymo_parquet",
        source_schema_version="waymo_v2_camera_image_v1",
        window_start_ts_us=1_000_000,
        window_end_ts_us=9_000_000,
        frame_count=80,
        cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        ingested_at="2026-05-25T00:00:00Z",
        pose_summary="Vehicle traveled 20m forward.",
        extra={"dataset": "waymo_open_dataset_v_2_0_1"},
    )
    restored = WindowManifest.from_json(original.to_json())
    assert restored.segment_id == original.segment_id
    assert restored.window_idx == original.window_idx
    assert restored.cameras == original.cameras
    assert restored.pose_summary == original.pose_summary
    assert restored.extra == original.extra


def test_window_manifest_null_pose_roundtrip() -> None:
    from pipeline.interfaces.window import WindowManifest
    original = WindowManifest(
        segment_id="seg_002",
        window_idx=0,
        source_format="waymo_parquet",
        source_schema_version="v1",
        window_start_ts_us=0,
        window_end_ts_us=8_000_000,
        frame_count=80,
        cameras=["FRONT"],
        ingested_at="2026-05-25T00:00:00Z",
        pose_summary=None,
    )
    restored = WindowManifest.from_json(original.to_json())
    assert restored.pose_summary is None


def test_window_manifest_window_key_property() -> None:
    from pipeline.interfaces.window import WindowKey, WindowManifest
    manifest = WindowManifest(
        segment_id="seg_001", window_idx=5,
        source_format="waymo_parquet", source_schema_version="v1",
        window_start_ts_us=0, window_end_ts_us=8_000_000,
        frame_count=80, cameras=["FRONT"], ingested_at="2026-05-25T00:00:00Z",
    )
    assert manifest.window_key == WindowKey(segment_id="seg_001", window_idx=5)


def test_dataset_manifest_roundtrip() -> None:
    from pipeline.interfaces.window import DatasetManifest, WindowKey
    original = DatasetManifest(
        bucket_uri="gs://my-bucket/verity",
        window_count=6,
        segment_count=3,
        created_at="2026-05-25T00:00:00Z",
        updated_at="2026-05-25T01:00:00Z",
        windows=[
            WindowKey(segment_id="seg_001", window_idx=0),
            WindowKey(segment_id="seg_001", window_idx=1),
            WindowKey(segment_id="seg_002", window_idx=0),
        ],
    )
    restored = DatasetManifest.from_json(original.to_json())
    assert restored.bucket_uri == original.bucket_uri
    assert restored.window_count == original.window_count
    assert len(restored.windows) == 3
    assert restored.windows[0] == original.windows[0]


def test_dataset_manifest_empty_windows_roundtrip() -> None:
    from pipeline.interfaces.window import DatasetManifest
    original = DatasetManifest(
        bucket_uri="gs://bucket", window_count=0, segment_count=0,
        created_at="2026-05-25T00:00:00Z", updated_at="2026-05-25T00:00:00Z",
    )
    restored = DatasetManifest.from_json(original.to_json())
    assert restored.windows == []


# ---------------------------------------------------------------------------
# schema_record.py
# ---------------------------------------------------------------------------

def test_schema_record_roundtrip() -> None:
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.interfaces.window import WindowKey
    original = SchemaRecord(
        window_id=WindowKey(segment_id="seg_001", window_idx=0),
        arm="reasoning",
        schema_version="1.0",
        prompt_template_id="v1_describe",
        fields={
            "agents": ["car", "pedestrian"],
            "environment": {"weather": "rain", "time_of_day": "night", "lighting_condition": "dim"},
            "road": {"geometry": "intersection", "lane_count": 4},
            "traffic_control": "traffic_light",
            "ego_task": "turning_left",
            "conditions": ["night_driving", "rain"],
        },
        failure_mode=None,
        cached=False,
        created_at="2026-05-25T00:00:00Z",
    )
    restored = SchemaRecord.from_json(original.to_json())
    assert str(restored.window_id) == str(original.window_id)
    assert restored.arm == original.arm
    assert restored.fields == original.fields
    assert restored.failure_mode is None
    assert restored.cached == original.cached


def test_schema_record_failed_roundtrip() -> None:
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.interfaces.window import WindowKey
    original = SchemaRecord(
        window_id=WindowKey(segment_id="seg_002", window_idx=1),
        arm="reasoning",
        schema_version="1.0",
        prompt_template_id="v1_describe",
        fields={"agents": None, "environment": {"weather": None, "time_of_day": None, "lighting_condition": None}, "road": {"geometry": None, "lane_count": None}, "traffic_control": None, "ego_task": None, "conditions": None},
        failure_mode="invalid_json",
        created_at="2026-05-25T00:00:00Z",
    )
    restored = SchemaRecord.from_json(original.to_json())
    assert restored.failure_mode == "invalid_json"
    assert not restored.succeeded


def test_schema_record_json_serializable() -> None:
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.interfaces.window import WindowKey
    record = SchemaRecord(
        window_id=WindowKey(segment_id="seg_001", window_idx=0),
        arm="reasoning", schema_version="1.0", prompt_template_id="v1_describe",
        fields={"agents": ["car"]}, failure_mode=None,
        created_at="2026-05-25T00:00:00Z",
    )
    assert json.dumps(record.to_json())  # must not raise


# ---------------------------------------------------------------------------
# proposal.py
# ---------------------------------------------------------------------------

def test_composition_proposal_roundtrip() -> None:
    from pipeline.interfaces.proposal import CompositionProposal
    from pipeline.interfaces.window import WindowKey
    original = CompositionProposal(
        composition_id="abc123",
        constituents=["night_driving", "fog"],
        marginal_frequencies={"night_driving": 0.15, "fog": 0.08},
        pairwise_frequencies={"night_driving|fog": 0.02},
        expected_joint=0.012,
        observed_joint=0.001,
        novelty_score=2.5,
        motivating_scene_ids=[WindowKey("seg_001", 0), WindowKey("seg_002", 3)],
        arm="reasoning",
    )
    restored = CompositionProposal.from_json(original.to_json())
    assert restored.composition_id == original.composition_id
    assert restored.constituents == original.constituents
    assert len(restored.motivating_scene_ids) == 2
    assert restored.novelty_score == original.novelty_score


def test_scored_proposal_roundtrip() -> None:
    from pipeline.interfaces.proposal import ScoredProposal
    from pipeline.interfaces.window import WindowKey
    original = ScoredProposal(
        composition_id="abc123",
        constituents=["night_driving", "fog"],
        marginal_frequencies={"night_driving": 0.15},
        pairwise_frequencies={},
        expected_joint=0.012,
        observed_joint=0.001,
        novelty_score=2.5,
        motivating_scene_ids=[WindowKey("seg_001", 0)],
        arm="reasoning",
        plausibility_score=0.85,
        plausibility_justification="Physically plausible combination.",
        frontier_difficulty_score=0.72,
        frontier_difficulty_signals={"variance": 0.4, "mean_confidence": 0.3},
        final_rank_score=0.78,
        accepted=True,
        rejection_reason=None,
    )
    restored = ScoredProposal.from_json(original.to_json())
    assert restored.accepted == original.accepted
    assert restored.plausibility_score == original.plausibility_score
    assert restored.rejection_reason is None


# ---------------------------------------------------------------------------
# rating.py
# ---------------------------------------------------------------------------

def test_rating_roundtrip() -> None:
    from pipeline.interfaces.rating import Rating
    from pipeline.interfaces.window import WindowKey
    original = Rating(
        rater_id="rater_001",
        proposal_id="abc123",
        arm="reasoning",
        coherence_score=4,
        usefulness_score=3,
        timestamp="2026-05-25T12:00:00Z",
        free_text_note="Plausible but unlikely.",
        seen_motivating_scenes=[WindowKey("seg_001", 0)],
    )
    restored = Rating.from_json(original.to_json())
    assert restored.rater_id == original.rater_id
    assert restored.coherence_score == original.coherence_score
    assert restored.free_text_note == original.free_text_note
    assert len(restored.seen_motivating_scenes) == 1


def test_rating_no_note_roundtrip() -> None:
    from pipeline.interfaces.rating import Rating
    original = Rating(
        rater_id="r1", proposal_id="p1", arm="reasoning",
        coherence_score=5, usefulness_score=5,
        timestamp="2026-05-25T00:00:00Z",
    )
    restored = Rating.from_json(original.to_json())
    assert restored.free_text_note is None
    assert restored.seen_motivating_scenes == []


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

def _make_seeded_recall() -> dict:
    return {
        "reasoning": {
            "overall":     {"@10": 0.60, "@30": 0.82, "@all": 0.90},
            "familiar":    {"@10": 0.70, "@30": 0.90, "@all": 0.95},
            "unfamiliar":  {"@10": 0.50, "@30": 0.71, "@all": 0.80},
        }
    }


def test_differential_example_roundtrip() -> None:
    from pipeline.interfaces.report import DifferentialExample
    original = DifferentialExample(
        proposal_id="abc123",
        constituents=["conditions:fog", "time_of_day:night"],
        arm_scores={"reasoning": 0.88, "visual": 0.42},
        arm_ranks={"reasoning": 3, "visual": 47},
        coherence_ratings={"reasoning": 4.2, "visual": 3.1},
        usefulness_ratings={"reasoning": 3.9, "visual": 2.8},
    )
    restored = DifferentialExample.from_json(original.to_json())
    assert restored.proposal_id == original.proposal_id
    assert restored.constituents == original.constituents
    assert restored.arm_scores == original.arm_scores
    assert restored.arm_ranks == original.arm_ranks
    assert restored.coherence_ratings == original.coherence_ratings
    assert restored.usefulness_ratings == original.usefulness_ratings


def test_differential_example_empty_ratings_roundtrip() -> None:
    from pipeline.interfaces.report import DifferentialExample
    original = DifferentialExample(
        proposal_id="def456",
        constituents=["agents:pedestrian", "conditions:rain"],
        arm_scores={"reasoning": 0.75},
        arm_ranks={"reasoning": 12},
        coherence_ratings={},
        usefulness_ratings={},
    )
    restored = DifferentialExample.from_json(original.to_json())
    assert restored.coherence_ratings == {}
    assert restored.usefulness_ratings == {}


def test_evaluation_report_roundtrip() -> None:
    from pipeline.interfaces.report import DifferentialExample, EvaluationReport
    original = EvaluationReport(
        seeded_recall=_make_seeded_recall(),
        recall_k_primary=30,
        mean_coherence={"reasoning": 3.8},
        mean_usefulness={"reasoning": 3.5},
        coherence_ci_95={"reasoning": (3.5, 4.1)},
        usefulness_ci_95={"reasoning": (3.2, 3.8)},
        n_ratings_per_arm={"reasoning": 87},
        inter_rater_agreement_coherence=0.74,
        inter_rater_agreement_usefulness=0.68,
        n_raters_overlapping=4,
        differential_examples=[
            DifferentialExample(
                proposal_id="abc123",
                constituents=["conditions:fog", "time_of_day:night"],
                arm_scores={"reasoning": 0.88},
                arm_ranks={"reasoning": 3},
                coherence_ratings={"reasoning": 4.2},
                usefulness_ratings={"reasoning": 3.9},
            )
        ],
        failure_mode_distribution={"invalid_json": 2, "vocabulary_violation": 1},
        n_proposals_per_arm={"reasoning": 30},
        n_proposals_filtered={"reasoning": 5},
        n_raters=5,
        seeded_set_size={"familiar": 10, "unfamiliar": 10},
    )
    restored = EvaluationReport.from_json(original.to_json())
    assert restored.n_raters == original.n_raters
    assert restored.recall_k_primary == 30
    assert restored.seeded_recall["reasoning"]["overall"]["@30"] == 0.82
    assert restored.inter_rater_agreement_coherence == original.inter_rater_agreement_coherence
    assert restored.coherence_ci_95["reasoning"] == original.coherence_ci_95["reasoning"]
    assert restored.failure_mode_distribution == original.failure_mode_distribution
    assert len(restored.differential_examples) == 1
    assert restored.differential_examples[0].proposal_id == "abc123"
    assert restored.n_ratings_per_arm == {"reasoning": 87}


def test_evaluation_report_none_ci_and_agreement_roundtrip() -> None:
    """EvaluationReport with None inter-rater agreement and None CI (insufficient data)."""
    from pipeline.interfaces.report import EvaluationReport
    original = EvaluationReport(
        seeded_recall=_make_seeded_recall(),
        recall_k_primary=30,
        mean_coherence={"reasoning": 3.8},
        mean_usefulness={"reasoning": 3.5},
        coherence_ci_95={"reasoning": None},
        usefulness_ci_95={"reasoning": None},
        n_ratings_per_arm={"reasoning": 10},
        inter_rater_agreement_coherence=None,
        inter_rater_agreement_usefulness=None,
        n_raters_overlapping=1,
        differential_examples=[],
        failure_mode_distribution={},
        n_proposals_per_arm={"reasoning": 30},
        n_proposals_filtered={"reasoning": 5},
        n_raters=1,
        seeded_set_size={"familiar": 10, "unfamiliar": 10},
    )
    restored = EvaluationReport.from_json(original.to_json())
    assert restored.inter_rater_agreement_coherence is None
    assert restored.inter_rater_agreement_usefulness is None
    assert restored.coherence_ci_95["reasoning"] is None
    assert restored.differential_examples == []
