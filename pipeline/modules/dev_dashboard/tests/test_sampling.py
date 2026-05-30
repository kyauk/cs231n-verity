"""Unit tests for the three-pool sampler."""

from __future__ import annotations

import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.dev_dashboard import (
    SampleResult,
    SamplingError,
    sample_three_pools,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_record(i: int, weather: str = "clear", agents: str = "car",
                 succeeded: bool = True) -> SchemaRecord:
    """Build one SchemaRecord with controllable atoms."""
    return SchemaRecord(
        window_id=WindowKey(segment_id=f"seg_{i:04d}", window_idx=0),
        arm="reasoning", schema_version="1.0",
        prompt_template_id="v1_describe",
        fields={
            "agents": [agents],
            "environment": {"weather": weather, "time_of_day": "day",
                            "lighting_condition": "well_lit"},
            "road": {"geometry": "straight", "lane_count": 2},
            "traffic_control": "none", "ego_task": "cruising",
            "conditions": [],
        } if succeeded else {},
        failure_mode=None if succeeded else "invalid_json",
    )


def _make_scored(i: int, *, rank: float, accepted: bool = True,
                 motivating: int = 1) -> ScoredProposal:
    return ScoredProposal(
        composition_id=f"comp_{i:04d}",
        constituents=["agents:car", "weather:clear"],
        marginal_frequencies={}, pairwise_frequencies={},
        expected_joint=0.0, observed_joint=0.0, novelty_score=1.0,
        motivating_scene_ids=[
            WindowKey(segment_id=f"verity_seg_{i:04d}", window_idx=j)
            for j in range(motivating)
        ],
        arm="reasoning",
        plausibility_score=0.8, plausibility_justification="",
        frontier_difficulty_score=None, frontier_difficulty_signals={},
        final_rank_score=rank,
        accepted=accepted,
        rejection_reason=None if accepted else "plausibility_below_threshold",
    )


def _make_dataset(n_records: int = 100, n_proposals: int = 50):
    # 90% common (clear/car), 10% rare (fog/pedestrian)
    records = []
    for i in range(n_records):
        if i < n_records // 10:
            records.append(_make_record(i, weather="fog", agents="pedestrian"))
        else:
            records.append(_make_record(i))
    # Accepted proposals with descending ranks
    proposals = [
        _make_scored(i, rank=float(n_proposals - i), accepted=True)
        for i in range(n_proposals)
    ]
    return records, proposals


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_three_pools_of_requested_size() -> None:
    records, proposals = _make_dataset(n_records=100, n_proposals=50)
    result = sample_three_pools(proposals, records, pool_size=30, seed=42)
    assert isinstance(result, SampleResult)
    assert len(result.verity) == 30
    assert len(result.random) == 30
    assert len(result.naive_rare) == 30


def test_verity_pool_is_top_k_by_final_rank_score() -> None:
    """Verity pool must be the top-pool_size accepted proposals' first
    motivating scene, in descending rank order."""
    records, proposals = _make_dataset(n_records=100, n_proposals=50)
    result = sample_three_pools(proposals, records, pool_size=10, seed=0)
    # Highest-ranked proposal had final_rank_score=50 (i=0), motivating
    # scene "verity_seg_0000". Lowest of top-10 was i=9, "verity_seg_0009".
    expected = [
        WindowKey(segment_id=f"verity_seg_{i:04d}", window_idx=0)
        for i in range(10)
    ]
    assert result.verity == expected


def test_naive_rare_atoms_returned() -> None:
    """The rare atoms actually used must be exposed (for the manifest)."""
    records, proposals = _make_dataset()
    result = sample_three_pools(proposals, records, pool_size=10, seed=0)
    assert isinstance(result.naive_rare_atoms, list)
    assert 1 <= len(result.naive_rare_atoms) <= 5
    # In our fixture, fog + pedestrian appear in only 10% of records — they
    # should be among the rarest atoms surfaced.
    rare_str = " ".join(result.naive_rare_atoms)
    assert "fog" in rare_str or "pedestrian" in rare_str


def test_naive_rare_pool_only_contains_windows_with_a_rare_atom() -> None:
    """Every window in the naive_rare pool must contain at least one of
    the surfaced rare atoms — that's the baseline's defining property.

    Uses top_k_rare_atoms=2 to ensure only the genuinely rare atoms
    (weather:fog, agents:pedestrian, each at 10% marginal) get selected.
    With top_k=5 in this fixture the rare set would expand to include
    ubiquitous atoms (e.g. time_of_day:day at 100% marginal) since they
    tie at the top of "lowest marginal among all atoms," which is real
    behavior worth keeping but not what this property test is checking.
    """
    records, proposals = _make_dataset()
    result = sample_three_pools(
        proposals, records, pool_size=10, seed=0, top_k_rare_atoms=2,
    )
    rare_window_ids = {
        r.window_id for r in records
        if r.fields.get("environment", {}).get("weather") == "fog"
        or "pedestrian" in r.fields.get("agents", [])
    }
    for w in result.naive_rare:
        assert w in rare_window_ids


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_pools() -> None:
    records, proposals = _make_dataset()
    a = sample_three_pools(proposals, records, pool_size=20, seed=123)
    b = sample_three_pools(proposals, records, pool_size=20, seed=123)
    assert a.verity == b.verity
    assert a.random == b.random
    assert a.naive_rare == b.naive_rare
    assert a.naive_rare_atoms == b.naive_rare_atoms


def test_different_seed_produces_different_random_pool() -> None:
    """Verity is deterministic (sort by score); Random must vary with seed."""
    records, proposals = _make_dataset()
    a = sample_three_pools(proposals, records, pool_size=20, seed=1)
    b = sample_three_pools(proposals, records, pool_size=20, seed=2)
    assert a.verity == b.verity  # deterministic in score
    assert a.random != b.random  # randomness keyed off seed


def test_verity_pool_breaks_ties_by_composition_id_not_list_order() -> None:
    """Pessimistic-review fix: ties on final_rank_score must resolve
    deterministically on composition_id, not on the operator's
    arbitrarily-ordered scored.json. Two runs over scrambled inputs with
    identical tied scores must produce identical Verity pools."""
    records, _ = _make_dataset(n_records=100)
    # 50 proposals all tied at score=10.0 — same score, different IDs
    tied_a = [_make_scored(i, rank=10.0, accepted=True) for i in range(50)]
    tied_b = list(reversed(tied_a))
    a = sample_three_pools(tied_a, records, pool_size=10, seed=0)
    b = sample_three_pools(tied_b, records, pool_size=10, seed=0)
    assert a.verity == b.verity, (
        "Verity pool changed when scored.json was re-ordered — "
        "tiebreaker on composition_id is broken."
    )


# ---------------------------------------------------------------------------
# Failure modes (each one has its own clear error)
# ---------------------------------------------------------------------------

def test_raises_if_pool_size_is_zero() -> None:
    with pytest.raises(SamplingError, match="pool_size"):
        sample_three_pools([], [], pool_size=0)


def test_raises_if_too_few_accepted_proposals() -> None:
    records, _ = _make_dataset(n_records=100)
    too_few = [_make_scored(i, rank=1.0) for i in range(5)]
    with pytest.raises(SamplingError, match="Verity pool"):
        sample_three_pools(too_few, records, pool_size=30, seed=0)


def test_raises_if_accepted_proposal_has_no_motivating_scenes() -> None:
    """Proposals with motivating_scene_ids=[] cannot contribute to Verity."""
    records, _ = _make_dataset(n_records=100)
    # 30 accepted proposals but ALL have zero motivating scenes
    starved = [
        _make_scored(i, rank=1.0, accepted=True, motivating=0)
        for i in range(30)
    ]
    with pytest.raises(SamplingError, match="Verity pool"):
        sample_three_pools(starved, records, pool_size=30, seed=0)


def test_raises_if_too_few_succeeded_records() -> None:
    """Random + Naive-rare both need succeeded records."""
    _, proposals = _make_dataset(n_proposals=50)
    too_few = [_make_record(i) for i in range(5)]
    with pytest.raises(SamplingError, match="Random pool"):
        sample_three_pools(proposals, too_few, pool_size=30, seed=0)


def test_raises_if_too_few_rare_windows() -> None:
    """If the rarest-K atoms together don't cover pool_size windows."""
    _, proposals = _make_dataset(n_proposals=50)
    # 100 records, all identical → only one set of atoms, all marginal=1.0
    # No "rare" atoms separate from common ones.
    # Build a dataset where even the top-5 rarest atoms only touch < 30 windows.
    records = [_make_record(i, weather="clear", agents="car") for i in range(100)]
    # All records have the exact same atom set → naive_rare candidate pool
    # = all 100 records, so this won't fail. Force a smaller candidate pool:
    records = records[:50] + [_make_record(i + 50, succeeded=False) for i in range(10)]
    # Make only 20 records with a rare atom present
    rare_records = [_make_record(i, weather="fog") for i in range(60, 80)]
    records = records[:50] + rare_records  # 50 common + 20 with fog
    # Naive-rare with top_k=1 will only pull from the 20 fog records.
    with pytest.raises(SamplingError, match="Naive-rare pool"):
        sample_three_pools(
            proposals, records, pool_size=30, seed=0, top_k_rare_atoms=1,
        )


# ---------------------------------------------------------------------------
# Pool independence (does not silently dedupe across pools)
# ---------------------------------------------------------------------------

def test_pools_may_overlap_without_error() -> None:
    """Same window can appear in multiple pools. The discrimination test
    is about pool-vs-pool means; double-rating is acceptable signal."""
    records, proposals = _make_dataset()
    result = sample_three_pools(proposals, records, pool_size=20, seed=0)
    # Just confirm no error and pools are full size — overlap is implicit.
    assert len(result.verity) == 20
    assert len(result.random) == 20
    assert len(result.naive_rare) == 20
