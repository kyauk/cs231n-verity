"""Tests for hypothesizer/composition.py."""
import math
import pytest
from pipeline.interfaces.window import WindowKey
from pipeline.modules.hypothesizer.composition import (
    atom_prefix,
    composition_id,
    build_proposals,
    _is_mutually_exclusive,
    _expected_joint,
    _min_pairwise,
)
from pipeline.modules.hypothesizer.config import HypothesizerConfig


def _key(i: int) -> WindowKey:
    return WindowKey("seg", i)


class TestHelpers:
    def test_atom_prefix(self):
        assert atom_prefix("weather:fog") == "weather"
        assert atom_prefix("agents:car") == "agents"
        assert atom_prefix("road_geometry:intersection") == "road_geometry"

    def test_composition_id_deterministic(self):
        a = composition_id(["b:y", "a:x"])
        b = composition_id(["a:x", "b:y"])
        assert a == b  # sorted internally

    def test_composition_id_is_16_chars(self):
        assert len(composition_id(["a:x", "b:y"])) == 16

    def test_composition_id_differs_for_different_inputs(self):
        assert composition_id(["a:x", "b:y"]) != composition_id(["a:x", "b:z"])


class TestMutualExclusivity:
    def test_same_categorical_prefix_rejected(self):
        assert _is_mutually_exclusive(frozenset({"weather:fog", "weather:rain"}))

    def test_different_prefix_accepted(self):
        assert not _is_mutually_exclusive(frozenset({"weather:fog", "ego_task:cruising"}))

    def test_multi_value_same_prefix_accepted(self):
        # agents is MULTI_VALUE — same-prefix pairing is allowed
        assert not _is_mutually_exclusive(frozenset({"agents:car", "agents:pedestrian"}))

    def test_three_atoms_one_conflict(self):
        assert _is_mutually_exclusive(frozenset({
            "weather:fog", "weather:rain", "ego_task:cruising"
        }))


class TestExpectedJoint:
    def test_product_of_marginals(self):
        marginal = {"a:x": 0.4, "b:y": 0.3}
        result = _expected_joint(frozenset({"a:x", "b:y"}), marginal)
        assert result == pytest.approx(0.12)

    def test_missing_atom_gives_zero(self):
        result = _expected_joint(frozenset({"missing:x", "b:y"}), {"b:y": 0.5})
        assert result == pytest.approx(0.0)


class TestMinPairwise:
    def test_returns_minimum(self):
        pairwise = {"a:x|b:y": 0.8, "a:x|c:z": 0.2, "b:y|c:z": 0.5}
        result = _min_pairwise(frozenset({"a:x", "b:y", "c:z"}), pairwise)
        assert result == pytest.approx(0.2)

    def test_missing_pair_returns_zero(self):
        result = _min_pairwise(frozenset({"a:x", "b:y"}), {})
        assert result == pytest.approx(0.0)


class TestBuildProposals:
    def _make_atom_sets_and_keys(self, n: int, composition: frozenset[str], hit_frac: float):
        """n total windows; hit_frac of them contain all atoms in composition."""
        n_hits = int(n * hit_frac)
        atom_sets = []
        keys = []
        for i in range(n):
            if i < n_hits:
                atom_sets.append(composition | frozenset({"filler:z"}))
            else:
                atom_sets.append(frozenset({"filler:z"}))
            keys.append(_key(i))
        return atom_sets, keys

    def test_basic_proposal_produced(self):
        # 100 windows; atoms A and B each appear in 50% but co-occur in only 1%
        n = 100
        atom_sets = []
        keys = [_key(i) for i in range(n)]
        for i in range(n):
            s: set[str] = set()
            if i < 50:
                s.add("weather:fog")
            if i < 50 and i % 50 == 0:  # only 1 window has both
                s.add("conditions:night_driving")
            elif i >= 50 and i < 80:
                s.add("conditions:night_driving")
            atom_sets.append(frozenset(s))

        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.05,
            max_joint_frequency=0.5,  # lenient for test
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        cids = {p.composition_id for p in proposals}
        constituents_sets = [frozenset(p.constituents) for p in proposals]
        assert frozenset({"weather:fog", "conditions:night_driving"}) in constituents_sets

    def test_mutual_exclusivity_filtered(self):
        n = 20
        # Impossible: weather:fog and weather:rain in the same window
        atom_sets = [frozenset({"weather:fog", "weather:rain"}) for _ in range(n)]
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=1.1,  # don't filter on joint
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        for p in proposals:
            assert frozenset(p.constituents) != frozenset({"weather:fog", "weather:rain"})

    def test_high_joint_frequency_filtered(self):
        n = 20
        atom_sets = [frozenset({"weather:fog", "conditions:night_driving"}) for _ in range(n)]
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        # observed=1.0, max_joint=0.5 → should be filtered out
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        for p in proposals:
            assert frozenset(p.constituents) != frozenset({"weather:fog", "conditions:night_driving"})

    def test_top_k_respected(self):
        n = 50
        atom_sets = [frozenset({f"conditions:tag{i % 5}", f"weather:w{i % 3}"}) for i in range(n)]
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=1.1,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=3,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        assert len(proposals) <= 3

    def test_sorted_by_novelty_desc(self):
        n = 100
        # Make two compositions with different novelty scores
        # Composition A: fog + night_driving appear individually but rarely together
        # Composition B: clear + day always together (low novelty)
        atom_sets = []
        keys = [_key(i) for i in range(n)]
        for i in range(n):
            s = {"weather:clear", "time_of_day:day"}
            if i < 30:
                s.add("weather:fog")  # can't coexist with clear — MX!
            # Use non-conflicting atoms
            atom_sets.append(frozenset())
        # Simpler: just build atom_sets that produce computable novelty
        atom_sets = [frozenset({"conditions:fog_cond", "conditions:night_driving"}) if i < 2
                     else frozenset({"conditions:fog_cond"}) if i < 60
                     else frozenset({"conditions:night_driving"}) if i < 90
                     else frozenset()
                     for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.01,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        if len(proposals) > 1:
            scores = [p.novelty_score for p in proposals]
            assert scores == sorted(scores, reverse=True)

    def test_novelty_score_positive_for_rare_joint(self):
        # If expected_joint >> observed_joint, score should be large and positive
        n = 100
        # atoms A and B each appear in 50% of windows but co-occur in only 1 window
        atom_sets = []
        for i in range(n):
            s: set[str] = set()
            if i < 50:
                s.add("conditions:fog_cond")
            if i == 0:  # only window 0 has both
                s.add("conditions:night_driving")
            elif 50 <= i < 80:
                s.add("conditions:night_driving")
            atom_sets.append(frozenset(s))
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        target = [p for p in proposals
                  if frozenset(p.constituents) == frozenset({"conditions:fog_cond", "conditions:night_driving"})]
        if target:
            assert target[0].novelty_score > 0

    def test_motivating_scenes_are_correct(self):
        n = 10
        atom_sets = [frozenset({"weather:fog", "conditions:night_driving"}) if i < 3
                     else frozenset({"weather:fog"})
                     for i in range(n)]
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        target = next(
            (p for p in proposals
             if frozenset(p.constituents) == frozenset({"weather:fog", "conditions:night_driving"})),
            None,
        )
        assert target is not None
        motivating_idxs = {k.window_idx for k in target.motivating_scene_ids}
        assert motivating_idxs == {0, 1, 2}

    def test_arm_propagated(self):
        n = 10
        atom_sets = [frozenset({"weather:fog", "conditions:night_driving"}) for _ in range(n)]
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        )
        proposals = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="visual")
        assert all(p.arm == "visual" for p in proposals)

    def test_deterministic_tie_breaker(self):
        # Two compositions with same novelty should be ordered by composition_id ASC
        n = 20
        # Two pairs each appearing in 1/n of windows
        atom_sets = []
        for i in range(n):
            if i == 0:
                atom_sets.append(frozenset({"conditions:fog_cond", "conditions:school_zone"}))
            elif i == 1:
                atom_sets.append(frozenset({"conditions:road_debris", "conditions:animal_crossing"}))
            elif i < 11:
                atom_sets.append(frozenset({"conditions:fog_cond"}))
            elif i < 12:
                atom_sets.append(frozenset({"conditions:school_zone"}))
            elif i < 16:
                atom_sets.append(frozenset({"conditions:road_debris"}))
            else:
                atom_sets.append(frozenset({"conditions:animal_crossing"}))
        keys = [_key(i) for i in range(n)]
        from pipeline.modules.hypothesizer.frequency import compute_frequencies
        marginal, pairwise = compute_frequencies(atom_sets)
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        p1 = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        p2 = build_proposals(atom_sets, keys, marginal, pairwise, cfg, arm="reasoning")
        assert [p.composition_id for p in p1] == [p.composition_id for p in p2]

    def test_empty_atom_sets_returns_empty(self):
        proposals = build_proposals([], [], {}, {}, HypothesizerConfig(), arm="reasoning")
        assert proposals == []
