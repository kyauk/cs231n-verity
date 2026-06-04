"""Round-trip serialization tests for DevRoundManifest.

If this fails, the dev_dashboard export endpoint and the round-manifest
filesystem layout will silently mismatch — a category-killer for the
discrimination test, since source labels must round-trip cleanly.
"""

from __future__ import annotations

import json

from pipeline.interfaces.dev_round import DevRoundManifest
from pipeline.interfaces.window import WindowKey


def _make_manifest() -> DevRoundManifest:
    return DevRoundManifest(
        round_id="round_2026-05-30T22-15-30Z",
        created_at="2026-05-30T22:15:30Z",
        dataset_label="waymo_val_split_1",
        pool_size=30,
        seed=42,
        pools={
            "verity": [WindowKey(segment_id="seg_a", window_idx=0),
                       WindowKey(segment_id="seg_a", window_idx=1)],
            "random": [WindowKey(segment_id="seg_b", window_idx=0),
                       WindowKey(segment_id="seg_b", window_idx=1)],
            "naive_rare": [WindowKey(segment_id="seg_c", window_idx=0),
                           WindowKey(segment_id="seg_c", window_idx=1)],
        },
        shuffled_order=[
            WindowKey(segment_id="seg_b", window_idx=0),
            WindowKey(segment_id="seg_a", window_idx=0),
            WindowKey(segment_id="seg_c", window_idx=0),
            WindowKey(segment_id="seg_a", window_idx=1),
            WindowKey(segment_id="seg_c", window_idx=1),
            WindowKey(segment_id="seg_b", window_idx=1),
        ],
        naive_rare_atoms=[
            "weather:fog", "conditions:icy_road", "traffic_control:yield",
            "agents:emergency_vehicle", "road_geometry:roundabout",
        ],
    )


def test_dev_round_manifest_roundtrip() -> None:
    original = _make_manifest()
    restored = DevRoundManifest.from_json(original.to_json())
    assert restored == original


def test_dev_round_manifest_json_serializable() -> None:
    """to_json() must produce a value json.dumps can handle."""
    m = _make_manifest()
    serialized = json.dumps(m.to_json())  # must not raise
    assert "round_2026-05-30T22-15-30Z" in serialized


def test_dev_round_manifest_pool_keys_preserved() -> None:
    """The three pool labels are load-bearing for blinding — they must survive."""
    original = _make_manifest()
    restored = DevRoundManifest.from_json(original.to_json())
    assert set(restored.pools.keys()) == {"verity", "random", "naive_rare"}


def test_dev_round_manifest_shuffled_order_preserved() -> None:
    """The presentation order is the blinding mechanism. Permutation must
    survive serialization — otherwise the analyst can't join ratings to
    source pools."""
    original = _make_manifest()
    restored = DevRoundManifest.from_json(original.to_json())
    assert restored.shuffled_order == original.shuffled_order


def test_dev_round_manifest_handles_empty_naive_rare_atoms() -> None:
    """default_factory=list means naive_rare_atoms defaults to []."""
    m = DevRoundManifest(
        round_id="r1", created_at="2026-05-30T22:15:30Z",
        dataset_label="ds", pool_size=30, seed=0,
        pools={"verity": [], "random": [], "naive_rare": []},
        shuffled_order=[],
    )
    assert m.naive_rare_atoms == []
    restored = DevRoundManifest.from_json(m.to_json())
    assert restored == m


def test_dev_round_manifest_seed_round_trips() -> None:
    """The seed must round-trip for reproducibility of rounds."""
    original = _make_manifest()
    restored = DevRoundManifest.from_json(original.to_json())
    assert restored.seed == 42 == original.seed
