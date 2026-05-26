"""Integration test: Encoder (Module 2) → Hypothesizer (Module 3) boundary.

These tests exercise the real Module 3 Hypothesizer with output from the real
Module 2 Encoder (using StubVLMClient for offline testing). The boundary
crossing is: Encoder.process_batch → list[SchemaRecord] → Hypothesizer.propose.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.interfaces.proposal import CompositionProposal
from pipeline.interfaces.window import WindowKey
from pipeline.modules.encoder.encoder import Encoder
from pipeline.modules.encoder.reasoning_arm import StubVLMClient, VLMUnavailableError
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
from pipeline.modules.hypothesizer.config import HypothesizerConfig, HypothesizerEmptyInputError
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(pose_summary: str | None = "Ego vehicle at intersection."):
    manifest = MagicMock()
    manifest.pose_summary = pose_summary
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/seg.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_encoder(tmp_path: Path) -> Encoder:
    return Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)


def _make_window(segment_id: str, window_idx: int):
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(
        segment_id=segment_id,
        window_idx=window_idx,
        storage=_make_storage(),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_encoder_batch_to_hypothesizer(tmp_path: Path) -> None:
    """Encoder batch output flows directly into Hypothesizer; proposals produced."""
    enc = _make_encoder(tmp_path)
    windows = [_make_window("integ_seg", i) for i in range(20)]
    records = enc.process_batch(windows)
    assert all(r.succeeded for r in records)

    hyp = Hypothesizer(HypothesizerConfig(
        min_marginal_frequency=0.0,
        max_joint_frequency=1.1,  # don't filter on joint for this test
        min_pairwise_frequency=0.0,
        composition_sizes=[2],
        top_k=5,
    ))
    proposals = hyp.propose(records, arm="reasoning")
    assert isinstance(proposals, list)
    for p in proposals:
        assert isinstance(p, CompositionProposal)


def test_window_keys_survive_boundary(tmp_path: Path) -> None:
    """WindowKey from encoder record is correctly passed through as motivating scene."""
    enc = _make_encoder(tmp_path)
    records = enc.process_batch([_make_window("boundary_seg", i) for i in range(20)])

    hyp = Hypothesizer(HypothesizerConfig(
        min_marginal_frequency=0.0,
        max_joint_frequency=1.1,
        min_pairwise_frequency=0.0,
        composition_sizes=[2],
        top_k=5,
    ))
    proposals = hyp.propose(records, arm="reasoning")
    for p in proposals:
        for wk in p.motivating_scene_ids:
            assert isinstance(wk, WindowKey)
            assert wk.segment_id == "boundary_seg"


def test_failed_records_skipped_by_hypothesizer(tmp_path: Path) -> None:
    """All-failure batch raises HypothesizerEmptyInputError."""
    class DeadVLM:
        model_id = "stub/dead"
        def complete(self, video_url: str, prompt: str) -> str:
            raise VLMUnavailableError("stub/dead", "offline")

    enc = Encoder(vlm=DeadVLM(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    records = enc.process_batch([_make_window("dead_seg", i) for i in range(5)])
    assert all(not r.succeeded for r in records)

    hyp = Hypothesizer()
    with pytest.raises(HypothesizerEmptyInputError):
        hyp.propose(records, arm="reasoning")


def test_json_serialization_survives_boundary(tmp_path: Path) -> None:
    """Proposals are JSON-serializable and round-trip correctly at the boundary."""
    enc = _make_encoder(tmp_path)
    records = enc.process_batch([_make_window("serial_seg", i) for i in range(20)])

    hyp = Hypothesizer(HypothesizerConfig(
        min_marginal_frequency=0.0,
        max_joint_frequency=1.1,
        min_pairwise_frequency=0.0,
        composition_sizes=[2],
        top_k=5,
    ))
    proposals = hyp.propose(records, arm="reasoning")
    for p in proposals:
        wire = p.to_json()
        restored = CompositionProposal.from_json(wire)
        assert restored.composition_id == p.composition_id
        assert restored.constituents == p.constituents
        assert restored.arm == p.arm
