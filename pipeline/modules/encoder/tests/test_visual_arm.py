"""Tests for Module 2: Encoder visual arm (Cosmos-Embed1).

All tests run without real network calls — StubEmbedClient is used throughout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> Any:
    manifest = MagicMock()
    manifest.pose_summary = "Vehicle traveled 10m."
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/video.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_window(segment_id: str = "seg_001", window_idx: int = 0):
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(segment_id=segment_id, window_idx=window_idx, storage=_make_storage())


def _make_encoder_with_visual(tmp_path: Path):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, VisualArm
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    visual = VisualArm(client=StubEmbedClient())
    return Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=visual,
    )


# ---------------------------------------------------------------------------
# EmbedClient protocol
# ---------------------------------------------------------------------------

def test_stub_embed_client_satisfies_protocol() -> None:
    from pipeline.modules.encoder.visual_arm import EmbedClient, StubEmbedClient
    stub = StubEmbedClient()
    assert isinstance(stub, EmbedClient)


def test_stub_embed_client_returns_correct_shape() -> None:
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, _EMBED_DIM_PER_CAMERA
    stub = StubEmbedClient()
    cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    embedding, used = stub.embed("seg_001", 0, _make_storage(), cameras)
    assert len(embedding) == len(cameras) * _EMBED_DIM_PER_CAMERA
    assert used == cameras
    assert all(v == 0.0 for v in embedding)


def test_stub_embed_client_single_camera() -> None:
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, _EMBED_DIM_PER_CAMERA
    stub = StubEmbedClient()
    embedding, used = stub.embed("seg_001", 0, _make_storage(), ["FRONT"])
    assert len(embedding) == _EMBED_DIM_PER_CAMERA
    assert used == ["FRONT"]


# ---------------------------------------------------------------------------
# VisualArm
# ---------------------------------------------------------------------------

def test_visual_arm_annotate_from_storage_returns_fields() -> None:
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, VisualArm
    arm = VisualArm(client=StubEmbedClient())
    fields, extra = arm.annotate_from_storage("seg_001", 0, _make_storage())
    assert extra is None
    assert "embedding" in fields
    assert "cameras" in fields
    assert "model_id" in fields
    assert isinstance(fields["embedding"], list)
    assert len(fields["embedding"]) > 0


def test_visual_arm_model_id_comes_from_client() -> None:
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, VisualArm
    stub = StubEmbedClient()
    arm = VisualArm(client=stub)
    assert arm.model_id == stub.model_id


# ---------------------------------------------------------------------------
# Encoder with visual arm configured
# ---------------------------------------------------------------------------

def test_encoder_process_with_visual_arm_returns_two_records(tmp_path: Path) -> None:
    enc = _make_encoder_with_visual(tmp_path)
    records = enc.process(_make_window())
    assert len(records) == 2
    arms = {r.arm for r in records}
    assert arms == {"reasoning", "visual"}


def test_encoder_reasoning_record_is_first(tmp_path: Path) -> None:
    enc = _make_encoder_with_visual(tmp_path)
    records = enc.process(_make_window())
    assert records[0].arm == "reasoning"
    assert records[1].arm == "visual"


def test_encoder_visual_record_has_embedding_field(tmp_path: Path) -> None:
    enc = _make_encoder_with_visual(tmp_path)
    records = enc.process(_make_window())
    visual = next(r for r in records if r.arm == "visual")
    assert visual.succeeded
    assert "embedding" in visual.fields
    assert isinstance(visual.fields["embedding"], list)


def test_encoder_visual_arm_failure_does_not_affect_reasoning(tmp_path: Path) -> None:
    """If the visual arm fails, the reasoning record must still succeed."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.visual_arm import EmbedClient, VisualArm
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    class FailingEmbedClient:
        model_id = "stub/failing-embed"
        def embed(self, segment_id, window_idx, storage, cameras):
            raise RuntimeError("embed endpoint down")

    visual = VisualArm(client=FailingEmbedClient())  # type: ignore[arg-type]
    enc = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=visual,
    )
    records = enc.process(_make_window())
    assert len(records) == 2

    reasoning = next(r for r in records if r.arm == "reasoning")
    visual_rec = next(r for r in records if r.arm == "visual")

    assert reasoning.succeeded, "Reasoning arm must succeed independently"
    assert not visual_rec.succeeded, "Visual arm must record its failure"
    assert visual_rec.failure_mode is not None


def test_encoder_visual_storage_error_classified_as_storage_error(tmp_path: Path) -> None:
    """A WindowStorageError from the visual arm maps to storage_error, not vlm_unavailable."""
    from pipeline.interfaces.errors import WindowStorageError
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.visual_arm import VisualArm
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    class StorageFailingEmbedClient:
        model_id = "stub/storage-fail"
        def embed(self, segment_id, window_idx, storage, cameras):
            raise WindowStorageError(f"{segment_id}/{window_idx:04d}", "blob not found")

    visual = VisualArm(client=StorageFailingEmbedClient())  # type: ignore[arg-type]
    enc = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=visual,
    )
    records = enc.process(_make_window())

    visual_rec = next(r for r in records if r.arm == "visual")
    reasoning = next(r for r in records if r.arm == "reasoning")
    assert visual_rec.failure_mode == "storage_error"
    assert reasoning.succeeded


def test_encoder_visual_cache_separate_from_reasoning(tmp_path: Path) -> None:
    """Reasoning and visual arm cache files must live in separate directories."""
    enc = _make_encoder_with_visual(tmp_path)
    enc.process(_make_window())

    reasoning_dir = tmp_path / "encoder" / "reasoning"
    visual_dir = tmp_path / "encoder" / "visual"

    assert len(list(reasoning_dir.glob("*.json"))) == 1
    assert len(list(visual_dir.glob("*.json"))) == 1

    # Keys must be different (different arm prefix in hash input)
    reasoning_key = list(reasoning_dir.glob("*.json"))[0].stem
    visual_key = list(visual_dir.glob("*.json"))[0].stem
    assert reasoning_key != visual_key


def test_encoder_process_batch_with_visual_arm_returns_2n_records(tmp_path: Path) -> None:
    enc = _make_encoder_with_visual(tmp_path)
    windows = [_make_window(window_idx=i) for i in range(4)]
    records = enc.process_batch(windows)
    assert len(records) == 8  # 2 arms × 4 windows
    reasoning = [r for r in records if r.arm == "reasoning"]
    visual = [r for r in records if r.arm == "visual"]
    assert len(reasoning) == 4
    assert len(visual) == 4


def test_encoder_without_visual_arm_returns_one_record(tmp_path: Path) -> None:
    """Default (no visual_arm) must still return a single-element list."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    enc = Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    records = enc.process(_make_window())
    assert len(records) == 1
    assert records[0].arm == "reasoning"
