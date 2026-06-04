"""Extractor tests — offline (stubs), pinning the evidence contract."""

from __future__ import annotations

import re

import pytest

from pipeline.interfaces.taxonomy import DEFAULT_AXES, RawDescriptor
from pipeline.modules.extractor import (
    Extractor,
    ExtractorConfig,
    StructuringError,
    StubEmbedder,
    StubReasonClient,
    StubStructureClient,
)


def _extractor(**cfg) -> Extractor:
    return Extractor(StubReasonClient(), StubStructureClient(), StubEmbedder(dim=16),
                     ExtractorConfig(**cfg))


def test_extract_produces_immutable_typed_evidence():
    descs = _extractor().extract("scene-1", "gs://bucket/scene-1.mp4")
    assert descs and all(isinstance(d, RawDescriptor) for d in descs)
    # every descriptor is typed to a known axis and carries an embedding + scene id
    for d in descs:
        assert d.axis in DEFAULT_AXES
        assert d.scene_id == "scene-1"
        assert len(d.embedding) == 16
        assert d.descriptor_id  # content-hash id assigned


def test_span_pointers_point_into_the_reasoning():
    # The fool-proofing: each structured atom is auditable to its source sentence.
    reasoning = StubReasonClient().describe("x", "p")
    descs = _extractor(require_span=True).extract("s", "ref")
    for d in descs:
        # span is a verbatim slice of the reasoning, or the closest sentence in it
        assert d.reasoning_span
        assert d.reasoning_span in reasoning or any(
            d.reasoning_span.strip() == s.strip()
            for s in re.split(r"(?<=[.!?])\s+", reasoning)
        )


def test_unknown_axes_are_dropped():
    class BadStructure:
        def structure(self, reasoning: str, prompt: str) -> str:
            import json
            return json.dumps({"descriptors": [
                {"axis": "made_up_axis", "text": "nope", "span": "x"},
                {"axis": "agents", "text": "car", "span": StubReasonClient().describe("", "")[:20]},
            ]})
    ex = Extractor(StubReasonClient(), BadStructure(), StubEmbedder(dim=8), ExtractorConfig())
    descs = ex.extract("s", "ref")
    assert {d.axis for d in descs} == {"agents"}   # the bogus axis is filtered out


def test_no_descriptors_raises():
    class Empty:
        def structure(self, reasoning: str, prompt: str) -> str:
            return '{"descriptors": []}'
    ex = Extractor(StubReasonClient(), Empty(), StubEmbedder(), ExtractorConfig())
    with pytest.raises(StructuringError):
        ex.extract("s", "ref")


def test_determinism_same_clip_same_evidence():
    # Same clip -> same evidence CONTENT (ids/axis/text/span/embedding). created_at
    # is wall-clock provenance and is intentionally excluded from the content id.
    def content(d):
        return (d.descriptor_id, d.axis, d.text, d.reasoning_span, d.embedding)
    a = _extractor().extract("s", "ref")
    b = _extractor().extract("s", "ref")
    assert [content(d) for d in a] == [content(d) for d in b]
