"""Extractor configuration and errors.

The extractor is the RawDescriptor *producer*: clip -> free-form reasoning ->
typed descriptors (each with a span pointer back to the reasoning) -> embeddings.
It is the only place the VLM is touched on this path; everything downstream
(curator, hypothesizer) consumes its immutable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.interfaces.taxonomy import DEFAULT_AXES


@dataclass(frozen=True)
class ExtractorConfig:
    axes: frozenset[str] = DEFAULT_AXES
    reason_prompt_id: str = "v1_reason"
    structure_prompt_id: str = "v1_structure"
    max_descriptors_per_scene: int = 32   # guard against runaway structuring output
    require_span: bool = True             # drop descriptors whose span isn't in the reasoning


class ExtractorError(Exception):
    """Base class for extractor failures."""


class ReasoningUnavailableError(ExtractorError):
    """The reasoning VLM could not be reached."""


class StructuringError(ExtractorError):
    """The structuring pass produced no parseable typed descriptors."""
