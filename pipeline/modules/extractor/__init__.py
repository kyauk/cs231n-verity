"""Module: Extractor — reason-first producer of immutable RawDescriptor evidence.

clip -> free-form reasoning -> typed descriptors (with span pointers) -> embeddings.
The only place the VLM is touched on the taxonomy path. Decides nothing about
labels (that's the curator, firewalled away); it only produces evidence.

Imports only pipeline.interfaces. Production clients are self-contained (openai /
google / requests), so no cross-module reach.

    from pipeline.modules.extractor import (
        Extractor, ExtractorConfig,
        CosmosReasonClient, NIMStructureClient, NIMEmbedder,   # production
        StubReasonClient, StubStructureClient, StubEmbedder,   # offline/tests
    )
"""

from pipeline.modules.extractor.clients import (
    CosmosReasonClient,
    Embedder,
    NIMEmbedder,
    NIMStructureClient,
    ReasonClient,
    StructureClient,
    StubEmbedder,
    StubReasonClient,
    StubStructureClient,
)
from pipeline.modules.extractor.config import (
    ExtractorConfig,
    ExtractorError,
    ReasoningUnavailableError,
    StructuringError,
)
from pipeline.modules.extractor.extractor import Extractor

__all__ = [
    "Extractor",
    "ExtractorConfig",
    "ExtractorError",
    "ReasoningUnavailableError",
    "StructuringError",
    "ReasonClient",
    "StructureClient",
    "Embedder",
    "CosmosReasonClient",
    "NIMStructureClient",
    "NIMEmbedder",
    "StubReasonClient",
    "StubStructureClient",
    "StubEmbedder",
]
