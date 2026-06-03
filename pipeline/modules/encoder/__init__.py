"""Module 2: Encoder — VLM annotation with caching.

Annotates each window with the reasoning arm (Cosmos-Reason2 → structured scene
JSON) and caches the result. The visual arm (Cosmos-Embed1 embeddings) was
removed in v1 because it produced output the categorical Hypothesizer could
not consume; if/when v2 brings continuous-space discovery online, the embed
infrastructure comes back as its own discovery channel, not as a parallel
encoder arm.

Public surface (import from the package root):
    from pipeline.modules.encoder import Encoder, CosmosReason2Client

The Stub* client returns valid output with no network call, for tests and
offline runs.
"""

from pipeline.modules.encoder.encoder import Encoder
from pipeline.modules.encoder.reasoning_arm import (
    CosmosReason2Client,
    ReasoningArm,
    StubVLMClient,
    VLMClient,
)
from pipeline.modules.encoder.schema import WindowInput
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY, Vocabulary

__all__ = [
    "Encoder",
    "WindowInput",
    "Vocabulary",
    "DEFAULT_VOCABULARY",
    # Reasoning arm
    "ReasoningArm",
    "VLMClient",
    "CosmosReason2Client",
    "StubVLMClient",
]
