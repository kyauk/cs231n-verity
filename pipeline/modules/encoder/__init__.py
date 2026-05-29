"""Module 2: Encoder — VLM annotation with caching.

Annotates each window via two independent arms and caches the results:
  - Reasoning arm (Cosmos-Reason2): structured scene annotation. Always on.
  - Visual arm   (Cosmos-Embed1):   1280-d embeddings. On only when an Encoder
                                     is constructed with visual_arm=...

Public surface (import from the package root):
    from pipeline.modules.encoder import Encoder, VisualArm, CosmosEmbed1Client

Each arm has a Stub* client that returns valid output with no network call,
for tests and offline runs.
"""

from pipeline.modules.encoder.encoder import Encoder
from pipeline.modules.encoder.reasoning_arm import (
    CosmosReason2Client,
    ReasoningArm,
    StubVLMClient,
    VLMClient,
)
from pipeline.modules.encoder.schema import WindowInput
from pipeline.modules.encoder.visual_arm import (
    CosmosEmbed1Client,
    EmbedClient,
    StubEmbedClient,
    VisualArm,
)
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
    # Visual arm
    "VisualArm",
    "EmbedClient",
    "CosmosEmbed1Client",
    "StubEmbedClient",
]
