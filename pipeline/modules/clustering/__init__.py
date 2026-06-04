"""Module 8: Clustering — embedding-based scenario discovery.

Embeds each window (continuous representation) and groups windows by density in
embedding space (UMAP -> HDBSCAN), surfacing clusters and GLOSH outliers. It is
the embedding-arm peer to the symbolic Judge path, and it shares Module 1's
ingestion: it reads windows over the WindowStorageBase Protocol and never
touches another module's internals (lego-block rule).

Public surface (import from the package root):
    from pipeline.modules.clustering import Clusterer, ClustererConfig, NIMEmbedClient

The embedder is an injected Protocol. NIMEmbedClient is the production
(Cosmos-Embed) impl; StubEmbedClient runs the full path offline / in tests.
"""

from pipeline.modules.clustering.config import (
    ClustererConfig,
    ClusteringError,
    EmbedClient,
    EmbedUnavailableError,
)
from pipeline.modules.clustering.embed import NIMEmbedClient, StubEmbedClient
from pipeline.modules.clustering.clusterer import Clusterer

__all__ = [
    "Clusterer",
    "ClustererConfig",
    "EmbedClient",
    "NIMEmbedClient",
    "StubEmbedClient",
    "ClusteringError",
    "EmbedUnavailableError",
]
