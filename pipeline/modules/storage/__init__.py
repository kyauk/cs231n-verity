"""Module 1: Storage — ingestion and retrieval of windowed fleet data.

Writes source footage into the canonical bucket layout (IngestionPipeline) and
serves windows back on demand. Two retrieval implementations:
  WindowStorage   — canonical ingested layout in GCS
  FlatMP4Storage  — flat bucket of MP4s, one window per file (quick analysis
                    path; no ingest required)

Both satisfy WindowStorageBase (pipeline.interfaces.window).

Public surface (import from the package root):
    from pipeline.modules.storage import WindowStorage, IngestionPipeline
    from pipeline.modules.storage import FlatMP4Storage

To add a new source format for the canonical path, implement the SourceAdapter
protocol (its Frame / RawSegment data types live in
pipeline.modules.storage.adapters.base). Cross-module error types live in
pipeline.interfaces.errors and are re-exported here for convenience.
"""

from pipeline.modules.storage.adapters.base import (
    IngestionError,
    IngestionRequest,
    SourceAdapter,
    SourceAdapterError,
    SourceSchemaVersionError,
    SourceUnreachableError,
    StorageError,
    WindowConfig,
    WindowStorageError,
)
from pipeline.modules.storage.adapters.parquet import WaymoParquetSource
from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource
from pipeline.modules.storage.client import WindowStorage
from pipeline.modules.storage.flat_mp4 import FlatMP4Storage
from pipeline.modules.storage.ingestion import IngestionPipeline

__all__ = [
    # Retrieval
    "WindowStorage",
    "FlatMP4Storage",
    # Ingestion
    "IngestionPipeline",
    "IngestionRequest",
    "WindowConfig",
    # Source adapters (extension point)
    "SourceAdapter",
    "WaymoParquetSource",
    "WaymoTFRecordSource",
    # Errors
    "StorageError",
    "WindowStorageError",
    "SourceUnreachableError",
    "SourceSchemaVersionError",
    "SourceAdapterError",
    "IngestionError",
]
