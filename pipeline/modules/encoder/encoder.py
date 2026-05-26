"""Module 2: Encoder — unified entry point with caching.

The Encoder is the only public class other modules should touch.
It owns the cache read/write, delegates annotation to ReasoningArm,
and returns SchemaRecord objects with the correct failure_mode set.

Cache contract
--------------
Key   : hash(segment_id + window_idx + arm + schema_version + prompt_template_id + model_id)
Format: JSON (SchemaRecord.to_json())
Path  : {cache_root}/encoder/reasoning/{cache_key}.json

On cache hit with matching key → return cached record (cached=True).
On cache miss → call arm → write cache → return fresh record.

Standalone usage (production):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import CosmosReason2Client
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.modules.encoder.schema import WindowInput
    from pipeline.modules.storage.client import WindowStorage
    import os

    storage = WindowStorage(bucket_uri="gs://my-bucket/verity", ...)
    vlm = CosmosReason2Client(api_key=os.environ["NVIDIA_API_KEY"])
    lib = Encoder(vlm=vlm, vocabulary=DEFAULT_VOCABULARY)
    record = lib.process(WindowInput(segment_id="seg_001", window_idx=0, storage=storage))

Standalone usage (offline / test):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.modules.encoder.schema import WindowInput

    lib = Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY)
    record = lib.process(WindowInput(segment_id="seg_001", window_idx=0, storage=mock_storage))
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.modules.encoder.reasoning_arm import (
    ReasoningArm,
    VLMClient,
    VLMUnavailableError,
)
from pipeline.modules.encoder.schema import (
    FAILURE_INVALID_JSON,
    FAILURE_STORAGE_ERROR,
    FAILURE_UNKNOWN,
    FAILURE_VLM_UNAVAILABLE,
    FAILURE_VOCABULARY_VIOLATION,
    NULL_FIELDS_V1,
    SchemaRecord,
    WindowInput,
)
from pipeline.modules.encoder.vocabulary import Vocabulary
from pipeline.modules.storage.adapters.base import WindowStorageError


_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache"


class Encoder:
    """Annotates windows using the reasoning arm. Owns the local cache.

    Parameters
    ----------
    vlm           VLMClient — CosmosReason2Client or StubVLMClient.
    vocabulary    Locked Vocabulary for validation.
    cache_root    Directory for caching results. Default: project/cache/
    max_retries   Passed through to ReasoningArm.
    camera        Camera to request video for. Default: "FRONT".
    """

    ARM = "reasoning"

    def __init__(
        self,
        vlm: VLMClient,
        vocabulary: Vocabulary,
        cache_root: Path | str | None = None,
        max_retries: int = 3,
        camera: str = "FRONT",
    ) -> None:
        self._arm = ReasoningArm(
            vlm=vlm,
            vocabulary=vocabulary,
            max_retries=max_retries,
            camera=camera,
        )
        self._vlm_model_id = vlm.model_id
        self._vocab = vocabulary
        self._cache_dir = (
            Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        ) / "encoder" / self.ARM
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, window: WindowInput) -> SchemaRecord:
        """Annotate one window. Returns a SchemaRecord.

        Never raises on VLM or vocabulary failure — those are recorded as
        failure_mode in the returned record. Raises only on transient
        infrastructure failures (storage timeout, disk full) that the
        caller must handle.
        """
        cache_key = self._cache_key(window)
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached.cached = True
            return cached

        record = self._annotate(window)
        # Don't cache transient infrastructure failures — they may succeed on retry.
        # Deterministic failures (invalid_json, vocabulary_violation) are cached so
        # they don't waste VLM calls on repeat runs.
        if record.failure_mode != FAILURE_VLM_UNAVAILABLE:
            self._write_cache(cache_key, record)
        return record

    def process_batch(self, windows: list[WindowInput]) -> list[SchemaRecord]:
        """Annotate a list of windows sequentially. Returns all records.

        Records with failure_mode set are included — callers filter them.
        """
        results: list[SchemaRecord] = []
        total = len(windows)
        for i, window in enumerate(windows, start=1):
            print(
                f"[Encoder] {i}/{total} — {window.window_id_str}",
                file=sys.stderr,
            )
            record = self.process(window)
            status = "ok" if record.succeeded else f"FAILED({record.failure_mode})"
            cached_tag = " [cached]" if record.cached else ""
            print(
                f"[Encoder] {i}/{total} — {window.window_id_str} → {status}{cached_tag}",
                file=sys.stderr,
            )
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _annotate(self, window: WindowInput) -> SchemaRecord:
        """Call the reasoning arm and return a SchemaRecord."""
        try:
            fields, raw = self._arm.annotate_from_storage(
                segment_id=window.segment_id,
                window_idx=window.window_idx,
                storage=window.storage,
                prompt_template_id=window.prompt_template_id,
            )
        except WindowStorageError as exc:
            print(
                f"\n[Encoder] STORAGE ERROR for {window.window_id_str}: {exc}",
                file=sys.stderr,
            )
            return SchemaRecord(
                window_id=window.window_key,
                arm=self.ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=FAILURE_STORAGE_ERROR,
            )
        except VLMUnavailableError as exc:
            print(
                f"\n[Encoder] VLM UNAVAILABLE for {window.window_id_str}: {exc}",
                file=sys.stderr,
            )
            return SchemaRecord(
                window_id=window.window_key,
                arm=self.ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=FAILURE_VLM_UNAVAILABLE,
            )
        except ValueError as exc:
            msg = str(exc)
            # Distinguish JSON parse failures from vocabulary violations
            if "JSON" in msg or "json" in msg:
                failure_mode = FAILURE_INVALID_JSON
            elif "vocabulary" in msg.lower() or "violation" in msg.lower():
                failure_mode = FAILURE_VOCABULARY_VIOLATION
            else:
                failure_mode = FAILURE_UNKNOWN
            print(
                f"\n[Encoder] ANNOTATION FAILED ({failure_mode}) "
                f"for {window.window_id_str}: {exc}",
                file=sys.stderr,
            )
            return SchemaRecord(
                window_id=window.window_key,
                arm=self.ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=failure_mode,
            )
        except Exception as exc:
            print(
                f"\n[Encoder] UNEXPECTED ERROR for {window.window_id_str}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return SchemaRecord(
                window_id=window.window_key,
                arm=self.ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=FAILURE_UNKNOWN,
                raw_vlm_response=None,
            )

        # Validate fill fraction (calibration warning, not a failure)
        fill = self._vocab.fill_fraction(fields)
        if fill < 0.8:
            print(
                f"[Encoder] WARNING: {window.window_id_str} fill fraction "
                f"{fill:.0%} < 80% calibration threshold.",
                file=sys.stderr,
            )

        return SchemaRecord(
            window_id=window.window_key,
            arm=self.ARM,
            schema_version=window.schema_version,
            prompt_template_id=window.prompt_template_id,
            fields=fields,
            failure_mode=None,
            raw_vlm_response=raw,
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, window: WindowInput) -> str:
        """Deterministic cache key from all inputs that affect the output.

        Includes model_id so that swapping VLM backends invalidates the cache
        even when schema_version and prompt_template_id are unchanged.
        """
        raw = (
            f"{window.segment_id}|{window.window_idx}|"
            f"{self.ARM}|{window.schema_version}|{window.prompt_template_id}|"
            f"{self._vlm_model_id}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> SchemaRecord | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SchemaRecord.from_json(data)
        except Exception as exc:
            print(
                f"[Encoder] WARNING: corrupt cache entry {path.name}: {exc}. "
                f"Discarding and re-annotating.",
                file=sys.stderr,
            )
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, record: SchemaRecord) -> None:
        path = self._cache_path(key)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(record.to_json(), indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except Exception as exc:
            print(
                f"[Encoder] WARNING: could not write cache {path.name}: {exc}",
                file=sys.stderr,
            )
            tmp.unlink(missing_ok=True)
