"""Module 2: Encoder — unified entry point with caching.

The Encoder is the only public class other modules should touch.
It owns the cache read/write, delegates annotation to ReasoningArm, and
returns SchemaRecord objects with the correct failure_mode set.

Cache contract
--------------
  Key   : hash(segment_id + window_idx + "reasoning" + schema_version
               + prompt_template_id + model_id)
  Path  : {cache_root}/encoder/reasoning/{key}.json

On cache hit with matching key → return cached record (cached=True).
On cache miss → call arm → write cache → return fresh record.

Concurrency
-----------
process_batch() dispatches windows concurrently via ThreadPoolExecutor
(max_workers, default 8). process() is thread-safe: VLM clients are
stateless per call, cache writes are atomic (.json.tmp → rename).

process() returns list[SchemaRecord] — currently always length 1 (the
reasoning arm). The list shape is preserved for forward compatibility with
v2, where a continuous-discovery channel may emit a parallel record per
window.

Standalone usage (production):
    import os
    from pipeline.modules.encoder import (
        Encoder, CosmosReason2Client, DEFAULT_VOCABULARY, WindowInput,
    )
    from pipeline.modules.storage import WindowStorage

    storage = WindowStorage(bucket_uri="gs://my-bucket/verity", ...)
    vlm = CosmosReason2Client(api_key=os.environ["NVIDIA_API_KEY"])
    enc = Encoder(vlm=vlm, vocabulary=DEFAULT_VOCABULARY)
    records = enc.process(WindowInput(segment_id="seg_001", window_idx=0,
                                      storage=storage))
    reasoning_record = records[0]

Standalone usage (offline / test):
    from pipeline.modules.encoder import Encoder, StubVLMClient, DEFAULT_VOCABULARY

    enc = Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY)
    records = enc.process(WindowInput(..., storage=mock_storage))
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.interfaces.errors import WindowStorageError
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


_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache"

_REASONING_ARM = "reasoning"


class Encoder:
    """Annotates windows using the reasoning arm.

    Parameters
    ----------
    vlm           VLMClient — CosmosReason2Client or StubVLMClient.
    vocabulary    Locked Vocabulary for validation.
    cache_root    Directory for caching results. Default: project/cache/
    max_retries   Passed through to ReasoningArm.
    camera        Camera to request video for. Default: "FRONT".
    max_workers   Max concurrent windows in process_batch(). Default: 8.
    """

    ARM = _REASONING_ARM

    def __init__(
        self,
        vlm: VLMClient,
        vocabulary: Vocabulary,
        cache_root: Path | str | None = None,
        max_retries: int = 3,
        camera: str = "FRONT",
        max_workers: int = 8,
    ) -> None:
        self._arm = ReasoningArm(
            vlm=vlm,
            vocabulary=vocabulary,
            max_retries=max_retries,
            camera=camera,
        )
        self._vlm_model_id = vlm.model_id
        self._vocab = vocabulary
        self._max_workers = max_workers

        cache_base = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._cache_dir = cache_base / "encoder" / _REASONING_ARM
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, window: WindowInput) -> list[SchemaRecord]:
        """Annotate one window. Returns a list of SchemaRecords.

        The list is length-1 in v1 (reasoning arm only). The list shape is
        preserved so callers don't need to change when v2 introduces a
        parallel continuous-discovery channel.

        Never raises on VLM or vocabulary failure — those become failure_mode
        on the returned record.
        """
        return [self._process_reasoning_arm(window)]

    def process_batch(self, windows: list[WindowInput]) -> list[SchemaRecord]:
        """Annotate a list of windows concurrently. Returns all records.

        Records with failure_mode set are included — callers filter them.
        """
        total = len(windows)
        results: list[SchemaRecord | None] = [None] * total

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self.process, w): i
                for i, w in enumerate(windows)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    records = future.result()
                    record = records[0]
                    status = "ok" if record.succeeded else f"FAILED({record.failure_mode})"
                    cached_tag = " [cached]" if record.cached else ""
                    print(
                        f"[Encoder] {i+1}/{total} [{record.arm}] "
                        f"— {windows[i].window_id_str} → {status}{cached_tag}",
                        file=sys.stderr,
                    )
                    results[i] = record
                except Exception as exc:
                    # Defensive: process() swallows arm failures into records and
                    # does not raise, so this is only reached on an unexpected
                    # infrastructure error. Synthesize one failure record so the
                    # returned batch keeps its slot and is never silently shortened.
                    print(
                        f"[Encoder] UNEXPECTED thread failure for window {i}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    results[i] = SchemaRecord(
                        window_id=windows[i].window_key,
                        arm=_REASONING_ARM,
                        schema_version=windows[i].schema_version,
                        prompt_template_id=windows[i].prompt_template_id,
                        fields=dict(NULL_FIELDS_V1),
                        failure_mode=FAILURE_UNKNOWN,
                    )

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Per-arm processing (cache + annotate, never raises)
    # ------------------------------------------------------------------

    def _process_reasoning_arm(self, window: WindowInput) -> SchemaRecord:
        cache_key = self._cache_key(window)
        cached = self._read_cache(self._cache_dir, cache_key)
        if cached is not None:
            cached.cached = True
            return cached

        record = self._annotate_reasoning(window)
        if record.failure_mode != FAILURE_VLM_UNAVAILABLE:
            self._write_cache(self._cache_dir, cache_key, record)
        return record

    # ------------------------------------------------------------------
    # Annotation (reasoning)
    # ------------------------------------------------------------------

    def _annotate_reasoning(self, window: WindowInput) -> SchemaRecord:
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
                arm=_REASONING_ARM,
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
                arm=_REASONING_ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=FAILURE_VLM_UNAVAILABLE,
            )
        except ValueError as exc:
            msg = str(exc)
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
                arm=_REASONING_ARM,
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
                arm=_REASONING_ARM,
                schema_version=window.schema_version,
                prompt_template_id=window.prompt_template_id,
                fields=dict(NULL_FIELDS_V1),
                failure_mode=FAILURE_UNKNOWN,
                raw_vlm_response=None,
            )

        fill = self._vocab.fill_fraction(fields)
        if fill < 0.8:
            print(
                f"[Encoder] WARNING: {window.window_id_str} fill fraction "
                f"{fill:.0%} < 80% calibration threshold.",
                file=sys.stderr,
            )

        return SchemaRecord(
            window_id=window.window_key,
            arm=_REASONING_ARM,
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
        raw = (
            f"{window.segment_id}|{window.window_idx}|"
            f"{_REASONING_ARM}|{window.schema_version}|{window.prompt_template_id}|"
            f"{self._vlm_model_id}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, cache_dir: Path, key: str) -> SchemaRecord | None:
        path = cache_dir / f"{key}.json"
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

    def _write_cache(self, cache_dir: Path, key: str, record: SchemaRecord) -> None:
        path = cache_dir / f"{key}.json"
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
