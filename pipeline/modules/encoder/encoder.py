"""Module 2: Encoder — unified entry point with caching.

The Encoder is the only public class other modules should touch.
It owns the cache read/write, delegates annotation to ReasoningArm (and
optionally VisualArm), and returns SchemaRecord objects with the correct
failure_mode set.

Cache contract
--------------
Reasoning arm:
  Key   : hash(segment_id + window_idx + "reasoning" + schema_version
               + prompt_template_id + model_id)
  Path  : {cache_root}/encoder/reasoning/{key}.json

Visual arm (when configured):
  Key   : hash(segment_id + window_idx + "visual" + schema_version
               + visual_model_id)
  Path  : {cache_root}/encoder/visual/{key}.json

On cache hit with matching key → return cached record (cached=True).
On cache miss → call arm → write cache → return fresh record.

Concurrency
-----------
process_batch() dispatches windows concurrently via ThreadPoolExecutor
(max_workers, default 8). process() is thread-safe: VLM clients are
stateless per call, cache writes are atomic (.json.tmp → rename).

When visual_arm is configured, process() dispatches both arms concurrently
for each window (ThreadPoolExecutor with 2 workers). Each arm fails
independently — a visual arm failure does not affect the reasoning record.

Return type change: process() returns list[SchemaRecord] (one record per
configured arm). Callers that only need the reasoning arm should filter:
    reasoning = [r for r in records if r.arm == "reasoning"]

Standalone usage (production, reasoning only):
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

Standalone usage (production, both arms):
    import os
    from pipeline.modules.encoder import CosmosEmbed1Client, VisualArm

    embed_client = CosmosEmbed1Client(cosmos_url=os.environ["COSMOS_EMBED1_URL"])
    visual = VisualArm(client=embed_client)
    enc = Encoder(vlm=vlm, vocabulary=DEFAULT_VOCABULARY, visual_arm=visual)
    records = enc.process(WindowInput(...))
    # records[0].arm == "reasoning", records[1].arm == "visual"

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
from typing import TYPE_CHECKING

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
from pipeline.interfaces.errors import WindowStorageError
from pipeline.modules.encoder.visual_arm import EmbedUnavailableError
from pipeline.modules.encoder.vocabulary import Vocabulary

if TYPE_CHECKING:
    from pipeline.modules.encoder.visual_arm import VisualArm


_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache"

_VISUAL_ARM = "visual"
_REASONING_ARM = "reasoning"


class Encoder:
    """Annotates windows using the reasoning arm (and optionally the visual arm).

    Parameters
    ----------
    vlm           VLMClient — CosmosReason2Client or StubVLMClient.
    vocabulary    Locked Vocabulary for validation.
    cache_root    Directory for caching results. Default: project/cache/
    max_retries   Passed through to ReasoningArm.
    camera        Camera to request video for reasoning arm. Default: "FRONT".
    max_workers   Max concurrent windows in process_batch(). Default: 8.
    visual_arm    Optional VisualArm (Cosmos-Embed1). When set, process()
                  dispatches reasoning + visual concurrently and returns
                  two SchemaRecords per window.
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
        visual_arm: "VisualArm | None" = None,
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
        self._visual_arm = visual_arm

        cache_base = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._cache_dir = cache_base / "encoder" / _REASONING_ARM
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._visual_cache_dir = cache_base / "encoder" / _VISUAL_ARM
        if visual_arm is not None:
            self._visual_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, window: WindowInput) -> list[SchemaRecord]:
        """Annotate one window. Returns a list of SchemaRecords (one per arm).

        When no visual_arm is configured: returns [reasoning_record].
        When visual_arm is configured: dispatches both arms concurrently and
        returns [reasoning_record, visual_record]. Each arm fails independently.

        Never raises on VLM or vocabulary failure — those become failure_mode
        on the returned record.

        Note on concurrency: the two arms hit different endpoints (Cosmos-Reason2
        vs Cosmos-Embed1), so running them together does not double the load on
        either. When called from process_batch, peak in-flight requests per
        endpoint is max_workers (not max_workers × 2).
        """
        if self._visual_arm is None:
            return [self._process_reasoning_arm(window)]

        with ThreadPoolExecutor(max_workers=2) as arm_pool:
            r_future = arm_pool.submit(self._process_reasoning_arm, window)
            v_future = arm_pool.submit(self._process_visual_arm, window)

        return [r_future.result(), v_future.result()]

    def process_batch(self, windows: list[WindowInput]) -> list[SchemaRecord]:
        """Annotate a list of windows concurrently. Returns all records.

        Records with failure_mode set are included — callers filter them.
        When both arms are configured, returns up to 2×len(windows) records.
        Filter by arm before passing to Hypothesizer:
            reasoning = [r for r in records if r.arm == "reasoning"]
        """
        total = len(windows)
        per_arm = 2 if self._visual_arm is not None else 1
        results: list[SchemaRecord | None] = [None] * (total * per_arm)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self.process, w): i
                for i, w in enumerate(windows)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    records = future.result()
                    for arm_idx, record in enumerate(records):
                        status = "ok" if record.succeeded else f"FAILED({record.failure_mode})"
                        cached_tag = " [cached]" if record.cached else ""
                        print(
                            f"[Encoder] {i+1}/{total} [{record.arm}] "
                            f"— {windows[i].window_id_str} → {status}{cached_tag}",
                            file=sys.stderr,
                        )
                        results[i * per_arm + arm_idx] = record
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
                    failure = SchemaRecord(
                        window_id=windows[i].window_key,
                        arm=_REASONING_ARM,
                        schema_version=windows[i].schema_version,
                        prompt_template_id=windows[i].prompt_template_id,
                        fields=dict(NULL_FIELDS_V1),
                        failure_mode=FAILURE_UNKNOWN,
                    )
                    results[i * per_arm] = failure

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

    def _process_visual_arm(self, window: WindowInput) -> SchemaRecord:
        assert self._visual_arm is not None
        cache_key = self._visual_cache_key(window)
        cached = self._read_cache(self._visual_cache_dir, cache_key)
        if cached is not None:
            cached.cached = True
            return cached

        record = self._annotate_visual(window)
        if record.failure_mode != FAILURE_VLM_UNAVAILABLE:
            self._write_cache(self._visual_cache_dir, cache_key, record)
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
    # Annotation (visual)
    # ------------------------------------------------------------------

    def _annotate_visual(self, window: WindowInput) -> SchemaRecord:
        """Call the visual arm and return a SchemaRecord."""
        assert self._visual_arm is not None
        try:
            fields, _ = self._visual_arm.annotate_from_storage(
                segment_id=window.segment_id,
                window_idx=window.window_idx,
                storage=window.storage,
            )
        except WindowStorageError as exc:
            print(
                f"\n[Encoder] VISUAL STORAGE ERROR for {window.window_id_str}: {exc}",
                file=sys.stderr,
            )
            return self._visual_failure_record(window, FAILURE_STORAGE_ERROR)
        except EmbedUnavailableError as exc:
            print(
                f"\n[Encoder] VISUAL ARM UNAVAILABLE for {window.window_id_str}: {exc}",
                file=sys.stderr,
            )
            return self._visual_failure_record(window, FAILURE_VLM_UNAVAILABLE)
        except Exception as exc:
            print(
                f"\n[Encoder] VISUAL ARM FAILED for {window.window_id_str}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return self._visual_failure_record(window, FAILURE_UNKNOWN)

        return SchemaRecord(
            window_id=window.window_key,
            arm=_VISUAL_ARM,
            schema_version=window.schema_version,
            prompt_template_id=None,
            fields=fields,
            failure_mode=None,
        )

    def _visual_failure_record(
        self, window: WindowInput, failure_mode: str
    ) -> SchemaRecord:
        return SchemaRecord(
            window_id=window.window_key,
            arm=_VISUAL_ARM,
            schema_version=window.schema_version,
            prompt_template_id=None,
            fields={},
            failure_mode=failure_mode,
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

    def _visual_cache_key(self, window: WindowInput) -> str:
        assert self._visual_arm is not None
        raw = (
            f"{window.segment_id}|{window.window_idx}|"
            f"{_VISUAL_ARM}|{window.schema_version}|"
            f"{self._visual_arm.model_id}"
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
