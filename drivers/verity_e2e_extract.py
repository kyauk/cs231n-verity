"""Fold the ingested E2E clips into the evidence store (salience-first extraction).

Reads outputs/waymo/e2e_clips.json (from verity_e2e_ingest.py), runs the extractor
on each clip, appends descriptors to the salience evidence store. Idempotent
(skips clips already extracted). After this, verity_hardnovel.py re-canonicalizes
the full corpus + E2E and re-ranks.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.modules.curator import TaxonomyStore
from pipeline.modules.extractor import (
    CosmosReasonClient, Extractor, ExtractorConfig, NIMEmbedder, NIMStructureClient,
)

STORE = Path("outputs/waymo/taxonomy_store_salience")
CLIPS = Path("outputs/waymo/e2e_clips.json")


def main() -> None:
    store = TaxonomyStore(STORE)
    clips = json.loads(CLIPS.read_text())
    already = {d.scene_id for d in store.load_descriptors()}
    pending = [c for c in clips if c["scene_id"] not in already]
    print(f"[e2e-x] {len(clips)} E2E clips; {len(pending)} to extract", file=sys.stderr)

    ex = Extractor(CosmosReasonClient(), NIMStructureClient(), NIMEmbedder(), ExtractorConfig())
    new, ok = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(ex.extract, c["scene_id"], c["ref"]): c for c in pending}
        for f in as_completed(futs):
            c = futs[f]
            try:
                ds = f.result(); new.extend(ds); ok += 1
                print(f"[e2e-x] {c['scene_id']} ({c.get('frames','?')} frames) -> {len(ds)} descriptors", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[e2e-x] {c['scene_id']} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    added = store.append_descriptors(new)
    print(f"\n[e2e-x] DONE — {ok}/{len(pending)} clips extracted, appended {added} descriptors "
          f"({len({d.scene_id for d in store.load_descriptors()})} total scenes in evidence)")


if __name__ == "__main__":
    main()
