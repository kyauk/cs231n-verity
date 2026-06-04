"""Scale run: evolve the emergent taxonomy on the 798 training segments, then
re-test the original 202 and measure whether labels improved.

Disk-safe streaming (the working-set / demand-paging model): each worker
  render FRONT mp4 -> upload to bucket -> extract salience descriptors -> DELETE local
so only ~workers segments are ever resident (21 GB disk is never threatened).
Checkpointed + resumable: re-running skips finished segments. Concurrent for
throughput: render (CPU) overlaps extract (GPU NIM) across workers.

Phases:
  0. BEFORE snapshot — canonical labels from just the original 202 (saved once).
  1. Stream render+extract the 798 training segments (concurrent, checkpointed).
  2. AFTER — canonicalize ALL evidence; compare the 202's labels before vs after.

    set -a && . ./.env && set +a
    .venv/bin/python verity_scale.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import pyarrow.parquet as pq
from google.cloud import storage

from waymo_pipeline.waymo_video_pipeline import decode_jpeg, encode_mp4_ffmpeg
from pipeline.interfaces.taxonomy import EMPTY_TAXONOMY
from pipeline.modules.curator import (
    CuratorConfig, TaxonomyStore, canonicalize, coverage, project,
)
from pipeline.modules.extractor import (
    CosmosReasonClient, Extractor, ExtractorConfig, NIMEmbedder, NIMStructureClient,
)

SRC_BUCKET = "waymo_open_dataset_v_2_0_1"
SRC_PREFIX = os.environ.get("WAYMO_SOURCE_PREFIX", "training/camera_image")
DEST_BUCKET = "nvidia-adr-waymo-segment-videos"
PROJECT = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

STORE_DIR = Path("outputs/waymo/taxonomy_store_salience")
EVIDENCE = STORE_DIR / "evidence.jsonl"
CKPT = Path("outputs/waymo/scale_checkpoint.json")
BEFORE = Path("outputs/waymo/scale_before.json")
REPORT = Path("outputs/waymo/scale_report.json")
CFG = CuratorConfig(cohesion_threshold=0.36, merge_threshold=0.20, support_threshold=2)
FRONT_CAM = 1  # Waymo camera_name int for FRONT

_lock = threading.Lock()


def _list_training_segments() -> list[str]:
    c = storage.Client(project=PROJECT)
    out = []
    for b in c.list_blobs(SRC_BUCKET, prefix=SRC_PREFIX + "/"):
        if b.name.endswith(".parquet"):
            out.append(b.name.rsplit("/", 1)[-1][: -len(".parquet")])
    return sorted(out)


def _render_front(seg: str, fs: gcsfs.GCSFileSystem, out_path: Path) -> None:
    pqpath = f"{SRC_BUCKET}/{SRC_PREFIX}/{seg}.parquet"
    with fs.open(pqpath, "rb") as f:
        names = pq.ParquetFile(f).schema_arrow.names

    def col(cands: list[str]) -> str:
        return next(c for c in cands if c in names)

    ci = col(["[CameraImageComponent].image", "image", "camera_image"])
    cc = col(["key.camera_name", "camera_name"])
    ct = col(["key.frame_timestamp_micros", "frame_timestamp_micros"])
    with fs.open(pqpath, "rb") as f:
        df = pq.read_table(f, columns=[ci, cc, ct]).to_pandas().sort_values(ct)
    front = df[df[cc] == FRONT_CAM].reset_index(drop=True)
    if front.empty:
        raise RuntimeError("no FRONT frames in parquet")
    frames = [decode_jpeg(bytes(r[ci])) for _, r in front.iterrows()]
    encode_mp4_ffmpeg(frames, str(out_path))


def _upload(local: str, seg: str) -> str:
    blob = f"segments/{seg}/{seg}_FRONT.mp4"
    storage.Client(project=PROJECT).bucket(DEST_BUCKET).blob(blob).upload_from_filename(
        local, content_type="video/mp4")
    return f"gs://{DEST_BUCKET}/{blob}"


def _persist(descs, seg: str, done: set[str]) -> None:
    with _lock:
        with EVIDENCE.open("a", encoding="utf-8") as f:
            for d in descs:
                f.write(json.dumps(d.to_json()) + "\n")
        done.add(seg)
        CKPT.write_text(json.dumps({"done": sorted(done)}))


def _worker(seg: str, fs, extractor: Extractor, done: set[str]) -> int:
    # render + upload, then free local immediately (working-set bound)
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / f"{seg}_FRONT.mp4"
        _render_front(seg, fs, mp4)
        ref = _upload(str(mp4), seg)
    # extract (GPU NIM) reads the uploaded clip; local already gone
    descs = extractor.extract(seg, ref)
    _persist(descs, seg, done)
    return len(descs)


def _snapshot_before(store: TaxonomyStore) -> dict:
    if BEFORE.exists():
        return json.loads(BEFORE.read_text())
    ds = store.load_descriptors()
    tax = canonicalize(ds, EMPTY_TAXONOMY, CFG)
    snap = {
        "scenes": sorted({d.scene_id for d in ds}),
        "n_descriptors": len(ds),
        "n_labels": len(tax.labels),
        "labels_by_axis": {ax: len(v) for ax, v in tax.labels_by_axis().items()},
        "coverage": coverage(ds, project(ds, tax, CFG))["coverage"],
    }
    BEFORE.write_text(json.dumps(snap, indent=2))
    return snap


def _compare(store: TaxonomyStore, before: dict) -> dict:
    ds = store.load_descriptors()
    tax = canonicalize(ds, EMPTY_TAXONOMY, CFG)
    before_scenes = set(before["scenes"])
    # re-project ONLY the original 202's descriptors onto the evolved taxonomy
    orig = [d for d in ds if d.scene_id in before_scenes]
    cov_after = coverage(orig, project(orig, tax, CFG))["coverage"]
    rep = {
        "before": before,
        "after": {
            "n_descriptors": len(ds),
            "n_scenes": len({d.scene_id for d in ds}),
            "n_labels": len(tax.labels),
            "labels_by_axis": {ax: len(v) for ax, v in tax.labels_by_axis().items()},
        },
        "original_202_reprojected": {
            "coverage_before": before["coverage"],
            "coverage_after": cov_after,
        },
        "label_growth": len(tax.labels) - before["n_labels"],
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 798")
    args = ap.parse_args()
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    store = TaxonomyStore(STORE_DIR)
    before = _snapshot_before(store)
    print(f"[scale] BEFORE: {before['n_descriptors']} descriptors, "
          f"{len(before['scenes'])} scenes, {before['n_labels']} labels, "
          f"coverage {before['coverage']:.0%}", file=sys.stderr)

    segs = _list_training_segments()
    if args.limit > 0:
        segs = segs[: args.limit]
    done = set(before["scenes"])
    if CKPT.exists():
        done |= set(json.loads(CKPT.read_text()).get("done", []))
    pending = [s for s in segs if s not in done]
    print(f"[scale] {len(segs)} training segs; {len(pending)} pending "
          f"(working set <= {args.workers} segments on disk)", file=sys.stderr)

    fs = gcsfs.GCSFileSystem()
    extractor = Extractor(CosmosReasonClient(), NIMStructureClient(), NIMEmbedder(), ExtractorConfig())
    n = ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, s, fs, extractor, done): s for s in pending}
        for f in as_completed(futs):
            s = futs[f]; n += 1
            try:
                k = f.result(); ok += 1
                print(f"[scale] {n}/{len(pending)} {s[:22]} -> {k} descriptors", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[scale] {n}/{len(pending)} {s[:22]} FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[scale] extraction done: {ok}/{len(pending)} ok. Canonicalizing all evidence...", file=sys.stderr)
    rep = _compare(store, before)
    print(f"\n[scale] DONE.")
    print(f"   labels:   {rep['before']['n_labels']} -> {rep['after']['n_labels']} "
          f"(+{rep['label_growth']})")
    print(f"   scenes:   {len(before['scenes'])} -> {rep['after']['n_scenes']}")
    print(f"   202 coverage: {rep['original_202_reprojected']['coverage_before']:.0%} "
          f"-> {rep['original_202_reprojected']['coverage_after']:.0%}")
    print(f"   report -> {REPORT}")


if __name__ == "__main__":
    main()
