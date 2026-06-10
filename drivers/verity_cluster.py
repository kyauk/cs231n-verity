"""Cluster-stage driver — embed segment videos via Cosmos-Embed1, then cluster.

Bypasses the signed-URL wall: the pipeline's NIMEmbedClient fetches each clip via
a v4 signed URL (which user-ADC can't generate). Here we download each clip with
ADC directly, base64 it, and POST to the local Cosmos-Embed1 NIM — then reuse the
REAL Clusterer.cluster() math (UMAP -> HDBSCAN -> coords_3d). Output is the same
clusters.json the runner's /cluster-space reads, so the UI Cluster Space renders.

    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_cluster --limit 20 --out outputs/waymo
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from google.cloud import storage

from pipeline.interfaces.window import WindowKey
from pipeline.modules.clustering import Clusterer, ClustererConfig
from pipeline.modules.clustering.clusterer import WindowEmbedding

BUCKET = os.environ.get("WAYMO_DEST_BUCKET", "nvidia-adr-waymo-segment-videos")
PROJECT = "nvidia-adr"
EMBED_URL = os.environ.get("COSMOS_EMBED1_URL", "http://localhost:8080").rstrip("/")
EMBED_MODEL = os.environ.get("COSMOS_EMBED1_MODEL_ID", "nvidia/cosmos-embed1")


def list_segments(limit: int, camera: str) -> list[tuple[str, str]]:
    c = storage.Client(project=PROJECT)
    seen: dict[str, str] = {}
    for blob in c.list_blobs(BUCKET, prefix="segments/"):
        if blob.name.endswith(f"_{camera}.mp4"):
            seg = blob.name.split("/")[1]
            seen.setdefault(seg, blob.name)
        if len(seen) >= limit:
            break
    return sorted(seen.items())[:limit]


def embed_clip(blob_name: str, width: int) -> list[float]:
    """Download via ADC, downscale, base64, POST to Cosmos-Embed1 -> vector."""
    c = storage.Client(project=PROJECT)
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.mp4")
        small = os.path.join(td, "s.mp4")
        c.bucket(BUCKET).blob(blob_name).download_to_filename(raw)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-vf", f"scale={width}:-2", "-an", "-c:v", "libx264",
                        "-crf", "30", small], capture_output=True, timeout=180)
        b64 = base64.b64encode(Path(small).read_bytes()).decode()
    payload = {"input": [f"data:video/mp4;base64,{b64}"], "request_type": "query",
               "encoding_format": "float", "model": EMBED_MODEL}
    r = requests.post(f"{EMBED_URL}/v1/embeddings", json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Embed1 {r.status_code}: {r.text[:200]}")
    return [float(x) for x in r.json()["data"][0]["embedding"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--out", default="outputs/waymo")
    args = ap.parse_args()

    segs = list_segments(args.limit, args.camera)
    print(f"[cluster] embedding {len(segs)} segments via Embed1 @ {EMBED_URL}", file=sys.stderr)

    embeddings: list[WindowEmbedding] = []
    for i, (seg, blob) in enumerate(segs):
        try:
            vec = embed_clip(blob, args.width)
            embeddings.append(WindowEmbedding(window_id=WindowKey(seg, 0), vector=vec, dim=len(vec)))
            print(f"[cluster] {i+1}/{len(segs)} {seg[:20]} -> dim={len(vec)}", file=sys.stderr)
        except Exception as exc:
            print(f"[cluster] {i+1}/{len(segs)} {seg[:20]} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not embeddings:
        print("[cluster] no embeddings produced — aborting.", file=sys.stderr)
        sys.exit(1)

    n = len(embeddings)
    # small-N-safe clustering config
    config = ClustererConfig(
        cameras=(args.camera,),
        umap_n_components=min(10, max(2, n - 2)),
        umap_n_neighbors=min(15, max(2, n - 1)),
        hdbscan_min_cluster_size=3,
    )
    report = Clusterer(embed_client=None, config=config).cluster(embeddings)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "clusters.json").write_text(json.dumps(report.to_json(), indent=2))
    print(f"\n[cluster] DONE — {report.n_clusters} clusters, {report.n_noise} noise "
          f"over {n} windows, dim={report.embedding_dim}", file=sys.stderr)
    print(f"[cluster] wrote {out}/clusters.json")


if __name__ == "__main__":
    main()
