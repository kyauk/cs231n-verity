"""Smoke test: fake embeddings -> clustering -> API endpoints.

Generates synthetic WindowEmbeddingRecord JSONL with deliberate cluster
structure, runs the clustering script, then hits the live runner endpoints
to confirm the UI will receive real data.

Usage:
    python smoke_test.py                  # run everything
    python smoke_test.py --skip-cluster   # skip clustering (reuse existing outputs)
    python smoke_test.py --keep           # don't delete fake embedding file afterwards

Requires:
    - .venv activated (umap-learn, hdbscan, scikit-learn, requests)
    - waymo-pipeline runner on :8000  (uvicorn waymo_pipeline.waymo_runner:app)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
OUTPUTS = PROJECT_ROOT / "outputs" / "waymo"
EMBED_JSONL = OUTPUTS / "waymo_window_embeddings.jsonl"
CLUSTERS_JSONL = OUTPUTS / "waymo_clusters.jsonl"
FLAGGED_JSONL = OUTPUTS / "flagged_windows.jsonl"
NPZ_PATH = OUTPUTS / "waymo_clusters.npz"

RUNNER_URL = "http://localhost:8000"

# 1280-d matches Cosmos Embed1 5-camera rig (5 * 256)
EMBED_DIM = 1280

# Cluster structure: 4 tight groups + some noise
# Enough points so UMAP and HDBSCAN have something real to work with
N_PER_CLUSTER = 20
N_NOISE = 8
N_CLUSTERS = 4

SCENARIO_TAGS = [
    ["night", "intersection"],
    ["rain", "pedestrian"],
    ["highway", "lane_change"],
    ["construction_zone", "low_visibility"],
]

WEATHER = ["Night", "Rain", "Clear", "Fog"]
TIME_OF_DAY = ["Night", "Day", "Dawn", "Dusk"]
ROAD_TYPE = ["Intersection", "Urban", "Highway", "Construction"]

SEED = 42


# ---------------------------------------------------------------------------
# Step 1: Generate fake embeddings
# ---------------------------------------------------------------------------

def generate_fake_embeddings(path: Path) -> list[str]:
    """Write synthetic WindowEmbeddingRecord JSONL. Returns list of window_ids."""
    rng = np.random.default_rng(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Each cluster is a tight cloud around a random centroid
    centroids = rng.standard_normal((N_CLUSTERS, EMBED_DIM)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    window_ids: list[str] = []
    records = []

    for cluster_idx in range(N_CLUSTERS):
        for point_idx in range(N_PER_CLUSTER):
            # Small gaussian noise around centroid -> tight cluster
            vec = centroids[cluster_idx] + rng.standard_normal(EMBED_DIM).astype(np.float32) * 0.05
            vec /= np.linalg.norm(vec) + 1e-12

            seg_id = f"segment_{cluster_idx:02d}{point_idx:03d}_abc123def456"
            window_id = f"{seg_id}_w000"
            window_ids.append(window_id)

            records.append({
                "window_id": window_id,
                "scene_token_hex": f"{seg_id}_w000",
                "log_id": seg_id,
                "scenario_tags": SCENARIO_TAGS[cluster_idx],
                "window_start_ts": 1_000_000 * (cluster_idx * N_PER_CLUSTER + point_idx),
                "window_end_ts": 1_000_000 * (cluster_idx * N_PER_CLUSTER + point_idx + 1),
                "camera_set": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
                "embedding": vec.tolist(),
                "quality": {
                    "total_ticks": 20,
                    "valid_ticks": 20,
                    "complete_tick_rate": 1.0,
                    "dropped_ticks": 0,
                },
                "metadata": {"dataset": "waymo_smoke_test", "cluster_hint": cluster_idx},
            })

    # Noise points: random unit vectors far from the centroids
    for noise_idx in range(N_NOISE):
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12

        seg_id = f"segment_noise{noise_idx:03d}_ffffff000000"
        window_id = f"{seg_id}_w000"
        window_ids.append(window_id)

        records.append({
            "window_id": window_id,
            "scene_token_hex": f"{seg_id}_w000",
            "log_id": seg_id,
            "scenario_tags": ["anomaly"],
            "window_start_ts": 999_000_000 + noise_idx * 1_000_000,
            "window_end_ts": 1_000_000_000 + noise_idx * 1_000_000,
            "camera_set": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
            "embedding": vec.tolist(),
            "quality": {
                "total_ticks": 20,
                "valid_ticks": 18,
                "complete_tick_rate": 0.9,
                "dropped_ticks": 2,
            },
            "metadata": {"dataset": "waymo_smoke_test", "cluster_hint": -1},
        })

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total = N_CLUSTERS * N_PER_CLUSTER + N_NOISE
    print(f"[smoke] Wrote {total} fake embeddings -> {path}")
    return window_ids


# ---------------------------------------------------------------------------
# Step 2: Run clustering
# ---------------------------------------------------------------------------

def run_clustering() -> bool:
    """Invoke waymo_cluster_embeddings as a subprocess. Returns True on success."""
    # Use small UMAP/HDBSCAN params appropriate for synthetic data size
    n_samples = N_CLUSTERS * N_PER_CLUSTER + N_NOISE  # 88
    umap_components = min(10, n_samples - 2)
    n_neighbors = min(5, n_samples - 2)
    min_cluster_size = max(3, N_PER_CLUSTER // 4)

    cmd = [
        sys.executable, "-u", "-m", "waymo_pipeline.waymo_cluster_embeddings",
        "--input-jsonl", str(EMBED_JSONL),
        "--output-npz", str(NPZ_PATH),
        "--output-jsonl", str(CLUSTERS_JSONL),
        "--flagged-jsonl", str(FLAGGED_JSONL),
        "--umap-components", str(umap_components),
        "--n-neighbors", str(n_neighbors),
        "--min-cluster-size", str(min_cluster_size),
        "--min-samples", "3",
        "--seed", str(SEED),
    ]

    print(f"\n[smoke] Running clustering  (n_samples={n_samples}, umap_components={umap_components})")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, text=True)
    if result.returncode != 0:
        print("[smoke] CLUSTERING FAILED")
        return False

    n_clusters = sum(1 for line in CLUSTERS_JSONL.read_text().splitlines()
                     if json.loads(line).get("cluster_label", -1) >= 0)
    n_noise = sum(1 for line in CLUSTERS_JSONL.read_text().splitlines()
                  if json.loads(line).get("cluster_label", -1) == -1)
    print(f"[smoke] Clustering done: {n_clusters} clustered, {n_noise} noise")
    return True


# ---------------------------------------------------------------------------
# Step 3: Hit runner endpoints
# ---------------------------------------------------------------------------

def check_runner() -> bool:
    """Quick connectivity check."""
    try:
        r = requests.get(f"{RUNNER_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"[smoke] Runner healthy: {r.json()}")
            return True
    except requests.ConnectionError:
        pass
    print(f"[smoke] Runner not reachable at {RUNNER_URL}. Start it with:")
    print(f"        source .venv/bin/activate && uvicorn waymo_pipeline.waymo_runner:app --port 8000")
    return False


def test_cluster_space() -> bool:
    r = requests.get(f"{RUNNER_URL}/cluster-space", timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] /cluster-space -> {r.status_code}")
        return False
    data = r.json()
    points = data.get("points", [])
    stats = data.get("clusterStats", [])
    if not points:
        print("[FAIL] /cluster-space returned 0 points")
        return False
    print(f"[PASS] /cluster-space  points={len(points)}  clusters={len(stats)}")

    # Spot-check point shape
    p = points[0]
    for key in ("id", "x", "y", "z", "clusterId", "sceneId", "isNoise"):
        if key not in p:
            print(f"[FAIL] Point missing field: {key}")
            return False
    print(f"       Sample point: id={p['id'][:30]}  cluster={p['clusterId']}  noise={p['isNoise']}")
    return True


def test_scene_endpoint(window_ids: list[str]) -> bool:
    # Pick a non-noise point to test
    clusters = [json.loads(l) for l in CLUSTERS_JSONL.read_text().splitlines() if l.strip()]
    non_noise = [c for c in clusters if c.get("cluster_label", -1) >= 0]
    wid = non_noise[0]["window_id"] if non_noise else window_ids[0]

    r = requests.get(f"{RUNNER_URL}/scenes/{wid}", timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] /scenes/{wid[:30]} -> {r.status_code}")
        return False
    data = r.json()
    for key in ("id", "videoUrl", "annotations"):
        if key not in data:
            print(f"[FAIL] Scene response missing field: {key}")
            return False
    print(f"[PASS] /scenes/{{id}}  id={data['id'][:30]}")
    print(f"       videoUrl={data['videoUrl']!r:.60}  (empty = expected, no real video)")
    print(f"       annotations={data['annotations']}")
    return True


def test_scenarios() -> bool:
    r = requests.get(f"{RUNNER_URL}/scenarios", timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] /scenarios -> {r.status_code}")
        return False
    data = r.json()
    scenarios = data.get("scenarios", [])
    print(f"[PASS] /scenarios  count={len(scenarios)}  (0 expected — no debate run yet)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cluster", action="store_true",
                        help="Skip clustering, reuse existing outputs/waymo cluster files.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep fake embedding JSONL after test.")
    args = parser.parse_args()

    print("=" * 60)
    print("Verity Waymo smoke test")
    print("=" * 60)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # --- Step 1: generate fake embeddings ---
    window_ids = generate_fake_embeddings(EMBED_JSONL)

    # --- Step 2: cluster ---
    if not args.skip_cluster:
        ok = run_clustering()
        if not ok:
            sys.exit(1)
    else:
        print("[smoke] Skipping clustering (--skip-cluster)")

    # --- Step 3: API checks ---
    print()
    if not check_runner():
        print("\n[smoke] Skipping API checks — start the runner first.")
        print("        Embeddings and cluster files are written, clustering worked.")
        sys.exit(0)

    results = [
        test_cluster_space(),
        test_scene_endpoint(window_ids),
        test_scenarios(),
    ]

    print()
    if all(results):
        print("=" * 60)
        print("ALL CHECKS PASSED — open http://localhost:3000 and go to Cluster Space")
        print("=" * 60)
    else:
        print("=" * 60)
        print(f"SOME CHECKS FAILED ({results.count(False)}/{len(results)})")
        print("=" * 60)
        sys.exit(1)

    if not args.keep:
        EMBED_JSONL.unlink(missing_ok=True)
        print(f"[smoke] Cleaned up {EMBED_JSONL.name}  (--keep to retain)")


if __name__ == "__main__":
    main()
