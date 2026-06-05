"""Ground-truth agent extraction from Waymo camera boxes — the judge-pipeline fix.

The reasoning VLM under-detects agents (verified: agents=[] for scenes full of
cars). Waymo ships ground-truth 2D camera boxes, so for agent fields we read the
labels instead of trusting the VLM. This:
  1. Quantifies the VLM's agent error (vs ground truth).
  2. Produces label-derived agent fields to replace the VLM `agents` list feeding
     the Hypothesizer.

FRONT camera only (camera_name==1), to match what the reasoning VLM saw.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from google.cloud import storage

SRC = "waymo_open_dataset_v_2_0_1"
PROJECT = "nvidia-adr"
FRONT = 1  # Waymo camera enum: FRONT=1
TYPE = {1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
# Map Waymo classes -> the encoder vocabulary's agents tags
VOCAB = {"vehicle": "car", "pedestrian": "pedestrian", "cyclist": "cyclist"}

_client = storage.Client(project=PROJECT)


def _read_camera_box(seg: str):
    import pyarrow.parquet as pq
    for split in ("validation", "training"):
        blob = _client.bucket(SRC).blob(f"{split}/camera_box/{seg}.parquet")
        if blob.exists():
            df = pq.read_table(io.BytesIO(blob.download_as_bytes())).to_pandas()
            return df, split
    return None, None


def gt_agents(seg: str) -> dict:
    df, split = _read_camera_box(seg)
    if df is None:
        return {"segment": seg, "error": "no camera_box parquet found"}
    f = df[df["key.camera_name"] == FRONT]
    tcol = "[CameraBoxComponent].type"
    idcol = "key.camera_object_id"
    fcol = "key.frame_timestamp_micros"

    distinct, perframe_max, perframe_mean = {}, {}, {}
    for code, name in TYPE.items():
        sub = f[f[tcol] == code]
        distinct[name] = int(sub[idcol].nunique())
        if len(sub):
            counts = sub.groupby(fcol)[idcol].nunique()
            perframe_max[name] = int(counts.max())
            perframe_mean[name] = round(float(counts.mean()), 1)
        else:
            perframe_max[name] = 0
            perframe_mean[name] = 0.0

    n_frames = int(f[fcol].nunique())
    # label-derived agent atoms for the schema (presence + density)
    present = [VOCAB[n] for n in ("vehicle", "pedestrian", "cyclist")
               if distinct.get(n, 0) > 0]
    max_simultaneous = sum(perframe_max.get(n, 0) for n in ("vehicle", "pedestrian", "cyclist"))
    if max_simultaneous == 0:
        density = "none"
    elif max_simultaneous <= 2:
        density = "sparse"
    elif max_simultaneous <= 6:
        density = "moderate"
    else:
        density = "heavy"
    return {
        "segment": seg,
        "split": split,
        "n_front_frames": n_frames,
        "distinct_objects": distinct,
        "per_frame_max": perframe_max,
        "per_frame_mean": perframe_mean,
        "agents_present_vocab": present,
        "multiple_agents": max_simultaneous >= 2,
        "agent_density": density,
    }


def main() -> None:
    records_path = Path("outputs/run10/schema_records.json")
    vlm = {}
    if records_path.exists():
        for r in json.loads(records_path.read_text()):
            vlm[r["window_id"]["segment_id"]] = r["fields"].get("agents", [])

    segs = list(vlm.keys()) if vlm else (sys.argv[1:] or [])
    out = []
    print(f"{'segment':<22} {'GT vehicles':<12} {'GT peds':<8} {'maxFrame':<9} {'VLM agents':<24} verdict")
    n_vlm_miss = 0
    for seg in segs:
        g = gt_agents(seg)
        out.append({**g, "vlm_agents": vlm.get(seg, [])})
        if "error" in g:
            print(f"{seg[:20]:<22} {g['error']}")
            continue
        d = g["distinct_objects"]
        gt_has = g["multiple_agents"] or d["vehicle"] > 0 or d["pedestrian"] > 0 or d["cyclist"] > 0
        vlm_has = bool(vlm.get(seg))
        miss = gt_has and not vlm_has
        n_vlm_miss += int(miss)
        verdict = "VLM MISSED ALL" if miss else ("ok" if vlm_has else "(both empty)")
        mx = g["per_frame_max"]
        print(f"{seg[:20]:<22} {d['vehicle']:<12} {d['pedestrian']:<8} "
              f"{mx['vehicle']+mx['pedestrian']+mx['cyclist']:<9} {str(vlm.get(seg,[]))[:22]:<24} {verdict}")

    Path("outputs/run10").mkdir(parents=True, exist_ok=True)
    Path("outputs/run10/agents_ground_truth.json").write_text(json.dumps(out, indent=2))
    n = len([o for o in out if "error" not in o])
    print(f"\nVLM missed all agents in {n_vlm_miss}/{n} segments that have ground-truth agents.")
    print("Wrote outputs/run10/agents_ground_truth.json")


if __name__ == "__main__":
    main()
