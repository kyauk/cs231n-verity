"""Ingest 20 SMOOTH front-camera clips from the Waymo End-to-End TFRecords.

Each record is ONE front frame tagged `$<scenario-hash>-<frameidx>`, and a
scenario's frames are shuffled across ALL ~266 shards. So: pick 20 scenario IDs,
stream every shard, keep ONLY those IDs' FRONT frames (discard the rest —
memory-tiny), then order each by frame index and encode a smooth mp4. No
tensorflow — raw JPEG-marker scan + a regex for the ID.

    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_e2e_ingest --clips 20 --max-shards 266
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from google.cloud import storage

from pipeline.modules.storage.media import encode_mp4_ffmpeg

SRC_BUCKET = "waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEST_BUCKET = "nvidia-adr-waymo-segment-videos"
PROJECT = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
SHARD_SERIES = "test_202504211836-202504220845.tfrecord-"   # one 266-shard test file
_ID = re.compile(rb"\$([0-9a-f]{32})-(\d+)")


def _records(data: bytes):
    off = 0
    while off + 12 <= len(data):
        L = struct.unpack("<Q", data[off:off + 8])[0]
        rs, rend = off + 12, off + 12 + L
        if rend + 4 > len(data):
            break
        yield data[rs:rend]
        off = rend + 4


def _front_jpeg(rec: bytes, cam_idx: int) -> bytes | None:
    out, i = [], 0
    while len(out) <= cam_idx:
        s = rec.find(b"\xff\xd8\xff", i)
        if s < 0:
            break
        e = rec.find(b"\xff\xd9", s + 2)
        if e < 0:
            break
        out.append(rec[s:e + 2]); i = e + 2
    return out[cam_idx] if cam_idx < len(out) else None


def _even(img):
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def _upload(local: str, seg: str) -> str:
    blob = f"segments/{seg}/{seg}_FRONT.mp4"
    storage.Client(project=PROJECT).bucket(DEST_BUCKET).blob(blob).upload_from_filename(
        local, content_type="video/mp4")
    return f"gs://{DEST_BUCKET}/{blob}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--cam-idx", type=int, default=0, help="which JPEG is FRONT (0 = first)")
    ap.add_argument("--max-shards", type=int, default=266)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    c = storage.Client(project=PROJECT)
    shards = sorted(c.list_blobs(SRC_BUCKET, prefix=SHARD_SERIES), key=lambda b: b.name)
    shards = shards[: args.max_shards]
    print(f"[e2e] {len(shards)} shards in series", file=sys.stderr)

    # --- pick target scenario IDs from shard 0 (the ones with most frames there) ---
    d0 = shards[0].download_as_bytes()
    cnt0: dict[str, int] = defaultdict(int)
    for rec in _records(d0):
        m = _ID.search(rec)
        if m:
            cnt0[m.group(1).decode()] += 1
    targets = {cid for cid, _ in sorted(cnt0.items(), key=lambda kv: -kv[1])[: args.clips]}
    print(f"[e2e] picked {len(targets)} target scenarios; collecting their frames across shards...", file=sys.stderr)

    # --- stream every shard CONCURRENTLY, keep ONLY target frames (memory-tiny) ---
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    def _proc(sh) -> dict[str, dict[int, bytes]]:
        data = sh.download_as_bytes()
        part: dict[str, dict[int, bytes]] = defaultdict(dict)
        for rec in _records(data):
            m = _ID.search(rec)
            if not m:
                continue
            cid = m.group(1).decode()
            if cid in targets:
                j = _front_jpeg(rec, args.cam_idx)
                if j is not None:
                    part[cid][int(m.group(2))] = j
        return part

    frames: dict[str, dict[int, bytes]] = {t: {} for t in targets}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(_proc, sh) for sh in shards]):
            for cid, fm in f.result().items():
                frames[cid].update(fm)
            done += 1
            if done % 20 == 0 or done == len(shards):
                tot = sum(len(v) for v in frames.values())
                print(f"[e2e] {done}/{len(shards)} shards | frames collected: {tot} "
                      f"(min/clip {min(len(v) for v in frames.values())}, max {max(len(v) for v in frames.values())})",
                      file=sys.stderr)

    # --- encode each scenario's frames (ordered by index) -> smooth mp4 ---
    refs = []
    for k, (cid, fmap) in enumerate(sorted(frames.items(), key=lambda kv: -len(kv[1]))):
        if not fmap:
            continue
        seg = f"e2e_{cid[:16]}"
        imgs = [_even(cv2.imdecode(np.frombuffer(fmap[i], np.uint8), cv2.IMREAD_COLOR))
                for i in sorted(fmap)]
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / f"{seg}_FRONT.mp4"
            encode_mp4_ffmpeg(imgs, str(mp4), fps=args.fps)
            ref = _upload(str(mp4), seg)
        refs.append({"scene_id": seg, "ref": ref, "frames": len(imgs)})
        print(f"[e2e] clip {k+1} {seg} ({len(imgs)} frames) -> {ref}", file=sys.stderr)

    Path("outputs/waymo").mkdir(parents=True, exist_ok=True)
    Path("outputs/waymo/e2e_clips.json").write_text(json.dumps(refs, indent=2))
    print(f"\n[e2e] DONE — {len(refs)} smooth FRONT clips; ids -> outputs/waymo/e2e_clips.json")


if __name__ == "__main__":
    main()
