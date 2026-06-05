"""Verity 10-video experiment driver.

The README's `python -m pipeline.run` CLI does not exist in this repo, so this
script wires the existing pipeline modules together for a real run on footage in
a GCS bucket.

It reuses the *actual* pipeline code:
  - pipeline.modules.encoder.reasoning_arm.ReasoningArm   (retry + JSON extract)
  - pipeline.modules.encoder.vocabulary.DEFAULT_VOCABULARY (validation)
  - pipeline.modules.hypothesizer                          (frequency statistics)

The only substitution is the VLM client. The intended model, Cosmos-Reason2, is
not accessible to this NVIDIA account (hosted 404 / nvcr Access Denied), so this
uses a frame-sampling client over an accessible vision model (Llama-3.2-vision).
The ReasoningArm only depends on the `complete(video_url, prompt) -> str`
protocol, so swapping Cosmos-Reason2 back in is a one-line change.

Usage:
    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_run --limit 10 --frames 4 --out outputs/run10
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.encoder.reasoning_arm import ReasoningArm, VLMUnavailableError
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
from pipeline.modules.hypothesizer import Hypothesizer
from pipeline.modules.hypothesizer.config import HypothesizerConfig, HypothesizerEmptyInputError
from pipeline.modules.hypothesizer.frequency import compute_frequencies, extract_atoms

BUCKET = os.environ.get("WAYMO_DEST_BUCKET", "nvidia-adr-waymo-segment-videos")
GCS_PROJECT = os.environ.get("GCS_PROJECT", "nvidia-adr")


# ---------------------------------------------------------------------------
# Frame-sampling VLM client (drop-in for CosmosReason2Client)
# ---------------------------------------------------------------------------

class LocalCosmosVideoClient:
    """VLMClient that sends the actual video to a (local) Cosmos NIM.

    `video_url` is interpreted as a LOCAL FILE PATH. The clip is downscaled /
    optionally trimmed with ffmpeg and sent as a base64 ``data:video/mp4`` URI,
    which the local Cosmos-Reason NIM decodes directly. This is the real
    video-reasoning path (no frame sampling).

    Identical API call to the pipeline's CosmosReason2Client.complete — the only
    difference is we build the payload from a local file. To target hosted
    Cosmos-Reason2 instead, point base_url at integrate.api.nvidia.com and pass
    a fetchable URL.
    """

    def __init__(self, model_id: str, base_url: str, api_key: str | None = None,
                 max_seconds: float | None = None, width: int = 512, crf: int = 30) -> None:
        self.model_id = model_id
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "local")
        self._max_seconds = max_seconds
        self._w = width
        self._crf = crf

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _prep_data_uri(self, path: str) -> str:
        """Downscale/transcode the clip and return a base64 data URI."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "clip.mp4")
            cmd = ["ffmpeg", "-y", "-loglevel", "error"]
            if self._max_seconds:
                cmd += ["-t", f"{self._max_seconds:.2f}"]
            cmd += ["-i", path, "-vf", f"scale={self._w}:-2", "-an",
                    "-c:v", "libx264", "-crf", str(self._crf), out]
            subprocess.run(cmd, capture_output=True, timeout=180)
            b = Path(out).read_bytes()
        return "data:video/mp4;base64," + base64.b64encode(b).decode()

    def complete(self, video_url: str, prompt: str) -> str:
        uri = self._prep_data_uri(video_url)
        client = self._client()
        last = None
        for _ in range(3):  # transient retry
            try:
                r = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url": uri}},
                        {"type": "text", "text": prompt}]}],
                    max_tokens=1024, temperature=0.0)
                return r.choices[0].message.content or ""
            except Exception as exc:
                last = exc
        raise VLMUnavailableError(self.model_id, f"{type(last).__name__}: {last}")


# ---------------------------------------------------------------------------
# Bucket listing + download
# ---------------------------------------------------------------------------

def list_segments(limit: int, camera: str) -> list[tuple[str, str]]:
    """Return [(segment_id, blob_name)] for the first `limit` segments' camera."""
    from google.cloud import storage
    c = storage.Client(project=GCS_PROJECT)
    seen: dict[str, str] = {}
    for blob in c.list_blobs(BUCKET, prefix="segments/"):
        if not blob.name.endswith(f"_{camera}.mp4"):
            continue
        seg = blob.name.split("/")[1]
        if seg not in seen:
            seen[seg] = blob.name
        if len(seen) >= limit:
            break
    return sorted(seen.items())[:limit]


def download(blob_name: str, dest: Path) -> Path:
    from google.cloud import storage
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    c = storage.Client(project=GCS_PROJECT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    c.bucket(BUCKET).blob(blob_name).download_to_filename(str(dest))
    return dest


# ---------------------------------------------------------------------------
# Per-segment annotation
# ---------------------------------------------------------------------------

def annotate_segment(seg_id: str, blob_name: str, arm: ReasoningArm,
                     cache_dir: Path) -> SchemaRecord:
    local = cache_dir / f"{seg_id}_{Path(blob_name).name}"
    try:
        download(blob_name, local)
    except Exception as exc:
        print(f"[run] DOWNLOAD FAIL {seg_id}: {exc}", file=sys.stderr)
        return SchemaRecord(WindowKey(seg_id, 0), "reasoning", "1.0",
                            "v1_describe", {}, failure_mode="storage_error")
    try:
        fields, raw = arm.annotate(video_url=str(local), pose_summary=None,
                                   prompt_template_id="v1_describe")
        rec = SchemaRecord(WindowKey(seg_id, 0), "reasoning", "1.0", "v1_describe",
                           fields, failure_mode=None, raw_vlm_response=raw)
        print(f"[run] OK  {seg_id}: {fields.get('environment',{}).get('weather')}, "
              f"agents={fields.get('agents')}, cond={fields.get('conditions')}", file=sys.stderr)
        return rec
    except VLMUnavailableError as exc:
        print(f"[run] VLM UNAVAILABLE {seg_id}: {exc}", file=sys.stderr)
        return SchemaRecord(WindowKey(seg_id, 0), "reasoning", "1.0", "v1_describe",
                            {}, failure_mode="vlm_unavailable")
    except Exception as exc:
        print(f"[run] ANNOTATE FAIL {seg_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return SchemaRecord(WindowKey(seg_id, 0), "reasoning", "1.0", "v1_describe",
                            {}, failure_mode="invalid_json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--model", default=os.environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason1-7b"))
    ap.add_argument("--base-url", default=os.environ.get("NVIDIA_BASE_URL", "http://localhost:8081/v1"))
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Trim each clip to N seconds before sending (default: whole clip)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/run10")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("/tmp/verity_vid")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] listing first {args.limit} segments ({args.camera})...", file=sys.stderr)
    segs = list_segments(args.limit, args.camera)
    print(f"[run] {len(segs)} segments. model={args.model} @ {args.base_url}", file=sys.stderr)

    vlm = LocalCosmosVideoClient(args.model, base_url=args.base_url,
                                 max_seconds=args.max_seconds, width=args.width)
    arm = ReasoningArm(vlm=vlm, vocabulary=DEFAULT_VOCABULARY, max_retries=3, camera=args.camera)

    records: list[SchemaRecord] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(annotate_segment, s, b, arm, cache_dir): s for s, b in segs}
        for f in as_completed(futs):
            records.append(f.result())

    records.sort(key=lambda r: r.window_id.segment_id)

    # Persist schema records
    (out / "schema_records.json").write_text(
        json.dumps([r.to_json() for r in records], indent=2))

    ok = [r for r in records if r.succeeded]
    print(f"\n[run] annotated {len(ok)}/{len(records)} windows successfully", file=sys.stderr)

    # --- raw frequency statistics (always computed, even for small N) ---
    atom_sets = [extract_atoms(r.fields, None, None, str(r.window_id)) for r in ok]
    atom_sets = [a for a in atom_sets if a]
    marginal, pairwise = compute_frequencies(atom_sets)
    stats = {
        "n_windows": len(records),
        "n_succeeded": len(ok),
        "marginal_frequencies": dict(sorted(marginal.items(), key=lambda x: -x[1])),
        "pairwise_frequencies": dict(sorted(pairwise.items(), key=lambda x: -x[1])),
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2))

    # --- Hypothesizer novelty proposals (may be empty at small N) ---
    proposals = []
    try:
        proposals = Hypothesizer(HypothesizerConfig()).propose(ok, arm="reasoning")
    except HypothesizerEmptyInputError as exc:
        print(f"[run] Hypothesizer: {exc}", file=sys.stderr)
    (out / "proposals.json").write_text(
        json.dumps([p.to_json() for p in proposals], indent=2))

    print("\n" + "=" * 64)
    print(f"RESULTS  ({len(ok)}/{len(records)} windows annotated)")
    print("=" * 64)
    print("\nTop marginal attribute frequencies (fraction of windows):")
    for atom, fr in list(stats["marginal_frequencies"].items())[:25]:
        print(f"  {fr*100:5.0f}%  {atom}")
    print(f"\nNovelty proposals (default thresholds): {len(proposals)}")
    print(f"\nWrote: {out}/schema_records.json, statistics.json, proposals.json")


if __name__ == "__main__":
    main()
