"""Run the discovery 'proposal algo' on pre-rendered segment MP4s.

Lists segment FRONT clips directly (handles the nested
`segments/{id}/{id}_FRONT.mp4` layout that FlatMP4Storage can't round-trip),
encodes each via the local Cosmos NIM, then runs the REAL pipeline modules:
Encoder -> Hypothesizer (proposal algo) -> Scorer. Writes the scored proposals
to outputs/waymo/judge_scored.json, which the Judge UI hot-reloads.

    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_judge --limit 30      # quick look
    .venv/bin/python -m drivers.verity_judge --limit 0       # all segments
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.modules.encoder.reasoning_arm import ReasoningArm
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
from pipeline.modules.hypothesizer import Hypothesizer
from pipeline.modules.hypothesizer.config import HypothesizerConfig, HypothesizerEmptyInputError
from pipeline.modules.scorer import NIMTextClient, Scorer

# Reuse the working local-NIM video client + bucket listing from verity_run.
from drivers.verity_run import LocalCosmosVideoClient, list_segments, annotate_segment

OUT = Path("outputs/waymo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="0 = all segments")
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--max-seconds", type=float, default=8.0,
                    help="Trim each clip to N seconds for the encoder (faster).")
    ap.add_argument("--out", default=str(OUT / "judge_scored.json"))
    args = ap.parse_args()

    cache_dir = Path("/tmp/verity_vid"); cache_dir.mkdir(parents=True, exist_ok=True)
    segs = list_segments(args.limit if args.limit else 10_000, args.camera)
    print(f"[judge] {len(segs)} segments | encoding via local NIM...", file=sys.stderr)

    vlm = LocalCosmosVideoClient(
        model_id=__import__("os").environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason1-7b"),
        base_url=__import__("os").environ.get("NVIDIA_BASE_URL", "http://localhost:8081/v1"),
        max_seconds=args.max_seconds, width=args.width)
    arm = ReasoningArm(vlm=vlm, vocabulary=DEFAULT_VOCABULARY, max_retries=3, camera=args.camera)

    # --- Stage 1: encode every segment -> SchemaRecord -------------------------
    records: list[SchemaRecord] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(annotate_segment, s, b, arm, cache_dir): s for s, b in segs}
        done = 0
        for f in as_completed(futs):
            records.append(f.result()); done += 1
            if done % 10 == 0:
                print(f"[judge] encoded {done}/{len(segs)}", file=sys.stderr)
    ok = [r for r in records if r.succeeded]
    print(f"[judge] {len(ok)}/{len(records)} encoded OK", file=sys.stderr)
    if not ok:
        print("[judge] no usable records — aborting.", file=sys.stderr); sys.exit(1)

    # --- Stage 2: proposal algo (Hypothesizer) --------------------------------
    try:
        proposals = Hypothesizer(HypothesizerConfig()).propose(ok)
    except HypothesizerEmptyInputError as exc:
        print(f"[judge] hypothesizer: {exc}", file=sys.stderr); proposals = []
    print(f"[judge] {len(proposals)} proposals from {len(ok)} windows", file=sys.stderr)

    # --- Stage 3: Scorer (plausibility + frontier difficulty) -----------------
    scored = []
    if proposals:
        client = NIMTextClient()  # NVIDIA_BASE_URL + SCORER_NIM_MODEL_ID (local NIM)
        scorer = Scorer(plausibility_client=client, difficulty_client=client)
        scored = scorer.score_batch(proposals)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s.to_json() for s in scored], indent=2))
    n_acc = sum(1 for s in scored if s.accepted)
    # Also persist the schema records so the proposal algo can be re-run instantly.
    (OUT / "judge_schema_records.json").write_text(
        json.dumps([r.to_json() for r in records], indent=2))
    print(f"\n[judge] DONE — {len(scored)} scored, {n_acc} accepted -> {out}")
    print("[judge] Judge UI will hot-reload these.")


if __name__ == "__main__":
    main()
