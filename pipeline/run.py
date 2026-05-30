"""Verity pipeline CLI — the consumer-facing entry point.

Drives the six pipeline modules end-to-end via three subcommands:

    python -m pipeline.run ingest  --source-format waymo_parquet \\
                                   --source-root /data/waymo \\
                                   --bucket gs://my-bucket/verity \\
                                   --segments all

    python -m pipeline.run analyze --bucket gs://my-bucket/verity \\
                                   --output outputs/session-1

    python -m pipeline.run report  --scored outputs/session-1/scored.json \\
                                   --ratings outputs/ratings/ \\
                                   --seeds outputs/seeds.json \\
                                   --output outputs/session-1

Pure consumer: imports only from each module's package-root public surface
(no submodule reaching, no internals). The composition root for the whole
pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# argparse — subcommand definitions
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description="Verity pipeline CLI — ingest fleet data, analyze for "
                    "underrepresented scenarios, generate evaluation reports.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    # --- ingest ----------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest",
        help="Slice raw fleet data into 8-second windows and upload to the bucket.",
        description="Run Module 1: Storage / IngestionPipeline.",
    )
    p_ingest.add_argument("--source-format", required=True,
                          choices=["waymo_parquet", "waymo_tfrecord"],
                          help="Source data format.")
    p_ingest.add_argument("--source-root", required=True,
                          help="Root directory containing the source data.")
    p_ingest.add_argument("--bucket", required=True,
                          help="Destination GCS bucket URI, e.g. gs://my-bucket/verity.")
    p_ingest.add_argument("--segments", required=True,
                          help="Comma-separated segment IDs, '@path/to/list.txt', or 'all'.")
    p_ingest.add_argument("--force", action="store_true",
                          help="Re-ingest segments that already exist in the bucket.")
    p_ingest.add_argument("--window-length-frames", type=int, default=80,
                          help="Window length in frames (default 80 = 8 s at 10 Hz).")
    p_ingest.add_argument("--target-fps", type=int, default=10,
                          help="Target FPS for encoded MP4s (default 10).")
    p_ingest.set_defaults(func=_run_ingest)

    # --- analyze ---------------------------------------------------------------
    p_analyze = sub.add_parser(
        "analyze",
        help="Annotate windows, find novel compositions, score them.",
        description="Run Modules 2–4: Encoder → Hypothesizer → Scorer.",
    )
    p_analyze.add_argument("--bucket", required=True,
                           help="Bucket URI written by `ingest`.")
    p_analyze.add_argument("--output", required=True,
                           help="Directory where schema_records / proposals / scored JSON go.")
    p_analyze.add_argument("--max-workers", type=int, default=8,
                           help="Concurrent VLM requests per arm (default 8).")
    p_analyze.add_argument("--stub", action="store_true",
                           help="Use stub clients instead of calling NVIDIA NIM "
                                "(offline / CI mode).")
    p_analyze.add_argument("--no-visual", action="store_true",
                           help="Skip the visual arm (reasoning arm only).")
    p_analyze.add_argument("--cache-root",
                           help="Override cache root (default: project /cache).")
    p_analyze.add_argument("--sign-as",
                           help="Service-account email to impersonate for signed URLs. "
                                "Required if ADC is a user refresh-token (cannot sign v4).")
    p_analyze.add_argument("--storage-mode", default="canonical",
                           choices=["canonical", "flat_mp4"],
                           help="canonical = ingested layout (default). "
                                "flat_mp4 = flat bucket of MP4 files, one per segment.")
    p_analyze.add_argument("--cameras",
                           help="Comma-separated camera names. REQUIRED with "
                                "--storage-mode flat_mp4. Example: 'FRONT' or "
                                "'FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT'.")
    p_analyze.set_defaults(func=_run_analyze)

    # --- report ----------------------------------------------------------------
    p_report = sub.add_parser(
        "report",
        help="Aggregate ratings + scored proposals into the final evaluation report.",
        description="Run Module 6: Evaluator.",
    )
    p_report.add_argument("--scored", required=True,
                          help="Path to scored.json (written by `analyze`).")
    p_report.add_argument("--seeds", required=True,
                          help="Path to seeds.json — pre-registered seeded windows + subset labels.")
    p_report.add_argument("--output", required=True,
                          help="Directory where report.json / report.md / report.html go.")
    # Ratings source: filesystem dir OR a running judge_ui HTTP endpoint.
    ratings = p_report.add_mutually_exclusive_group(required=True)
    ratings.add_argument("--ratings",
                         help="Filesystem ratings directory ({rater_id}/{proposal_id}.json).")
    ratings.add_argument("--ratings-url",
                         help="URL of a running judge_ui server (e.g. http://localhost:8001) "
                              "— reads from GET /judge/ratings/export.")
    p_report.add_argument("--schema-records",
                          help="Optional: path to schema_records.json for failure-mode stats.")
    p_report.add_argument("--recall-k", type=int, default=30,
                          help="Primary recall threshold K (default 30).")
    p_report.set_defaults(func=_run_report)

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers — implemented in later slices
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers — wiring + arg parsing
# ---------------------------------------------------------------------------

def _build_source(source_format: str, source_root: str) -> Any:
    """Construct the appropriate SourceAdapter for the requested format.

    Conventions for `source_root` by format:
      waymo_parquet:  gs://<bucket>/<prefix>   (GCS-backed)
      waymo_tfrecord: /local/path/to/dir        (local directory)
    """
    from pipeline.modules.storage import WaymoParquetSource, WaymoTFRecordSource

    if source_format == "waymo_parquet":
        if not source_root.startswith("gs://"):
            raise ValueError(
                f"waymo_parquet requires --source-root as 'gs://bucket/prefix', got {source_root!r}"
            )
        no_scheme = source_root.removeprefix("gs://").rstrip("/")
        bucket, _, prefix = no_scheme.partition("/")
        if not bucket or not prefix:
            raise ValueError(
                "--source-root must include both bucket and prefix: gs://bucket/prefix"
            )
        return WaymoParquetSource(bucket=bucket, prefix=prefix)

    if source_format == "waymo_tfrecord":
        if source_root.startswith("gs://"):
            raise ValueError(
                f"waymo_tfrecord requires a local --source-root, got {source_root!r}"
            )
        return WaymoTFRecordSource(directory=source_root)

    raise ValueError(f"Unknown source format: {source_format!r}")


def _parse_segments_arg(segments_arg: str, source: Any) -> list[str]:
    """Resolve --segments into a list of segment IDs.

    Accepted forms:
      "all"                  → source.list_segments()
      "@path/to/file.txt"    → one ID per non-empty line
      "id_a,id_b,id_c"       → comma-separated
    """
    if segments_arg == "all":
        ids = list(source.list_segments())
        if not ids:
            raise ValueError("source.list_segments() returned no segments.")
        return ids

    if segments_arg.startswith("@"):
        path = Path(segments_arg[1:])
        if not path.exists():
            raise FileNotFoundError(f"--segments @{path} does not exist.")
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not ids:
            raise ValueError(f"--segments @{path} is empty.")
        return ids

    ids = [s.strip() for s in segments_arg.split(",") if s.strip()]
    if not ids:
        raise ValueError("--segments is empty.")
    return ids


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _run_ingest(args: argparse.Namespace) -> int:
    from pipeline.modules.storage import (
        IngestionPipeline,
        IngestionRequest,
        SourceUnreachableError,
        SourceSchemaVersionError,
        WindowConfig,
    )

    try:
        source = _build_source(args.source_format, args.source_root)
        segment_ids = _parse_segments_arg(args.segments, source)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[pipeline.run ingest] {exc}", file=sys.stderr)
        return 2

    window_config = WindowConfig(
        length_frames=args.window_length_frames,
        stride_frames=args.window_length_frames,  # non-overlapping by default
        target_fps=args.target_fps,
    )
    request = IngestionRequest(
        segment_ids=segment_ids,
        bucket_uri=args.bucket,
        window_config=window_config,
        source=source,
        force_reingest=args.force,
    )

    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(request)
    except (SourceUnreachableError, SourceSchemaVersionError) as exc:
        print(f"[pipeline.run ingest] fatal: {exc}", file=sys.stderr)
        return 2

    print(
        f"[pipeline.run ingest] DONE — bucket={result.bucket_uri} "
        f"segments succeeded={result.segments_succeeded} "
        f"failed={result.segments_failed} skipped={result.segments_skipped} "
        f"windows_total={result.windows_total}"
    )
    return 0


def _build_encoder(
    stub: bool,
    no_visual: bool,
    cache_root: str | None,
    cameras: list[str] | None = None,
) -> Any:
    """Build the Encoder with production or stub clients per --stub.

    Visual arm is on by default and off with --no-visual. Stub mode swaps in
    the offline clients but keeps the visual arm wired (so the offline path
    exercises the same code as production).

    `cameras`, when set, configures the visual arm to embed only those cameras
    (used with --storage-mode flat_mp4 where the storage has a fixed camera
    set). Embedding dimension becomes len(cameras) * 256.
    """
    import os
    from pipeline.modules.encoder import (
        CosmosEmbed1Client, CosmosReason2Client, DEFAULT_VOCABULARY, Encoder,
        StubEmbedClient, StubVLMClient, VisualArm,
    )

    if stub:
        vlm: Any = StubVLMClient()
        embed_client: Any = StubEmbedClient()
    else:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Either export it (or load .env) "
                "or pass --stub to use offline clients."
            )
        vlm = CosmosReason2Client(api_key=api_key)
        embed_client = CosmosEmbed1Client(url=os.environ.get("COSMOS_EMBED1_URL", ""))

    if no_visual:
        visual = None
    elif cameras is not None:
        visual = VisualArm(client=embed_client, cameras=cameras)
    else:
        visual = VisualArm(client=embed_client)
    return Encoder(
        vlm=vlm,
        vocabulary=DEFAULT_VOCABULARY,
        visual_arm=visual,
        cache_root=Path(cache_root) if cache_root else None,
    )


def _build_scorer(stub: bool, cache_root: str | None) -> Any:
    from pipeline.modules.scorer import (
        NIMTextClient, Scorer, StubDifficultyClient, StubPlausibilityClient,
    )
    import os

    if stub:
        return Scorer(
            plausibility_client=StubPlausibilityClient(),
            difficulty_client=StubDifficultyClient(),
            cache_root=Path(cache_root) if cache_root else None,
        )

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Either export it (or load .env) "
            "or pass --stub to use offline clients."
        )
    client = NIMTextClient(api_key=api_key)
    return Scorer(
        plausibility_client=client,
        difficulty_client=client,
        cache_root=Path(cache_root) if cache_root else None,
    )


def _write_json_list(path: Path, items: list[Any]) -> None:
    """Atomically write [item.to_json() for item in items] to `path`."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps([it.to_json() for it in items], indent=2))
    tmp.replace(path)


def _run_analyze(args: argparse.Namespace) -> int:
    from pipeline.interfaces.errors import WindowStorageError
    from pipeline.modules.encoder import WindowInput
    from pipeline.modules.hypothesizer import Hypothesizer
    from pipeline.modules.storage import FlatMP4Storage, WindowStorage

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Storage + window list ----------------------------------------
    if args.storage_mode == "flat_mp4":
        if not args.cameras:
            print(
                "[pipeline.run analyze] --storage-mode flat_mp4 requires "
                "--cameras (comma-separated, e.g. --cameras FRONT). The customer "
                "must declare which cameras their MP4 bucket contains so the "
                "visual-arm embedding dimensionality is explicit.",
                file=sys.stderr,
            )
            return 2
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
        storage: Any = FlatMP4Storage(
            bucket_uri=args.bucket, cameras=cameras, sign_as=args.sign_as,
        )
    else:
        cameras = None
        storage = WindowStorage(bucket_uri=args.bucket, sign_as=args.sign_as)

    try:
        windows = storage.list_windows()
    except WindowStorageError as exc:
        print(
            f"[pipeline.run analyze] cannot list windows at {args.bucket}: {exc}\n"
            f"  → Check the bucket URI, your GCS credentials (gcloud auth "
            f"application-default login), and that the prefix actually contains "
            f"ingested windows (or MP4s for --storage-mode flat_mp4).",
            file=sys.stderr,
        )
        return 2
    if not windows:
        print(
            f"[pipeline.run analyze] no windows found at {args.bucket} — "
            f"did you run `ingest` first?",
            file=sys.stderr,
        )
        return 2
    print(f"[pipeline.run analyze] found {len(windows)} windows", file=sys.stderr)

    # --- 2. Encoder ------------------------------------------------------
    try:
        encoder = _build_encoder(
            args.stub, args.no_visual, args.cache_root, cameras=cameras,
        )
    except RuntimeError as exc:
        print(f"[pipeline.run analyze] {exc}", file=sys.stderr)
        return 2

    inputs = [
        WindowInput(segment_id=w.segment_id, window_idx=w.window_idx, storage=storage)
        for w in windows
    ]
    records = encoder.process_batch(inputs)
    _write_json_list(output_dir / "schema_records.json", records)

    # --- 3. Hypothesizer (reasoning records only) ------------------------
    reasoning_records = [r for r in records if r.arm == "reasoning"]
    n_succeeded = sum(1 for r in reasoning_records if r.succeeded)
    if n_succeeded == 0:
        print(
            f"[pipeline.run analyze] 0/{len(reasoning_records)} reasoning records "
            f"succeeded — cannot hypothesize. Check encoder failure_modes in "
            f"schema_records.json.",
            file=sys.stderr,
        )
        return 2

    proposals = Hypothesizer().propose(reasoning_records)
    _write_json_list(output_dir / "proposals.json", proposals)
    print(
        f"[pipeline.run analyze] {n_succeeded}/{len(reasoning_records)} reasoning "
        f"records succeeded → {len(proposals)} proposals",
        file=sys.stderr,
    )

    if not proposals:
        print(
            "[pipeline.run analyze] no proposals passed the Hypothesizer filters — "
            "writing empty scored.json and exiting cleanly.",
            file=sys.stderr,
        )
        _write_json_list(output_dir / "scored.json", [])
        return 0

    # --- 4. Scorer -------------------------------------------------------
    try:
        scorer = _build_scorer(args.stub, args.cache_root)
    except RuntimeError as exc:
        print(f"[pipeline.run analyze] {exc}", file=sys.stderr)
        return 2

    scored = scorer.score_batch(proposals)
    _write_json_list(output_dir / "scored.json", scored)

    n_accepted = sum(1 for s in scored if s.accepted)
    print(
        f"[pipeline.run analyze] DONE — {len(scored)} scored, {n_accepted} accepted. "
        f"Outputs: {output_dir}/"
    )
    return 0


def _load_json(path: Path) -> Any:
    """Load a JSON file. Raises FileNotFoundError or json.JSONDecodeError loudly."""
    import json
    return json.loads(path.read_text())


def _load_ratings_from_dir(ratings_dir: Path) -> list[Any]:
    """Walk {ratings_dir}/{rater_id}/{proposal_id}.json and parse each as Rating."""
    from pipeline.interfaces.rating import Rating

    if not ratings_dir.exists() or not ratings_dir.is_dir():
        raise FileNotFoundError(f"--ratings directory does not exist: {ratings_dir}")

    ratings: list[Any] = []
    for rater_dir in sorted(ratings_dir.iterdir()):
        if not rater_dir.is_dir():
            continue
        for rating_file in sorted(rater_dir.glob("*.json")):
            ratings.append(Rating.from_json(_load_json(rating_file)))
    return ratings


def _load_ratings_from_url(ratings_url: str) -> list[Any]:
    """GET {ratings_url}/judge/ratings/export and parse each row as Rating."""
    from pipeline.interfaces.rating import Rating
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("--ratings-url requires `requests`. pip install requests.") from exc

    url = ratings_url.rstrip("/") + "/judge/ratings/export"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [Rating.from_json(d) for d in resp.json()]


def _load_seeds(seeds_path: Path) -> tuple[list[Any], dict[Any, str]]:
    """Parse a seeds JSON file → (seeded_window_ids, seeded_subset_labels).

    Expected shape:
        {
          "seeded_windows": [
            {"window": "seg_001/0000", "subset": "familiar"},
            {"window": {"segment_id": "seg_002", "window_idx": 1}, "subset": "unfamiliar"}
          ]
        }

    `window` may be the canonical "segment_id/window_idx" string OR the
    dict form. `subset` must be exactly "familiar" or "unfamiliar".
    """
    from pipeline.interfaces.window import WindowKey

    if not seeds_path.exists():
        raise FileNotFoundError(f"--seeds file does not exist: {seeds_path}")
    raw = _load_json(seeds_path)
    entries = raw.get("seeded_windows", [])
    if not entries:
        raise ValueError(f"--seeds: 'seeded_windows' is empty or missing in {seeds_path}")

    ids: list[Any] = []
    labels: dict[Any, str] = {}
    for entry in entries:
        win_raw = entry["window"]
        subset = entry["subset"]
        if subset not in ("familiar", "unfamiliar"):
            raise ValueError(
                f"--seeds: subset must be 'familiar' or 'unfamiliar', got {subset!r}"
            )
        key = WindowKey.from_str(win_raw) if isinstance(win_raw, str) else WindowKey.from_json(win_raw)
        ids.append(key)
        labels[key] = subset
    return ids, labels


def _group_scored_by_arm(scored: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for sp in scored:
        grouped.setdefault(sp.arm, []).append(sp)
    return grouped


def _run_report(args: argparse.Namespace) -> int:
    from pipeline.interfaces.proposal import ScoredProposal
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.evaluation import (
        ArmMismatchError, EvaluationInput, Evaluator, MissingSubsetLabelsError,
    )

    output_dir = Path(args.output)

    # --- 1. Scored proposals --------------------------------------------
    try:
        scored_raw = _load_json(Path(args.scored))
    except FileNotFoundError as exc:
        print(f"[pipeline.run report] {exc}", file=sys.stderr)
        return 2
    scored = [ScoredProposal.from_json(d) for d in scored_raw]
    proposals_by_arm = _group_scored_by_arm(scored)
    if not proposals_by_arm:
        print(
            f"[pipeline.run report] no scored proposals in {args.scored} — nothing to evaluate.",
            file=sys.stderr,
        )
        return 2

    # --- 2. Seeds --------------------------------------------------------
    try:
        seeded_ids, seeded_labels = _load_seeds(Path(args.seeds))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[pipeline.run report] seeds: {exc}", file=sys.stderr)
        return 2

    # --- 3. Ratings (filesystem OR HTTP) ---------------------------------
    try:
        if args.ratings:
            ratings = _load_ratings_from_dir(Path(args.ratings))
        else:
            ratings = _load_ratings_from_url(args.ratings_url)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[pipeline.run report] ratings: {exc}", file=sys.stderr)
        return 2

    # --- 4. Optional schema records (for failure_mode_distribution) ------
    schema_records: list[Any] | None = None
    if args.schema_records:
        try:
            schema_records = [
                SchemaRecord.from_json(d) for d in _load_json(Path(args.schema_records))
            ]
        except FileNotFoundError as exc:
            print(f"[pipeline.run report] {exc}", file=sys.stderr)
            return 2

    # --- 5. Evaluate -----------------------------------------------------
    evaluator = Evaluator()
    try:
        report = evaluator.evaluate(EvaluationInput(
            proposals_by_arm=proposals_by_arm,
            ratings=ratings,
            seeded_window_ids=seeded_ids,
            seeded_subset_labels=seeded_labels,
            schema_records=schema_records,
            recall_k=args.recall_k,
        ))
    except (MissingSubsetLabelsError, ArmMismatchError) as exc:
        print(f"[pipeline.run report] {exc}", file=sys.stderr)
        return 2

    # --- 6. Save ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    written_path = evaluator.save(report, output_dir)
    print(
        f"[pipeline.run report] DONE — {len(ratings)} ratings, "
        f"{sum(len(v) for v in proposals_by_arm.values())} proposals across "
        f"{len(proposals_by_arm)} arm(s). Report at {written_path}"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
