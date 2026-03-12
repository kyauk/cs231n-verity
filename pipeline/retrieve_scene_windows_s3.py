"""CLI to retrieve nuPlan logs from S3 and emit scene-window JSONL artifacts.

Usage:
  python -m pipeline.retrieve_scene_windows_s3 \\
      --bucket my-nuplan-bucket \\
      --output-jsonl outputs/scene_windows.jsonl \\
      --max-logs 1
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.parse import urlparse
from typing import Any

from dotenv import load_dotenv

from pipeline.extract_scene_windows import extract_scene_windows
from pipeline.s3_retrieval import batched_logs, download_db_to_tempfile, list_log_db_objects, make_s3_client


def parse_args() -> argparse.Namespace:
    """Purpose: Parse CLI options for retrieval/extraction pipeline.
    Parameters: None.
    Returns: argparse.Namespace of parsed options.
    Called by: main().
    Calls: argparse.ArgumentParser().
    """
    parser = argparse.ArgumentParser(description="Retrieve scene windows from S3 nuPlan logs.")
    parser.add_argument("--bucket", required=False, help="S3 bucket containing logs and images.")
    parser.add_argument(
        "--db-prefix",
        default=os.getenv("S3_NUPLAN_DB_PREFIX", "mini-split/data/cache/"),
        help="S3 prefix for sqlite .db objects.",
    )
    parser.add_argument(
        "--camera-prefix-root",
        default=os.getenv(
            "S3_NUPLAN_CAMERA_PREFIX_ROOT", "camera_0/nuplan-v1.1_mini_camera_0"
        ),
        help="S3 camera root prefix before <log_id>/.",
    )
    parser.add_argument("--output-jsonl", default="outputs/scene_windows.jsonl")
    parser.add_argument("--max-logs", type=int, default=1)
    parser.add_argument("--logs-per-batch", type=int, default=1)
    parser.add_argument("--sync-tolerance-us", type=int, default=50_000)
    parser.add_argument("--min-complete-tick-rate", type=float, default=0.9)
    parser.add_argument(
        "--materialize-images-dir",
        default="",
        help="Optional local directory to download referenced image frames.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _scene_to_json(scene: Any) -> str:
    """Purpose: Serialize Pydantic scene model to a JSON line.
    Parameters:
        scene (Any): SceneWindow object.
    Returns:
        str: JSON string with deterministic key ordering.
    Called by: main().
    Calls: model_dump(), json.dumps().
    """
    return json.dumps(scene.model_dump(), sort_keys=True)


def _download_scene_images(
    s3_client: Any, scene: Any, materialize_images_dir: str
) -> None:
    """Purpose: Optionally download all referenced scene images locally.
    Parameters:
        s3_client (Any): boto3 S3 client.
        scene (Any): SceneWindow model.
        materialize_images_dir (str): Target local root directory.
    Returns:
        None.
    Called by: main().
    Calls: s3_client.download_file().
    """
    if not materialize_images_dir:
        return
    for tick in scene.ticks:
        for channel, uri in tick.frames_by_channel.items():
            parsed = urlparse(uri)
            if parsed.scheme != "s3":
                continue
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            target = os.path.join(
                materialize_images_dir,
                scene.log_id,
                scene.scene_token_hex,
                str(tick.lidar_timestamp),
                channel + os.path.splitext(key)[1],
            )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                s3_client.download_file(bucket, key, target)


def main() -> None:
    """Purpose: Run end-to-end S3 retrieval plus scene extraction JSONL emission.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: list_log_db_objects(), download_db_to_tempfile(), extract_scene_windows().
    """
    load_dotenv()
    args = parse_args()

    bucket = args.bucket or os.getenv("S3_NUPLAN_BUCKET")
    if not bucket:
        raise ValueError("Provide --bucket or set S3_NUPLAN_BUCKET in environment.")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    s3_client = make_s3_client()
    logs = list_log_db_objects(
        s3_client=s3_client,
        bucket=bucket,
        db_prefix=args.db_prefix,
        max_logs=args.max_logs,
    )
    print(f"Discovered {len(logs)} log db objects from s3://{bucket}/{args.db_prefix}")

    scene_count = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for batch_idx, batch in enumerate(batched_logs(logs, args.logs_per_batch), start=1):
            print(f"Processing batch {batch_idx} with {len(batch)} logs")
            for log_obj in batch:
                db_local_path = download_db_to_tempfile(s3_client, log_obj)
                try:
                    # nuPlan DB image.filename_jpg already includes log_id (e.g. log_id/CAM_B0/file.jpg).
                    # Do not append log_id again or S3 key becomes .../log_id/log_id/CAM_B0/...
                    image_prefix = f"s3://{bucket}/{args.camera_prefix_root}"
                    scenes = extract_scene_windows(
                        db_path=db_local_path,
                        log_id=log_obj.log_id,
                        image_s3_prefix=image_prefix,
                        tolerance_us=args.sync_tolerance_us,
                        min_complete_tick_rate=args.min_complete_tick_rate,
                    )
                    print(
                        f"log_id={log_obj.log_id} scenes_extracted={len(scenes)}"
                    )
                    if not args.dry_run:
                        for scene in scenes:
                            _download_scene_images(
                                s3_client=s3_client,
                                scene=scene,
                                materialize_images_dir=args.materialize_images_dir,
                            )
                            out.write(_scene_to_json(scene) + "\n")
                    scene_count += len(scenes)
                finally:
                    if os.path.exists(db_local_path):
                        os.remove(db_local_path)
    print(
        f"Completed retrieval. scene_count={scene_count}, dry_run={args.dry_run}, "
        f"output_jsonl={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()

