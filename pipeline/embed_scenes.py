"""Scene-window JSONL to embedding-vector JSONL transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Iterable

import numpy as np
from dotenv import load_dotenv

from pipeline.models.scene_window import SceneWindow


def parse_args() -> argparse.Namespace:
    """Purpose: Parse CLI arguments for embedding stage.
    Parameters: None.
    Returns: argparse.Namespace with input/output and encoder options.
    Called by: main().
    Calls: argparse.ArgumentParser().
    """
    parser = argparse.ArgumentParser(description="Embed scene-window artifacts.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--encoder", default="fake")
    parser.add_argument("--embedding-dim", type=int, default=256)
    return parser.parse_args()


def iter_scene_windows(path: str) -> Iterable[SceneWindow]:
    """Purpose: Stream parse SceneWindow records from JSONL.
    Parameters:
        path (str): JSONL file path containing scene artifacts.
    Returns:
        Iterable[SceneWindow]: Parsed scene-window objects.
    Called by: main().
    Calls: json.loads(), SceneWindow.model_validate().
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield SceneWindow.model_validate(json.loads(line))


def fake_encode_scene(scene: SceneWindow, embedding_dim: int) -> np.ndarray:
    """Purpose: Deterministic placeholder encoder for pipeline bring-up.
    Parameters:
        scene (SceneWindow): Extracted scene artifact.
        embedding_dim (int): Target vector dimension.
    Returns:
        np.ndarray: Deterministic float32 embedding vector.
    Called by: encode_scene().
    Calls: hashlib.sha256(), numpy.random.default_rng().
    """
    seed_material = (
        f"{scene.scene_token_hex}|{scene.log_id}|{len(scene.ticks)}|"
        f"{','.join(scene.scenario_tags)}"
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(size=(embedding_dim,), dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


def encode_scene(scene: SceneWindow, encoder: str, embedding_dim: int) -> np.ndarray:
    """Purpose: Dispatch scene encoding by configured encoder backend.
    Parameters:
        scene (SceneWindow): Scene artifact with temporal tick payload.
        encoder (str): Encoder backend id (`fake` for now).
        embedding_dim (int): Target vector dimension.
    Returns:
        np.ndarray: Scene embedding vector.
    Called by: main().
    Calls: fake_encode_scene().
    """
    if encoder == "fake":
        return fake_encode_scene(scene, embedding_dim)
    raise NotImplementedError(
        f"Encoder '{encoder}' not implemented yet. Use --encoder fake for retrieval pipeline validation."
    )


def main() -> None:
    """Purpose: Generate embeddings for each scene and write JSONL output.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: iter_scene_windows(), encode_scene(), json.dumps().
    """
    load_dotenv()
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    count = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for scene in iter_scene_windows(args.input_jsonl):
            emb = encode_scene(
                scene=scene,
                encoder=args.encoder,
                embedding_dim=args.embedding_dim,
            )
            record = {
                "scene_token_hex": scene.scene_token_hex,
                "log_id": scene.log_id,
                "scenario_tags": scene.scenario_tags,
                "quality": scene.quality.model_dump(),
                "embedding": emb.astype(float).tolist(),
            }
            out.write(json.dumps(record) + "\n")
            count += 1
    print(f"Embedded {count} scenes -> {args.output_jsonl}")


if __name__ == "__main__":
    main()

