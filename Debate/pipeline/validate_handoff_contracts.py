"""Validate JSONL files against locked handoff Pydantic contracts."""

from __future__ import annotations

import argparse
import json
from typing import Type

from pydantic import BaseModel, ValidationError

from pipeline.models.handoff_contracts import (
    AnomalyResultRecord,
    DebateOutputRecord,
    DebateInputRecord,
    EmbeddingContractRecord,
    SceneDescriptionOutputRecord,
    SceneDescriptionInputRecord,
)


MODEL_BY_NAME: dict[str, Type[BaseModel]] = {
    "embedding": EmbeddingContractRecord,
    "anomaly": AnomalyResultRecord,
    "description_input": SceneDescriptionInputRecord,
    "description_output": SceneDescriptionOutputRecord,
    "debate_input": DebateInputRecord,
    "debate_output": DebateOutputRecord,
}


def parse_args() -> argparse.Namespace:
    """
    Purpose: Parse CLI args for contract validation utility.
    Parameters:
        None
    Returns:
        argparse.Namespace: Parsed args.
    Called by: main()
    Calls: argparse.ArgumentParser.parse_args()
    """

    parser = argparse.ArgumentParser(
        description="Validate JSONL against handoff contract models.",
    )
    parser.add_argument("--input-jsonl", required=True, help="JSONL file to validate.")
    parser.add_argument(
        "--contract",
        required=True,
        choices=sorted(MODEL_BY_NAME.keys()),
        help="Contract type to validate against.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Purpose: Validate each JSONL row against selected contract model.
    Parameters:
        None
    Returns:
        int: Exit code (0 if all valid, 1 otherwise).
    Called by: CLI entrypoint
    Calls: parse_args(), json.loads(), BaseModel.model_validate()
    """

    args = parse_args()
    model = MODEL_BY_NAME[args.contract]
    total = 0
    failed = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                model.model_validate(row)
            except (json.JSONDecodeError, ValidationError) as error:
                failed += 1
                print(f"line {line_no} invalid: {error}")

    print(f"validated rows: {total}")
    print(f"invalid rows: {failed}")
    if failed > 0:
        return 1
    print("contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
