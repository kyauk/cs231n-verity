# Handoff Contracts (Embeddings -> Anomaly -> Description -> Debate)

This file locks boundary schemas so re-encoded embeddings can be swapped in without breaking downstream stages.

## Source of truth

All handoff models live in:

- `pipeline/models/handoff_contracts.py`

Contracts are Pydantic models and should be used at module boundaries (no raw dict contracts).

## 1) Embeddings input contract

Model:

- `EmbeddingContractRecord`

Primary fields:

- `window_id`
- `scene_token_hex`
- `log_id`
- `scenario_tags`
- `window_start_ts`
- `window_end_ts`
- `camera_set`
- `embedding` (list[float], fixed dimension per run)
- `quality`
- `metadata`

Typical file:

- `outputs/window_embeddings_cosmos.jsonl`

## 2) Anomaly results contract

Model:

- `AnomalyResultRecord`

Primary fields:

- `window_id`
- `scene_token_hex`
- `log_id`
- `cluster_label`
- `is_noise`
- `cluster_probability`
- `outlier_score`
- `anomaly_rank`
- plus pass-through context (`scenario_tags`, `quality`, `metadata`)

Produced by:

- `python -m pipeline.anomaly_detect`

Typical file:

- `outputs/flagged_windows.jsonl`

Companion summary:

- `outputs/anomaly_summary.json`

## 3) Scene description input contract

Model:

- `SceneDescriptionInputRecord`

Primary fields:

- `run_id`
- `window_id`
- `scene_token_hex`
- `log_id`
- `cluster_label`
- `is_noise`
- `outlier_score`
- `anomaly_rank`
- `media_refs` (paths/URIs to grid/mp4/frames)
- `prompt_context`

Notes:

- Use this as the contract for Cosmos-Reason2 scene-description stage input.
- Keep `media_refs` populated from visual artifact manifest when available.

Scene description output contract:

- `SceneDescriptionOutputRecord`

Primary fields:

- `run_id`
- `window_id`
- `scene_description`
- `anomaly_rationale`
- `confidence`
- `model_source`

## 4) Debate input contract

Model:

- `DebateInputRecord`

Primary fields:

- `run_id`
- `window_id`
- `scene_token_hex`
- `log_id`
- `scene_description`
- `anomaly_rationale`
- `severity_hint`
- `regression_suite`
- `recommendation_question`
- `metadata`

Notes:

- Use this as the contract for the multi-agent debate stage input.
- Debate node should output recommendation + rationale while preserving `window_id`.

Debate output contract:

- `DebateOutputRecord`

Primary fields:

- `run_id`
- `window_id`
- `decision` (`yes`/`no`)
- `recommendation` (`add_immediately`/`already_covered`/`not_critical`)
- `priority_score`
- `rationale`
- `model_source`

## Contract enforcement rule

At boundary read/write points, validate rows with the corresponding model from
`pipeline/models/handoff_contracts.py` before writing files or handing data to the next stage.
