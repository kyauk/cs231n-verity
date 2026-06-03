"""Waymo scene-description + multi-agent debate stage.

Mirrors ``relevant_video_debate_files/pipeline/stage_describe_and_debate.py``.
Given an anomaly-ranked window (from waymo_cluster_embeddings) it:
  1. Describes the scene from the Waymo FRONT-camera clip via a VLM.
  2. Runs proponent / critic / judge debate turns via NVIDIA NIM.
  3. Emits PIPELINE_PROGRESS lines for the SSE stream and writes JSONL outputs
     (description_outputs, debate_outputs, proposals, summary.json).

Description backend selection (env COSMOS_DESCRIBE_BACKEND):
  - "nim_vlm" (default): NVIDIA NIM hosted vision model -- no local GPU needed.
  - "hf":               local HuggingFace VLM (matches the reference runner).

Usage:
  python -m waymo_pipeline.waymo_describe_and_debate \
      --flagged-jsonl outputs/flagged_windows.jsonl \
      --visual-manifest-jsonl outputs/flagged_visuals/manifest.jsonl \
      --regression-suite-json outputs/regression_suite.json \
      --output-dir outputs/reasoning --top-k 1 --debate-rounds 2
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

from waymo_pipeline.models.handoff_contracts import (
    AnomalyResultRecord,
    DebateInputRecord,
    DebateOutputRecord,
    RegressionCaseProposal,
    SceneDescriptionInputRecord,
    SceneDescriptionOutputRecord,
)
from waymo_pipeline.debate_actors import run_tool_augmented_debate
from waymo_pipeline.proposal_builder import build_proposal_from_debate_output

PROGRESS_PREFIX = "PIPELINE_PROGRESS:"


# ---------------------------------------------------------------------------
# Progress / IO helpers
# ---------------------------------------------------------------------------

def _emit_pipeline_progress(step: str, title: str, detail: str = "") -> None:
    """Emit a structured stdout line consumed by the SSE-streaming runner."""
    payload = json.dumps({"step": step, "title": title, "detail": detail}, ensure_ascii=False)
    print(f"{PROGRESS_PREFIX}{payload}", flush=True)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of rows."""
    rows: list[dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL output path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_regression_suite(path: str) -> list[str]:
    """Load an optional regression suite list from disk."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [str(item) for item in payload] if isinstance(payload, list) else []


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the Waymo description + debate stage."""
    parser = argparse.ArgumentParser(
        description="Run scene-description and debate stages for Waymo windows."
    )
    parser.add_argument("--flagged-jsonl", default="outputs/flagged_windows.jsonl")
    parser.add_argument("--visual-manifest-jsonl", default="outputs/flagged_visuals/manifest.jsonl")
    parser.add_argument("--regression-suite-json", default="")
    parser.add_argument("--output-dir", default="outputs/reasoning")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--hf-max-new-tokens", type=int, default=2400)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Severity / input builders
# ---------------------------------------------------------------------------

def _severity_from_outlier(outlier_score: float, is_noise: bool) -> str:
    """Convert anomaly signals into a coarse severity hint for the debate."""
    if is_noise and outlier_score >= 0.25:
        return "critical"
    if outlier_score >= 0.2:
        return "high"
    if outlier_score >= 0.1:
        return "medium"
    return "low"


def _build_description_inputs(
    run_id: str,
    anomaly_rows: list[AnomalyResultRecord],
    manifest_by_window: dict[str, dict[str, Any]],
) -> list[SceneDescriptionInputRecord]:
    """Build description-stage inputs from anomaly rows and the media manifest."""
    results: list[SceneDescriptionInputRecord] = []
    for row in anomaly_rows:
        manifest = manifest_by_window.get(row.window_id, {})
        media_refs: list[str] = []
        if isinstance(manifest.get("mp4_path"), str) and manifest["mp4_path"]:
            media_refs.append(manifest["mp4_path"])
        if isinstance(manifest.get("grid_path"), str) and manifest["grid_path"]:
            media_refs.append(manifest["grid_path"])
        results.append(
            SceneDescriptionInputRecord(
                run_id=run_id,
                window_id=row.window_id,
                scene_token_hex=row.scene_token_hex,
                log_id=row.log_id,
                scenario_tags=row.scenario_tags,
                cluster_label=row.cluster_label,
                is_noise=row.is_noise,
                outlier_score=row.outlier_score,
                anomaly_rank=row.anomaly_rank,
                media_refs=media_refs,
                prompt_context={
                    "cluster_label": row.cluster_label,
                    "is_noise": row.is_noise,
                    "outlier_score": row.outlier_score,
                    "anomaly_rank": row.anomaly_rank,
                    "quality": row.quality,
                    "metadata": row.metadata,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# Model callers (NVIDIA NIM)
# ---------------------------------------------------------------------------

def _nim_text_chat_completion(messages: list[dict[str, Any]]) -> str:
    """Run an NVIDIA NIM text completion (used for debate turns)."""
    api_key = os.getenv("NVIDIA_API_KEY", "")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model_id = os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct")
    temperature = float(os.getenv("DEBATE_TEMPERATURE", "0.2"))
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY missing for debate stage.")

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model_id, "messages": messages, "temperature": temperature},
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"NIM debate call failed: HTTP {response.status_code}: {response.text[:300]}"
        )
    return str(response.json()["choices"][0]["message"]["content"]).strip()


def _nim_vlm_describe(video_path: str, anomaly_priors: dict[str, Any]) -> str:
    """Describe a Waymo clip via an NVIDIA NIM hosted vision model.

    Sends the FRONT-camera clip as a base64 data URI. Returns the raw model
    text (parsed downstream by _parse_description).
    """
    api_key = os.getenv("NVIDIA_API_KEY", "")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model_id = os.getenv("DESCRIBE_NIM_MODEL_ID", "nvidia/nemotron-nano-12b-v2-vl")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY missing for description stage.")

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    system_prompt = (
        "You are an expert autonomous-driving scene-understanding model analyzing a short "
        "Waymo dash-cam style clip from an ego vehicle. Observe environment, ego behavior, "
        "other road users, traffic controls, and temporal dynamics. "
        "Output ONLY a strict JSON object (no markdown fences, no prose) with keys: "
        '{"scene_description": str, "anomaly_rationale": str, "confidence": "low"|"medium"|"high"}'
    )
    user_content = [
        {"type": "text", "text": (
            "Anomaly signals for this clip (use as priors, verify against the video):\n"
            + json.dumps(anomaly_priors, indent=2)
        )},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
    ]
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        },
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"NIM VLM describe failed: HTTP {response.status_code}: {response.text[:300]}"
        )
    return str(response.json()["choices"][0]["message"]["content"]).strip()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    work = text.strip()
    if work.startswith("```"):
        work = re.sub(r"^```(?:json)?\s*", "", work, flags=re.IGNORECASE)
        work = re.sub(r"\s*```$", "", work).strip()
    return work


def _extract_json_object(text: str, stage_name: str) -> dict[str, Any]:
    """Extract the first valid JSON object from model output."""
    work = _strip_fences(text)
    try:
        parsed = json.loads(work)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = work.find("{")
    if start >= 0:
        try:
            parsed, _ = decoder.raw_decode(work[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"{stage_name} output must contain a JSON object. Head: {text[:300]!r}")


def _parse_description(raw: str) -> dict[str, Any]:
    """Parse description output into the strict required schema."""
    parsed = _extract_json_object(raw, "Description stage")
    for key in ("scene_description", "anomaly_rationale", "confidence"):
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Description stage output missing non-empty string key: {key}")
    if parsed["confidence"].lower() not in {"low", "medium", "high"}:
        raise RuntimeError("Description confidence must be one of: low, medium, high")
    return {
        "scene_description": parsed["scene_description"].strip(),
        "anomaly_rationale": parsed["anomaly_rationale"].strip(),
        "confidence": parsed["confidence"].strip().lower(),
    }


def _parse_judge(raw: str) -> dict[str, Any]:
    """Parse the judge output into normalized decision fields."""
    parsed = _extract_json_object(raw, "Debate stage")
    if parsed.get("decision") not in {"yes", "no"}:
        raise RuntimeError("Debate decision must be 'yes' or 'no'")
    if parsed.get("recommendation") not in {"add_immediately", "already_covered", "not_critical"}:
        raise RuntimeError("Debate recommendation must be add_immediately|already_covered|not_critical")
    if not isinstance(parsed.get("priority_score"), (int, float)):
        raise RuntimeError("Debate priority_score must be numeric")
    if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
        raise RuntimeError("Debate rationale must be a non-empty string")
    if not isinstance(parsed.get("capability_tag"), str) or not parsed["capability_tag"].strip():
        raise RuntimeError("Debate capability_tag must be a non-empty string")
    return parsed


# ---------------------------------------------------------------------------
# Multi-agent debate
# ---------------------------------------------------------------------------

def build_debate_input(
    description: SceneDescriptionOutputRecord, regression_suite: list[str]
) -> DebateInputRecord:
    """Convert scene-description output into a debate-input contract row."""
    severity_hint = _severity_from_outlier(
        float(description.metadata.get("outlier_score", 0.0)),
        bool(description.metadata.get("is_noise", False)),
    )
    return DebateInputRecord(
        run_id=description.run_id,
        window_id=description.window_id,
        scene_token_hex=description.scene_token_hex,
        log_id=description.log_id,
        scene_description=description.scene_description,
        anomaly_rationale=description.anomaly_rationale,
        severity_hint=severity_hint,
        regression_suite=regression_suite,
        metadata=description.metadata,
    )


def multi_agent_debate(record: DebateInputRecord, rounds: int) -> DebateOutputRecord:
    """Run proponent / critic rounds and a final judge decision via NIM."""
    rounds = max(1, rounds)
    suite_text = "\n".join(
        f"{i + 1}. {item}" for i, item in enumerate(record.regression_suite)
    )
    _emit_pipeline_progress(
        "debate", "Regression suite debate",
        f"{rounds} round(s): proponent, critic, then judge verdict via NIM.",
    )

    history: list[str] = []
    latest_proponent = "No proponent argument yet."
    latest_critic = "No critic argument yet."

    for round_index in range(1, rounds + 1):
        _emit_pipeline_progress(
            "debate_round", f"Debate round {round_index} of {rounds}",
            "Proponent is drafting the regression-suite proposal...",
        )
        latest_proponent = _nim_text_chat_completion([
            {"role": "system", "content": (
                "You are a safety-critical AV scenario analyst proposing inclusion in the "
                "regression suite. Focus on capability tested, failure mode exposed, coverage "
                "gap, and severity; consider context from previous rounds."
            )},
            {"role": "user", "content": (
                f"Round: {round_index}\nWindow ID: {record.window_id}\n"
                f"Severity hint: {record.severity_hint}\n\n"
                f"Scenario:\n{record.scene_description}\n\n"
                f"Regression-value rationale:\n{record.anomaly_rationale}\n\n"
                f"Existing suite:\n{suite_text}\n\n"
                f"Latest critic argument:\n{latest_critic}\n\n"
                f"Transcript so far:\n{chr(10).join(history) or 'No prior turns yet.'}"
            )},
        ])
        history.append(f"Round {round_index} - Proponent: {latest_proponent}")

        _emit_pipeline_progress(
            "debate_round", f"Debate round {round_index} of {rounds}",
            "Critic is reviewing the proposal...",
        )
        latest_critic = _nim_text_chat_completion([
            {"role": "system", "content": (
                "You are a regression-suite quality controller. Challenge redundancy, "
                "reproducibility, signal quality, and actionability; consider previous rounds."
            )},
            {"role": "user", "content": (
                f"Round: {round_index}\n\nProposal:\n{latest_proponent}\n\n"
                f"Scenario description:\n{record.scene_description}\n\n"
                f"Existing suite:\n{suite_text}\n\n"
                f"Transcript so far:\n{chr(10).join(history)}"
            )},
        ])
        history.append(f"Round {round_index} - Critic: {latest_critic}")

    _emit_pipeline_progress(
        "debate_judge", "Judge verdict",
        "Collecting final JSON decision from the judge model...",
    )
    judge_raw = _nim_text_chat_completion([
        {"role": "system", "content": (
            "You are the final AV regression-suite arbiter.\n"
            "Return STRICT JSON only with keys:\n"
            '{"decision":"yes|no","recommendation":"add_immediately|already_covered|not_critical",'
            '"priority_score":<float 0..1>,"rationale":"<one paragraph>","capability_tag":"<tag>"}'
        )},
        {"role": "user", "content": (
            f"Scenario:\n{record.scene_description}\n\nProposal:\n{latest_proponent}\n\n"
            f"Critique:\n{latest_critic}\n\nExisting suite:\n{suite_text}\n\n"
            f"Full debate transcript:\n{chr(10).join(history)}"
        )},
    ])
    parsed = _parse_judge(judge_raw)
    priority_score = max(0.0, min(1.0, float(parsed["priority_score"])))

    enriched_metadata = dict(record.metadata)
    enriched_metadata.update({
        "debate_mode": "multi_agent",
        "debate_model_id": os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct"),
        "debate_rounds": rounds,
        "debate_history": history,
        "judge_raw_output": judge_raw,
        "capability_tag": parsed["capability_tag"].strip(),
    })
    return DebateOutputRecord(
        run_id=record.run_id,
        window_id=record.window_id,
        scene_token_hex=record.scene_token_hex,
        log_id=record.log_id,
        decision=parsed["decision"],
        recommendation=parsed["recommendation"],
        priority_score=priority_score,
        rationale=parsed["rationale"].strip(),
        model_source="nim_text_debate_model",
        metadata=enriched_metadata,
    )


# ---------------------------------------------------------------------------
# Proposal builder
# ---------------------------------------------------------------------------

def build_proposal(debate: DebateOutputRecord, run_id: str) -> RegressionCaseProposal:
    """Assemble a RegressionCaseProposal from a debate output record."""
    metadata = debate.metadata
    history = metadata.get("debate_history", [])
    severity_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
    risk_level = severity_map.get(
        _severity_from_outlier(
            float(metadata.get("outlier_score", 0.0)),
            bool(metadata.get("is_noise", False)),
        ),
        "low",
    )
    decision_map = {
        "add_immediately": "add_to_suite",
        "already_covered": "dismiss",
        "not_critical": "monitor",
    }
    return RegressionCaseProposal(
        case_id=f"case_{uuid.uuid4().hex[:8]}",
        run_id=run_id,
        window_id=debate.window_id,
        scene_token_hex=debate.scene_token_hex,
        log_id=debate.log_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        failure_mode=metadata.get("capability_tag", "unspecified"),
        why_anomalous=debate.rationale,
        evidence_summary=debate.rationale,
        risk_level=risk_level,
        affected_capability=metadata.get("capability_tag", "unspecified"),
        affected_odds=[],
        counterarguments=[h for h in history if "Critic" in h],
        rebuttal_summary=history[-1] if history else "",
        decision=decision_map.get(debate.recommendation, "monitor"),
        recommended_test_spec=debate.rationale,
        scenario_variants=[],
        confidence=debate.priority_score,
        uncertainty_factors=[],
        debate_transcript=history,
        model_source=debate.model_source,
        metadata={"dataset": "waymo"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run description then debate stages for the selected Waymo anomaly rows."""
    load_dotenv()
    args = parse_args()
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    _emit_pipeline_progress("start", "Pipeline started", "Loading anomaly inputs and media paths.")

    anomaly_rows = [AnomalyResultRecord.model_validate(r) for r in _read_jsonl(args.flagged_jsonl)]
    anomaly_rows = sorted(anomaly_rows, key=lambda r: r.anomaly_rank)[: max(1, args.top_k)]
    if not anomaly_rows:
        print("No anomaly rows selected for description/debate stages.")
        return 1

    manifest_by_window: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(args.visual_manifest_jsonl):
        manifest_by_window[str(row.get("window_id", ""))] = row

    regression_suite = _load_regression_suite(args.regression_suite_json)
    description_inputs = _build_description_inputs(run_id, anomaly_rows, manifest_by_window)
    backend = os.getenv("COSMOS_DESCRIBE_BACKEND", "nim_vlm").strip().lower()

    description_outputs: list[SceneDescriptionOutputRecord] = []
    total = len(description_inputs)
    for idx, record in enumerate(description_inputs, start=1):
        try:
            _emit_pipeline_progress(
                "describe", "Scene description",
                f"Window {record.window_id} ({idx}/{total}): analyzing Waymo clip...",
            )
            video_ref = next(
                (m for m in record.media_refs if m.lower().endswith(
                    (".mp4", ".mov", ".mkv", ".avi", ".webm"))),
                None,
            )
            anomaly_priors = {
                "window_id": record.window_id,
                "scene_token_hex": record.scene_token_hex,
                "log_id": record.log_id,
                "scenario_tags": record.scenario_tags,
                "cluster_label": record.cluster_label,
                "is_noise": record.is_noise,
                "outlier_score": record.outlier_score,
                "anomaly_rank": record.anomaly_rank,
            }
            if backend == "hf":
                # Local HF VLM path -- delegate to the reference runner's module
                # if present in the environment.
                from pipeline.stage_describe_and_debate import _hf_chat_completion  # type: ignore

                abs_path = os.path.abspath(video_ref) if video_ref else ""
                raw = _hf_chat_completion(messages=[
                    {"role": "user", "content": (
                        [{"type": "video", "video": abs_path, "fps": 8}] if abs_path else []
                    ) + [{"type": "text", "text": (
                        "Describe this AV clip. Output strict JSON with keys "
                        "scene_description, anomaly_rationale, confidence.\n"
                        + json.dumps(anomaly_priors)
                    )}]},
                ])
            else:
                if not video_ref or not os.path.isfile(os.path.abspath(video_ref)):
                    raise RuntimeError(
                        f"No local video file for window {record.window_id}; "
                        f"expected media ref. Got: {record.media_refs}"
                    )
                raw = _nim_vlm_describe(os.path.abspath(video_ref), anomaly_priors)

            parsed = _parse_description(raw)
            description_outputs.append(
                SceneDescriptionOutputRecord(
                    run_id=record.run_id,
                    window_id=record.window_id,
                    scene_token_hex=record.scene_token_hex,
                    log_id=record.log_id,
                    scene_description=parsed["scene_description"],
                    anomaly_rationale=parsed["anomaly_rationale"],
                    confidence=parsed["confidence"],
                    model_source="nim_vlm" if backend != "hf" else "hf_vlm",
                    media_refs=record.media_refs,
                    metadata=record.prompt_context,
                )
            )
        except Exception as error:  # noqa: BLE001
            print(f"COSMOS_BLOCKED: true during description stage: {error}")
            return 1

    _emit_pipeline_progress(
        "describe_done", "Scene description complete",
        f"Processed {len(description_outputs)} window(s). Starting debate stage...",
    )

    debate_inputs = [build_debate_input(d, regression_suite) for d in description_outputs]
    debate_outputs: list[DebateOutputRecord] = []
    for record, description in zip(debate_inputs, description_outputs):
        try:
            # Tool-augmented four-actor ReAct debate. VLM follow-ups re-query the
            # clip through the hosted NIM vision API (no local model load).
            debate_output, _proposal_metadata = run_tool_augmented_debate(
                record, description.media_refs, rounds=args.debate_rounds
            )
            debate_outputs.append(debate_output)
        except Exception as error:  # noqa: BLE001
            print(f"COSMOS_BLOCKED: true during debate stage: {error}")
            return 1

    _emit_pipeline_progress("save", "Saving results", "Writing JSONL outputs and summary.")
    os.makedirs(args.output_dir, exist_ok=True)
    desc_in_path = os.path.join(args.output_dir, "description_inputs.jsonl")
    desc_out_path = os.path.join(args.output_dir, "description_outputs.jsonl")
    debate_in_path = os.path.join(args.output_dir, "debate_inputs.jsonl")
    debate_out_path = os.path.join(args.output_dir, "debate_outputs.jsonl")
    proposals_path = os.path.join(args.output_dir, "proposals.jsonl")

    _write_jsonl(desc_in_path, [r.model_dump() for r in description_inputs])
    _write_jsonl(desc_out_path, [r.model_dump() for r in description_outputs])
    _write_jsonl(debate_in_path, [r.model_dump() for r in debate_inputs])
    _write_jsonl(debate_out_path, [r.model_dump() for r in debate_outputs])

    proposals = [build_proposal_from_debate_output(d, run_id) for d in debate_outputs]
    _write_jsonl(proposals_path, [r.model_dump() for r in proposals])

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_rows": len(anomaly_rows),
        "cosmos_blocked": False,
        "dataset": "waymo_open_dataset_v_2_0_1",
        "description_backend": backend,
        "debate_backend": "nim_text_tool_augmented",
        "vlm_followup_backend": "nim_vlm_api",
        "debate_model_id": os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct"),
        "description_outputs": desc_out_path,
        "debate_outputs": debate_out_path,
        "proposals": proposals_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("COSMOS_BLOCKED: false (Waymo description/debate path succeeded).")
    print(f"Saved summary: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
