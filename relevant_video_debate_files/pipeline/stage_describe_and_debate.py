"""Stage pipeline for scene description + debate (video upload flow). Anomaly/seveity scores logic to be changed """

from __future__ import annotations

import argparse
import re
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

from pipeline.models.handoff_contracts import (
    AnomalyResultRecord,
    DebateInputRecord,
    DebateOutputRecord,
    RegressionCaseProposal,
    SceneDescriptionInputRecord,
    SceneDescriptionOutputRecord,
)
from pipeline.debate_actors import run_tool_augmented_debate
from pipeline.proposal_builder import build_proposal_from_debate_output

PROGRESS_PREFIX = "PIPELINE_PROGRESS:"


def _emit_pipeline_progress(step: str, title: str, detail: str = "") -> None:
    """Structured stdout line for streaming UIs (parsed by remote runner / Next route)."""
    payload = json.dumps(
        {"step": step, "title": title, "detail": detail},
        ensure_ascii=False,
    )
    print(f"{PROGRESS_PREFIX}{payload}", flush=True)


def _patch_transformers_video_backend_pyav() -> None:
    """
    Hugging Face video processors default to torchcodec, then torchvision.io.read_video.
    torchvision 0.26+ removed read_video; torchcodec often needs extra system CUDA/FFmpeg bits.
    PyAV (`pip install av`) is reliable for decoding local paths used here.
    """
    if getattr(_patch_transformers_video_backend_pyav, "_done", False):
        return
    import transformers.video_processing_utils as vpu
    from transformers.video_utils import load_video

    def fetch_videos_pyav(self, video_url_or_urls, sample_indices_fn=None):  # noqa: ANN001
        if isinstance(video_url_or_urls, list):
            return list(
                zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls])
            )
        return load_video(
            video_url_or_urls,
            backend="pyav",
            sample_indices_fn=sample_indices_fn,
        )

    vpu.BaseVideoProcessor.fetch_videos = fetch_videos_pyav  # type: ignore[method-assign]
    setattr(_patch_transformers_video_backend_pyav, "_done", True)


def parse_args() -> argparse.Namespace:
    """
    Purpose: Parse CLI options used by the upload-video reasoning stage.
    Parameters:
        None
    Returns:
        argparse.Namespace: Parsed args.
    Called by: main()
    Calls: argparse.ArgumentParser.parse_args()
    """

    parser = argparse.ArgumentParser(
        description="Run scene-description and debate stages for uploaded videos.",
    )
    parser.add_argument(
        "--flagged-jsonl",
        default="outputs/flagged_windows.jsonl",
        help="Anomaly input rows.",
    )
    parser.add_argument(
        "--visual-manifest-jsonl",
        default="outputs/flagged_visuals/manifest.jsonl",
        help="Manifest mapping window_id to media paths.",
    )
    parser.add_argument(
        "--regression-suite-json",
        default="",
        help="Optional JSON file with list[str] regression scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reasoning",
        help="Directory for JSONL outputs and summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of anomaly rows to process by anomaly_rank.",
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=3200,
        help="Max generated tokens for scene-description model.",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=2,
        help="Proponent/Critic rounds before final judge decision.",
    )
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """
    Purpose: Read a JSONL file into a list of rows.
    Parameters:
        path (str): JSONL file path.
    Returns:
        list[dict[str, Any]]: Parsed rows.
    Called by: main()
    Calls: open(), json.loads()
    """

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """
    Purpose: Write rows to JSONL output path.
    Parameters:
        path (str): Output path.
        rows (list[dict[str, Any]]): Rows to write.
    Returns:
        None
    Called by: main()
    Calls: os.makedirs(), json.dumps()
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_regression_suite(path: str) -> list[str]:
    """
    Purpose: Load optional regression suite list from disk.
    Parameters:
        path (str): JSON file path.
    Returns:
        list[str]: Scenario list or empty when missing/invalid.
    Called by: main()
    Calls: os.path.isfile(), open(), json.load()
    """

    if not path or not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]

#placeholder for now will be changed later !
def _severity_from_outlier(outlier_score: float, is_noise: bool) -> str:
    """
    Purpose: Convert anomaly signals to a coarse severity hint for debate.
    Parameters:
        outlier_score (float): Outlier score from anomaly stage.
        is_noise (bool): Whether point is in noise cluster.
    Returns:
        str: low|medium|high|critical.
    Called by: build_debate_input()
    Calls: None
    """

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
    """
    Purpose: Build description-stage inputs from anomaly rows and media manifest.
    Parameters:
        run_id (str): Current run id.
        anomaly_rows (list[AnomalyResultRecord]): Selected anomalies.
        manifest_by_window (dict[str, dict[str, Any]]): Manifest rows by window_id.
    Returns:
        list[SceneDescriptionInputRecord]: Description input records.
    Called by: main()
    Calls: SceneDescriptionInputRecord()
    """

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


def _hf_chat_completion(
    messages: list[dict[str, Any]],
    max_new_tokens_override: int | None = None,
) -> str:
    """
    Purpose: Run local HF chat completion for scene description.
    Parameters:
        messages (list[dict[str, Any]]): Role/content message list.
        max_new_tokens_override (int | None): Override default token budget, used by retry logic.
    Returns:
        str: Generated text.
    Called by: main()
    Calls: transformers.AutoProcessor.from_pretrained(), model.generate()
    """

    try:
        import torch  # type: ignore[import-not-found]
        import torchvision  # type: ignore[import-not-found]  # noqa: F401
        import transformers  # type: ignore[import-not-found]
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "Description stage requires transformers, torch, and torchvision in current environment."
        ) from error

    _patch_transformers_video_backend_pyav()

    model_id = os.getenv("COSMOS_HF_MODEL_ID", "nvidia/Cosmos-Reason2-8B")
    max_new_tokens = max_new_tokens_override or int(os.getenv("COSMOS_HF_MAX_NEW_TOKENS", "3200"))
    video_fps = float(os.getenv("COSMOS_HF_VIDEO_FPS", "8"))
    dtype_name = os.getenv("COSMOS_HF_TORCH_DTYPE", "float16")
    dtype = getattr(torch, dtype_name, torch.float16)

    cache = getattr(_hf_chat_completion, "_cache", None)
    if cache is None or cache.get("model_id") != model_id:
        _emit_pipeline_progress(
            "model_load",
            "Loading vision-language model",
            "Downloading or loading weights (first run is slower); please wait.",
        )
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        cache = {"model_id": model_id, "model": model, "processor": processor}
        setattr(_hf_chat_completion, "_cache", cache)
        _emit_pipeline_progress(
            "model_ready",
            "Model ready",
            f"{model_id} is loaded and ready for inference.",
        )

    model = cache["model"]
    processor = cache["processor"]

    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        normalized_content: list[dict[str, Any]] = []

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append({"type": "text", "text": str(item)})
                    continue

                item_type = str(item.get("type", "text")).lower()
                if item_type == "video" and item.get("video"):
                    normalized_content.append(
                        {
                            "type": "video",
                            "video": str(item["video"]),
                            "fps": float(item.get("fps", video_fps)),
                        }
                    )
                elif item_type == "image" and item.get("image"):
                    normalized_content.append(
                        {
                            "type": "image",
                            "image": str(item["image"]),
                        }
                    )
                else:
                    normalized_content.append(
                        {
                            "type": "text",
                            "text": str(item.get("text", "")),
                        }
                    )
        else:
            normalized_content = [{"type": "text", "text": str(content)}]

        if not normalized_content:
            normalized_content = [{"type": "text", "text": ""}]

        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": normalized_content,
            }
        )

    inputs = processor.apply_chat_template(
        normalized,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"fps": video_fps},
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(output_text[0]).strip() if output_text else ""


def _nim_text_chat_completion(messages: list[dict[str, Any]]) -> str:
    """
    Purpose: Run NVIDIA NIM text completion for debate turns.
    Parameters:
        messages (list[dict[str, Any]]): OpenAI-style chat messages.
    Returns:
        str: Model text output.
    Called by: cosmos_multi_agent_debate()
    Calls: requests.post()
    """

    api_key = os.getenv("NVIDIA_API_KEY", "")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model_id = os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct")
    temperature = float(os.getenv("DEBATE_TEMPERATURE", "0.2"))

    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY missing for debate stage.")

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f"NIM debate call failed: HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    return str(body["choices"][0]["message"]["content"]).strip()


def _parse_strict_json_object(raw: str, stage_name: str) -> dict[str, Any]:
    text = raw.strip()

    # Accept common fenced-json wrapper while still requiring valid JSON object.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    def _extract_first_json_object(candidate: str) -> str | None:
        start = candidate.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(candidate)):
            char = candidate[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start : idx + 1]
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise RuntimeError(
                f"{stage_name} output must be strict JSON object. Raw head: {raw[:300]!r}"
            ) from error
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as nested_error:
            raise RuntimeError(
                f"{stage_name} output must be strict JSON object. Raw head: {raw[:300]!r}"
            ) from nested_error

    if not isinstance(parsed, dict):
        raise RuntimeError(f"{stage_name} output must be a JSON object.")
    return parsed


class DescriptionTruncatedError(RuntimeError):
    """Raised when description generation output is incomplete (not valid JSON and no closing brace)."""


def _looks_truncated(raw: str) -> bool:
    """
    Purpose: Heuristically detect if model output was cut off mid-JSON.
    Parameters:
        raw (str): Raw model output.
    Returns:
        bool: True when output appears truncated (unbalanced braces / unterminated string).
    Called by: _parse_cosmos_description()
    Calls: None
    """

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()

    if not stripped:
        return True

    if stripped.endswith("}"):
        return False

    depth = 0
    in_string = False
    escaped = False
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

    return in_string or depth > 0


def _parse_json_object_from_markup(text: str, stage_name: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output that may include <redacted_thinking>, prose, and <answer> wrappers.
    """
    work = text.strip()
    if work.startswith("```"):
        work = re.sub(r"^```(?:json)?\s*", "", work, flags=re.IGNORECASE)
        work = re.sub(r"\s*```$", "", work).strip()

    m = re.search(r"<answer>\s*", work, flags=re.IGNORECASE | re.DOTALL)
    if m:
        work = work[m.end() :].strip()
    work = re.sub(r"</answer>\s*$", "", work, flags=re.IGNORECASE | re.DOTALL).strip()

    candidates: list[str] = [work]
    for index, char in enumerate(work):
        if char == "{":
            candidates.append(work[index:])
            break

    decoder = json.JSONDecoder()
    for candidate in candidates:
        chunk = candidate.strip()
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed, _ = decoder.raw_decode(chunk)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        f"{stage_name} output must contain a JSON object. Raw head: {text[:300]!r}"
    )


def _parse_cosmos_description(raw: str) -> dict[str, Any]:
    """
    Purpose: Parse description output into the strict required schema.
    Parameters:
        raw (str): Raw model output.
    Returns:
        dict[str, Any]: scene_description, anomaly_rationale, confidence.
    Called by: main()
    Calls: _parse_json_object_from_markup(), _looks_truncated()
    """
    try:
        parsed = _parse_json_object_from_markup(raw, "Description stage")
    except RuntimeError as error:
        if _looks_truncated(raw):
            raise DescriptionTruncatedError(
                f"Description stage output appears truncated. Raw tail: {raw[-300:]!r}"
            ) from error
        raise
    required_keys = ("scene_description", "anomaly_rationale", "confidence")
    for key in required_keys:
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Description stage output missing non-empty string key: {key}")

    if parsed["confidence"].lower() not in {"low", "medium", "high"}:
        raise RuntimeError("Description stage confidence must be one of: low, medium, high")

    return {
        "scene_description": parsed["scene_description"].strip(),
        "anomaly_rationale": parsed["anomaly_rationale"].strip(),
        "confidence": parsed["confidence"].strip().lower(),
    }


def _parse_cosmos_debate(raw: str) -> dict[str, Any]:
    """
    Purpose: Parse judge output into normalized decision fields.
    Parameters:
        raw (str): Raw judge output.
    Returns:
        dict[str, Any]: decision, recommendation, priority_score, rationale, capability_tag.
    Called by: cosmos_multi_agent_debate()
    Calls: _parse_strict_json_object()
    """
    parsed = _parse_strict_json_object(raw, "Debate stage")

    decision = parsed.get("decision")
    if decision not in {"yes", "no"}:
        raise RuntimeError("Debate stage key decision must be 'yes' or 'no'")

    recommendation = parsed.get("recommendation")
    if recommendation not in {"add_immediately", "already_covered", "not_critical"}:
        raise RuntimeError(
            "Debate stage key recommendation must be add_immediately|already_covered|not_critical"
        )

    priority_score = parsed.get("priority_score")
    if not isinstance(priority_score, (int, float)):
        raise RuntimeError("Debate stage key priority_score must be numeric")

    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise RuntimeError("Debate stage key rationale must be a non-empty string")

    capability_tag = parsed.get("capability_tag")
    if not isinstance(capability_tag, str) or not capability_tag.strip():
        raise RuntimeError("Debate stage key capability_tag must be a non-empty string")

    return parsed


def build_debate_input(
    description: SceneDescriptionOutputRecord,
    regression_suite: list[str],
) -> DebateInputRecord:
    """
    Purpose: Convert scene-description output into debate input contract row.
    Parameters:
        description (SceneDescriptionOutputRecord): Description output row.
        regression_suite (list[str]): Regression suite context.
    Returns:
        DebateInputRecord: Debate input row.
    Called by: main()
    Calls: DebateInputRecord(), _severity_from_outlier()
    """

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


def cosmos_multi_agent_debate(record: DebateInputRecord, rounds: int) -> DebateOutputRecord:
    """
    Purpose: Run proponent/critic rounds and a final judge decision.
    Parameters:
        record (DebateInputRecord): Debate input row.
        rounds (int): Number of rounds.
    Returns:
        DebateOutputRecord: Final debate output row.
    Called by: main()
    Calls: _nim_text_chat_completion(), _parse_cosmos_debate()
    """

    rounds = max(1, rounds)
    suite_text = "\n".join([f"{index + 1}. {item}" for index, item in enumerate(record.regression_suite)])

    _emit_pipeline_progress(
        "debate",
        "Regression suite debate",
        f"{rounds} round(s): proponent, critic, then judge verdict via NIM.",
    )

    history: list[str] = []
    latest_proponent = "No proponent argument yet."
    latest_critic = "No critic argument yet."

    for round_index in range(1, rounds + 1):
        _emit_pipeline_progress(
            "debate_round",
            f"Debate round {round_index} of {rounds}",
            "Proponent is drafting the regression-suite proposal…",
        )
        transcript_so_far = "\n".join(history) if history else "No prior turns yet."
        latest_proponent = _nim_text_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a safety-critical AV scenario analyst proposing inclusion in "
                        "the regression suite. Focus on capability tested, failure mode exposed, "
                        "coverage gap, and severity, and consider context of previous rounds if there are any.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Round: {round_index}\n"
                        f"Window ID: {record.window_id}\n"
                        f"Severity hint: {record.severity_hint}\n\n"
                        f"Scenario:\n{record.scene_description}\n\n"
                        f"Regression-value rationale:\n{record.anomaly_rationale}\n\n"
                        f"Existing suite:\n{suite_text}\n\n"
                        f"Latest critic argument:\n{latest_critic}\n\n"
                        f"Full transcript so far:\n{transcript_so_far}"
                    ),
                },
            ]
        ).strip()
        history.append(f"Round {round_index} - Proponent: {latest_proponent}")

        _emit_pipeline_progress(
            "debate_round",
            f"Debate round {round_index} of {rounds}",
            "Critic is reviewing the proposal…",
        )
        transcript_so_far = "\n".join(history)
        latest_critic = _nim_text_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a regression-suite quality controller. Challenge redundancy, "
                        "reproducibility, signal quality, and actionability, and consider context of previous rounds if there are any.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Round: {round_index}\n\n"
                        f"Proposal:\n{latest_proponent}\n\n"
                        f"Scenario description:\n{record.scene_description}\n\n"
                        f"Existing suite:\n{suite_text}\n\n"
                        f"Full transcript so far:\n{transcript_so_far}"
                    ),
                },
            ]
        ).strip()
        history.append(f"Round {round_index} - Critic: {latest_critic}")

    _emit_pipeline_progress(
        "debate_judge",
        "Judge verdict",
        "Collecting final JSON decision from the judge model…",
    )
    judge_raw = _nim_text_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the final AV regression-suite arbiter.\n"
                    "Return STRICT JSON only with keys:\n"
                    "{\n"
                    '  "decision": "yes|no",\n'
                    '  "recommendation": "add_immediately|already_covered|not_critical",\n'
                    '  "priority_score": <float 0..1>,\n'
                    '  "rationale": "<one paragraph>",\n'
                    '  "capability_tag": "<tag>"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scenario:\n{record.scene_description}\n\n"
                    f"Proposal:\n{latest_proponent}\n\n"
                    f"Critique:\n{latest_critic}\n\n"
                    f"Existing suite:\n{suite_text}\n\n"
                    "Full debate transcript:\n" + "\n".join(history)
                ),
            },
        ]
    )

    parsed = _parse_cosmos_debate(judge_raw)
    decision = parsed["decision"]
    recommendation = parsed["recommendation"]
    priority_score = max(0.0, min(1.0, float(parsed["priority_score"])))

    enriched_metadata = dict(record.metadata)
    enriched_metadata.update(
        {
            "debate_mode": "multi_agent",
            "debate_model_id": os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct"),
            "debate_rounds": rounds,
            "debate_history": history,
            "judge_raw_output": judge_raw,
            "capability_tag": parsed["capability_tag"].strip(),
        }
    )

    return DebateOutputRecord(
        run_id=record.run_id,
        window_id=record.window_id,
        scene_token_hex=record.scene_token_hex,
        log_id=record.log_id,
        decision=decision,
        recommendation=recommendation,
        priority_score=priority_score,
        rationale=parsed["rationale"].strip(),
        model_source="nim_text_debate_model",
        metadata=enriched_metadata,
    )


def main() -> int:
    """
    Purpose: Run description stage then debate stage for selected anomaly rows.
    Parameters:
        None
    Returns:
        int: 0 on success, 1 on failure.
    Called by: CLI entrypoint
    Calls: parse_args(), _hf_chat_completion(), cosmos_multi_agent_debate()
    """

    args = parse_args()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    os.environ["COSMOS_HF_MAX_NEW_TOKENS"] = str(args.hf_max_new_tokens)
    video_fps = float(os.getenv("COSMOS_HF_VIDEO_FPS", "8"))

    _emit_pipeline_progress(
        "start",
        "Pipeline started",
        "Loading anomaly inputs and media paths.",
    )

    anomaly_rows = [AnomalyResultRecord.model_validate(row) for row in _read_jsonl(args.flagged_jsonl)]
    anomaly_rows = sorted(anomaly_rows, key=lambda row: row.anomaly_rank)[: max(1, args.top_k)]
    if not anomaly_rows:
        print("No anomaly rows selected for description/debate stages.")
        return 1

    manifest_by_window: dict[str, dict[str, Any]] = {}
    if os.path.isfile(args.visual_manifest_jsonl):
        for row in _read_jsonl(args.visual_manifest_jsonl):
            manifest_by_window[str(row.get("window_id", ""))] = row

    regression_suite = _load_regression_suite(args.regression_suite_json)
    description_inputs = _build_description_inputs(run_id, anomaly_rows, manifest_by_window)

    description_prompt = (
        "You are an expert autonomous-driving scene-understanding model. "
        "Analyze the provided media and return ONLY a strict JSON object. "
        "Do not include markdown fences, tags, commentary, or chain-of-thought. "
        "Required schema (all values are strings): "
        '{"scene_description": str, "anomaly_rationale": str, "confidence": "low"|"medium"|"high"}. '
        "Field meanings: scene_description is a concise factual summary of what is visible. "
        "The key anomaly_rationale is required by the API only: its CONTENT must explain why "
        "this real-world clip is a valuable edge case for autonomous vehicle regression testing "
        "(causal chain, rare interaction, challenging ODD, or safety-relevant ambiguity worth "
        "a test engineer's review). Never claim the footage is fake, synthetic, a glitch, or "
        "unrealistic physics; treat everything as a real driving event to characterize. "
        "confidence is low|medium|high for your confidence in scene_description plus that rationale. "
        "Keep scene_description under 120 words; keep the anomaly_rationale string under 80 words."
    )
    corrective_prompt = (
        "Your previous response was not valid JSON matching the required schema. "
        "Return ONLY the JSON object with keys scene_description, anomaly_rationale, confidence. "
        "anomaly_rationale must describe valuable edge-case value for AV regression testing "
        "(real events only; no fake-footage or glitch theories). "
        "No prose, no markdown, no commentary. Keep strings concise so the JSON closes within the token budget."
    )

    base_token_budget = int(os.getenv("COSMOS_HF_MAX_NEW_TOKENS", str(args.hf_max_new_tokens)))
    max_parse_retries = int(os.getenv("DESCRIPTION_MAX_PARSE_RETRIES", "2"))

    description_outputs: list[SceneDescriptionOutputRecord] = []
    total_desc = len(description_inputs)
    for desc_index, record in enumerate(description_inputs, start=1):
        try:
            _emit_pipeline_progress(
                "describe",
                "Scene description",
                f"Window {record.window_id} ({desc_index}/{total_desc}): decoding video and generating structured output…",
            )
            media_blocks: list[dict[str, Any]] = []
            for media_ref in record.media_refs:
                abs_path = os.path.abspath(media_ref)
                lowered = media_ref.lower()
                if lowered.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                    media_blocks.append({"type": "video", "video": abs_path, "fps": video_fps})
                elif lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    media_blocks.append({"type": "image", "image": abs_path})

            media_blocks.append(
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "window_id": record.window_id,
                            "scene_token_hex": record.scene_token_hex,
                            "log_id": record.log_id,
                            "scenario_tags": record.scenario_tags,
                            "cluster_label": record.cluster_label,
                            "is_noise": record.is_noise,
                            "outlier_score": record.outlier_score,
                            "anomaly_rank": record.anomaly_rank,
                        }
                    ),
                }
            )

            conversation: list[dict[str, Any]] = [
                {"role": "system", "content": description_prompt},
                {"role": "user", "content": media_blocks},
            ]

            parsed: dict[str, Any] | None = None
            last_error: Exception | None = None
            token_budget = base_token_budget

            for attempt in range(max_parse_retries + 1):
                raw_description = _hf_chat_completion(
                    messages=conversation,
                    max_new_tokens_override=token_budget,
                )
                try:
                    parsed = _parse_cosmos_description(raw_description)
                    break
                except DescriptionTruncatedError as error:
                    last_error = error
                    print(
                        f"Description attempt {attempt + 1} truncated; "
                        f"retrying with larger token budget.",
                        flush=True,
                    )
                    token_budget = min(token_budget * 2, 8192)
                    conversation = [
                        {"role": "system", "content": description_prompt},
                        {"role": "user", "content": media_blocks},
                        {"role": "assistant", "content": raw_description},
                        {"role": "user", "content": corrective_prompt},
                    ]
                except RuntimeError as error:
                    last_error = error
                    print(
                        f"Description attempt {attempt + 1} invalid: {error}; re-prompting.",
                        flush=True,
                    )
                    conversation = [
                        {"role": "system", "content": description_prompt},
                        {"role": "user", "content": media_blocks},
                        {"role": "assistant", "content": raw_description},
                        {"role": "user", "content": corrective_prompt},
                    ]

            if parsed is None:
                raise last_error or RuntimeError("Description stage failed after retries.")

            description_outputs.append(
                SceneDescriptionOutputRecord(
                    run_id=record.run_id,
                    window_id=record.window_id,
                    scene_token_hex=record.scene_token_hex,
                    log_id=record.log_id,
                    scene_description=parsed["scene_description"],
                    anomaly_rationale=parsed["anomaly_rationale"],
                    confidence=parsed["confidence"],
                    model_source="cosmos_reason2",
                    media_refs=record.media_refs,
                    metadata=record.prompt_context,
                )
            )
        except Exception as error:  # noqa: BLE001
            print(f"COSMOS_BLOCKED: true during description stage: {error}")
            return 1

    _emit_pipeline_progress(
        "describe_done",
        "Scene description complete",
        f"Processed {len(description_outputs)} window(s). Starting debate stage…",
    )

    debate_inputs = [build_debate_input(record, regression_suite) for record in description_outputs]
    debate_outputs: list[DebateOutputRecord] = []
    for record, desc_out in zip(debate_inputs, description_outputs):
        try:
            debate_output, _proposal_meta = run_tool_augmented_debate(record, desc_out.media_refs)
            debate_outputs.append(debate_output)
        except Exception as error:  # noqa: BLE001
            print(f"COSMOS_BLOCKED: true during debate stage: {error}")
            return 1

    _emit_pipeline_progress(
        "save",
        "Saving results",
        "Writing JSONL outputs and summary.",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    description_inputs_path = os.path.join(args.output_dir, "description_inputs.jsonl")
    description_outputs_path = os.path.join(args.output_dir, "description_outputs.jsonl")
    debate_inputs_path = os.path.join(args.output_dir, "debate_inputs.jsonl")
    debate_outputs_path = os.path.join(args.output_dir, "debate_outputs.jsonl")

    _write_jsonl(description_inputs_path, [row.model_dump() for row in description_inputs])
    _write_jsonl(description_outputs_path, [row.model_dump() for row in description_outputs])
    _write_jsonl(debate_inputs_path, [row.model_dump() for row in debate_inputs])
    _write_jsonl(debate_outputs_path, [row.model_dump() for row in debate_outputs])

    proposals: list[RegressionCaseProposal] = [
        build_proposal_from_debate_output(row, run_id) for row in debate_outputs
    ]
    proposals_path = os.path.join(args.output_dir, "proposals.jsonl")
    _write_jsonl(proposals_path, [row.model_dump() for row in proposals])

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_rows": len(anomaly_rows),
        "cosmos_blocked": False,
        "description_backend": "hf_cosmos_reason2",
        "debate_backend": "nim_text",
        "debate_model_id": os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct"),
        "description_inputs": description_inputs_path,
        "description_outputs": description_outputs_path,
        "debate_inputs": debate_inputs_path,
        "debate_outputs": debate_outputs_path,
        "proposals": proposals_path,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("COSMOS_BLOCKED: false (Cosmos path succeeded).")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
