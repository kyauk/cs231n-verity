import { promises as fs } from "fs";
import path from "path";
import { spawn } from "child_process";

import { NextResponse } from "next/server";
import { Agent, fetch as undiciFetch } from "undici";

export const runtime = "nodejs";

const LONG_LIVED_REMOTE_AGENT = new Agent({
  headersTimeout: 0,
  bodyTimeout: 0,
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 3_600_000,
});

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const OUTPUTS_ROOT = path.resolve(PROJECT_ROOT, "outputs");
const INPUTS_ROOT = path.resolve(PROJECT_ROOT, "inputs");
const REMOTE_GPU_RUN_URL = process.env.REMOTE_GPU_RUN_URL ?? "";
const FAST_PROFILE_VIDEO_FPS = process.env.WORKSPACE_VIDEO_FPS ?? "8";
const FAST_PROFILE_MAX_NEW_TOKENS = process.env.WORKSPACE_MAX_NEW_TOKENS ?? "2400";
const FAST_PROFILE_DEBATE_ROUNDS = process.env.WORKSPACE_DEBATE_ROUNDS ?? "2";

type RunResult = {
  code: number;
  stdout: string;
  stderr: string;
};

type JsonRow = Record<string, unknown>;

function sanitize_filename(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function as_string(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function as_number(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function as_string_array(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function artifact_url(raw_path: unknown): string | null {
  if (typeof raw_path !== "string" || !raw_path.startsWith("outputs/")) {
    return null;
  }
  const relative = raw_path.replace(/^outputs\//, "");
  return `/api/workspace/artifact?path=${encodeURIComponent(relative)}`;
}

async function read_jsonl_rows(path_from_outputs: string): Promise<JsonRow[]> {
  const target = path.resolve(OUTPUTS_ROOT, path_from_outputs);
  try {
    const content = await fs.readFile(target, "utf-8");
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as JsonRow);
  } catch {
    return [];
  }
}

async function load_latest_outputs(window_id: string): Promise<{
  latestReasoning: JsonRow | null;
  latestFlagged: JsonRow | null;
  latestProposal: JsonRow | null;
}> {
  const [description_rows, debate_rows, flagged_rows, manifest_rows, proposal_rows] = await Promise.all([
    read_jsonl_rows("reasoning/description_outputs.jsonl"),
    read_jsonl_rows("reasoning/debate_outputs.jsonl"),
    read_jsonl_rows("flagged_windows.jsonl"),
    read_jsonl_rows("flagged_visuals/manifest.jsonl"),
    read_jsonl_rows("reasoning/proposals.jsonl")
  ]);

  const latest_description =
    [...description_rows].reverse().find((row) => as_string(row.window_id) === window_id) ?? null;
  const latest_debate = [...debate_rows].reverse().find((row) => as_string(row.window_id) === window_id) ?? null;
  const latest_flagged_raw =
    [...flagged_rows].reverse().find((row) => as_string(row.window_id) === window_id) ?? null;
  const latest_manifest_raw =
    [...manifest_rows].reverse().find((row) => as_string(row.window_id) === window_id) ?? null;
  const latest_proposal_raw =
    [...proposal_rows].reverse().find((row) => as_string(row.window_id) === window_id) ?? null;

  const latestReasoning =
    latest_debate && latest_description
      ? {
        windowId: as_string(latest_debate.window_id),
        sceneDescription: as_string(latest_description.scene_description, "Description pending."),
        anomalyRationale: as_string(latest_description.anomaly_rationale, "Rationale pending."),
        decision: as_string(latest_debate.decision, "no"),
        recommendation: as_string(latest_debate.recommendation, "not_critical"),
        priorityScore: as_number(latest_debate.priority_score, 0),
        modelSource: as_string(latest_debate.model_source, "unknown"),
        capabilityTag: as_string((latest_debate.metadata as JsonRow | undefined)?.capability_tag),
        debateHistory: as_string_array((latest_debate.metadata as JsonRow | undefined)?.debate_history),
        judgeRawOutput: as_string((latest_debate.metadata as JsonRow | undefined)?.judge_raw_output)
      }
      : null;

  const latestFlagged =
    latest_flagged_raw !== null
      ? {
        windowId: as_string(latest_flagged_raw.window_id),
        sceneTokenHex: as_string(latest_flagged_raw.scene_token_hex),
        logId: as_string(latest_flagged_raw.log_id),
        clusterLabel: as_number(latest_flagged_raw.cluster_label, -1),
        isNoise: Boolean(latest_flagged_raw.is_noise),
        outlierScore: as_number(latest_flagged_raw.outlier_score, 0),
        anomalyRank: as_number(latest_flagged_raw.anomaly_rank, 0),
        gridUrl: artifact_url(latest_manifest_raw?.grid_path),
        mp4Url: artifact_url(latest_manifest_raw?.mp4_path)
      }
      : null;

  const latestProposal =
    latest_proposal_raw !== null
      ? {
        caseId: as_string(latest_proposal_raw.case_id),
        windowId: as_string(latest_proposal_raw.window_id),
        generatedAt: as_string(latest_proposal_raw.generated_at),
        failureMode: as_string(latest_proposal_raw.failure_mode),
        whyAnomalous: as_string(latest_proposal_raw.why_anomalous),
        evidenceSummary: as_string(latest_proposal_raw.evidence_summary),
        riskLevel: as_string(latest_proposal_raw.risk_level, "low"),
        affectedCapability: as_string(latest_proposal_raw.affected_capability),
        affectedOdds: as_string_array(latest_proposal_raw.affected_odds),
        counterarguments: as_string_array(latest_proposal_raw.counterarguments),
        rebuttalSummary: as_string(latest_proposal_raw.rebuttal_summary),
        decision: as_string(latest_proposal_raw.decision, "monitor"),
        recommendedTestSpec: as_string(latest_proposal_raw.recommended_test_spec),
        scenarioVariants: as_string_array(latest_proposal_raw.scenario_variants),
        confidence: as_number(latest_proposal_raw.confidence, 0),
        uncertaintyFactors: as_string_array(latest_proposal_raw.uncertainty_factors),
        debateTranscript: as_string_array(latest_proposal_raw.debate_transcript)
      }
      : null;

  return { latestReasoning, latestFlagged, latestProposal };
}

function is_mock_model_source(value: unknown): boolean {
  const normalized = as_string(value).toLowerCase();
  return normalized.includes("mock");
}

function run_process(command: string, args: string[], cwd: string): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        COSMOS_HF_VIDEO_FPS: FAST_PROFILE_VIDEO_FPS,
        COSMOS_HF_MAX_NEW_TOKENS: FAST_PROFILE_MAX_NEW_TOKENS
      },
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      resolve({
        code: code ?? 1,
        stdout,
        stderr
      });
    });
  });
}

async function ensure_default_regression_suite(path_from_outputs: string): Promise<void> {
  const target = path.resolve(OUTPUTS_ROOT, path_from_outputs);
  try {
    await fs.access(target);
  } catch {
    const defaults = [
      "Night-time right turn at signalized intersection.",
      "Pedestrian crossing in rain with limited visibility.",
      "Unprotected left turn with cross traffic.",
      "Vehicle emerging from occluded driveway."
    ];
    await fs.writeFile(target, JSON.stringify(defaults, null, 2), "utf-8");
  }
}

async function resolve_python_bin(): Promise<string> {
  const configured = process.env.WORKSPACE_PYTHON_BIN;
  if (configured) {
    return configured;
  }
  const venv_python = path.resolve(PROJECT_ROOT, ".venv", "bin", "python");
  try {
    await fs.access(venv_python);
    return venv_python;
  } catch {
    return "python3";
  }
}

async function forward_to_remote_gpu(file: File): Promise<Response> {
  const form = new FormData();
  form.append("video", file, file.name);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60 * 60 * 1000);
  try {
    const response = await undiciFetch(REMOTE_GPU_RUN_URL, {
      method: "POST",
      body: form as unknown as BodyInit,
      signal: controller.signal,
      dispatcher: LONG_LIVED_REMOTE_AGENT,
    });
    const response_text = await response.text();
    let parsed: unknown;
    try {
      parsed = response_text ? JSON.parse(response_text) : {};
    } catch {
      parsed = {
        detail: response_text || "Remote runner returned non-JSON response."
      };
    }
    return NextResponse.json(parsed, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        detail: `Remote GPU runner request failed: ${String(error)}`
      },
      { status: 500 }
    );
  } finally {
    clearTimeout(timeout);
  }
}

export async function POST(request: Request): Promise<Response> {
  const form_data = await request.formData();
  const file = form_data.get("video");
  if (!(file instanceof File)) {
    return NextResponse.json({ detail: "Missing video file." }, { status: 400 });
  }

  const lower_name = file.name.toLowerCase();
  const allowed = [".mp4", ".mov", ".mkv", ".avi", ".webm"];
  const is_allowed = allowed.some((ext) => lower_name.endsWith(ext));
  if (!is_allowed) {
    return NextResponse.json({ detail: "Unsupported video type." }, { status: 400 });
  }

  if (REMOTE_GPU_RUN_URL) {
    return forward_to_remote_gpu(file);
  }

  await fs.mkdir(OUTPUTS_ROOT, { recursive: true });
  await fs.mkdir(path.resolve(OUTPUTS_ROOT, "flagged_visuals"), { recursive: true });
  await fs.mkdir(path.resolve(OUTPUTS_ROOT, "reasoning"), { recursive: true });
  await fs.mkdir(INPUTS_ROOT, { recursive: true });

  const safe_name = sanitize_filename(file.name);
  const stored_name = `upload_${Date.now()}_${safe_name}`;
  const absolute_video_path = path.resolve(INPUTS_ROOT, stored_name);
  const relative_video_path = path.posix.join("inputs", stored_name);

  const bytes = Buffer.from(await file.arrayBuffer());
  await fs.writeFile(absolute_video_path, bytes);

  const window_id = `upload_window_${Date.now()}`;
  const flagged_row = {
    window_id,
    scene_token_hex: `upload_${Date.now()}`,
    log_id: "manual_upload",
    scenario_tags: ["manual_upload"],
    window_start_ts: 0,
    window_end_ts: 0,
    cluster_label: -1,
    is_noise: true,
    cluster_probability: 0.0,
    outlier_score: 0.9,
    anomaly_rank: 1,
    quality: {},
    metadata: {
      upload_source: "frontend_drag_drop"
    }
  };
  const manifest_row = {
    window_id,
    grid_path: "",
    mp4_path: relative_video_path
  };

  const flagged_path = path.resolve(OUTPUTS_ROOT, "flagged_windows.jsonl");
  const manifest_path = path.resolve(OUTPUTS_ROOT, "flagged_visuals", "manifest.jsonl");
  const suite_relative = "regression_suite.json";
  const suite_path = path.resolve(OUTPUTS_ROOT, suite_relative);

  await fs.writeFile(flagged_path, `${JSON.stringify(flagged_row)}\n`, "utf-8");
  await fs.writeFile(manifest_path, `${JSON.stringify(manifest_row)}\n`, "utf-8");
  await ensure_default_regression_suite(suite_relative);

  const python_bin = await resolve_python_bin();
  const args = [
    "-u",
    "-m",
    "pipeline.stage_describe_and_debate",
    "--flagged-jsonl",
    "outputs/flagged_windows.jsonl",
    "--visual-manifest-jsonl",
    "outputs/flagged_visuals/manifest.jsonl",
    "--regression-suite-json",
    `outputs/${suite_relative}`,
    "--output-dir",
    "outputs/reasoning",
    "--hf-max-new-tokens",
    FAST_PROFILE_MAX_NEW_TOKENS,
    "--top-k",
    "1",
    "--debate-rounds",
    FAST_PROFILE_DEBATE_ROUNDS
  ];

  const run = await run_process(python_bin, args, PROJECT_ROOT);
  if (run.code !== 0) {
    const stderr_tail = run.stderr.trim().slice(-2500);
    const stdout_tail = run.stdout.trim().slice(-1200);
    const detail = [
      "Pipeline run failed.",
      `python: ${python_bin}`,
      stderr_tail ? `stderr: ${stderr_tail}` : "",
      stdout_tail ? `stdout: ${stdout_tail}` : ""
    ]
      .filter((line) => line.length > 0)
      .join("\n");
    return NextResponse.json(
      {
        detail,
        pythonBin: python_bin,
        stdout: run.stdout.slice(-8000),
        stderr: run.stderr.slice(-8000)
      },
      { status: 500 }
    );
  }

  let reasoning_summary: unknown = null;
  try {
    const summary_text = await fs.readFile(path.resolve(OUTPUTS_ROOT, "reasoning", "summary.json"), "utf-8");
    reasoning_summary = JSON.parse(summary_text);
  } catch {
    reasoning_summary = null;
  }
  const { latestReasoning, latestFlagged, latestProposal } = await load_latest_outputs(window_id);
  if (latestReasoning && is_mock_model_source(latestReasoning.modelSource)) {
    return NextResponse.json(
      {
        detail:
          "Run completed but returned mock output. Real scene description is required. " +
          "Sync latest pipeline files to remote and re-run.",
        latestReasoning,
        latestFlagged,
        latestProposal
      },
      { status: 500 }
    );
  }

  return NextResponse.json({
    ok: true,
    windowId: window_id,
    videoPath: relative_video_path,
    stdout: run.stdout.slice(-4000),
    stderr: run.stderr.slice(-4000),
    reasoningSummary: reasoning_summary,
    latestReasoning,
    latestFlagged,
    latestProposal,
    message: "Video uploaded and description/debate pipeline completed."
  });
}

