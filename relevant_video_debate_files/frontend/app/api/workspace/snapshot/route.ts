import { promises as fs } from "fs";
import path from "path";

import { NextResponse } from "next/server";

type JsonObject = Record<string, unknown>;

const OUTPUTS_ROOT = path.resolve(process.cwd(), "..", "outputs");

async function read_json(path_from_outputs: string): Promise<JsonObject | null> {
  const target = path.resolve(OUTPUTS_ROOT, path_from_outputs);
  try {
    const content = await fs.readFile(target, "utf-8");
    return JSON.parse(content) as JsonObject;
  } catch {
    return null;
  }
}

async function read_jsonl(path_from_outputs: string): Promise<JsonObject[]> {
  const target = path.resolve(OUTPUTS_ROOT, path_from_outputs);
  try {
    const content = await fs.readFile(target, "utf-8");
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as JsonObject);
  } catch {
    return [];
  }
}

function as_number(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function as_string(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function as_object(value: unknown): JsonObject {
  return typeof value === "object" && value !== null ? (value as JsonObject) : {};
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

export async function GET(): Promise<Response> {
  const anomaly_rows = await read_jsonl("flagged_windows.jsonl");
  const visual_rows = await read_jsonl("flagged_visuals/manifest.jsonl");
  const debate_rows = await read_jsonl("reasoning/debate_outputs.jsonl");
  const description_rows = await read_jsonl("reasoning/description_outputs.jsonl");
  const proposal_rows = await read_jsonl("reasoning/proposals.jsonl");

  const anomaly_summary = await read_json("anomaly_summary.json");
  const visual_summary = await read_json("flagged_visuals/summary.json");
  const reasoning_summary = await read_json("reasoning/summary.json");

  const visual_by_window = new Map<string, JsonObject>();
  for (const row of visual_rows) {
    visual_by_window.set(as_string(row.window_id), row);
  }

  const description_by_window = new Map<string, JsonObject>();
  for (const row of description_rows) {
    description_by_window.set(as_string(row.window_id), row);
  }

  const flagged_items = anomaly_rows
    .sort((a, b) => as_number(a.anomaly_rank, 10 ** 9) - as_number(b.anomaly_rank, 10 ** 9))
    .slice(0, 25)
    .map((row) => {
      const window_id = as_string(row.window_id);
      const visual = visual_by_window.get(window_id);
      return {
        windowId: window_id,
        sceneTokenHex: as_string(row.scene_token_hex),
        logId: as_string(row.log_id),
        clusterLabel: as_number(row.cluster_label, -1),
        isNoise: Boolean(row.is_noise),
        outlierScore: as_number(row.outlier_score, 0),
        anomalyRank: as_number(row.anomaly_rank, 0),
        gridUrl: artifact_url(visual?.grid_path),
        mp4Url: artifact_url(visual?.mp4_path)
      };
    });

  const reasoning_items = debate_rows.slice(0, 25).map((row) => {
    const window_id = as_string(row.window_id);
    const description = description_by_window.get(window_id);
    const metadata = as_object(row.metadata);
    return {
      windowId: window_id,
      sceneDescription: as_string(description?.scene_description, "Description pending."),
      anomalyRationale: as_string(description?.anomaly_rationale, "Rationale pending."),
      decision: as_string(row.decision, "no"),
      recommendation: as_string(row.recommendation, "not_critical"),
      priorityScore: as_number(row.priority_score, 0),
      modelSource: as_string(row.model_source, "unknown"),
      capabilityTag: as_string(metadata.capability_tag),
      debateHistory: as_string_array(metadata.debate_history),
      judgeRawOutput: as_string(metadata.judge_raw_output)
    };
  });

  const proposals = proposal_rows.map((row) => ({
    caseId: as_string(row.case_id),
    windowId: as_string(row.window_id),
    generatedAt: as_string(row.generated_at),
    failureMode: as_string(row.failure_mode),
    whyAnomalous: as_string(row.why_anomalous),
    evidenceSummary: as_string(row.evidence_summary),
    riskLevel: as_string(row.risk_level, "low"),
    affectedCapability: as_string(row.affected_capability),
    affectedOdds: as_string_array(row.affected_odds),
    counterarguments: as_string_array(row.counterarguments),
    rebuttalSummary: as_string(row.rebuttal_summary),
    decision: as_string(row.decision, "monitor"),
    recommendedTestSpec: as_string(row.recommended_test_spec),
    scenarioVariants: as_string_array(row.scenario_variants),
    confidence: as_number(row.confidence, 0),
    uncertaintyFactors: as_string_array(row.uncertainty_factors),
    debateTranscript: as_string_array(row.debate_transcript)
  }));

  return NextResponse.json({
    generatedAt: new Date().toISOString(),
    flaggedItems: flagged_items,
    reasoningItems: reasoning_items,
    anomalySummary: anomaly_summary,
    visualSummary: visual_summary,
    reasoningSummary: reasoning_summary,
    proposals
  });
}
