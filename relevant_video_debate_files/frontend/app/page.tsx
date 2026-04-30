"use client";

import { type DragEvent, useEffect, useMemo, useState } from "react";

import { ApiError, append_run_metrics_log, fetch_workspace_snapshot, run_video_pipeline_stream } from "@/lib/api";
import type {
  PipelineProgressPayload,
  RegressionCaseProposal,
  RunMetricStatus,
  RunMetricsLogEntry,
  WorkspaceSnapshotResponse
} from "@/types/api";

function decision_chip_class(decision: RegressionCaseProposal["decision"]): string {
  switch (decision) {
    case "add_to_suite":
      return "severity-low";
    case "monitor":
      return "severity-medium";
    case "dismiss":
    default:
      return "severity-unknown";
  }
}

function risk_chip_class(risk: RegressionCaseProposal["riskLevel"]): string {
  switch (risk) {
    case "critical":
    case "high":
      return "severity-high";
    case "medium":
      return "severity-medium";
    case "low":
    default:
      return "severity-low";
  }
}

type ProgressRow = {
  key: string;
  step: string;
  title: string;
  detail: string;
  time_label: string;
};

function format_progress_time(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

type CaseRunSummary = {
  runId: string;
  videoName: string;
  startedAt: string;
  completedAt: string | null;
  status: "success" | "failed";
  message: string;
  error: string | null;
  runtimeSec: number | null;
  firstUiSec: number | null;
  progressLog: ProgressRow[];
  snapshot: WorkspaceSnapshotResponse | null;
  selectedWindowId: string | null;
};

const CASE_HISTORY_STORAGE_KEY = "workspace_case_history_v1";

function metric_status_label(status: RunMetricStatus): string {
  return status.replace(/_/g, " ");
}

function derive_stage(step: string, title: string): RunMetricStatus {
  const text = `${step} ${title}`.toLowerCase();
  if (text.includes("embed")) {
    return "embedding";
  }
  if (text.includes("cluster") || text.includes("anomaly")) {
    return "clustering";
  }
  if (text.includes("final") || text.includes("save") || text.includes("proposal")) {
    return "finalizing";
  }
  if (
    text.includes("debate") ||
    text.includes("describe") ||
    text.includes("scene") ||
    text.includes("agent")
  ) {
    return "agent_debate";
  }
  return "queued";
}

function classify_error_reason(message: string): string {
  const normalized = message.toLowerCase();
  if (normalized.includes("timeout")) {
    return "timeout";
  }
  if (normalized.includes("network") || normalized.includes("fetch")) {
    return "network";
  }
  if (normalized.includes("mock")) {
    return "mock_output";
  }
  if (normalized.includes("python") || normalized.includes("pipeline")) {
    return "pipeline";
  }
  return "unknown";
}

function parse_tool_failures(message: string): Record<string, number> {
  const lower = message.toLowerCase();
  return {
    vision_model: lower.includes("vision") ? 1 : 0,
    risk_taxonomy: lower.includes("taxonomy") ? 1 : 0,
    arbiter: lower.includes("arbiter") ? 1 : 0
  };
}

export default function HomePage(): JSX.Element {
  const [workspace_snapshot, set_workspace_snapshot] = useState<WorkspaceSnapshotResponse | null>(null);
  const [snapshot_loading, set_snapshot_loading] = useState<boolean>(false);
  const [snapshot_error, set_snapshot_error] = useState<string | null>(null);
  const [selected_window_id, set_selected_window_id] = useState<string | null>(null);
  const [video_file, set_video_file] = useState<File | null>(null);
  const [video_run_loading, set_video_run_loading] = useState<boolean>(false);
  const [video_run_message, set_video_run_message] = useState<string | null>(null);
  const [video_run_error, set_video_run_error] = useState<string | null>(null);
  const [pipeline_stdout, set_pipeline_stdout] = useState<string>("");
  const [progress_log, set_progress_log] = useState<ProgressRow[]>([]);
  const [case_history, set_case_history] = useState<CaseRunSummary[]>([]);
  const [selected_case_id, set_selected_case_id] = useState<string | null>(null);
  const [current_run_status, set_current_run_status] = useState<RunMetricStatus>("queued");
  const [review_actions, set_review_actions] = useState<RunMetricsLogEntry["userReviewActions"]>({
    openedPreviousCaseCount: 0,
    openedReasoningDetailsCount: 0,
    viewedFinalProposalCount: 0
  });

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(CASE_HISTORY_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        set_case_history(parsed as CaseRunSummary[]);
      }
    } catch {
      /* ignore local cache parse failures */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(CASE_HISTORY_STORAGE_KEY, JSON.stringify(case_history));
    } catch {
      /* ignore storage write failures */
    }
  }, [case_history]);

  function append_progress(payload: PipelineProgressPayload): void {
    set_progress_log((prev) => {
      const row: ProgressRow = {
        key: `${payload.step}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        step: payload.step,
        title: payload.title,
        detail: payload.detail,
        time_label: format_progress_time()
      };
      return [...prev, row].slice(-40);
    });
    set_current_run_status(derive_stage(payload.step, payload.title));
  }

  async function load_workspace_snapshot(): Promise<void> {
    set_snapshot_loading(true);
    set_snapshot_error(null);
    try {
      const response = await fetch_workspace_snapshot();
      set_workspace_snapshot(response);
    } catch (error) {
      if (error instanceof ApiError) {
        set_snapshot_error(`Workspace snapshot API error (${error.status}): ${error.message}`);
      } else {
        set_snapshot_error("Failed to load workspace snapshot artifacts.");
      }
    } finally {
      set_snapshot_loading(false);
    }
  }

  useEffect(() => {
    void load_workspace_snapshot();
  }, []);

  const selected_case = useMemo(
    () => case_history.find((item) => item.runId === selected_case_id) ?? null,
    [case_history, selected_case_id]
  );

  const active_snapshot = selected_case?.snapshot ?? workspace_snapshot;
  const active_selected_window_id = selected_case?.selectedWindowId ?? selected_window_id;

  const selected_flagged_item = useMemo(() => {
    if (!active_snapshot?.flaggedItems.length) {
      return null;
    }
    if (!active_selected_window_id) {
      return active_snapshot.flaggedItems[0];
    }
    return (
      active_snapshot.flaggedItems.find((item) => item.windowId === active_selected_window_id) ??
      active_snapshot.flaggedItems[0]
    );
  }, [active_snapshot, active_selected_window_id]);

  const selected_reasoning_item = useMemo(() => {
    if (!active_snapshot?.reasoningItems.length || !selected_flagged_item) {
      return null;
    }
    return (
      active_snapshot.reasoningItems.find(
        (item) => item.windowId === selected_flagged_item.windowId
      ) ?? null
    );
  }, [active_snapshot, selected_flagged_item]);

  const selected_proposal = useMemo<RegressionCaseProposal | null>(() => {
    if (!active_snapshot?.proposals?.length || !selected_flagged_item) {
      return null;
    }
    return (
      active_snapshot.proposals.find(
        (item) => item.windowId === selected_flagged_item.windowId
      ) ?? null
    );
  }, [active_snapshot, selected_flagged_item]);

  const summary_text = useMemo(() => {
    const summary = active_snapshot?.anomalySummary;
    if (!summary || typeof summary !== "object") {
      return "No anomaly summary loaded.";
    }
    const rows = typeof summary.rows_processed === "number" ? summary.rows_processed : "-";
    const clusters = typeof summary.cluster_count === "number" ? summary.cluster_count : "-";
    const noise = typeof summary.noise_count === "number" ? summary.noise_count : "-";
    const noise_ratio =
      typeof summary.noise_ratio === "number"
        ? `${(summary.noise_ratio * 100).toFixed(1)}%`
        : "-";
    return `rows ${rows} | clusters ${clusters} | noise ${noise} (${noise_ratio})`;
  }, [active_snapshot]);

  const detailed_report_text = useMemo(() => {
    if (!selected_flagged_item || !selected_reasoning_item) {
      return "Select a processed window to see the detailed scene report.";
    }
    const transcript = selected_reasoning_item.debateHistory.length
      ? selected_reasoning_item.debateHistory.map((line) => `- ${line}`).join("\n")
      : "- Debate history not available.";
    return [
      `Window: ${selected_flagged_item.windowId}`,
      `Log: ${selected_flagged_item.logId}`,
      `Anomaly: rank #${selected_flagged_item.anomalyRank}, score ${selected_flagged_item.outlierScore.toFixed(4)}, ${selected_flagged_item.isNoise ? "noise" : `cluster ${selected_flagged_item.clusterLabel}`}`,
      "",
      "Scene Description:",
      selected_reasoning_item.sceneDescription,
      "",
      "Anomaly Rationale:",
      selected_reasoning_item.anomalyRationale,
      "",
      "Debate Decision:",
      `- Decision: ${selected_reasoning_item.decision.toUpperCase()}`,
      `- Recommendation: ${selected_reasoning_item.recommendation}`,
      `- Priority Score: ${selected_reasoning_item.priorityScore.toFixed(2)}`,
      `- Capability Tag: ${selected_reasoning_item.capabilityTag || "n/a"}`,
      "",
      "Debate Transcript:",
      transcript
    ].join("\n");
  }, [selected_flagged_item, selected_reasoning_item]);

  async function on_run_video_pipeline(): Promise<void> {
    if (!video_file) {
      set_video_run_error("Select a video file first.");
      return;
    }
    set_video_run_loading(true);
    set_video_run_error(null);
    set_video_run_message(null);
    set_pipeline_stdout("");
    set_progress_log([]);
    set_selected_case_id(null);
    set_current_run_status("queued");
    set_review_actions({
      openedPreviousCaseCount: 0,
      openedReasoningDetailsCount: 0,
      viewedFinalProposalCount: 0
    });
    const run_started_at = new Date();
    const run_started_ms = Date.now();
    const run_id = `${run_started_at.toISOString()}-${Math.random().toString(36).slice(2, 7)}`;
    let first_ui_response_ms: number | null = null;
    let first_meaningful_result_ms: number | null = null;
    let progress_count = 0;
    let stage_started_at = run_started_ms;
    let active_stage: RunMetricStatus = "queued";
    const stage_times_ms: Record<string, number> = {
      queued: 0,
      embedding: 0,
      clustering: 0,
      agent_debate: 0,
      finalizing: 0
    };
    try {
      const response = await run_video_pipeline_stream(video_file, (payload) => {
        const now = Date.now();
        if (first_ui_response_ms === null) {
          first_ui_response_ms = now;
        }
        progress_count += 1;
        const next_stage = derive_stage(payload.step, payload.title);
        stage_times_ms[active_stage] = (stage_times_ms[active_stage] ?? 0) + (now - stage_started_at);
        active_stage = next_stage;
        stage_started_at = now;
        if (
          first_meaningful_result_ms === null &&
          (next_stage === "clustering" || next_stage === "agent_debate" || payload.title.toLowerCase().includes("anomaly"))
        ) {
          first_meaningful_result_ms = now;
        }
        append_progress(payload);
      });
      const completed_ms = Date.now();
      stage_times_ms[active_stage] = (stage_times_ms[active_stage] ?? 0) + (completed_ms - stage_started_at);
      set_video_run_message(response.message);
      set_pipeline_stdout(response.stdout || response.stderr || "");
      set_selected_window_id(response.windowId);
      set_current_run_status("success");
      let refreshed_snapshot: WorkspaceSnapshotResponse | null = null;
      try {
        refreshed_snapshot = await fetch_workspace_snapshot();
        set_workspace_snapshot(refreshed_snapshot);
      } catch {
        refreshed_snapshot = null;
      }
      const runtime_sec = (completed_ms - run_started_ms) / 1000;
      const first_ui_sec = first_ui_response_ms === null ? null : (first_ui_response_ms - run_started_ms) / 1000;
      const first_meaningful_sec =
        first_meaningful_result_ms === null ? null : (first_meaningful_result_ms - run_started_ms) / 1000;
      set_case_history((prev) => [
        {
          runId: run_id,
          videoName: video_file.name,
          startedAt: run_started_at.toISOString(),
          completedAt: new Date(completed_ms).toISOString(),
          status: "success",
          message: response.message,
          error: null,
          runtimeSec: runtime_sec,
          firstUiSec: first_ui_sec,
          progressLog: [],
          snapshot: refreshed_snapshot,
          selectedWindowId: response.windowId
        },
        ...prev
      ]);
      const stage_times_sec = Object.fromEntries(
        Object.entries(stage_times_ms)
          .filter(([key]) => key !== "queued")
          .map(([key, value]) => [key, Number((value / 1000).toFixed(3))])
      );
      try {
        await append_run_metrics_log({
          runId: run_id,
          videoName: video_file.name,
          status: "success",
          startedAt: run_started_at.toISOString(),
          firstUiResponseAt: first_ui_response_ms ? new Date(first_ui_response_ms).toISOString() : null,
          firstMeaningfulResultAt: first_meaningful_result_ms
            ? new Date(first_meaningful_result_ms).toISOString()
            : null,
          completedAt: new Date(completed_ms).toISOString(),
          timeToFirstUiResponseSec: first_ui_sec,
          firstMeaningfulResultSec: first_meaningful_sec,
          totalRuntimeSec: runtime_sec,
          stageTimesSec: stage_times_sec,
          progressUpdateCount: progress_count,
          error: null,
          errorCategory: null,
          agentToolFailureCounts: {
            vision_model: 0,
            risk_taxonomy: 0,
            arbiter: 0
          },
          userReviewActions: review_actions
        });
      } catch {
        /* metrics backup failure should not fail the run UI */
      }
    } catch (error) {
      const completed_ms = Date.now();
      stage_times_ms[active_stage] = (stage_times_ms[active_stage] ?? 0) + (completed_ms - stage_started_at);
      set_current_run_status("failed");
      const message =
        error instanceof ApiError ? `Run failed (${error.status}): ${error.message}` : "Run failed due to an unexpected error.";
      if (error instanceof ApiError) {
        set_video_run_error(message);
      } else {
        set_video_run_error(message);
      }
      const runtime_sec = (completed_ms - run_started_ms) / 1000;
      const first_ui_sec = first_ui_response_ms === null ? null : (first_ui_response_ms - run_started_ms) / 1000;
      set_case_history((prev) => [
        {
          runId: run_id,
          videoName: video_file.name,
          startedAt: run_started_at.toISOString(),
          completedAt: new Date(completed_ms).toISOString(),
          status: "failed",
          message: "Run failed.",
          error: message,
          runtimeSec: runtime_sec,
          firstUiSec: first_ui_sec,
          progressLog: [],
          snapshot: null,
          selectedWindowId: null
        },
        ...prev
      ]);
      const stage_times_sec = Object.fromEntries(
        Object.entries(stage_times_ms)
          .filter(([key]) => key !== "queued")
          .map(([key, value]) => [key, Number((value / 1000).toFixed(3))])
      );
      try {
        await append_run_metrics_log({
          runId: run_id,
          videoName: video_file.name,
          status: "failed",
          startedAt: run_started_at.toISOString(),
          firstUiResponseAt: first_ui_response_ms ? new Date(first_ui_response_ms).toISOString() : null,
          firstMeaningfulResultAt: first_meaningful_result_ms
            ? new Date(first_meaningful_result_ms).toISOString()
            : null,
          completedAt: new Date(completed_ms).toISOString(),
          timeToFirstUiResponseSec: first_ui_sec,
          firstMeaningfulResultSec:
            first_meaningful_result_ms === null ? null : (first_meaningful_result_ms - run_started_ms) / 1000,
          totalRuntimeSec: runtime_sec,
          stageTimesSec: stage_times_sec,
          progressUpdateCount: progress_count,
          error: message,
          errorCategory: classify_error_reason(message),
          agentToolFailureCounts: parse_tool_failures(message),
          userReviewActions: review_actions
        });
      } catch {
        /* metrics backup failure should not fail the run UI */
      }
    } finally {
      set_video_run_loading(false);
    }
  }

  function on_drop_video(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      set_video_file(file);
      set_video_run_error(null);
    }
  }

  function on_new_input(): void {
    set_video_file(null);
    set_video_run_error(null);
    set_video_run_message(null);
    set_pipeline_stdout("");
    set_progress_log([]);
    set_current_run_status("queued");
    set_selected_case_id(null);
  }

  const kpi_summary = useMemo(() => {
    const total_runs = case_history.length;
    const successful_runs = case_history.filter((item) => item.status === "success").length;
    const runtime_values = case_history
      .map((item) => item.runtimeSec)
      .filter((value): value is number => typeof value === "number");
    const first_response_values = case_history
      .map((item) => item.firstUiSec)
      .filter((value): value is number => typeof value === "number");
    const avg_runtime = runtime_values.length
      ? runtime_values.reduce((acc, value) => acc + value, 0) / runtime_values.length
      : null;
    const avg_first_response = first_response_values.length
      ? first_response_values.reduce((acc, value) => acc + value, 0) / first_response_values.length
      : null;
    return {
      totalRuns: total_runs,
      successfulRuns: successful_runs,
      failedRuns: total_runs - successful_runs,
      avgRuntimeSec: avg_runtime,
      avgFirstResponseSec: avg_first_response
    };
  }, [case_history]);

  return (
    <main className="workspace-page">
      <header className="workspace-header">
        <span className="eyebrow">Debugging Copilot Workspace</span>
        <h1>Edge Case Discovery Dashboard</h1>
        <p>
          End-to-end workspace view for anomalies, visual QA artifacts, and staged reasoning outputs.
        </p>
        <div className="pipeline-chips">
          <span className="pipeline-chip pipeline-chip-done">UMAP + HDBSCAN</span>
          <span className="pipeline-chip pipeline-chip-done">Visual QA</span>
          <span className="pipeline-chip pipeline-chip-running">Description + Debate</span>
        </div>
      </header>

      <section className="status-bar">
        <div className="status-tile">
          <strong>Total Runs</strong>
          <span className="status-value">{kpi_summary.totalRuns}</span>
        </div>
        <div className="status-tile">
          <strong>Success Rate</strong>
          <span className="status-value">
            {kpi_summary.successfulRuns}/{kpi_summary.totalRuns}
          </span>
        </div>
        <div className="status-tile">
          <strong>Avg Runtime</strong>
          <span className="status-value">
            {kpi_summary.avgRuntimeSec === null ? "-" : `${kpi_summary.avgRuntimeSec.toFixed(1)}s`}
          </span>
        </div>
        <div className="status-tile">
          <strong>Avg First Response</strong>
          <span className="status-value">
            {kpi_summary.avgFirstResponseSec === null ? "-" : `${kpi_summary.avgFirstResponseSec.toFixed(2)}s`}
          </span>
        </div>
        <div className="status-tile">
          <strong>Current Status</strong>
          <span className="status-value">
            {video_run_loading ? metric_status_label(current_run_status) : "idle"}
          </span>
        </div>
        <div className="status-tile">
          <strong>Previous Cases Saved</strong>
          <span className="status-value">{case_history.length}</span>
        </div>
        <div className="status-tile">
          <strong>Flagged Items</strong>
          <span className="status-value">{active_snapshot?.flaggedItems.length ?? 0}</span>
        </div>
        <div className="status-tile">
          <strong>Reasoning Items</strong>
          <span className="status-value">{active_snapshot?.reasoningItems.length ?? 0}</span>
        </div>
        <div className="status-tile">
          <strong>Snapshot Time</strong>
          <span className="status-value">
            {active_snapshot?.generatedAt
              ? new Date(active_snapshot.generatedAt).toLocaleTimeString()
              : "not loaded"}
          </span>
        </div>
      </section>

      <div className="alerts-row">
        {snapshot_error ? <div className="alert alert-error">{snapshot_error}</div> : null}
        {video_run_error ? <div className="alert alert-error">{video_run_error}</div> : null}
        {video_run_message ? <div className="alert alert-success">{video_run_message}</div> : null}
      </div>

      <section className="section-card upload-card">
        <div className="section-header">
          <h2>Run From Video (Drag + Drop)</h2>
          <p>
            Drop a single video, run scene description + debate pipeline, and inspect outputs
            immediately. While the run is in progress, a live timeline appears below with each major
            step (model load, scene description, debate rounds, save).
          </p>
        </div>
        <div
          className="video-dropzone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={on_drop_video}
        >
          <strong>{video_file ? video_file.name : "Drop video file here"}</strong>
          <span>{video_file ? `${(video_file.size / (1024 * 1024)).toFixed(2)} MB` : "or pick manually below"}</span>
        </div>
        <div className="upload-controls">
          <input
            type="file"
            accept=".mp4,.mov,.mkv,.avi,.webm"
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              set_video_file(file);
              set_video_run_error(null);
            }}
          />
          <button
            type="button"
            onClick={() => {
              void on_run_video_pipeline();
            }}
            disabled={video_run_loading}
          >
            {video_run_loading ? "Running..." : "Run Description + Debate"}
          </button>
          <button type="button" className="secondary-action" onClick={on_new_input} disabled={video_run_loading}>
            New Input
          </button>
        </div>

        {video_run_loading || progress_log.length > 0 ? (
          <div className="pipeline-progress-panel" style={{ marginTop: 16 }} aria-live="polite">
            <div className="pipeline-progress-header">
              {video_run_loading ? "Pipeline progress" : "Last run timeline"}
            </div>
            <ul className="pipeline-progress-list">
              {progress_log.map((row, index) => (
                <li
                  key={row.key}
                  className={`pipeline-progress-row${video_run_loading && index === progress_log.length - 1 ? " is-active" : ""
                    }`}
                >
                  <time dateTime={row.time_label}>{row.time_label}</time>
                  <div>
                    <div className="pipeline-progress-title">{row.title}</div>
                    {row.detail ? <div className="pipeline-progress-detail">{row.detail}</div> : null}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {pipeline_stdout ? (
          <pre className="run-log">{pipeline_stdout}</pre>
        ) : null}
      </section>

      <div className="workspace-columns">
        <div className="workspace-column workspace-column-primary">
          <h2 className="column-title">Flagged Anomaly Queue</h2>
          <section className="section-card">
            <div className="section-header">
              <h2>Top Flagged Windows</h2>
              <p>Select one row to inspect its visual and reasoning outputs.</p>
            </div>
            <div className="panel-toolbar">
              <button
                type="button"
                className="secondary-action"
                onClick={() => {
                  void load_workspace_snapshot();
                }}
                disabled={snapshot_loading}
              >
                {snapshot_loading ? "Refreshing..." : "Refresh Snapshot"}
              </button>
              <span className="panel-meta">{summary_text}</span>
            </div>
            {!active_snapshot || active_snapshot.flaggedItems.length === 0 ? (
              <div className="empty-state">No flagged windows found.</div>
            ) : (
              <ul className="recent-runs-list">
                {active_snapshot.flaggedItems.slice(0, 15).map((item) => (
                  <li key={item.windowId}>
                    <button
                      type="button"
                      className="run-select-button"
                      onClick={() => {
                        set_selected_window_id(item.windowId);
                        set_review_actions((prev) => ({
                          ...prev,
                          openedReasoningDetailsCount: prev.openedReasoningDetailsCount + 1
                        }));
                      }}
                    >
                      <strong>{item.windowId}</strong>
                      <span>
                        rank #{item.anomalyRank} | {item.isNoise ? "noise" : `cluster ${item.clusterLabel}`} |
                        score {item.outlierScore.toFixed(4)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="section-card">
            <div className="section-header">
              <h2>Previous Cases</h2>
              <p>Click any completed run to re-open its artifacts without refreshing.</p>
            </div>
            {case_history.length === 0 ? (
              <div className="empty-state">No previous cases yet.</div>
            ) : (
              <ul className="recent-runs-list">
                {case_history.map((item) => (
                  <li key={item.runId}>
                    <button
                      type="button"
                      className="run-select-button"
                      onClick={() => {
                        set_selected_case_id(item.runId);
                        set_review_actions((prev) => ({
                          ...prev,
                          openedPreviousCaseCount: prev.openedPreviousCaseCount + 1
                        }));
                      }}
                    >
                      <strong>{item.videoName}</strong>
                      <span>
                        {item.status.toUpperCase()} | {item.runtimeSec ? `${item.runtimeSec.toFixed(1)}s` : "-"} |{" "}
                        {new Date(item.startedAt).toLocaleTimeString()}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        <div className="workspace-column workspace-column-secondary">
          <h2 className="column-title">Inspection + Reasoning</h2>
          <section className="section-card">
            <div className="section-header">
              <h2>Visual Verification</h2>
              <p>Frame grid and MP4 for currently selected flagged window.</p>
            </div>
            {!selected_flagged_item ? (
              <div className="empty-state">Select a flagged window from the left panel.</div>
            ) : (
              <div className="artifact-preview">
                <div className="artifact-preview-header">
                  <strong>{selected_flagged_item.windowId}</strong>
                  <span>
                    {selected_flagged_item.isNoise
                      ? "noise"
                      : `cluster ${selected_flagged_item.clusterLabel}`}{" "}
                    | score {selected_flagged_item.outlierScore.toFixed(4)}
                  </span>
                </div>
                {selected_flagged_item.gridUrl ? (
                  <img
                    src={selected_flagged_item.gridUrl}
                    alt={`Grid for ${selected_flagged_item.windowId}`}
                    className="artifact-grid-image"
                  />
                ) : (
                  <div className="empty-state">Grid artifact pending.</div>
                )}
                {selected_flagged_item.mp4Url ? (
                  <video
                    className="artifact-video"
                    controls
                    preload="metadata"
                    src={selected_flagged_item.mp4Url}
                  />
                ) : (
                  <div className="empty-state">MP4 artifact pending.</div>
                )}
                <div className="artifact-links">
                  {selected_flagged_item.gridUrl ? (
                    <a href={selected_flagged_item.gridUrl} target="_blank" rel="noreferrer">
                      Open Grid
                    </a>
                  ) : null}
                  {selected_flagged_item.mp4Url ? (
                    <a href={selected_flagged_item.mp4Url} target="_blank" rel="noreferrer">
                      Open MP4
                    </a>
                  ) : null}
                </div>
              </div>
            )}
          </section>

          <section className="section-card">
            <div className="section-header">
              <h2>Description + Debate</h2>
              <p>Stage output for the selected window from the reasoning pipeline.</p>
            </div>
            {!selected_reasoning_item ? (
              <div className="empty-state">No reasoning row found for this window yet.</div>
            ) : (
              <div className="artifact-list">
                <article className="artifact-card">
                  <header>
                    <strong>{selected_reasoning_item.windowId}</strong>
                    <span>{selected_reasoning_item.decision.toUpperCase()}</span>
                  </header>
                  <p>{selected_reasoning_item.sceneDescription}</p>
                  <p className="artifact-subtext">{selected_reasoning_item.anomalyRationale}</p>
                  <p className="artifact-subtext">
                    {selected_reasoning_item.recommendation} | priority{" "}
                    {selected_reasoning_item.priorityScore.toFixed(2)} |{" "}
                    {selected_reasoning_item.modelSource}
                  </p>
                </article>
              </div>
            )}
          </section>

          <section className="section-card">
            <div className="section-header">
              <h2>Detailed Scene Report</h2>
              <p>Combined anomaly + description + debate report for handoff and review.</p>
            </div>
            <pre className="detailed-report">{detailed_report_text}</pre>
          </section>

          {selected_proposal ? (
            <section className="section-card">
              <div className="section-header">
                <h2>Regression-Case Proposal</h2>
                <p>
                  Structured, review-ready proposal assembled from the four-actor tool-augmented
                  debate. Scroll through each pane to validate before pushing to the suite.
                </p>
              </div>

              <div className="artifact-preview-header">
                <strong>{selected_proposal.caseId || selected_proposal.windowId}</strong>
                <span
                  className={`severity-chip ${decision_chip_class(selected_proposal.decision)}`}
                >
                  {selected_proposal.decision.replace(/_/g, " ")}
                </span>
              </div>
              <div className="panel-toolbar">
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() =>
                    set_review_actions((prev) => ({
                      ...prev,
                      viewedFinalProposalCount: prev.viewedFinalProposalCount + 1
                    }))
                  }
                >
                  Track Proposal View
                </button>
              </div>

              <div className="capsule-block">
                <h3>Failure Mode &amp; Evidence</h3>
                <p>
                  <strong>Failure mode: </strong>
                  {selected_proposal.failureMode}
                </p>
                <p>
                  <strong>Why this is valuable: </strong>
                  {selected_proposal.whyAnomalous}
                </p>
                <p>
                  <strong>Evidence summary: </strong>
                  {selected_proposal.evidenceSummary}
                </p>
              </div>

              <div className="capsule-meta">
                <div>
                  <strong>Risk level</strong>
                  <span
                    className={`severity-chip ${risk_chip_class(selected_proposal.riskLevel)}`}
                  >
                    {selected_proposal.riskLevel.toUpperCase()}
                  </span>
                </div>
                <div>
                  <strong>Affected capability</strong>
                  <span className="tag">
                    {selected_proposal.affectedCapability || "unspecified"}
                  </span>
                </div>
                <div>
                  <strong>ODD conditions</strong>
                  {selected_proposal.affectedOdds.length ? (
                    <span className="tag-row">
                      {selected_proposal.affectedOdds.map((odd, index) => (
                        <span key={`${odd}-${index}`} className="tag">
                          {odd}
                        </span>
                      ))}
                    </span>
                  ) : (
                    <span>n/a</span>
                  )}
                </div>
              </div>

              <div className="capsule-block">
                <h3>Counterarguments (Coverage Analyst)</h3>
                {selected_proposal.counterarguments.length ? (
                  <ol>
                    {selected_proposal.counterarguments.map((item, index) => (
                      <li key={`counter-${index}`}>{item}</li>
                    ))}
                  </ol>
                ) : (
                  <p>No counterarguments raised.</p>
                )}
                {selected_proposal.rebuttalSummary ? (
                  <p>
                    <strong>Rebuttal: </strong>
                    {selected_proposal.rebuttalSummary}
                  </p>
                ) : null}
              </div>

              <div className="capsule-block">
                <h3>Recommended Action</h3>
                <p>
                  <strong>Test spec: </strong>
                  {selected_proposal.recommendedTestSpec}
                </p>
                {selected_proposal.scenarioVariants.length ? (
                  <p>
                    <strong>Variants to also test: </strong>
                    <span className="tag-row">
                      {selected_proposal.scenarioVariants.map((variant, index) => (
                        <span key={`variant-${index}`} className="tag">
                          {variant}
                        </span>
                      ))}
                    </span>
                  </p>
                ) : null}
              </div>

              <div className="capsule-meta">
                <div>
                  <strong>Confidence</strong>
                  <span>{`${(selected_proposal.confidence * 100).toFixed(1)}%`}</span>
                </div>
                <div>
                  <strong>Uncertainty factors</strong>
                  {selected_proposal.uncertaintyFactors.length ? (
                    <ul>
                      {selected_proposal.uncertaintyFactors.map((item, index) => (
                        <li key={`uncertain-${index}`}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <span>None flagged.</span>
                  )}
                </div>
              </div>

              <details>
                <summary>Debate transcript</summary>
                <pre className="detailed-report">
                  {selected_proposal.debateTranscript.length
                    ? selected_proposal.debateTranscript.join("\n")
                    : "No transcript captured."}
                </pre>
              </details>
            </section>
          ) : null}
        </div>
      </div>
    </main>
  );
}
