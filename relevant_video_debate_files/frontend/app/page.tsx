"use client";

import { type DragEvent, useEffect, useMemo, useState } from "react";

import { ApiError, fetch_workspace_snapshot, run_video_pipeline_stream } from "@/lib/api";
import type { PipelineProgressPayload, WorkspaceSnapshotResponse } from "@/types/api";

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

  const selected_flagged_item = useMemo(() => {
    if (!workspace_snapshot?.flaggedItems.length) {
      return null;
    }
    if (!selected_window_id) {
      return workspace_snapshot.flaggedItems[0];
    }
    return (
      workspace_snapshot.flaggedItems.find((item) => item.windowId === selected_window_id) ??
      workspace_snapshot.flaggedItems[0]
    );
  }, [workspace_snapshot, selected_window_id]);

  const selected_reasoning_item = useMemo(() => {
    if (!workspace_snapshot?.reasoningItems.length || !selected_flagged_item) {
      return null;
    }
    return (
      workspace_snapshot.reasoningItems.find(
        (item) => item.windowId === selected_flagged_item.windowId
      ) ?? null
    );
  }, [workspace_snapshot, selected_flagged_item]);

  const summary_text = useMemo(() => {
    const summary = workspace_snapshot?.anomalySummary;
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
  }, [workspace_snapshot]);

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
    try {
      const response = await run_video_pipeline_stream(video_file, append_progress);
      set_video_run_message(response.message);
      set_pipeline_stdout(response.stdout || response.stderr || "");
      await load_workspace_snapshot();
      set_selected_window_id(response.windowId);
    } catch (error) {
      if (error instanceof ApiError) {
        set_video_run_error(`Run failed (${error.status}): ${error.message}`);
      } else {
        set_video_run_error("Run failed due to an unexpected error.");
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
          <strong>Flagged Items</strong>
          <span className="status-value">{workspace_snapshot?.flaggedItems.length ?? 0}</span>
        </div>
        <div className="status-tile">
          <strong>Reasoning Items</strong>
          <span className="status-value">{workspace_snapshot?.reasoningItems.length ?? 0}</span>
        </div>
        <div className="status-tile">
          <strong>Snapshot Time</strong>
          <span className="status-value">
            {workspace_snapshot?.generatedAt
              ? new Date(workspace_snapshot.generatedAt).toLocaleTimeString()
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
            {!workspace_snapshot || workspace_snapshot.flaggedItems.length === 0 ? (
              <div className="empty-state">No flagged windows found.</div>
            ) : (
              <ul className="recent-runs-list">
                {workspace_snapshot.flaggedItems.slice(0, 15).map((item) => (
                  <li key={item.windowId}>
                    <button
                      type="button"
                      className="run-select-button"
                      onClick={() => set_selected_window_id(item.windowId)}
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
        </div>
      </div>
    </main>
  );
}
