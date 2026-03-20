"use client";

import { useState } from "react";

import { ApiError, fetch_workspace_snapshot, run_video_pipeline } from "@/lib/api";
import type { WorkspaceFlaggedItem, WorkspaceReasoningItem } from "@/types/api";

type DebateTurn = {
  round: number;
  role: "Proponent" | "Critic";
  content: string;
};

function sanitize_run_log(raw: string): string {
  return raw
    .split("\n")
    .filter((line) => !line.trim().startsWith("COSMOS_BLOCKED:"))
    .join("\n")
    .trim();
}

function build_debate_turns(history: string[]): DebateTurn[] {
  return history.map((entry, index) => ({
    round: Math.floor(index / 2) + 1,
    role: index % 2 === 0 ? "Proponent" : "Critic",
    content: entry
  }));
}

function build_scene_report_ticket(
  reasoning: WorkspaceReasoningItem,
  flagged: WorkspaceFlaggedItem | null
): string {
  const turns = build_debate_turns(reasoning.debateHistory);
  const transcript = turns.length
    ? turns
        .map((turn) => `[Round ${turn.round}] ${turn.role}: ${turn.content}`)
        .join("\n\n")
    : "No debate transcript captured.";

  return [
    `Title: Edge-Case Review - ${reasoning.windowId}`,
    "",
    "Summary:",
    `${reasoning.sceneDescription}`,
    "",
    "Anomaly Rationale:",
    `${reasoning.anomalyRationale}`,
    "",
    "Signal Metadata:",
    `- Window ID: ${reasoning.windowId}`,
    `- Cluster Label: ${flagged ? flagged.clusterLabel : "unknown"}`,
    `- Is Noise: ${flagged ? String(flagged.isNoise) : "unknown"}`,
    `- Outlier Score: ${flagged ? flagged.outlierScore.toFixed(3) : "unknown"}`,
    "",
    "Debate Outcome:",
    `- Decision: ${reasoning.decision.toUpperCase()}`,
    `- Recommendation: ${reasoning.recommendation}`,
    `- Priority Score: ${reasoning.priorityScore.toFixed(3)}`,
    `- Capability Tag: ${reasoning.capabilityTag || "none"}`,
    "",
    "Debate Transcript:",
    transcript,
    "",
    "Judge Raw Output:",
    reasoning.judgeRawOutput || "No raw judge output captured.",
    "",
    "Proposed Action:",
    reasoning.decision === "yes"
      ? "Add this scenario to the regression suite and include weather/visibility variants."
      : "Keep as monitored anomaly and revisit after additional neighbor comparisons."
  ].join("\n");
}

export default function VideoLabPage(): JSX.Element {
  const [video_file, set_video_file] = useState<File | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [stdout, setStdout] = useState<string>("");
  const [reasoning_summary, set_reasoning_summary] = useState<Record<string, unknown> | null>(null);
  const [latest_reasoning, set_latest_reasoning] = useState<WorkspaceReasoningItem | null>(null);
  const [latest_flagged, set_latest_flagged] = useState<WorkspaceFlaggedItem | null>(null);

  async function on_run(): Promise<void> {
    if (!video_file) {
      setError("Select a video file first.");
      return;
    }
    setRunning(true);
    setStatus("");
    setError("");
    setStdout("");
    set_reasoning_summary(null);
    set_latest_reasoning(null);
    set_latest_flagged(null);
    try {
      const response = await run_video_pipeline(video_file);
      setStatus(response.message);
      setStdout(sanitize_run_log(response.stdout || response.stderr || ""));
      set_reasoning_summary(response.reasoningSummary);
      if (response.latestReasoning || response.latestFlagged) {
        set_latest_reasoning(response.latestReasoning);
        set_latest_flagged(response.latestFlagged);
        return;
      }

      // Pull full artifacts so the page renders description + debate outputs.
      const snapshot = await fetch_workspace_snapshot();
      const reasoning_item =
        snapshot.reasoningItems.find((item) => item.windowId === response.windowId) ??
        snapshot.reasoningItems[0] ??
        null;
      const flagged_item =
        snapshot.flaggedItems.find((item) => item.windowId === response.windowId) ??
        snapshot.flaggedItems[0] ??
        null;
      set_latest_reasoning(reasoning_item);
      set_latest_flagged(flagged_item);
    } catch (run_error) {
      if (run_error instanceof ApiError) {
        setError(`Run failed (${run_error.status}): ${run_error.message}`);
      } else {
        setError("Run failed due to an unexpected error.");
      }
    } finally {
      setRunning(false);
    }
  }

  const debate_turns = latest_reasoning ? build_debate_turns(latest_reasoning.debateHistory) : [];
  const scene_ticket =
    latest_reasoning !== null ? build_scene_report_ticket(latest_reasoning, latest_flagged) : "";

  return (
    <main className="video-lab-page">
      <section className="video-lab-card">
        <h1>Video Lab (Standalone)</h1>
        <p>
          Upload one video, run scene description + debate, and inspect pipeline logs directly.
        </p>

        <div className="video-lab-dropzone">
          <strong>{video_file ? video_file.name : "Choose a video to run"}</strong>
          <span>
            {video_file ? `${(video_file.size / (1024 * 1024)).toFixed(2)} MB` : "mp4 / mov / mkv / avi / webm"}
          </span>
        </div>

        <div className="video-lab-controls">
          <input
            type="file"
            accept=".mp4,.mov,.mkv,.avi,.webm"
            onChange={(event) => set_video_file(event.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            onClick={() => {
              void on_run();
            }}
            disabled={running}
          >
            {running ? "Running..." : "Run Description + Debate"}
          </button>
        </div>

        {status ? <div className="alert alert-success">{status}</div> : null}
        {error ? <div className="alert alert-error">{error}</div> : null}

        {latest_flagged ? (
          <div className="video-card">
            <h3>Flagged Window</h3>
            <p>
              <strong>Window:</strong> {latest_flagged.windowId}
            </p>
            <p>
              <strong>Cluster:</strong> {latest_flagged.clusterLabel} | <strong>Noise:</strong>{" "}
              {latest_flagged.isNoise ? "yes" : "no"} | <strong>Outlier score:</strong>{" "}
              {latest_flagged.outlierScore.toFixed(3)}
            </p>
            {latest_flagged.mp4Url ? (
              <video controls style={{ width: "100%", borderRadius: "8px" }} src={latest_flagged.mp4Url} />
            ) : null}
          </div>
        ) : null}

        {latest_reasoning ? (
          <div className="video-card">
            <h3>Description + Debate Result</h3>
            <p>
              <strong>Decision:</strong> {latest_reasoning.decision.toUpperCase()} | <strong>Recommendation:</strong>{" "}
              {latest_reasoning.recommendation} | <strong>Priority:</strong>{" "}
              {latest_reasoning.priorityScore.toFixed(3)}
            </p>
            <p>
              <strong>Scene description:</strong> {latest_reasoning.sceneDescription}
            </p>
            <p>
              <strong>Anomaly rationale:</strong> {latest_reasoning.anomalyRationale}
            </p>
            <p>
              <strong>Debate rounds captured:</strong> {Math.ceil(debate_turns.length / 2)}
            </p>

            {debate_turns.length ? (
              <div className="detailed-report">
                <strong>Debate Transcript (organized by round)</strong>
                {"\n\n"}
                {debate_turns
                  .map((turn) => `[Round ${turn.round}] ${turn.role}\n${turn.content}`)
                  .join("\n\n")}
              </div>
            ) : null}

            <div className="detailed-report">
              <strong>Detailed Scene Report Ticket</strong>
              {"\n\n"}
              {scene_ticket}
            </div>
          </div>
        ) : null}

        {reasoning_summary ? (
          <details>
            <summary>Reasoning Summary JSON</summary>
            <pre className="run-log">{JSON.stringify(reasoning_summary, null, 2)}</pre>
          </details>
        ) : null}

        {stdout ? <pre className="run-log">{stdout}</pre> : null}
      </section>
    </main>
  );
}

