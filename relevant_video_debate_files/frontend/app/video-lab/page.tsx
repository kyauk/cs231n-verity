"use client";

import { useState } from "react";

import { ApiError, fetch_workspace_snapshot, run_video_pipeline_stream } from "@/lib/api";
import type {
  PipelineProgressPayload,
  RegressionCaseProposal,
  WorkspaceFlaggedItem,
  WorkspaceReasoningItem
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
  flagged: WorkspaceFlaggedItem | null,
  proposal: RegressionCaseProposal | null
): string {
  const turns = build_debate_turns(reasoning.debateHistory);
  const transcript = turns.length
    ? turns
      .map((turn) => `[Round ${turn.round}] ${turn.role}: ${turn.content}`)
      .join("\n\n")
    : "No debate transcript captured.";

  const lines: string[] = [
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
    reasoning.judgeRawOutput || "No raw judge output captured."
  ];

  if (proposal) {
    lines.push(
      "",
      "Structured Proposal:",
      `- Case ID: ${proposal.caseId}`,
      `- Proposal Decision: ${proposal.decision}`,
      `- Risk Level: ${proposal.riskLevel.toUpperCase()}`,
      `- Affected Capability: ${proposal.affectedCapability || "unspecified"}`,
      `- ODD Conditions: ${proposal.affectedOdds.length ? proposal.affectedOdds.join(", ") : "n/a"}`,
      `- Confidence: ${(proposal.confidence * 100).toFixed(1)}%`,
      "",
      "Failure Mode:",
      proposal.failureMode,
      "",
      "Why Valuable:",
      proposal.whyAnomalous,
      "",
      "Evidence Summary:",
      proposal.evidenceSummary,
      "",
      "Recommended Test Spec:",
      proposal.recommendedTestSpec,
      "",
      "Scenario Variants:",
      proposal.scenarioVariants.length
        ? proposal.scenarioVariants.map((v) => `- ${v}`).join("\n")
        : "- n/a",
      "",
      "Counterarguments:",
      proposal.counterarguments.length
        ? proposal.counterarguments.map((v) => `- ${v}`).join("\n")
        : "- none raised",
      "",
      "Rebuttal Summary:",
      proposal.rebuttalSummary || "n/a",
      "",
      "Uncertainty Factors:",
      proposal.uncertaintyFactors.length
        ? proposal.uncertaintyFactors.map((v) => `- ${v}`).join("\n")
        : "- none flagged"
    );
  }

  lines.push(
    "",
    "Proposed Action:",
    reasoning.decision === "yes"
      ? "Add this scenario to the regression suite and include weather/visibility variants."
      : "Keep as monitored anomaly and revisit after additional neighbor comparisons."
  );

  return lines.join("\n");
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

export default function VideoLabPage(): JSX.Element {
  const [video_file, set_video_file] = useState<File | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [stdout, setStdout] = useState<string>("");
  const [reasoning_summary, set_reasoning_summary] = useState<Record<string, unknown> | null>(null);
  const [latest_reasoning, set_latest_reasoning] = useState<WorkspaceReasoningItem | null>(null);
  const [latest_flagged, set_latest_flagged] = useState<WorkspaceFlaggedItem | null>(null);
  const [latest_proposal, set_latest_proposal] = useState<RegressionCaseProposal | null>(null);
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

  async function on_run(): Promise<void> {
    if (!video_file) {
      setError("Select a video file first.");
      return;
    }
    setRunning(true);
    setStatus("");
    setError("");
    setStdout("");
    set_progress_log([]);
    set_reasoning_summary(null);
    set_latest_reasoning(null);
    set_latest_flagged(null);
    set_latest_proposal(null);
    try {
      const response = await run_video_pipeline_stream(video_file, append_progress);
      setStatus(response.message);
      setStdout(sanitize_run_log(response.stdout || response.stderr || ""));
      set_reasoning_summary(response.reasoningSummary);
      if (response.latestReasoning || response.latestFlagged) {
        set_latest_reasoning(response.latestReasoning);
        set_latest_flagged(response.latestFlagged);
        set_latest_proposal(response.latestProposal ?? null);
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
      const proposal_item =
        snapshot.proposals?.find((item) => item.windowId === response.windowId) ??
        snapshot.proposals?.[0] ??
        null;
      set_latest_reasoning(reasoning_item);
      set_latest_flagged(flagged_item);
      set_latest_proposal(proposal_item);
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
    latest_reasoning !== null
      ? build_scene_report_ticket(latest_reasoning, latest_flagged, latest_proposal)
      : "";

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

        {running || progress_log.length > 0 ? (
          <div className="pipeline-progress-panel" aria-live="polite">
            <div className="pipeline-progress-header">
              {running ? "Pipeline progress" : "Last run timeline"}
            </div>
            <ul className="pipeline-progress-list">
              {progress_log.map((row, index) => (
                <li
                  key={row.key}
                  className={`pipeline-progress-row${running && index === progress_log.length - 1 ? " is-active" : ""}`}
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

        {latest_proposal ? (
          <div className="video-card">
            <div className="artifact-preview-header">
              <strong>
                Regression-Case Proposal: {latest_proposal.caseId || latest_proposal.windowId}
              </strong>
              <span
                className={`severity-chip ${decision_chip_class(latest_proposal.decision)}`}
              >
                {latest_proposal.decision.replace(/_/g, " ")}
              </span>
            </div>

            <div className="capsule-block">
              <h4>Failure Mode &amp; Evidence</h4>
              <p>
                <strong>Failure mode: </strong>
                {latest_proposal.failureMode}
              </p>
              <p>
                <strong>Why this is valuable: </strong>
                {latest_proposal.whyAnomalous}
              </p>
              <p>
                <strong>Evidence summary: </strong>
                {latest_proposal.evidenceSummary}
              </p>
            </div>

            <div className="capsule-meta">
              <div>
                <strong>Risk level</strong>
                <span
                  className={`severity-chip ${risk_chip_class(latest_proposal.riskLevel)}`}
                >
                  {latest_proposal.riskLevel.toUpperCase()}
                </span>
              </div>
              <div>
                <strong>Affected capability</strong>
                <span className="tag">
                  {latest_proposal.affectedCapability || "unspecified"}
                </span>
              </div>
              <div>
                <strong>ODD conditions</strong>
                {latest_proposal.affectedOdds.length ? (
                  <span className="tag-row">
                    {latest_proposal.affectedOdds.map((odd, index) => (
                      <span key={`odd-${index}`} className="tag">
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
              <h4>Counterarguments (Coverage Analyst)</h4>
              {latest_proposal.counterarguments.length ? (
                <ol>
                  {latest_proposal.counterarguments.map((item, index) => (
                    <li key={`cb-${index}`}>{item}</li>
                  ))}
                </ol>
              ) : (
                <p>No counterarguments raised.</p>
              )}
              {latest_proposal.rebuttalSummary ? (
                <p>
                  <strong>Rebuttal: </strong>
                  {latest_proposal.rebuttalSummary}
                </p>
              ) : null}
            </div>

            <div className="capsule-block">
              <h4>Recommended Action</h4>
              <p>
                <strong>Test spec: </strong>
                {latest_proposal.recommendedTestSpec}
              </p>
              {latest_proposal.scenarioVariants.length ? (
                <p>
                  <strong>Variants to also test: </strong>
                  <span className="tag-row">
                    {latest_proposal.scenarioVariants.map((variant, index) => (
                      <span key={`var-${index}`} className="tag">
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
                <span>{`${(latest_proposal.confidence * 100).toFixed(1)}%`}</span>
              </div>
              <div>
                <strong>Uncertainty factors</strong>
                {latest_proposal.uncertaintyFactors.length ? (
                  <ul>
                    {latest_proposal.uncertaintyFactors.map((item, index) => (
                      <li key={`uf-${index}`}>{item}</li>
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
                {latest_proposal.debateTranscript.length
                  ? latest_proposal.debateTranscript.join("\n")
                  : "No transcript captured."}
              </pre>
            </details>
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

