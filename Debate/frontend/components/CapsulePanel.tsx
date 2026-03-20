"use client";

import { useState } from "react";

import { SectionCard } from "@/components/SectionCard";
import type { FailureCapsuleResponse } from "@/types/api";

type CapsulePanelProps = {
  capsule: FailureCapsuleResponse | null;
};

function format_datetime(value: string | null): string {
  if (!value) {
    return "N/A";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

export function CapsulePanel({ capsule }: CapsulePanelProps): JSX.Element {
  const [copy_status, set_copy_status] = useState<"idle" | "copied" | "error">("idle");

  async function copy_capsule_json(): Promise<void> {
    if (!capsule) {
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(capsule, null, 2));
      set_copy_status("copied");
      setTimeout(() => set_copy_status("idle"), 1500);
    } catch {
      set_copy_status("error");
      setTimeout(() => set_copy_status("idle"), 1500);
    }
  }

  async function copy_capsule_id(): Promise<void> {
    if (!capsule) {
      return;
    }
    try {
      await navigator.clipboard.writeText(capsule.capsuleId);
      set_copy_status("copied");
      setTimeout(() => set_copy_status("idle"), 1500);
    } catch {
      set_copy_status("error");
      setTimeout(() => set_copy_status("idle"), 1500);
    }
  }

  function download_capsule_json(): void {
    if (!capsule) {
      return;
    }
    const blob = new Blob([JSON.stringify(capsule, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `capsule-${capsule.capsuleId}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  return (
    <SectionCard
      title="Failure Capsule"
      subtitle="Structured triage artifact generated from ticket evidence."
    >
      <div className="capsule-actions">
        <button type="button" onClick={copy_capsule_json} disabled={!capsule}>
          Copy JSON
        </button>
        <button type="button" onClick={download_capsule_json} disabled={!capsule}>
          Export JSON
        </button>
        <button type="button" onClick={copy_capsule_id} disabled={!capsule}>
          Copy Capsule ID
        </button>
        {copy_status !== "idle" ? (
          <span className={`copy-state copy-state-${copy_status}`}>
            {copy_status === "copied" ? "Copied" : "Copy failed"}
          </span>
        ) : null}
      </div>

      {!capsule ? (
        <div className="empty-state">
          <p>No capsule generated yet.</p>
          <span>Submit a ticket or fetch an existing capsule ID.</span>
        </div>
      ) : (
        <div className="capsule-grid">
          <div className="capsule-block">
            <h3>Summary</h3>
            <p>{capsule.triageSummary}</p>
          </div>
          <div className="capsule-meta">
            <div>
              <strong>Capsule ID</strong>
              <span>{capsule.capsuleId}</span>
            </div>
            <div>
              <strong>Ticket ID</strong>
              <span>{capsule.ticketId}</span>
            </div>
            <div>
              <strong>Scenario Type</strong>
              <span>{capsule.scenarioType ?? "unknown"}</span>
            </div>
            <div>
              <strong>Likely Subsystem</strong>
              <span>{capsule.likelySubsystem ?? "unknown"}</span>
            </div>
            <div>
              <strong>Severity</strong>
              <span className={`severity-chip severity-${capsule.severityCue}`}>
                {capsule.severityCue}
              </span>
            </div>
            <div>
              <strong>Key Timestamp</strong>
              <span>{format_datetime(capsule.keyTimestamp)}</span>
            </div>
            <div>
              <strong>Created At</strong>
              <span>{format_datetime(capsule.createdAt)}</span>
            </div>
          </div>
          <div className="capsule-block">
            <h3>Failure Mode Hints</h3>
            <ul>
              {capsule.failureModeHints.length === 0 ? <li>None</li> : null}
              {capsule.failureModeHints.map((hint) => (
                <li key={hint}>{hint}</li>
              ))}
            </ul>
          </div>
          <div className="capsule-block">
            <h3>Tags</h3>
            <div className="tag-row">
              {capsule.tags.length === 0 ? <span className="tag">none</span> : null}
              {capsule.tags.map((tag) => (
                <span className="tag" key={tag}>
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </SectionCard>
  );
}
