export type SourceType = "jira" | "slack" | "manual";

export type SeverityCue = "critical" | "high" | "medium" | "low" | "unknown";

export type IngestFailureTicketRequest = {
  sourceType: SourceType;
  sourceRef: string;
  title: string;
  rawText: string;
  eventTimestamp: string;
  agentId: string | null;
  scenarioId: string | null;
  artifactsRef: string[] | null;
};

export type IngestFailureTicketResponse = {
  ticketId: string;
  status: "received" | "processing" | "failed";
  message: string;
};

export type GenerateFailureCapsuleRequest = {
  ticketId: string;
};

export type FailureCapsuleResponse = {
  capsuleId: string;
  ticketId: string;
  triageSummary: string;
  scenarioType: string | null;
  failureModeHints: string[];
  likelySubsystem: string | null;
  severityCue: SeverityCue;
  keyTimestamp: string | null;
  tags: string[];
  createdAt: string;
};

export type WorkspaceFlaggedItem = {
  windowId: string;
  sceneTokenHex: string;
  logId: string;
  clusterLabel: number;
  isNoise: boolean;
  outlierScore: number;
  anomalyRank: number;
  gridUrl: string | null;
  mp4Url: string | null;
};

export type WorkspaceReasoningItem = {
  windowId: string;
  sceneDescription: string;
  anomalyRationale: string;
  decision: "yes" | "no";
  recommendation: "add_immediately" | "already_covered" | "not_critical";
  priorityScore: number;
  modelSource: string;
  capabilityTag: string;
  debateHistory: string[];
  judgeRawOutput: string;
};

export type RegressionCaseProposal = {
  caseId: string;
  windowId: string;
  generatedAt: string;

  failureMode: string;
  whyAnomalous: string;
  evidenceSummary: string;

  riskLevel: "critical" | "high" | "medium" | "low";
  affectedCapability: string;
  affectedOdds: string[];

  counterarguments: string[];
  rebuttalSummary: string;

  decision: "add_to_suite" | "monitor" | "dismiss";
  recommendedTestSpec: string;
  scenarioVariants: string[];
  confidence: number;
  uncertaintyFactors: string[];

  debateTranscript: string[];
};

export type WorkspaceSnapshotResponse = {
  generatedAt: string;
  flaggedItems: WorkspaceFlaggedItem[];
  reasoningItems: WorkspaceReasoningItem[];
  anomalySummary: Record<string, unknown> | null;
  visualSummary: Record<string, unknown> | null;
  reasoningSummary: Record<string, unknown> | null;
  proposals: RegressionCaseProposal[];
};

export type RunVideoResponse = {
  ok: boolean;
  windowId: string;
  videoPath: string;
  stdout: string;
  stderr: string;
  reasoningSummary: Record<string, unknown> | null;
  latestReasoning: WorkspaceReasoningItem | null;
  latestFlagged: WorkspaceFlaggedItem | null;
  latestProposal: RegressionCaseProposal | null;
  message: string;
};

/** Structured line from pipeline stdout (matches Python PIPELINE_PROGRESS payload). */
export type PipelineProgressPayload = {
  step: string;
  title: string;
  detail: string;
};
