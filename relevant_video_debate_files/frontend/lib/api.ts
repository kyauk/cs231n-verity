import type {
  FailureCapsuleResponse,
  GenerateFailureCapsuleRequest,
  IngestFailureTicketRequest,
  IngestFailureTicketResponse,
  PipelineProgressPayload,
  RunVideoResponse,
  WorkspaceSnapshotResponse
} from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function parse_response<T>(response: Response): Promise<T> {
  const response_text = await response.text();
  const parsed_body: unknown = response_text ? JSON.parse(response_text) : {};

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    if (
      parsed_body &&
      typeof parsed_body === "object" &&
      "detail" in parsed_body &&
      typeof parsed_body.detail === "string"
    ) {
      message = parsed_body.detail;
    }
    throw new ApiError(message, response.status);
  }

  return parsed_body as T;
}

export async function ingest_failure_ticket(
  payload: IngestFailureTicketRequest
): Promise<IngestFailureTicketResponse> {
  const response = await fetch(`${API_BASE_URL}/ingest/failure_ticket`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parse_response<IngestFailureTicketResponse>(response);
}

export async function generate_failure_capsule(
  payload: GenerateFailureCapsuleRequest
): Promise<FailureCapsuleResponse> {
  const response = await fetch(`${API_BASE_URL}/generate/failure_capsule`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parse_response<FailureCapsuleResponse>(response);
}

export async function fetch_failure_capsule(capsule_id: string): Promise<FailureCapsuleResponse> {
  const response = await fetch(`${API_BASE_URL}/fetch/failure_capsule/${capsule_id}`, {
    method: "GET"
  });
  return parse_response<FailureCapsuleResponse>(response);
}

export async function fetch_workspace_snapshot(): Promise<WorkspaceSnapshotResponse> {
  const response = await fetch("/api/workspace/snapshot", {
    method: "GET"
  });
  return parse_response<WorkspaceSnapshotResponse>(response);
}

export async function run_video_pipeline(video_file: File): Promise<RunVideoResponse> {
  const form_data = new FormData();
  form_data.append("video", video_file);
  const response = await fetch("/api/workspace/run-video", {
    method: "POST",
    body: form_data
  });
  return parse_response<RunVideoResponse>(response);
}

/**
 * Streaming pipeline run: emits structured progress via callback, then resolves with the same shape as run_video_pipeline.
 */
export async function run_video_pipeline_stream(
  video_file: File,
  on_progress: (payload: PipelineProgressPayload) => void
): Promise<RunVideoResponse> {
  const form_data = new FormData();
  form_data.append("video", video_file);
  const response = await fetch("/api/workspace/run-video-stream", {
    method: "POST",
    body: form_data
  });

  if (!response.ok) {
    const response_text = await response.text();
    let message = `Request failed with status ${response.status}`;
    try {
      const parsed = response_text ? JSON.parse(response_text) : {};
      if (parsed && typeof parsed === "object" && "detail" in parsed && typeof parsed.detail === "string") {
        message = parsed.detail;
      }
    } catch {
      if (response_text) {
        message = response_text;
      }
    }
    throw new ApiError(message, response.status);
  }

  const body = response.body;
  if (!body) {
    throw new ApiError("No response body from stream.", 500);
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed: RunVideoResponse | null = null;

  const process_blocks = (): void => {
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!block.startsWith("data: ")) {
        continue;
      }
      const json_text = block.slice(6).trim();
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(json_text) as Record<string, unknown>;
      } catch {
        continue;
      }
      const kind = parsed.kind;
      if (kind === "progress") {
        const payload = parsed.payload as PipelineProgressPayload | undefined;
        if (payload && typeof payload.title === "string") {
          on_progress({
            step: typeof payload.step === "string" ? payload.step : "progress",
            title: payload.title,
            detail: typeof payload.detail === "string" ? payload.detail : ""
          });
        }
      } else if (kind === "error") {
        const detail =
          typeof parsed.detail === "string"
            ? parsed.detail
            : typeof parsed.logTail === "string"
              ? parsed.logTail.slice(-2000)
              : "Pipeline failed.";
        throw new ApiError(detail, 500);
      } else if (kind === "complete" && parsed.ok === true) {
        completed = {
          ok: true,
          windowId: String(parsed.windowId ?? ""),
          videoPath: String(parsed.videoPath ?? ""),
          stdout: String(parsed.stdout ?? ""),
          stderr: String(parsed.stderr ?? ""),
          reasoningSummary: (parsed.reasoningSummary as Record<string, unknown> | null) ?? null,
          latestReasoning: (parsed.latestReasoning as RunVideoResponse["latestReasoning"]) ?? null,
          latestFlagged: (parsed.latestFlagged as RunVideoResponse["latestFlagged"]) ?? null,
          message: String(parsed.message ?? "Done.")
        };
      }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
      process_blocks();
    }
    if (done) {
      buffer += decoder.decode();
      process_blocks();
      break;
    }
  }

  if (!completed) {
    throw new ApiError("Stream ended without a complete event.", 500);
  }

  return completed;
}

export { ApiError };
