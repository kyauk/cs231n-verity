import type {
  FailureCapsuleResponse,
  GenerateFailureCapsuleRequest,
  IngestFailureTicketRequest,
  IngestFailureTicketResponse,
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

export { ApiError };
