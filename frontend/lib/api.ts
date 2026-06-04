/**
 * API client for the Waymo discovery runner (waymo-pipeline/waymo_runner.py).
 *
 * The runner returns shapes that map directly onto the UI types in
 * `lib/types.ts`, so no transformation layer is needed beyond what is here.
 *
 * Base URL comes from NEXT_PUBLIC_API_URL (defaults to http://localhost:8000).
 */

import type {
  BatchJob,
  ClusterPoint,
  ClusterStats,
  Scene,
  FlaggedScenario,
  JudgeProposalRow,
  JudgeProposalDetail,
  JudgeVideoUrl,
  JudgeRatingSubmission,
  JudgeSessionSummary,
} from './types'

// Default to RELATIVE (empty base) so every call goes through the single Next
// origin and its proxy rewrites. Absolute backend URLs break under port-
// forwarding — an absolute backend port resolves to the USER's machine, not the
// server's. Override with NEXT_PUBLIC_* only for split-origin deployments.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? ''

const JUDGE_API_URL =
  process.env.NEXT_PUBLIC_JUDGE_API_URL ?? ''

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  const text = await response.text()
  const body: unknown = text ? JSON.parse(text) : {}

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    if (
      body &&
      typeof body === 'object' &&
      'detail' in body &&
      typeof (body as { detail: unknown }).detail === 'string'
    ) {
      message = (body as { detail: string }).detail
    }
    throw new ApiError(message, response.status)
  }
  return body as T
}

/* ------------------------------------------------------------------ */
/* Ingest tab                                                          */
/* ------------------------------------------------------------------ */

/** Fetch all embedding batch jobs, newest first. */
export async function fetchBatchJobs(): Promise<BatchJob[]> {
  const response = await fetch(`${API_BASE_URL}/batches`, { cache: 'no-store' })
  const data = await parseResponse<{ batches: BatchJob[] }>(response)
  return data.batches
}

/** Probe a GCS path and return how many segments are available. */
export async function probePath(uri: string): Promise<{ valid: boolean; segmentCount: number; detail: string }> {
  const response = await fetch(`${API_BASE_URL}/probe-path?uri=${encodeURIComponent(uri)}`, { cache: 'no-store' })
  return parseResponse<{ valid: boolean; segmentCount: number; detail: string }>(response)
}

/** Launch a new embedding batch and return the created record. */
export async function launchBatch(
  dataSourceUri: string,
  label: string,
  region: string,
  maxSegments: number = 5,
  mode: 'cluster' | 'reason' | 'both' = 'both',
): Promise<BatchJob> {
  const response = await fetch(`${API_BASE_URL}/batches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataSourceUri, label, region, maxSegments, mode }),
  })
  const data = await parseResponse<{ batch: BatchJob }>(response)
  return data.batch
}

/* ------------------------------------------------------------------ */
/* Cluster Space tab                                                   */
/* ------------------------------------------------------------------ */

/** Fetch 3D cluster points and per-cluster statistics. */
export async function fetchClusterSpace(): Promise<{
  points: ClusterPoint[]
  clusterStats: ClusterStats[]
}> {
  const response = await fetch(`${API_BASE_URL}/cluster-space`, {
    cache: 'no-store',
  })
  return parseResponse<{ points: ClusterPoint[]; clusterStats: ClusterStats[] }>(
    response,
  )
}

/** Fetch one scene's detail (video URL + annotations). */
export async function fetchScene(sceneId: string): Promise<Scene> {
  const response = await fetch(
    `${API_BASE_URL}/scenes/${encodeURIComponent(sceneId)}`,
    { cache: 'no-store' },
  )
  const data = await parseResponse<{
    id: string
    videoUrl: string
    thumbnail: string
    annotations: Scene['annotations']
  }>(response)
  return {
    id: data.id,
    videoUrl: data.videoUrl,
    thumbnail: data.thumbnail,
    annotations: data.annotations,
  }
}

/* ------------------------------------------------------------------ */
/* Dashboard tab                                                       */
/* ------------------------------------------------------------------ */

/** Fetch flagged scenarios for the results dashboard. */
export async function fetchScenarios(): Promise<FlaggedScenario[]> {
  const response = await fetch(`${API_BASE_URL}/scenarios`, {
    cache: 'no-store',
  })
  const data = await parseResponse<{ scenarios: FlaggedScenario[] }>(response)
  return data.scenarios
}

/* ------------------------------------------------------------------ */
/* Analysis tab — SSE-streamed agentic analysis                        */
/* ------------------------------------------------------------------ */

/** Structured progress line emitted by the pipeline subprocess. */
export interface AnalysisProgress {
  step: string
  title: string
  detail: string
}

/** Final analysis result returned when the stream completes. */
export interface AnalysisResultPayload {
  sceneId: string
  agentOutputs: {
    proposer: string
    critic: string
    judge: string
  }
  conclusion: {
    sceneId: string
    verdict: string
    priorityScore: number
    simulationSpec: string
  }
  sceneDescription: string
}

/**
 * Run agentic analysis for a scene. Progress events are delivered to
 * `onProgress`; the promise resolves with the final analysis result.
 */
export async function runAnalysisStream(
  sceneId: string,
  onProgress: (progress: AnalysisProgress) => void,
): Promise<AnalysisResultPayload> {
  const response = await fetch(`${API_BASE_URL}/analysis/run-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sceneId }),
  })

  if (!response.ok || !response.body) {
    const text = await response.text()
    let message = `Request failed with status ${response.status}`
    try {
      const parsed = text ? JSON.parse(text) : {}
      if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
        message = String((parsed as { detail: unknown }).detail)
      }
    } catch {
      if (text) message = text
    }
    throw new ApiError(message, response.status)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed: AnalysisResultPayload | null = null

  // Accumulate any SSE error so it can be thrown after the stream drains,
  // rather than inside processBlocks where it would abandon buffered bytes.
  let sseError: ApiError | null = null

  const processBlocks = (): void => {
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, sep).trim()
      buffer = buffer.slice(sep + 2)
      if (!block.startsWith('data: ')) continue

      let parsed: Record<string, unknown>
      try {
        parsed = JSON.parse(block.slice(6).trim()) as Record<string, unknown>
      } catch {
        continue
      }

      if (parsed.kind === 'progress') {
        const payload = parsed.payload as AnalysisProgress | undefined
        if (payload && typeof payload.title === 'string') {
          onProgress({
            step: typeof payload.step === 'string' ? payload.step : 'progress',
            title: payload.title,
            detail: typeof payload.detail === 'string' ? payload.detail : '',
          })
        }
      } else if (parsed.kind === 'error') {
        sseError = new ApiError(
          typeof parsed.detail === 'string' ? parsed.detail : 'Analysis failed.',
          500,
        )
      } else if (parsed.kind === 'complete' && parsed.ok === true) {
        completed = {
          sceneId: String(parsed.sceneId ?? ''),
          agentOutputs: parsed.agentOutputs as AnalysisResultPayload['agentOutputs'],
          conclusion: parsed.conclusion as AnalysisResultPayload['conclusion'],
          sceneDescription: String(parsed.sceneDescription ?? ''),
        }
      }
    }
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: true })
      processBlocks()
    }
    if (done) {
      buffer += decoder.decode()
      processBlocks()
      break
    }
  }

  if (sseError) throw sseError

  if (!completed) {
    throw new ApiError('Stream ended without a complete event.', 500)
  }
  return completed
}

/* ------------------------------------------------------------------ */
/* Judge UI tab (Module 5)                                             */
/* ------------------------------------------------------------------ */

export async function fetchJudgeProposals(): Promise<JudgeProposalRow[]> {
  // cache-buster query param defeats any intermediary/tunnel cache, on top of no-store
  const response = await fetch(`${JUDGE_API_URL}/judge/proposals?_t=${Date.now()}`, { cache: 'no-store' })
  return parseResponse<JudgeProposalRow[]>(response)
}

export async function fetchJudgeProposalDetail(proposalId: string): Promise<JudgeProposalDetail> {
  const response = await fetch(
    `${JUDGE_API_URL}/judge/proposals/${encodeURIComponent(proposalId)}`,
    { cache: 'no-store' },
  )
  return parseResponse<JudgeProposalDetail>(response)
}

export async function fetchJudgeVideoUrl(
  segmentId: string,
  windowIdx: number,
  camera: string = 'FRONT',
): Promise<JudgeVideoUrl> {
  const params = new URLSearchParams({
    segment_id: segmentId,
    window_idx: String(windowIdx),
    camera,
  })
  const response = await fetch(`${JUDGE_API_URL}/judge/video-url?${params}`, { cache: 'no-store' })
  return parseResponse<JudgeVideoUrl>(response)
}

export async function submitJudgeRating(
  submission: JudgeRatingSubmission,
): Promise<{ ok: boolean }> {
  const response = await fetch(`${JUDGE_API_URL}/judge/ratings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(submission),
  })
  return parseResponse<{ ok: boolean }>(response)
}

export async function fetchJudgeSession(raterId: string): Promise<JudgeSessionSummary> {
  const response = await fetch(
    `${JUDGE_API_URL}/judge/session/${encodeURIComponent(raterId)}`,
    { cache: 'no-store' },
  )
  return parseResponse<JudgeSessionSummary>(response)
}
