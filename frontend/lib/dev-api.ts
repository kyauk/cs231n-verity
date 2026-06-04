// Fetch wrappers for the dev_dashboard backend (Module 7).
// All endpoints are private — the bundle only includes this code path when
// NEXT_PUBLIC_DEV_DASHBOARD_URL is set at build time.

import type {
  AccuracyReport,
  CreateRoundRequest,
  CreateRoundResponse,
  ExportResponse,
  NextWindowResponse,
  RoundListEntry,
  RoundStatus,
  SubmitRatingRequest,
  VideoUrlResponse,
} from './dev-types'

export const DEV_API_URL =
  process.env.NEXT_PUBLIC_DEV_DASHBOARD_URL ?? 'http://localhost:8002'

async function _json<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`HTTP ${resp.status}: ${text || resp.statusText}`)
  }
  return resp.json()
}

// ---------- Discrimination test ----------

export async function createRound(
  req: CreateRoundRequest,
): Promise<CreateRoundResponse> {
  return _json(
    await fetch(`${DEV_API_URL}/dev/rounds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  )
}

export async function listRounds(): Promise<RoundListEntry[]> {
  return _json(await fetch(`${DEV_API_URL}/dev/rounds`))
}

export async function getRoundStatus(roundId: string): Promise<RoundStatus> {
  return _json(await fetch(`${DEV_API_URL}/dev/rounds/${roundId}`))
}

export async function getNextWindow(
  roundId: string,
): Promise<NextWindowResponse> {
  return _json(await fetch(`${DEV_API_URL}/dev/rounds/${roundId}/next`))
}

export async function getVideoUrl(
  roundId: string,
  segmentId: string,
  windowIdx: number,
  camera = 'FRONT',
): Promise<VideoUrlResponse> {
  const params = new URLSearchParams({
    segment_id: segmentId,
    window_idx: String(windowIdx),
    camera,
  })
  return _json(
    await fetch(`${DEV_API_URL}/dev/rounds/${roundId}/video-url?${params}`),
  )
}

export async function submitRating(
  roundId: string,
  req: SubmitRatingRequest,
): Promise<void> {
  await _json(
    await fetch(`${DEV_API_URL}/dev/rounds/${roundId}/ratings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  )
}

export async function exportRound(roundId: string): Promise<ExportResponse> {
  return _json(await fetch(`${DEV_API_URL}/dev/rounds/${roundId}/export`))
}

// ---------- Accuracy ----------

export async function getAccuracyTemplate(): Promise<unknown> {
  return _json(await fetch(`${DEV_API_URL}/dev/accuracy/template`))
}

export async function postAccuracyDiff(
  gold: unknown,
  schemaRecords: unknown[],
): Promise<AccuracyReport> {
  return _json(
    await fetch(`${DEV_API_URL}/dev/accuracy/diff`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gold, schema_records: schemaRecords }),
    }),
  )
}
