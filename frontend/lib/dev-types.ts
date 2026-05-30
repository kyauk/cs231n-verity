// TypeScript types for the Module 7: Dev Dashboard backend.
// These mirror the Pydantic shapes in pipeline/modules/dev_dashboard/server.py.
// If you change one, change the other.

export type WindowKeyDTO = {
  segment_id: string
  window_idx: number
}

// ----- Discrimination test -----

export type CreateRoundRequest = {
  dataset_label: string
  pool_size: number
  seed: number
  top_k_rare_atoms: number
  scored: unknown[]
  schema_records: unknown[]
}

export type CreateRoundResponse = {
  round_id: string
  total_windows: number
  naive_rare_atoms: string[]
}

export type RoundListEntry = {
  round_id: string
  created_at: string
  dataset_label: string
}

export type RoundStatus = {
  round_id: string
  created_at: string
  dataset_label: string
  pool_size: number
  total_windows: number
  rated_count: number
  complete: boolean
}

export type NextWindowResponse = {
  complete: boolean
  window: WindowKeyDTO | null
  progress_idx: number
  total_windows: number
}

export type VideoUrlResponse = {
  url: string
  generated_at: string
}

export type SubmitRatingRequest = {
  rater_id: string
  window: WindowKeyDTO
  safety_relevance: number
  perceived_rarity: number
  free_text_note?: string | null
}

export type ExportRow = {
  rater_id: string
  window: WindowKeyDTO
  source_pool: 'verity' | 'random' | 'naive_rare'
  safety_relevance: number
  perceived_rarity: number
  timestamp: string
  free_text_note: string | null
}

export type ExportResponse = {
  round_id: string
  dataset_label: string
  pool_size: number
  seed: number
  naive_rare_atoms: string[]
  complete: boolean
  ratings: ExportRow[]
}

// ----- Accuracy diff -----

export type FieldDiff = {
  field_path: string
  gold: unknown
  vlm: unknown
  match: boolean
  precision: number | null
  recall: number | null
  f1: number | null
}

export type WindowDiff = {
  window_id: WindowKeyDTO
  fields: FieldDiff[]
}

export type MissingEntry = {
  window_id: WindowKeyDTO
  direction: 'missing_in_records' | 'missing_in_gold'
}

export type AccuracyReport = {
  schema_version: string
  windows: WindowDiff[]
  field_aggregates: Record<string, [number, number]>
  missing_entries: MissingEntry[]
}
