export interface BatchJob {
  id: string
  label: string
  dataSourceUri: string
  region: string
  status: 'running' | 'completed' | 'failed'
  scenesProcessed: number
  totalScenes: number | null
  startedAt: string
  completedAt: string | null
}

export interface ClusterPoint {
  id: string
  x: number
  y: number
  z: number
  clusterId: number
  sceneId: string
  isNoise: boolean
}

export interface ClusterStats {
  id: number
  sceneCount: number
  density: number
}

export interface Scene {
  id: string
  videoUrl: string
  thumbnail: string
  annotations: {
    weather: string
    timeOfDay: string
    roadType: string
    actors: string[]
    events: string[]
  }
}

export interface AgentStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'complete'
  output: string
}

export interface AnalysisResult {
  sceneId: string
  verdict: string
  priorityScore: number
  simulationSpec: string
}

export interface AnalysisHistoryEntry {
  id: string
  sceneId: string
  ranAt: string
  verdict: string
  priorityScore: number
  simulationSpec: string
  agentOutputs: {
    proposer: string
    critic: string
    judge: string
  }
}

export interface FlaggedScenario {
  id: string
  scenarioName: string
  clusterId: number
  priorityScore: number
  definingConditions: string
  hasSimulationSpec: boolean
  region: string
}

// ---------------------------------------------------------------------------
// Module 5: Judge UI types
// ---------------------------------------------------------------------------

export interface JudgeScoreBadges {
  novelty_score: number
  plausibility_score: number
  frontier_difficulty_score: number | null
  final_rank_score: number
}

export interface JudgeMotivatingScene {
  segment_id: string
  window_idx: number
}

export interface JudgeProposalRow {
  proposal_id: string
  constituents: string[]
  scores: JudgeScoreBadges
  motivating_scene_count: number
}

export interface JudgeProposalDetail {
  proposal_id: string
  constituents: string[]
  scores: JudgeScoreBadges
  plausibility_justification: string
  motivating_scenes: JudgeMotivatingScene[]
  rejection_reason: string | null
}

export interface JudgeVideoUrl {
  url: string
  generated_at: string
}

export interface JudgeRatingSubmission {
  rater_id: string
  proposal_id: string
  coherence_score: number
  usefulness_score: number
  free_text_note: string | null
  seen_motivating_scenes: JudgeMotivatingScene[]
}

export interface JudgeSessionSummary {
  rater_id: string
  rated_proposal_ids: string[]
  total_accepted: number
  coherence_distribution: Record<number, number>
  usefulness_distribution: Record<number, number>
  mean_coherence: number | null
  mean_usefulness: number | null
}
