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

export interface FlaggedScenario {
  id: string
  scenarioName: string
  clusterId: number
  priorityScore: number
  definingConditions: string
  hasSimulationSpec: boolean
  region: string
}
