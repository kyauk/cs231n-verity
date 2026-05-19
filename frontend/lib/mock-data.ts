import type { BatchJob, ClusterPoint, ClusterStats, Scene, FlaggedScenario } from './types'

export const mockBatchJobs: BatchJob[] = [
  { 
    id: 'batch-001', 
    label: 'Phoenix Q4 Highway Collection', 
    dataSourceUri: 's3://verity-data/phoenix/2024-q4/highway/', 
    region: 'US-West',
    status: 'completed',
    scenesProcessed: 12847,
    totalScenes: 12847,
    startedAt: '2024-12-15T08:30:00Z',
    completedAt: '2024-12-15T14:22:00Z'
  },
  { 
    id: 'batch-002', 
    label: 'Austin Urban Intersections', 
    dataSourceUri: 's3://verity-data/austin/urban/intersections/', 
    region: 'US-West',
    status: 'completed',
    scenesProcessed: 8432,
    totalScenes: 8432,
    startedAt: '2024-12-14T10:15:00Z',
    completedAt: '2024-12-14T13:45:00Z'
  },
  { 
    id: 'batch-003', 
    label: 'Munich Winter Conditions', 
    dataSourceUri: 'gs://verity-eu/munich/winter-2024/', 
    region: 'EU-West',
    status: 'running',
    scenesProcessed: 3201,
    totalScenes: null,
    startedAt: '2024-12-16T06:00:00Z',
    completedAt: null
  },
  { 
    id: 'batch-004', 
    label: 'SF Construction Zones', 
    dataSourceUri: 's3://verity-data/sf/construction/', 
    region: 'US-West',
    status: 'failed',
    scenesProcessed: 1502,
    totalScenes: 4200,
    startedAt: '2024-12-13T09:00:00Z',
    completedAt: '2024-12-13T11:30:00Z'
  },
  { 
    id: 'batch-005', 
    label: 'Boston Night Driving', 
    dataSourceUri: 's3://verity-data/boston/night/', 
    region: 'US-East',
    status: 'completed',
    scenesProcessed: 5621,
    totalScenes: 5621,
    startedAt: '2024-12-12T22:00:00Z',
    completedAt: '2024-12-13T02:15:00Z'
  },
]

// Generate cluster points for 3D visualization
function generateClusterPoints(): ClusterPoint[] {
  const points: ClusterPoint[] = []
  const clusterCenters = [
    { x: -3, y: 2, z: 1, count: 45 },
    { x: 2, y: -1, z: 3, count: 32 },
    { x: 0, y: 3, z: -2, count: 28 },
    { x: -2, y: -2, z: -1, count: 18 },
    { x: 3, y: 1, z: -3, count: 22 },
  ]

  clusterCenters.forEach((center, clusterId) => {
    for (let i = 0; i < center.count; i++) {
      const spread = clusterId < 2 ? 0.4 : 0.8 // Dense vs sparse clusters
      points.push({
        id: `point-${clusterId}-${i}`,
        x: center.x + (Math.random() - 0.5) * spread * 2,
        y: center.y + (Math.random() - 0.5) * spread * 2,
        z: center.z + (Math.random() - 0.5) * spread * 2,
        clusterId,
        sceneId: `scene-${clusterId}-${i}`,
        isNoise: false,
      })
    }
  })

  // Add noise points
  for (let i = 0; i < 12; i++) {
    points.push({
      id: `noise-${i}`,
      x: (Math.random() - 0.5) * 10,
      y: (Math.random() - 0.5) * 10,
      z: (Math.random() - 0.5) * 10,
      clusterId: -1,
      sceneId: `noise-scene-${i}`,
      isNoise: true,
    })
  }

  return points
}

export const mockClusterPoints = generateClusterPoints()

export const mockClusterStats: ClusterStats[] = [
  { id: 0, sceneCount: 45, density: 0.92 },
  { id: 1, sceneCount: 32, density: 0.88 },
  { id: 2, sceneCount: 28, density: 0.65 },
  { id: 3, sceneCount: 18, density: 0.54 },
  { id: 4, sceneCount: 22, density: 0.71 },
]

export const mockScene: Scene = {
  id: 'scene-0-15',
  videoUrl: '/placeholder-video.mp4',
  thumbnail: '/placeholder.svg?height=180&width=320',
  annotations: {
    weather: 'Clear',
    timeOfDay: 'Dusk',
    roadType: 'Highway Merge',
    actors: ['Lead Vehicle', 'Adjacent Vehicle', 'Motorcycle'],
    events: ['Lane Change', 'Hard Brake', 'Cut-in'],
  },
}

export const mockFlaggedScenarios: FlaggedScenario[] = [
  { id: '1', scenarioName: 'Aggressive Lane Change', clusterId: 0, priorityScore: 92, definingConditions: 'Speed > 65mph, Gap < 2s, Rain', hasSimulationSpec: true, region: 'US-West' },
  { id: '2', scenarioName: 'Pedestrian Occlusion', clusterId: 2, priorityScore: 88, definingConditions: 'Urban intersection, Low visibility, Multiple pedestrians', hasSimulationSpec: true, region: 'US-East' },
  { id: '3', scenarioName: 'Construction Merge', clusterId: 1, priorityScore: 85, definingConditions: 'Lane closure, Heavy traffic, Night', hasSimulationSpec: false, region: 'US-West' },
  { id: '4', scenarioName: 'Emergency Vehicle Response', clusterId: 3, priorityScore: 79, definingConditions: 'Ambulance approaching, Multi-lane, Urban', hasSimulationSpec: false, region: 'EU-West' },
  { id: '5', scenarioName: 'Cyclist Cut-through', clusterId: 4, priorityScore: 76, definingConditions: 'Bike lane adjacent, Right turn, Blind spot', hasSimulationSpec: true, region: 'US-East' },
  { id: '6', scenarioName: 'Sensor Degradation Rain', clusterId: 0, priorityScore: 94, definingConditions: 'Heavy rain, Highway speed, Sensor noise spike', hasSimulationSpec: true, region: 'US-West' },
  { id: '7', scenarioName: 'School Zone Transition', clusterId: 2, priorityScore: 81, definingConditions: 'Speed limit change, Children present, Crosswalk', hasSimulationSpec: false, region: 'EU-West' },
]

export const agentOutputs = {
  proposer: `[PROPOSER AGENT] Initializing scene analysis...
Loading scene context: scene-0-15
Detected scenario type: Highway Merge with Cut-in Event

Analyzing frame sequence 0-240...
- Lead vehicle distance: 45m → 12m (rapid closure)
- Adjacent vehicle speed delta: +15 km/h
- Ego vehicle response latency: 340ms

Proposing adversarial modifications:
1. Reduce initial gap to 35m
2. Increase closure rate by 20%
3. Add sensor occlusion from motorcycle
4. Introduce rain degradation to LiDAR

Confidence: 0.87
Proposed priority: HIGH`,

  critic: `[CRITIC AGENT] Reviewing proposer analysis...

Validating scenario modifications:
✓ Gap reduction: Within physical bounds
✓ Closure rate: Plausible vehicle dynamics
✗ LiDAR degradation: Intensity may exceed realistic bounds

Counter-arguments:
- Original scenario already captures core risk
- Motorcycle occlusion redundant with existing blind spot
- Rain model assumes uniform degradation (unrealistic)

Suggested refinements:
1. Use spatially-varying rain model
2. Remove motorcycle occlusion (low impact)
3. Add wind gust perturbation instead

Risk assessment: Scenario valid but over-specified
Recommendation: ACCEPT with modifications`,

  judge: `[JUDGE AGENT] Synthesizing analysis...

Evaluating proposer-critic debate:
- Proposer identified valid coverage gap
- Critic raised legitimate physics concerns
- Compromise reached on rain modeling

Final determination:
This scenario represents a genuine edge case in the
current test coverage. The highway merge with rapid
closure event occurs in 0.3% of training data but
accounts for 12% of disengagement events.

VERDICT: PRIORITY SCENARIO
Priority Score: 92/100

Simulation specification approved with critic modifications.
Generating ASAM OpenSCENARIO 2.0 output...`,
}
