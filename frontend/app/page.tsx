'use client'

import { useState, useEffect, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Brain,
  ClipboardCheck,
  Database,
  FlaskConical,
  Gavel,
  LayoutDashboard,
  ScatterChart,
} from 'lucide-react'
import { IngestTab } from '@/components/ingest-tab'
import { ClusterSpaceTab } from '@/components/cluster-space-tab'
import { AnalysisTab } from '@/components/analysis-tab'
import { DashboardTab } from '@/components/dashboard-tab'
import { JudgeTab } from '@/components/judge-tab'
import { DevAccuracyTab } from '@/components/dev-accuracy-tab'
import { DevDiscriminationTab } from '@/components/dev-discrimination-tab'

// Dev-dashboard tabs are conditionally rendered. They only appear when the
// build was given NEXT_PUBLIC_DEV_DASHBOARD_URL — keeps them off customer
// deployments. The backend additionally refuses to start without
// VERITY_DEV_MODE=1 on the server side.
const DEV_DASHBOARD_ENABLED = !!process.env.NEXT_PUBLIC_DEV_DASHBOARD_URL
import {
  fetchBatchJobs,
  launchBatch,
  fetchClusterSpace,
  fetchScenarios,
} from '@/lib/api'
import type {
  BatchJob,
  ClusterPoint,
  ClusterStats,
  FlaggedScenario,
  Scene,
} from '@/lib/types'

export default function Home() {
  const [activeTab, setActiveTab] = useState('ingest')

  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([])
  const [clusterPoints, setClusterPoints] = useState<ClusterPoint[]>([])
  const [clusterStats, setClusterStats] = useState<ClusterStats[]>([])
  const [scenarios, setScenarios] = useState<FlaggedScenario[]>([])
  const [analysisScene, setAnalysisScene] = useState<Scene | null>(null)

  const loadBatchJobs = useCallback(async () => {
    try {
      setBatchJobs(await fetchBatchJobs())
    } catch {
      /* keep last-known data — runner offline */
    }
  }, [])

  const loadClusterSpace = useCallback(async () => {
    try {
      const { points, clusterStats: stats } = await fetchClusterSpace()
      setClusterPoints(points)
      setClusterStats(stats)
    } catch {
      /* runner offline — keep last-known state */
    }
  }, [])

  const loadScenarios = useCallback(async () => {
    try {
      setScenarios(await fetchScenarios())
    } catch {
      /* runner offline — keep last-known state */
    }
  }, [])

  // Initial load.
  useEffect(() => {
    loadBatchJobs()
    loadClusterSpace()
    loadScenarios()
  }, [loadBatchJobs, loadClusterSpace, loadScenarios])

  // Poll batch jobs while any are running (drives the running/completed state).
  useEffect(() => {
    const hasRunning = batchJobs.some((job) => job.status === 'running')
    if (!hasRunning) return
    const interval = setInterval(() => {
      loadBatchJobs()
      loadClusterSpace()
      loadScenarios()
    }, 5000)
    return () => clearInterval(interval)
  }, [batchJobs, loadBatchJobs, loadClusterSpace, loadScenarios])

  const handleLaunchBatch = async (
    dataSourceUri: string,
    label: string,
    region: string,
    maxSegments: number,
    mode: 'cluster' | 'reason' | 'both',
  ) => {
    // Let errors propagate — IngestTab catches and displays them inline.
    const created = await launchBatch(dataSourceUri, label, region, maxSegments, mode)
    // Show the new batch immediately rather than waiting on the refetch, which
    // can momentarily lag the just-written record. loadBatchJobs then reconciles,
    // and the running-state poll (below) picks up subsequent status changes.
    setBatchJobs((prev) => [created, ...prev.filter((b) => b.id !== created.id)])
    await loadBatchJobs()
  }

  const handleViewClusterSpace = (_batchId: string) => {
    loadClusterSpace()
    setActiveTab('cluster')
  }

  const handleAnalyzeScene = (scene: Scene) => {
    setAnalysisScene(scene)
    setActiveTab('analysis')
  }

  const handleViewScenario = (_scenarioId: string) => {
    setActiveTab('analysis')
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b bg-card px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">AV</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-foreground">Verity Platform</h1>
            <p className="text-xs text-muted-foreground">Autonomous Vehicle Safety Validation</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          System Online
        </div>
      </header>

      {/* Main Content with Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
        <div className="border-b bg-card">
          <TabsList className="h-12 w-full justify-start rounded-none border-none bg-transparent px-6 gap-1">
            <TabsTrigger
              value="ingest"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
            >
              <Database className="w-4 h-4 mr-2" />
              Ingest
            </TabsTrigger>
            <TabsTrigger
              value="cluster"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
            >
              <ScatterChart className="w-4 h-4 mr-2" />
              Cluster Space
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
            >
              <Brain className="w-4 h-4 mr-2" />
              Analysis
            </TabsTrigger>
            <TabsTrigger
              value="dashboard"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
            >
              <LayoutDashboard className="w-4 h-4 mr-2" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger
              value="judge"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
            >
              <Gavel className="w-4 h-4 mr-2" />
              Judge
            </TabsTrigger>
            {DEV_DASHBOARD_ENABLED && (
              <>
                <TabsTrigger
                  value="dev-accuracy"
                  className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
                >
                  <ClipboardCheck className="w-4 h-4 mr-2" />
                  Dev · Accuracy
                </TabsTrigger>
                <TabsTrigger
                  value="dev-discrimination"
                  className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4"
                >
                  <FlaskConical className="w-4 h-4 mr-2" />
                  Dev · Discrimination
                </TabsTrigger>
              </>
            )}
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="ingest" className="h-full m-0 data-[state=inactive]:hidden">
            <IngestTab
              batchJobs={batchJobs}
              onLaunchBatch={handleLaunchBatch}
              onViewClusterSpace={handleViewClusterSpace}
            />
          </TabsContent>

          <TabsContent value="cluster" className="h-full m-0 data-[state=inactive]:hidden">
            <ClusterSpaceTab
              points={clusterPoints}
              clusterStats={clusterStats}
              onAnalyzeScene={handleAnalyzeScene}
            />
          </TabsContent>

          <TabsContent value="analysis" forceMount className="h-full m-0 data-[state=inactive]:hidden">
            <AnalysisTab
              scene={analysisScene}
            />
          </TabsContent>

          <TabsContent value="dashboard" className="h-full m-0 data-[state=inactive]:hidden">
            <DashboardTab
              scenarios={scenarios}
              onViewScenario={handleViewScenario}
            />
          </TabsContent>

          <TabsContent value="judge" className="h-full m-0 data-[state=inactive]:hidden">
            <JudgeTab />
          </TabsContent>

          {DEV_DASHBOARD_ENABLED && (
            <>
              <TabsContent
                value="dev-accuracy"
                className="h-full m-0 data-[state=inactive]:hidden"
              >
                <DevAccuracyTab />
              </TabsContent>
              <TabsContent
                value="dev-discrimination"
                className="h-full m-0 data-[state=inactive]:hidden"
              >
                <DevDiscriminationTab />
              </TabsContent>
            </>
          )}
        </div>
      </Tabs>
    </div>
  )
}
