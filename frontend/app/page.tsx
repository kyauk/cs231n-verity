'use client'

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Database, ScatterChart, Brain, LayoutDashboard } from 'lucide-react'
import { IngestTab } from '@/components/ingest-tab'
import { ClusterSpaceTab } from '@/components/cluster-space-tab'
import { AnalysisTab } from '@/components/analysis-tab'
import { DashboardTab } from '@/components/dashboard-tab'
import { 
  mockBatchJobs, 
  mockClusterPoints, 
  mockClusterStats, 
  mockScene,
  mockFlaggedScenarios,
  agentOutputs 
} from '@/lib/mock-data'

export default function Home() {
  const [activeTab, setActiveTab] = useState('ingest')

  const handleLaunchBatch = (dataSourceUri: string, label: string, region: string) => {
    // In a real app, this would trigger the batch job
    console.log('Launching batch:', { dataSourceUri, label, region })
  }

  const handleViewClusterSpace = (batchId: string) => {
    setActiveTab('cluster')
  }

  const handleAnalyzeScene = (sceneId: string) => {
    setActiveTab('analysis')
  }

  const handleViewScenario = (scenarioId: string) => {
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
            <h1 className="text-lg font-semibold text-foreground">Adversarial Environment Generator</h1>
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
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="ingest" className="h-full m-0 data-[state=inactive]:hidden">
            <IngestTab 
              batchJobs={mockBatchJobs}
              onLaunchBatch={handleLaunchBatch}
              onViewClusterSpace={handleViewClusterSpace}
            />
          </TabsContent>

          <TabsContent value="cluster" className="h-full m-0 data-[state=inactive]:hidden">
            <ClusterSpaceTab 
              points={mockClusterPoints}
              clusterStats={mockClusterStats}
              scene={mockScene}
              onAnalyzeScene={handleAnalyzeScene}
            />
          </TabsContent>

          <TabsContent value="analysis" className="h-full m-0 data-[state=inactive]:hidden">
            <AnalysisTab 
              scene={mockScene}
              agentOutputs={agentOutputs}
            />
          </TabsContent>

          <TabsContent value="dashboard" className="h-full m-0 data-[state=inactive]:hidden">
            <DashboardTab 
              scenarios={mockFlaggedScenarios}
              onViewScenario={handleViewScenario}
            />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}
