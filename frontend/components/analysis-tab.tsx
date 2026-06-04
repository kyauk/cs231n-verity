'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Play, CheckCircle2, Circle, Loader2, Sparkles, Gavel, MessageSquareWarning, MousePointerClick, History, X, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Scene, AgentStep, AnalysisResult, AnalysisHistoryEntry } from '@/lib/types'
import { runAnalysisStream, ApiError } from '@/lib/api'

const HISTORY_KEY = (sceneId: string) => `verity:analysis-history:${sceneId}`

function loadHistory(sceneId: string): AnalysisHistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY(sceneId))
    return raw ? (JSON.parse(raw) as AnalysisHistoryEntry[]) : []
  } catch {
    return []
  }
}

function saveHistory(sceneId: string, entry: AnalysisHistoryEntry) {
  const existing = loadHistory(sceneId)
  localStorage.setItem(HISTORY_KEY(sceneId), JSON.stringify([entry, ...existing].slice(0, 20)))
}

interface AnalysisTabProps {
  scene: Scene | null
}

// Map a progress event to one of the three agent panels by keyword. The
// backend has multiple debate implementations (proponent/critic/judge AND the
// tool-augmented Scene Analyst / Coverage Analyst / Synthesis Arbiter), so match
// on the step+title text rather than exact step names that drift between them.
function progressToAgentIndex(step: string, title: string): number {
  const t = `${step} ${title}`.toLowerCase()
  if (t.includes('judge') || t.includes('arbiter') || t.includes('synthesis') || t.includes('verdict')) return 2
  if (t.includes('critic') || t.includes('coverage') || t.includes('risk')) return 1
  if (t.includes('proponent') || t.includes('analyst') || t.includes('describ')) return 0
  return -1
}

const AGENT_ICONS = {
  proposer: Sparkles,
  critic: MessageSquareWarning,
  judge: Gavel,
}

export function AnalysisTab({ scene }: AnalysisTabProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [steps, setSteps] = useState<AgentStep[]>([
    { id: 'proposer', name: 'Proposer Agent', status: 'pending', output: '' },
    { id: 'critic', name: 'Critic Agent', status: 'pending', output: '' },
    { id: 'judge', name: 'Judge Agent', status: 'pending', output: '' },
  ])
  const [conclusion, setConclusion] = useState<AnalysisResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [liveStatus, setLiveStatus] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [history, setHistory] = useState<AnalysisHistoryEntry[]>([])
  const [expandedHistory, setExpandedHistory] = useState<string | null>(null)
  const terminalRefs = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    if (scene) setHistory(loadHistory(scene.id))
  }, [scene])

  const resetSteps = (): AgentStep[] => [
    { id: 'proposer', name: 'Proposer Agent', status: 'pending', output: '' },
    { id: 'critic', name: 'Critic Agent', status: 'pending', output: '' },
    { id: 'judge', name: 'Judge Agent', status: 'pending', output: '' },
  ]

  const runAnalysis = async () => {
    if (!scene) return
    setIsRunning(true)
    setConclusion(null)
    setErrorMessage(null)
    setLiveStatus('Starting analysis…')
    setSteps(resetSteps())

    try {
      const result = await runAnalysisStream(scene.id, (progress) => {
        // Always surface the latest activity so it's obvious work is happening.
        setLiveStatus(
          progress.detail ? `${progress.title} — ${progress.detail}` : progress.title,
        )
        // Light up the matching agent panel; mark earlier ones complete.
        const agentIdx = progressToAgentIndex(progress.step, progress.title)
        if (agentIdx < 0) return
        setSteps(prev =>
          prev.map((s, idx) => {
            if (idx < agentIdx && s.status !== 'complete') return { ...s, status: 'complete' }
            if (idx === agentIdx) return { ...s, status: 'running', output: `${s.output}${progress.detail}\n` }
            return s
          }),
        )
      })

      // Populate each agent terminal with its real transcript.
      setSteps([
        { id: 'proposer', name: 'Proposer Agent', status: 'complete', output: result.agentOutputs.proposer },
        { id: 'critic', name: 'Critic Agent', status: 'complete', output: result.agentOutputs.critic },
        { id: 'judge', name: 'Judge Agent', status: 'complete', output: result.agentOutputs.judge },
      ])

      const conclusionData = {
        sceneId: result.conclusion.sceneId,
        verdict: result.conclusion.verdict,
        priorityScore: result.conclusion.priorityScore,
        simulationSpec: result.conclusion.simulationSpec,
      }
      setConclusion(conclusionData)

      const entry: AnalysisHistoryEntry = {
        id: crypto.randomUUID(),
        sceneId: scene.id,
        ranAt: new Date().toISOString(),
        verdict: conclusionData.verdict,
        priorityScore: conclusionData.priorityScore,
        simulationSpec: conclusionData.simulationSpec,
        agentOutputs: result.agentOutputs,
      }
      saveHistory(scene.id, entry)
      setHistory(loadHistory(scene.id))
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : 'Analysis failed to run.'
      setErrorMessage(message)
      setSteps(resetSteps())
    } finally {
      setIsRunning(false)
      setLiveStatus(null)
    }
  }

  // Auto-scroll terminals
  useEffect(() => {
    steps.forEach((step, i) => {
      if (step.status === 'running' && terminalRefs.current[i]) {
        terminalRefs.current[i]!.scrollTop = terminalRefs.current[i]!.scrollHeight
      }
    })
  }, [steps])

  const getStatusIcon = (status: AgentStep['status']) => {
    switch (status) {
      case 'pending':
        return <Circle className="w-5 h-5 text-muted-foreground" />
      case 'running':
        return <Loader2 className="w-5 h-5 text-primary animate-spin" />
      case 'complete':
        return <CheckCircle2 className="w-5 h-5 text-primary" />
    }
  }

  if (!scene) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-12">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
          <MousePointerClick className="w-8 h-8 text-muted-foreground" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">No Scene Selected</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Go to Cluster Space, click a point, and choose &ldquo;Analyze this scene&rdquo; to begin.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
      {/* Error — pinned to the top so it's seen immediately */}
      {errorMessage && (
        <Card className="sticky top-0 z-10 border-destructive/30 bg-destructive/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium flex items-center gap-2 text-destructive">
              <MessageSquareWarning className="w-5 h-5" />
              Analysis Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{errorMessage}</p>
          </CardContent>
        </Card>
      )}

      {/* Live status — shows the current step so it's clear the agents are working */}
      {isRunning && liveStatus && (
        <Card className="sticky top-0 z-10 border-primary/30 bg-primary/5">
          <CardContent className="py-3 flex items-center gap-3">
            <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0" />
            <p className="text-sm text-foreground truncate">{liveStatus}</p>
          </CardContent>
        </Card>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Agentic Analysis</h2>
          <p className="text-sm text-muted-foreground">Multi-agent debate for adversarial scenario generation</p>
        </div>
        <div className="flex items-center gap-2">
          {history.length > 0 && (
            <Button variant="outline" onClick={() => setHistoryOpen(v => !v)}>
              <History className="w-4 h-4 mr-2" />
              History ({history.length})
            </Button>
          )}
          <Button onClick={runAnalysis} disabled={isRunning}>
            {isRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
            {isRunning ? 'Running...' : 'Generate Analysis'}
          </Button>
        </div>
      </div>

      {/* History Panel */}
      {historyOpen && (
        <Card className="border-muted">
          <CardHeader className="pb-2 pt-4 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <History className="w-4 h-4 text-muted-foreground" />
                Past Analyses — {scene.id}
              </CardTitle>
              <button onClick={() => setHistoryOpen(false)}>
                <X className="w-4 h-4 text-muted-foreground hover:text-foreground" />
              </button>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-4 space-y-2">
            {history.map(entry => (
              <div key={entry.id} className="rounded-lg border bg-muted/20 overflow-hidden">
                <button
                  className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-muted/40 transition-colors"
                  onClick={() => setExpandedHistory(expandedHistory === entry.id ? null : entry.id)}
                >
                  <div className="flex items-center gap-3">
                    <Badge className="bg-primary/10 text-primary border-primary/20 font-mono text-xs">
                      {entry.priorityScore}/100
                    </Badge>
                    <span className="text-sm font-medium">{entry.verdict}</span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(entry.ranAt).toLocaleString()}
                    </span>
                  </div>
                  {expandedHistory === entry.id
                    ? <ChevronUp className="w-4 h-4 text-muted-foreground shrink-0" />
                    : <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />}
                </button>
                {expandedHistory === entry.id && (
                  <div className="px-3 pb-3 space-y-2 border-t pt-2">
                    <p className="text-xs text-muted-foreground">{entry.simulationSpec}</p>
                    {(['proposer', 'critic', 'judge'] as const).map(agent => (
                      entry.agentOutputs[agent] && (
                        <div key={agent}>
                          <p className="text-xs font-medium capitalize text-muted-foreground mb-1">{agent}</p>
                          <div className="rounded p-2 font-mono text-xs overflow-auto max-h-32"
                            style={{ backgroundColor: 'oklch(0.15 0.02 240)', color: 'oklch(0.65 0.2 125)' }}>
                            <pre className="whitespace-pre-wrap">{entry.agentOutputs[agent]}</pre>
                          </div>
                        </div>
                      )
                    ))}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Scene Preview */}
      <Card>
        <CardContent className="p-4">
          <div className="flex gap-4">
            <div className="w-40 h-24 bg-muted rounded-lg flex items-center justify-center">
              <Play className="w-8 h-8 text-muted-foreground" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-medium text-foreground">Selected Scene</h3>
                <Badge variant="outline" className="font-mono text-xs">{scene.id}</Badge>
              </div>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Weather: </span>
                  <span className="font-medium">{scene.annotations.weather}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Time: </span>
                  <span className="font-medium">{scene.annotations.timeOfDay}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Road: </span>
                  <span className="font-medium">{scene.annotations.roadType}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Events: </span>
                  <span className="font-medium">{scene.annotations.events.length}</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Agent Stepper */}
      <div className="flex-1 space-y-4">
        {steps.map((step, index) => {
          const Icon = AGENT_ICONS[step.id as keyof typeof AGENT_ICONS]
          return (
            <Card 
              key={step.id}
              className={cn(
                'transition-all',
                step.status === 'running' && 'ring-2 ring-primary/20'
              )}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      'w-10 h-10 rounded-full flex items-center justify-center',
                      step.status === 'complete' ? 'bg-primary/10' : 'bg-muted'
                    )}>
                      <Icon className={cn(
                        'w-5 h-5',
                        step.status === 'complete' ? 'text-primary' : 'text-muted-foreground'
                      )} />
                    </div>
                    <div>
                      <CardTitle className="text-base font-medium flex items-center gap-2">
                        {step.name}
                        {step.status === 'running' && (
                          <span className="text-xs text-primary animate-pulse-subtle">Running...</span>
                        )}
                      </CardTitle>
                      <p className="text-xs text-muted-foreground">
                        Step {index + 1} of 3
                      </p>
                    </div>
                  </div>
                  {getStatusIcon(step.status)}
                </div>
              </CardHeader>
              <CardContent>
                {step.status === 'pending' ? (
                  <div className="rounded-lg border border-dashed border-muted-foreground/20 h-40 flex items-center justify-center">
                    <p className="text-xs text-muted-foreground/40 italic">Waiting for previous step...</p>
                  </div>
                ) : (
                  <div
                    ref={el => { terminalRefs.current[index] = el }}
                    className="rounded-lg p-4 font-mono text-sm h-40 overflow-auto"
                    style={{ backgroundColor: 'oklch(0.15 0.02 240)' }}
                  >
                    <pre className="whitespace-pre-wrap" style={{ color: 'oklch(0.65 0.2 125)' }}>
                      {step.output}
                      {step.status === 'running' && (
                        <span className="inline-block w-2 h-4 bg-terminal-text ml-0.5 animate-blink" />
                      )}
                    </pre>
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Conclusion */}
      {conclusion && (
        <Card className="border-primary/30 bg-primary/5">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-primary" />
                Analysis Complete
              </CardTitle>
              <Badge className="bg-primary text-primary-foreground text-lg px-3">
                {conclusion.priorityScore}/100
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <span className="text-sm text-muted-foreground">Verdict: </span>
                <span className="font-semibold text-primary">{conclusion.verdict}</span>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Simulation Specification: </span>
                <p className="text-sm mt-1">{conclusion.simulationSpec}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
