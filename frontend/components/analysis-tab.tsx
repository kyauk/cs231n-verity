'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Play, CheckCircle2, Circle, Loader2, Sparkles, Gavel, MessageSquareWarning, MousePointerClick } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Scene, AgentStep, AnalysisResult } from '@/lib/types'
import { runAnalysisStream, ApiError } from '@/lib/api'

interface AnalysisTabProps {
  scene: Scene | null
}

// Maps a pipeline progress `step` to the index of the agent it belongs to.
function stepToAgentIndex(step: string): number {
  if (step.startsWith('debate_judge')) return 2
  if (step.startsWith('debate')) return 0 // proponent/critic rounds light up step 1+
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
  const terminalRefs = useRef<(HTMLDivElement | null)[]>([])

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
    setSteps(resetSteps())

    try {
      const result = await runAnalysisStream(scene.id, (progress) => {
        // Progress events keep the stepper alive while the pipeline runs.
        const agentIdx = stepToAgentIndex(progress.step)
        setSteps(prev =>
          prev.map((s, idx) => {
            if (progress.step === 'debate_judge' && idx === 2) {
              return { ...s, status: 'running', output: `${s.output}${progress.detail}\n` }
            }
            if (agentIdx === 0 && progress.step.startsWith('debate_round')) {
              // Proponent + critic share rounds; light steps 0 and 1.
              const target = progress.detail.toLowerCase().includes('critic') ? 1 : 0
              if (idx === target) {
                return { ...s, status: 'running', output: `${s.output}${progress.detail}\n` }
              }
            }
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

      setConclusion({
        sceneId: result.conclusion.sceneId,
        verdict: result.conclusion.verdict,
        priorityScore: result.conclusion.priorityScore,
        simulationSpec: result.conclusion.simulationSpec,
      })
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : 'Analysis failed to run.'
      setErrorMessage(message)
      setSteps(resetSteps())
    } finally {
      setIsRunning(false)
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Agentic Analysis</h2>
          <p className="text-sm text-muted-foreground">Multi-agent debate for adversarial scenario generation</p>
        </div>
        <Button onClick={runAnalysis} disabled={isRunning}>
          <Play className="w-4 h-4 mr-2" />
          Generate Analysis
        </Button>
      </div>

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
                <div
                  ref={el => { terminalRefs.current[index] = el }}
                  className={cn(
                    'rounded-lg p-4 font-mono text-sm h-40 overflow-auto',
                    step.status === 'pending' && 'opacity-50'
                  )}
                  style={{ backgroundColor: 'oklch(0.15 0.02 240)' }}
                >
                  {step.output ? (
                    <pre className="whitespace-pre-wrap" style={{ color: 'oklch(0.65 0.2 125)' }}>
                      {step.output}
                      {step.status === 'running' && (
                        <span className="inline-block w-2 h-4 bg-terminal-text ml-0.5 animate-blink" />
                      )}
                    </pre>
                  ) : (
                    <p className="text-muted-foreground/50 italic">Waiting for previous step...</p>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Error */}
      {errorMessage && (
        <Card className="border-destructive/30 bg-destructive/5">
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
