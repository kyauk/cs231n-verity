'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Play, CheckCircle2, Circle, Loader2, Sparkles, Gavel, MessageSquareWarning } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Scene, AgentStep, AnalysisResult } from '@/lib/types'

interface AnalysisTabProps {
  scene: Scene
  agentOutputs: {
    proposer: string
    critic: string
    judge: string
  }
}

const AGENT_ICONS = {
  proposer: Sparkles,
  critic: MessageSquareWarning,
  judge: Gavel,
}

export function AnalysisTab({ scene, agentOutputs }: AnalysisTabProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [steps, setSteps] = useState<AgentStep[]>([
    { id: 'proposer', name: 'Proposer Agent', status: 'pending', output: '' },
    { id: 'critic', name: 'Critic Agent', status: 'pending', output: '' },
    { id: 'judge', name: 'Judge Agent', status: 'pending', output: '' },
  ])
  const [conclusion, setConclusion] = useState<AnalysisResult | null>(null)
  const terminalRefs = useRef<(HTMLDivElement | null)[]>([])

  // Simulate streaming text
  const streamText = (text: string, onUpdate: (partial: string) => void, onComplete: () => void) => {
    let index = 0
    const interval = setInterval(() => {
      if (index < text.length) {
        const chunkSize = Math.floor(Math.random() * 3) + 1
        index = Math.min(index + chunkSize, text.length)
        onUpdate(text.slice(0, index))
      } else {
        clearInterval(interval)
        onComplete()
      }
    }, 15)
    return () => clearInterval(interval)
  }

  const runAnalysis = async () => {
    setIsRunning(true)
    setConclusion(null)
    setSteps([
      { id: 'proposer', name: 'Proposer Agent', status: 'pending', output: '' },
      { id: 'critic', name: 'Critic Agent', status: 'pending', output: '' },
      { id: 'judge', name: 'Judge Agent', status: 'pending', output: '' },
    ])

    const outputs = ['proposer', 'critic', 'judge'] as const

    for (let i = 0; i < outputs.length; i++) {
      const agentId = outputs[i]
      
      // Set current step to running
      setSteps(prev => prev.map((s, idx) => 
        idx === i ? { ...s, status: 'running' } : s
      ))

      await new Promise<void>((resolve) => {
        streamText(
          agentOutputs[agentId],
          (partial) => {
            setSteps(prev => prev.map((s, idx) => 
              idx === i ? { ...s, output: partial } : s
            ))
          },
          () => {
            setSteps(prev => prev.map((s, idx) => 
              idx === i ? { ...s, status: 'complete' } : s
            ))
            resolve()
          }
        )
      })

      // Small delay between agents
      await new Promise(r => setTimeout(r, 500))
    }

    // Show conclusion
    setConclusion({
      sceneId: scene.id,
      verdict: 'PRIORITY SCENARIO',
      priorityScore: 92,
      simulationSpec: 'Highway merge with aggressive cut-in and sensor degradation. Generated OpenSCENARIO 2.0 specification with modified rain model and wind gust perturbations.',
    })

    setIsRunning(false)
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
                    'bg-terminal-bg rounded-lg p-4 font-mono text-sm h-40 overflow-auto',
                    step.status === 'pending' && 'opacity-50'
                  )}
                >
                  {step.output ? (
                    <pre className="text-terminal-text whitespace-pre-wrap">
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
