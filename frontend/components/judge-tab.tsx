'use client'

import { useState, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  CheckCircle2,
  ChevronLeft,
  Play,
  Loader2,
  Gavel,
  AlertCircle,
  RefreshCw,
  BarChart2,
  Eye,
  ClipboardList,
  TrendingUp,
  ListChecks,
  Clock,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  fetchJudgeProposals,
  fetchJudgeProposalDetail,
  fetchJudgeVideoUrl,
  submitJudgeRating,
  fetchJudgeSession,
} from '@/lib/api'
import type {
  JudgeProposalRow,
  JudgeProposalDetail,
  JudgeMotivatingScene,
  JudgeSessionSummary,
} from '@/lib/types'

type Screen = 'setup' | 'list' | 'detail' | 'summary'

interface VideoState {
  url: string | null
  generatedAt: string | null
  loading: boolean
  error: string | null
  attempts: number
}

function sceneKey(s: JudgeMotivatingScene): string {
  return `${s.segment_id}/${s.window_idx}`
}

// ---------------------------------------------------------------------------
// Score badges — using design tokens throughout
// ---------------------------------------------------------------------------
function ScoreBadgeRow({ scores }: { scores: JudgeProposalRow['scores'] }) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      <Badge variant="outline" className="text-xs font-mono">
        N {scores.novelty_score.toFixed(2)}
      </Badge>
      <Badge
        variant="secondary"
        className="text-xs font-mono bg-primary/10 text-primary border-primary/20"
      >
        P {Math.round(scores.plausibility_score * 100)}%
      </Badge>
      {scores.frontier_difficulty_score !== null && (
        <Badge variant="outline" className="text-xs font-mono bg-amber-100 text-amber-700 border-amber-200">
          D {Math.round(scores.frontier_difficulty_score * 100)}%
        </Badge>
      )}
      <Badge variant="outline" className="text-xs font-mono bg-blue-50 text-blue-600 border-blue-200">
        R {scores.final_rank_score.toFixed(2)}
      </Badge>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 1–5 score selector — styled like the existing button patterns
// ---------------------------------------------------------------------------
function ScoreSelector({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint: string
  value: number | null
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-2">
      <div>
        <Label className="text-sm font-medium">{label}</Label>
        <p className="text-xs text-muted-foreground mt-0.5">{hint}</p>
      </div>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className={cn(
              'w-10 h-10 rounded-lg border text-sm font-medium transition-colors',
              value === n
                ? 'bg-primary text-primary-foreground border-primary'
                : 'border-border text-muted-foreground hover:border-primary/50 hover:text-foreground bg-card',
            )}
          >
            {n}
          </button>
        ))}
        <span className="flex items-center text-xs text-muted-foreground ml-1">
          {value === null ? 'select' : value === 1 ? 'poor' : value === 2 ? 'fair' : value === 3 ? 'ok' : value === 4 ? 'good' : 'excellent'}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function JudgeTab() {
  const [screen, setScreen] = useState<Screen>('setup')
  const [raterId, setRaterId] = useState('')
  const [raterInput, setRaterInput] = useState('')

  const [proposals, setProposals] = useState<JudgeProposalRow[]>([])
  const [ratedIds, setRatedIds] = useState<Set<string>>(new Set())
  const [session, setSession] = useState<JudgeSessionSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selectedProposal, setSelectedProposal] = useState<JudgeProposalDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const [expandedScene, setExpandedScene] = useState<string | null>(null)
  const [seenScenes, setSeenScenes] = useState<JudgeMotivatingScene[]>([])
  const videoStates = useRef<Record<string, VideoState>>({})
  const [, setVideoVersion] = useState(0)

  const [coherenceScore, setCoherenceScore] = useState<number | null>(null)
  const [usefulnessScore, setUsefulnessScore] = useState<number | null>(null)
  const [freeText, setFreeText] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // -------------------------------------------------------------------------
  // Data loading
  // -------------------------------------------------------------------------

  const loadSession = useCallback(async (id: string) => {
    setLoading(true)
    setError(null)
    try {
      const [propsData, sessionData] = await Promise.all([
        fetchJudgeProposals(),
        fetchJudgeSession(id),
      ])
      setProposals(propsData)
      setRatedIds(new Set(sessionData.rated_proposal_ids))
      setSession(sessionData)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load proposals')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleStartSession = async () => {
    const id = raterInput.trim()
    if (!id) return
    setRaterId(id)
    await loadSession(id)
    setScreen('list')
  }

  const handleReview = async (proposalId: string) => {
    setDetailLoading(true)
    setSelectedProposal(null)
    setExpandedScene(null)
    setSeenScenes([])
    setCoherenceScore(null)
    setUsefulnessScore(null)
    setFreeText('')
    setSubmitError(null)
    try {
      const detail = await fetchJudgeProposalDetail(proposalId)
      setSelectedProposal(detail)
      setScreen('detail')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load proposal')
    } finally {
      setDetailLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Video URL management
  // -------------------------------------------------------------------------

  const loadVideoUrl = useCallback(async (segment_id: string, window_idx: number, camera = 'FRONT') => {
    const key = `${segment_id}/${window_idx}/${camera}`
    const current = videoStates.current[key]
    const attempts = current?.attempts ?? 0
    if (attempts >= 2) return

    videoStates.current[key] = { url: null, generatedAt: null, loading: true, error: null, attempts }
    setVideoVersion((v) => v + 1)

    try {
      const { url, generated_at } = await fetchJudgeVideoUrl(segment_id, window_idx, camera)
      videoStates.current[key] = { url, generatedAt: generated_at, loading: false, error: null, attempts }
    } catch (e) {
      videoStates.current[key] = {
        url: null, generatedAt: null, loading: false,
        error: attempts >= 1 ? 'Video unavailable. Try refreshing.' : (e instanceof Error ? e.message : 'Video unavailable'),
        attempts: attempts + 1,
      }
    }
    setVideoVersion((v) => v + 1)
  }, [])

  const handleVideoError = useCallback((segment_id: string, window_idx: number, camera = 'FRONT') => {
    const key = `${segment_id}/${window_idx}/${camera}`
    const current = videoStates.current[key]
    if (!current || current.attempts >= 2) return
    videoStates.current[key] = { ...current, attempts: current.attempts + 1, url: null, loading: true }
    setVideoVersion((v) => v + 1)
    loadVideoUrl(segment_id, window_idx, camera)
  }, [loadVideoUrl])

  const handleExpandScene = (scene: JudgeMotivatingScene) => {
    const key = sceneKey(scene)
    if (expandedScene === key) { setExpandedScene(null); return }
    setExpandedScene(key)
    if (!seenScenes.find((s) => sceneKey(s) === key)) setSeenScenes((prev) => [...prev, scene])
    const videoKey = `${scene.segment_id}/${scene.window_idx}/FRONT`
    if (!videoStates.current[videoKey]?.url) loadVideoUrl(scene.segment_id, scene.window_idx)
  }

  // -------------------------------------------------------------------------
  // Rating submission
  // -------------------------------------------------------------------------

  const handleSubmitRating = async () => {
    if (!selectedProposal || coherenceScore === null || usefulnessScore === null) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      await submitJudgeRating({
        rater_id: raterId,
        proposal_id: selectedProposal.proposal_id,
        coherence_score: coherenceScore,
        usefulness_score: usefulnessScore,
        free_text_note: freeText.trim() || null,
        seen_motivating_scenes: seenScenes,
      })
      const newRatedIds = new Set(ratedIds)
      newRatedIds.add(selectedProposal.proposal_id)
      setRatedIds(newRatedIds)
      const sessionData = await fetchJudgeSession(raterId)
      setSession(sessionData)
      if (newRatedIds.size >= proposals.length) {
        setScreen('summary')
      } else {
        setScreen('list')
      }
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'Failed to submit rating')
    } finally {
      setSubmitting(false)
    }
  }

  // -------------------------------------------------------------------------
  // Screen: Setup
  // -------------------------------------------------------------------------
  if (screen === 'setup') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-12">
        <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
          <Gavel className="w-8 h-8 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Judge UI — Proposal Rating</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Enter your rater ID to start or resume your session. Arm identity is hidden during rating.
          </p>
        </div>
        <Card className="w-full max-w-sm text-left mt-2">
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-2">
              <Label htmlFor="rater-id">Rater ID</Label>
              <Input
                id="rater-id"
                placeholder="e.g. alice"
                value={raterInput}
                onChange={(e) => setRaterInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleStartSession() }}
                autoFocus
              />
            </div>
            {error && (
              <div className="flex items-start gap-2 text-destructive text-xs">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <Button
              className="w-full"
              onClick={handleStartSession}
              disabled={!raterInput.trim() || loading}
            >
              {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Start Session
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Screen: Proposal List
  // -------------------------------------------------------------------------
  if (screen === 'list') {
    const unratedCount = proposals.filter((p) => !ratedIds.has(p.proposal_id)).length
    const progressPct = proposals.length > 0
      ? Math.round((ratedIds.size / proposals.length) * 100)
      : 0

    return (
      <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-foreground">Proposal Queue</h2>
            <p className="text-sm text-muted-foreground">
              Rater: <span className="font-medium text-foreground">{raterId}</span>
              {' · '}arm identity blinded during rating
            </p>
          </div>
          {ratedIds.size > 0 && (
            <Button variant="outline" size="sm" onClick={() => setScreen('summary')}>
              <BarChart2 className="w-4 h-4 mr-2" />
              View Summary
            </Button>
          )}
        </div>

        {error && (
          <div className="flex items-start gap-2 text-destructive text-xs">
            <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Metric cards */}
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <ClipboardList className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{proposals.length}</p>
                  <p className="text-sm text-muted-foreground">Total Proposals</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <CheckCircle2 className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{ratedIds.size}</p>
                  <p className="text-sm text-muted-foreground">Rated</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-amber-100 flex items-center justify-center">
                  <Clock className="w-6 h-6 text-amber-600" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{unratedCount}</p>
                  <p className="text-sm text-muted-foreground">Remaining</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-blue-100 flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{progressPct}%</p>
                  <p className="text-sm text-muted-foreground">Progress</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Proposals table */}
        <Card className="flex-1 min-h-0 flex flex-col">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium">Proposals</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 p-0">
            <div className="overflow-auto h-full">
              {proposals.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-40 gap-2 text-center">
                  <ListChecks className="w-8 h-8 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    No proposals available. Run Module 4 (Scorer) to generate proposals.
                  </p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50">
                      <TableHead className="w-8">#</TableHead>
                      <TableHead>Composition</TableHead>
                      <TableHead className="w-[280px]">Scores</TableHead>
                      <TableHead className="w-24 text-center">Scenes</TableHead>
                      <TableHead className="w-28 text-right">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {proposals.map((p, idx) => {
                      const rated = ratedIds.has(p.proposal_id)
                      return (
                        <TableRow key={p.proposal_id}>
                          <TableCell className="text-muted-foreground text-xs font-mono">
                            {idx + 1}
                          </TableCell>
                          <TableCell>
                            <p className="font-medium text-sm">
                              {p.constituents.map((c) => c.split(':')[1] ?? c).join(' + ')}
                            </p>
                            <p className="text-xs text-muted-foreground font-mono truncate max-w-[220px]">
                              {p.constituents.join(', ')}
                            </p>
                          </TableCell>
                          <TableCell>
                            <ScoreBadgeRow scores={p.scores} />
                          </TableCell>
                          <TableCell className="text-center">
                            <span className="text-sm text-muted-foreground">
                              {p.motivating_scene_count > 0 ? p.motivating_scene_count : '—'}
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            {rated ? (
                              <Badge className="bg-primary/10 text-primary border-primary/20">
                                <CheckCircle2 className="w-3 h-3 mr-1" />
                                Rated
                              </Badge>
                            ) : (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleReview(p.proposal_id)}
                                disabled={detailLoading}
                                className="text-primary hover:text-primary"
                              >
                                {detailLoading
                                  ? <Loader2 className="w-3 h-3 animate-spin" />
                                  : <>Review <Eye className="w-3 h-3 ml-1" /></>
                                }
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Screen: Proposal Detail + Rating Widget
  // -------------------------------------------------------------------------
  if (screen === 'detail' && selectedProposal) {
    return (
      <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
        {/* Back nav + title */}
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => setScreen('list')} className="-ml-2">
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back to Queue
          </Button>
          <div className="h-4 w-px bg-border" />
          <div>
            <h2 className="text-xl font-semibold text-foreground">
              {selectedProposal.constituents.map((c) => c.split(':')[1] ?? c).join(' + ')}
            </h2>
            <p className="text-sm text-muted-foreground">Review and rate this composition</p>
          </div>
        </div>

        {/* Composition */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium">Composition</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {selectedProposal.constituents.map((c) => (
                <Badge key={c} variant="secondary" className="text-xs font-mono">
                  {c}
                </Badge>
              ))}
            </div>
            <ScoreBadgeRow scores={selectedProposal.scores} />
            <div className="bg-muted/50 rounded-lg px-4 py-3 text-sm text-muted-foreground italic">
              "{selectedProposal.plausibility_justification}"
            </div>
          </CardContent>
        </Card>

        {/* Motivating scenes */}
        {selectedProposal.motivating_scenes.length > 0 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <Eye className="w-4 h-4 text-muted-foreground" />
                Motivating Scenes
                <Badge variant="outline" className="ml-1 font-mono text-xs">
                  {selectedProposal.motivating_scenes.length}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {selectedProposal.motivating_scenes.map((scene) => {
                const key = sceneKey(scene)
                const videoKey = `${scene.segment_id}/${scene.window_idx}/FRONT`
                const vState = videoStates.current[videoKey]
                const isExpanded = expandedScene === key
                const isSeen = seenScenes.some((s) => sceneKey(s) === key)

                return (
                  <div key={key} className="border rounded-lg overflow-hidden">
                    <button
                      className="w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/40 transition-colors text-left"
                      onClick={() => handleExpandScene(scene)}
                    >
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <Play className="w-3.5 h-3.5 text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground font-mono">
                          {scene.segment_id} / {String(scene.window_idx).padStart(4, '0')}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {isExpanded ? 'Click to collapse' : 'Click to play'}
                        </p>
                      </div>
                      {isSeen && (
                        <Badge className="bg-primary/10 text-primary border-primary/20 text-xs shrink-0">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Watched
                        </Badge>
                      )}
                    </button>

                    {isExpanded && (
                      <div className="border-t bg-black">
                        {vState?.loading && (
                          <div className="flex items-center justify-center h-36">
                            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                          </div>
                        )}
                        {!vState?.loading && vState?.error && (
                          <div className="flex flex-col items-center justify-center h-36 gap-2">
                            <AlertCircle className="w-5 h-5 text-destructive" />
                            <p className="text-xs text-muted-foreground">{vState.error}</p>
                            {vState.attempts < 2 && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => loadVideoUrl(scene.segment_id, scene.window_idx)}
                              >
                                <RefreshCw className="w-3 h-3 mr-1" /> Retry
                              </Button>
                            )}
                          </div>
                        )}
                        {vState?.url && !vState.loading && (
                          <video
                            key={vState.url}
                            src={vState.url}
                            controls
                            autoPlay
                            className="w-full max-h-64 object-contain"
                            onError={() => handleVideoError(scene.segment_id, scene.window_idx)}
                          />
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </CardContent>
          </Card>
        )}

        {/* Rating widget */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              <Gavel className="w-4 h-4 text-muted-foreground" />
              Rate This Proposal
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <ScoreSelector
              label="Coherence"
              hint="Does this make physical and behavioral sense as a scenario?"
              value={coherenceScore}
              onChange={setCoherenceScore}
            />
            <ScoreSelector
              label="Usefulness"
              hint="Would this composition expose model failures in testing?"
              value={usefulnessScore}
              onChange={setUsefulnessScore}
            />
            <div className="space-y-2">
              <Label className="text-sm font-medium">Notes</Label>
              <p className="text-xs text-muted-foreground">Optional observations about this composition</p>
              <Textarea
                placeholder="Any observations about this composition..."
                value={freeText}
                onChange={(e) => setFreeText(e.target.value)}
                className="resize-none"
                rows={3}
              />
            </div>
            {submitError && (
              <div className="flex items-start gap-2 text-destructive text-xs">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>{submitError}</span>
              </div>
            )}
            <Button
              className="w-full sm:w-auto"
              disabled={coherenceScore === null || usefulnessScore === null || submitting}
              onClick={handleSubmitRating}
            >
              {submitting
                ? <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                : <Gavel className="w-4 h-4 mr-2" />
              }
              Submit Rating
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Screen: Session Summary
  // -------------------------------------------------------------------------
  if (screen === 'summary' && session) {
    const allRated = ratedIds.size >= proposals.length
    const coh = session.coherence_distribution
    const use = session.usefulness_distribution
    const cohTotal = Object.values(coh).reduce((a, b) => a + b, 0)
    const useTotal = Object.values(use).reduce((a, b) => a + b, 0)

    return (
      <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-foreground">
              {allRated ? 'Session Complete' : 'Session Summary'}
            </h2>
            <p className="text-sm text-muted-foreground">
              {allRated
                ? `All ${proposals.length} proposals rated — thank you!`
                : `${ratedIds.size} of ${session.total_accepted} proposals rated so far`}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => setScreen('list')}>
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back to Queue
          </Button>
        </div>

        {/* Metric cards */}
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <CheckCircle2 className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{ratedIds.size}</p>
                  <p className="text-sm text-muted-foreground">Proposals Rated</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <BarChart2 className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">
                    {session.mean_coherence !== null ? session.mean_coherence.toFixed(1) : '—'}
                  </p>
                  <p className="text-sm text-muted-foreground">Mean Coherence</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-blue-100 flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">
                    {session.mean_usefulness !== null ? session.mean_usefulness.toFixed(1) : '—'}
                  </p>
                  <p className="text-sm text-muted-foreground">Mean Usefulness</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className={cn(
                  'w-12 h-12 rounded-lg flex items-center justify-center',
                  allRated ? 'bg-primary/10' : 'bg-amber-100',
                )}>
                  <ClipboardList className={cn('w-6 h-6', allRated ? 'text-primary' : 'text-amber-600')} />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">
                    {proposals.length > 0
                      ? Math.round((ratedIds.size / proposals.length) * 100)
                      : 0}%
                  </p>
                  <p className="text-sm text-muted-foreground">Complete</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Distribution tables */}
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: 'Coherence Distribution', dist: coh, total: cohTotal },
            { label: 'Usefulness Distribution', dist: use, total: useTotal },
          ].map(({ label, dist, total }) => (
            <Card key={label}>
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-medium">{label}</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50">
                      <TableHead>Score</TableHead>
                      <TableHead className="text-right">Count</TableHead>
                      <TableHead className="text-right">Share</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[1, 2, 3, 4, 5].map((score) => {
                      const count = dist[score] ?? 0
                      const pct = total > 0 ? Math.round((count / total) * 100) : 0
                      return (
                        <TableRow key={score}>
                          <TableCell className="font-medium">{score}</TableCell>
                          <TableCell className="text-right font-mono">{count}</TableCell>
                          <TableCell className="text-right">
                            {count > 0 ? (
                              <Badge className="bg-primary/10 text-primary border-primary/20 font-mono text-xs">
                                {pct}%
                              </Badge>
                            ) : (
                              <span className="text-muted-foreground text-xs">—</span>
                            )}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center h-full">
      <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
    </div>
  )
}
