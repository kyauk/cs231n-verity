'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  createRound,
  exportRound,
  getNextWindow,
  getRoundStatus,
  getVideoUrl,
  listRounds,
  submitRating,
} from '@/lib/dev-api'
import type {
  CreateRoundResponse,
  ExportResponse,
  NextWindowResponse,
  RoundListEntry,
  RoundStatus,
} from '@/lib/dev-types'
import { Download, FlaskConical, Play, Plus } from 'lucide-react'

/**
 * Dev Dashboard — Discrimination Test tab.
 *
 * Stages:
 *   (a) Setup: pick existing round or create a new one (upload scored.json
 *       + schema_records.json, set dataset_label, pool_size, seed).
 *   (b) Rate: one window at a time, video + two 1–5 sliders.
 *   (c) Complete: download the export JSON for offline Mann-Whitney.
 *
 * Blinding: the rater never sees which pool a window came from. Source
 * labels are only revealed in the export endpoint.
 */
export function DevDiscriminationTab() {
  const [rounds, setRounds] = useState<RoundListEntry[]>([])
  const [activeRoundId, setActiveRoundId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function refreshRounds() {
    try {
      setRounds(await listRounds())
    } catch (e) {
      setError(String(e))
    }
  }

  useEffect(() => {
    refreshRounds()
  }, [])

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-semibold flex items-center gap-2">
            <FlaskConical className="w-6 h-6" />
            Discrimination Test
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            One round = three pools of 30 windows (Verity / Random /
            Naive-rare), blind-shuffled. You rate each on safety-relevance
            and rarity. Export reveals source pools for offline
            Mann-Whitney analysis.
          </p>
        </div>

        {error && (
          <div className="text-sm text-red-500 whitespace-pre-wrap">{error}</div>
        )}

        {!activeRoundId ? (
          <SetupView
            rounds={rounds}
            onPickRound={setActiveRoundId}
            onCreate={async (r) => {
              setActiveRoundId(r.round_id)
              await refreshRounds()
            }}
            onError={setError}
          />
        ) : (
          <RatingView
            roundId={activeRoundId}
            onExit={() => {
              setActiveRoundId(null)
              refreshRounds()
            }}
            onError={setError}
          />
        )}
      </div>
    </div>
  )
}

// ---------- Setup ----------

function SetupView({
  rounds,
  onPickRound,
  onCreate,
  onError,
}: {
  rounds: RoundListEntry[]
  onPickRound: (rid: string) => void
  onCreate: (r: CreateRoundResponse) => void
  onError: (msg: string) => void
}) {
  const [datasetLabel, setDatasetLabel] = useState('')
  const [poolSize, setPoolSize] = useState(30)
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1_000_000))
  const [scoredFile, setScoredFile] = useState<File | null>(null)
  const [recordsFile, setRecordsFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleCreate() {
    if (!scoredFile || !recordsFile || !datasetLabel) {
      onError('Provide all three inputs: dataset label, scored.json, schema_records.json.')
      return
    }
    setBusy(true)
    try {
      const scored = JSON.parse(await scoredFile.text())
      const records = JSON.parse(await recordsFile.text())
      const r = await createRound({
        dataset_label: datasetLabel,
        pool_size: poolSize,
        seed,
        top_k_rare_atoms: 5,
        scored,
        schema_records: records,
      })
      onCreate(r)
    } catch (e) {
      onError(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {/* Existing rounds */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-3">Existing rounds</h3>
        {rounds.length === 0 ? (
          <p className="text-sm text-muted-foreground">No rounds yet.</p>
        ) : (
          <ul className="space-y-1">
            {rounds.map((r) => (
              <li key={r.round_id}>
                <button
                  onClick={() => onPickRound(r.round_id)}
                  className="w-full text-left px-3 py-2 rounded hover:bg-muted text-sm flex items-center justify-between"
                >
                  <span className="font-mono text-xs">{r.round_id}</span>
                  <span className="text-muted-foreground">
                    {r.dataset_label} · {r.created_at.slice(0, 16)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {/* Create new round */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Create a new round
        </h3>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Dataset label</span>
            <input
              type="text"
              value={datasetLabel}
              onChange={(e) => setDatasetLabel(e.target.value)}
              placeholder="waymo_val_split_1"
              className="border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Pool size</span>
            <input
              type="number"
              value={poolSize}
              onChange={(e) => setPoolSize(parseInt(e.target.value) || 30)}
              min={1}
              className="border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Seed</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
              className="border rounded px-2 py-1 text-sm"
            />
          </label>
          <div />
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">scored.json</span>
            <input
              type="file"
              accept="application/json,.json"
              onChange={(e) => setScoredFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">schema_records.json</span>
            <input
              type="file"
              accept="application/json,.json"
              onChange={(e) => setRecordsFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
          </label>
        </div>
        <Button
          onClick={handleCreate}
          disabled={busy}
          className="mt-3"
        >
          <Play className="w-4 h-4 mr-2" />
          {busy ? 'Creating…' : 'Create round'}
        </Button>
      </Card>
    </>
  )
}

// ---------- Rating ----------

function RatingView({
  roundId,
  onExit,
  onError,
}: {
  roundId: string
  onExit: () => void
  onError: (msg: string) => void
}) {
  const [status, setStatus] = useState<RoundStatus | null>(null)
  const [next, setNext] = useState<NextWindowResponse | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [raterId, setRaterId] = useState(() => {
    if (typeof window === 'undefined') return ''
    return window.localStorage.getItem('verity_dev_rater_id') ?? ''
  })
  const [safety, setSafety] = useState(3)
  const [rarity, setRarity] = useState(3)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [exportData, setExportData] = useState<ExportResponse | null>(null)

  async function refresh() {
    try {
      const [s, n] = await Promise.all([
        getRoundStatus(roundId),
        getNextWindow(roundId),
      ])
      setStatus(s)
      setNext(n)
      if (n.window && !n.complete) {
        try {
          const v = await getVideoUrl(roundId, n.window.segment_id, n.window.window_idx)
          setVideoUrl(v.url)
        } catch (e) {
          // Video URL is optional — bucket may not be configured
          setVideoUrl(null)
          console.warn('video URL not available:', e)
        }
      } else {
        setVideoUrl(null)
      }
    } catch (e) {
      onError(String(e))
    }
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roundId])

  useEffect(() => {
    if (typeof window !== 'undefined' && raterId) {
      window.localStorage.setItem('verity_dev_rater_id', raterId)
    }
  }, [raterId])

  async function handleSubmit() {
    if (!next?.window || !raterId) {
      onError('Set a rater ID before submitting.')
      return
    }
    setBusy(true)
    try {
      await submitRating(roundId, {
        rater_id: raterId,
        window: next.window,
        safety_relevance: safety,
        perceived_rarity: rarity,
        free_text_note: note || null,
      })
      setSafety(3)
      setRarity(3)
      setNote('')
      await refresh()
    } catch (e) {
      onError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleExport() {
    try {
      const data = await exportRound(roundId)
      setExportData(data)
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${roundId}-export.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      onError(String(e))
    }
  }

  if (!status || !next) {
    return <p className="text-sm text-muted-foreground">Loading round…</p>
  }

  return (
    <>
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground font-mono">{roundId}</p>
            <p className="text-sm">
              {status.dataset_label} · pool_size {status.pool_size}
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm tabular-nums">
              {status.rated_count} / {status.total_windows} rated
            </p>
            {status.complete && <Badge>Complete</Badge>}
          </div>
        </div>
        <div className="flex gap-2 mt-3">
          <Button variant="outline" onClick={onExit}>
            ← Back to rounds
          </Button>
          <Button onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />
            Download export JSON
          </Button>
        </div>
      </Card>

      {next.complete ? (
        <Card className="p-6">
          <h3 className="text-lg font-semibold">Round complete</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Download the export JSON above. The source-pool label for each
            rating is included so you can compute per-pool means + run
            Mann-Whitney offline.
          </p>
          {exportData && (
            <p className="text-xs text-muted-foreground mt-2">
              Naive-rare atoms used: {exportData.naive_rare_atoms.join(', ')}
            </p>
          )}
        </Card>
      ) : (
        <Card className="p-4 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm">
              Window {next.progress_idx} / {next.total_windows}
            </p>
            <code className="text-xs text-muted-foreground">
              {next.window?.segment_id}/
              {String(next.window?.window_idx).padStart(4, '0')}
            </code>
          </div>

          {videoUrl ? (
            <video
              key={videoUrl}
              src={videoUrl}
              controls
              autoPlay
              className="w-full rounded bg-black aspect-video"
            />
          ) : (
            <div className="w-full aspect-video rounded bg-muted flex items-center justify-center text-sm text-muted-foreground">
              Video URL unavailable (set DEV_DASHBOARD_BUCKET_URI on the server)
            </div>
          )}

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Rater ID</span>
            <input
              type="text"
              value={raterId}
              onChange={(e) => setRaterId(e.target.value)}
              placeholder="your_name"
              className="border rounded px-2 py-1 text-sm"
            />
          </label>

          <SliderField
            label="Safety relevance"
            help="Would you want your AV stack tested against this?"
            value={safety}
            onChange={setSafety}
          />
          <SliderField
            label="Perceived rarity"
            help="How unusual does this feel?"
            value={rarity}
            onChange={setRarity}
          />

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Note (optional)</span>
            <input
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
            />
          </label>

          <Button onClick={handleSubmit} disabled={busy || !raterId}>
            {busy ? 'Submitting…' : 'Submit rating'}
          </Button>
        </Card>
      )}
    </>
  )
}

function SliderField({
  label,
  help,
  value,
  onChange,
}: {
  label: string
  help: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm tabular-nums">{value} / 5</span>
      </div>
      <p className="text-xs text-muted-foreground mb-2">{help}</p>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((v) => (
          <button
            key={v}
            onClick={() => onChange(v)}
            className={`flex-1 py-2 rounded text-sm border ${
              v === value
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-background hover:bg-muted'
            }`}
          >
            {v}
          </button>
        ))}
      </div>
    </div>
  )
}
