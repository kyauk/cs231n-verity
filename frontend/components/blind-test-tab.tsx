'use client'

import { useEffect, useMemo, useState } from 'react'

// A blinded, quiz-style rating flow. Reads the eval feed (/eval_feed.json),
// shows ONE scene description at a time with its source hidden, and collects a
// THREE-way rating designed to see Verity's contribution (which lives in a
// quadrant flat "usefulness" averages away):
//   severity      — how bad if it happened
//   coverage      — how valuable to ADD to a test suite vs what's already tested
//   thinkability  — would this come up in a standard brainstorm (LOW = nobody
//                   would enumerate it = the thing Verity should win by losing)
// At the end it reveals per-source averages for all three.

type Item = { id: string; arm: string; description: string }
type Score = { id: string; arm: string; severity: number; coverage: number; thinkability: number }

const ARM_LABEL: Record<string, string> = {
  verity: 'Verity (ours)',
  ungrounded_llm: 'Ungrounded LLM',
  compositional_rarity: 'Compositional Rarity',
}

const QUESTIONS: { key: keyof Omit<Score, 'id' | 'arm'>; label: string; hint: string }[] = [
  { key: 'severity', label: 'Severity', hint: '1 = minor · 5 = catastrophic if it happened' },
  { key: 'coverage', label: 'Coverage value', hint: '1 = redundant/already tested · 5 = fills a real gap' },
  { key: 'thinkability', label: 'Thinkability', hint: '1 = no one would brainstorm it · 5 = obvious, everyone lists it' },
]

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function DescriptionCard({ text }: { text: string }) {
  const lines = text.split('\n').filter(Boolean)
  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        const idx = line.indexOf(':')
        if (idx > 0 && idx < 16) {
          return (
            <p key={i} className="text-sm leading-relaxed">
              <span className="font-semibold text-foreground">{line.slice(0, idx)}:</span>
              <span className="text-muted-foreground">{line.slice(idx + 1)}</span>
            </p>
          )
        }
        return <p key={i} className="text-sm text-muted-foreground leading-relaxed">{line}</p>
      })}
    </div>
  )
}

function Likert({
  label, hint, value, onChange,
}: { label: string; hint: string; value: number | null; onChange: (n: number) => void }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">{hint}</span>
      </div>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map(n => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className={`w-11 h-11 rounded-md border text-base font-semibold transition-colors
              ${value === n
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-background text-foreground border-border hover:bg-muted'}`}
          >
            {n}
          </button>
        ))}
      </div>
    </div>
  )
}

export function BlindTestTab() {
  const [items, setItems] = useState<Item[] | null>(null)
  const [rater, setRater] = useState('')
  const [started, setStarted] = useState(false)
  const [idx, setIdx] = useState(0)
  const [scores, setScores] = useState<Score[]>([])
  const [cur, setCur] = useState<{ severity: number | null; coverage: number | null; thinkability: number | null }>(
    { severity: null, coverage: null, thinkability: null })
  const [done, setDone] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch('/eval_feed.json')
      .then(r => r.json())
      .then((d: Item[]) => setItems(shuffle(d)))
      .catch(e => setErr(String(e)))
  }, [])

  const summary = useMemo(() => {
    const by: Record<string, { s: number; c: number; t: number; n: number }> = {}
    for (const x of scores) {
      by[x.arm] = by[x.arm] || { s: 0, c: 0, t: 0, n: 0 }
      by[x.arm].s += x.severity; by[x.arm].c += x.coverage; by[x.arm].t += x.thinkability; by[x.arm].n += 1
    }
    return Object.entries(by)
      .map(([arm, v]) => ({ arm, severity: v.s / v.n, coverage: v.c / v.n, thinkability: v.t / v.n, n: v.n }))
      .sort((a, b) => b.coverage - a.coverage)
  }, [scores])

  if (err) return <div className="p-8 text-sm text-red-500">Failed to load eval feed: {err}</div>
  if (!items) return <div className="p-8 text-sm text-muted-foreground">Loading blind test…</div>

  if (!started) {
    return (
      <div className="max-w-xl mx-auto p-8 space-y-4">
        <h2 className="text-xl font-semibold text-foreground">Blind Scenario Rating</h2>
        <p className="text-sm text-muted-foreground">
          {items.length} scenario descriptions, one at a time, source hidden. Rate each on three
          axes. At the end you&apos;ll get per-source averages to record.
        </p>
        <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-5">
          <li><b>Severity</b> — how bad if it happened.</li>
          <li><b>Coverage value</b> — how valuable to add to a test suite vs what&apos;s already tested.</li>
          <li><b>Thinkability</b> — would this come up in a normal brainstorm? (low = nobody would list it).</li>
        </ul>
        <input
          value={rater}
          onChange={e => setRater(e.target.value)}
          placeholder="Your name (for your notes)"
          className="w-full px-3 py-2 rounded-md border border-border bg-background text-sm"
        />
        <button onClick={() => setStarted(true)}
          className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium">
          Start ({items.length} scenarios)
        </button>
      </div>
    )
  }

  if (done) {
    const copy = () => {
      const rows = summary.map(s =>
        `${ARM_LABEL[s.arm] ?? s.arm}\tseverity ${s.severity.toFixed(2)}\tcoverage ${s.coverage.toFixed(2)}\tthinkability ${s.thinkability.toFixed(2)}\tn=${s.n}`)
      navigator.clipboard?.writeText(`Rater: ${rater || '(unnamed)'}\n` + rows.join('\n'))
    }
    return (
      <div className="max-w-2xl mx-auto p-8 space-y-6">
        <h2 className="text-xl font-semibold text-foreground">Results{rater ? ` — ${rater}` : ''}</h2>
        <p className="text-xs text-muted-foreground">
          Pre-registered prediction: Verity wins on <b>coverage</b> and <i>loses</i> on <b>thinkability</b>
          (low thinkability = nobody would have enumerated it = the point).
        </p>
        <table className="w-full text-sm border border-border rounded-md overflow-hidden">
          <thead className="bg-muted">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Source</th>
              <th className="text-right px-3 py-2 font-medium">Severity</th>
              <th className="text-right px-3 py-2 font-medium">Coverage</th>
              <th className="text-right px-3 py-2 font-medium">Thinkability</th>
              <th className="text-right px-3 py-2 font-medium">n</th>
            </tr>
          </thead>
          <tbody>
            {summary.map(s => (
              <tr key={s.arm} className="border-t border-border">
                <td className="px-3 py-2 text-foreground">{ARM_LABEL[s.arm] ?? s.arm}</td>
                <td className="px-3 py-2 text-right tabular-nums">{s.severity.toFixed(2)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{s.coverage.toFixed(2)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{s.thinkability.toFixed(2)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{s.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button onClick={copy} className="px-4 py-2 rounded-md border border-border text-sm hover:bg-muted">
          Copy results
        </button>
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Per-scenario breakdown ({scores.length})</summary>
          <table className="w-full mt-2 border border-border">
            <thead className="bg-muted"><tr>
              <th className="text-left px-2 py-1">id</th><th className="text-left px-2 py-1">source</th>
              <th className="px-2 py-1">sev</th><th className="px-2 py-1">cov</th><th className="px-2 py-1">think</th>
            </tr></thead>
            <tbody>
              {scores.map(s => (
                <tr key={s.id} className="border-t border-border">
                  <td className="px-2 py-1 font-mono">{s.id}</td>
                  <td className="px-2 py-1">{ARM_LABEL[s.arm] ?? s.arm}</td>
                  <td className="px-2 py-1 text-center">{s.severity}</td>
                  <td className="px-2 py-1 text-center">{s.coverage}</td>
                  <td className="px-2 py-1 text-center">{s.thinkability}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </div>
    )
  }

  const item = items[idx]
  const ready = cur.severity != null && cur.coverage != null && cur.thinkability != null
  const submit = () => {
    if (!ready) return
    setScores(s => [...s, { id: item.id, arm: item.arm, severity: cur.severity!, coverage: cur.coverage!, thinkability: cur.thinkability! }])
    setCur({ severity: null, coverage: null, thinkability: null })
    if (idx + 1 >= items.length) setDone(true)
    else setIdx(idx + 1)
  }

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Scenario {idx + 1} of {items.length}</span>
        <div className="h-1.5 w-40 bg-muted rounded-full overflow-hidden">
          <div className="h-full bg-primary" style={{ width: `${(idx / items.length) * 100}%` }} />
        </div>
      </div>
      <div className="rounded-lg border border-border p-5 bg-card">
        <DescriptionCard text={item.description} />
      </div>
      <div className="space-y-4">
        {QUESTIONS.map(q => (
          <Likert key={q.key} label={q.label} hint={q.hint}
            value={cur[q.key]} onChange={n => setCur(c => ({ ...c, [q.key]: n }))} />
        ))}
      </div>
      <button onClick={submit} disabled={!ready}
        className="w-full py-2.5 rounded-md bg-primary text-primary-foreground text-sm font-medium
          disabled:opacity-40 disabled:cursor-not-allowed">
        {idx + 1 >= items.length ? 'Finish & see scores' : 'Next'}
      </button>
    </div>
  )
}
