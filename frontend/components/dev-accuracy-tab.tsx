'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getAccuracyTemplate, postAccuracyDiff } from '@/lib/dev-api'
import type { AccuracyReport, FieldDiff } from '@/lib/dev-types'
import { Copy, FileUp, CheckCircle2, XCircle } from 'lucide-react'

/**
 * Dev Dashboard — VLM Accuracy tab.
 *
 * Two file uploads, one button: a hand-labeled gold-set JSON + the
 * schema_records.json from a `pipeline.run analyze` run. The server
 * computes per-field diff + per-field aggregates (match counts) and the
 * UI renders them. No statistics in-UI — operator hand-aggregates from
 * the visible totals (or downloads the raw report).
 */
export function DevAccuracyTab() {
  const [goldFile, setGoldFile] = useState<File | null>(null)
  const [recordsFile, setRecordsFile] = useState<File | null>(null)
  const [report, setReport] = useState<AccuracyReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  async function handleCopyTemplate() {
    try {
      const tpl = await getAccuracyTemplate()
      await navigator.clipboard.writeText(JSON.stringify(tpl, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (e) {
      setError(String(e))
    }
  }

  async function handleRunDiff() {
    if (!goldFile || !recordsFile) {
      setError('Upload both files first.')
      return
    }
    setError(null)
    setBusy(true)
    try {
      const goldText = await goldFile.text()
      const recordsText = await recordsFile.text()
      const gold = JSON.parse(goldText)
      const records = JSON.parse(recordsText)
      if (!Array.isArray(records)) {
        throw new Error('schema_records.json must be a JSON array.')
      }
      const r = await postAccuracyDiff(gold, records)
      setReport(r)
    } catch (e) {
      setError(String(e))
      setReport(null)
    } finally {
      setBusy(false)
    }
  }

  function downloadReport() {
    if (!report) return
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'accuracy-diff.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-semibold">VLM Accuracy</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Upload a hand-labeled gold set and the encoder&apos;s{' '}
            <code>schema_records.json</code>. The server computes per-field
            match counts. No statistics are computed here — note the
            aggregates below, or download the raw report.
          </p>
        </div>

        {/* Uploads + actions */}
        <Card className="p-4">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium">Gold-set JSON</span>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => setGoldFile(e.target.files?.[0] ?? null)}
                className="text-sm"
              />
              {goldFile && (
                <span className="text-xs text-muted-foreground">
                  {goldFile.name} ({Math.round(goldFile.size / 1024)} KB)
                </span>
              )}
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium">
                schema_records.json
              </span>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => setRecordsFile(e.target.files?.[0] ?? null)}
                className="text-sm"
              />
              {recordsFile && (
                <span className="text-xs text-muted-foreground">
                  {recordsFile.name} ({Math.round(recordsFile.size / 1024)} KB)
                </span>
              )}
            </label>
          </div>

          <div className="flex gap-2 mt-4">
            <Button onClick={handleRunDiff} disabled={busy || !goldFile || !recordsFile}>
              <FileUp className="w-4 h-4 mr-2" />
              {busy ? 'Computing…' : 'Run Diff'}
            </Button>
            <Button variant="outline" onClick={handleCopyTemplate}>
              <Copy className="w-4 h-4 mr-2" />
              {copied ? 'Copied!' : 'Copy Gold-set Template'}
            </Button>
            {report && (
              <Button variant="outline" onClick={downloadReport}>
                Download Report JSON
              </Button>
            )}
          </div>

          {error && (
            <div className="mt-3 text-sm text-red-500 whitespace-pre-wrap">
              {error}
            </div>
          )}
        </Card>

        {report && <ReportView report={report} />}
      </div>
    </div>
  )
}

function ReportView({ report }: { report: AccuracyReport }) {
  return (
    <>
      {/* Per-field aggregates */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-3">Per-field match counts</h3>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(report.field_aggregates).map(([path, [m, t]]) => {
            const rate = t > 0 ? (m / t) * 100 : 0
            return (
              <div
                key={path}
                className="flex items-center justify-between border rounded px-3 py-2"
              >
                <code className="text-xs">{path}</code>
                <span className="text-sm tabular-nums">
                  {m}/{t}
                  {t > 0 && (
                    <span className="text-muted-foreground ml-2">
                      ({rate.toFixed(0)}%)
                    </span>
                  )}
                </span>
              </div>
            )
          })}
        </div>
      </Card>

      {/* Missing entries */}
      {report.missing_entries.length > 0 && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold mb-3">
            Missing entries ({report.missing_entries.length})
          </h3>
          <ul className="text-sm space-y-1">
            {report.missing_entries.slice(0, 20).map((m, i) => (
              <li key={i} className="flex gap-2">
                <Badge variant="outline">{m.direction}</Badge>
                <code className="text-xs">
                  {m.window_id.segment_id}/
                  {String(m.window_id.window_idx).padStart(4, '0')}
                </code>
              </li>
            ))}
            {report.missing_entries.length > 20 && (
              <li className="text-xs text-muted-foreground">
                … and {report.missing_entries.length - 20} more
              </li>
            )}
          </ul>
        </Card>
      )}

      {/* Per-window diff */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-3">
          Per-window diff ({report.windows.length} windows)
        </h3>
        <div className="space-y-3 max-h-[600px] overflow-y-auto">
          {report.windows.map((w, i) => (
            <WindowDiffRow key={i} window={w} />
          ))}
        </div>
      </Card>
    </>
  )
}

function WindowDiffRow({
  window,
}: {
  window: { window_id: { segment_id: string; window_idx: number }; fields: FieldDiff[] }
}) {
  const matches = window.fields.filter((f) => f.match).length
  const total = window.fields.length
  return (
    <div className="border rounded p-3">
      <div className="flex items-center justify-between mb-2">
        <code className="text-xs font-medium">
          {window.window_id.segment_id}/
          {String(window.window_id.window_idx).padStart(4, '0')}
        </code>
        <span className="text-xs tabular-nums">
          {matches}/{total} fields match
        </span>
      </div>
      <div className="grid gap-1 text-xs">
        {window.fields.map((f, j) => (
          <div
            key={j}
            className={`flex items-center gap-2 px-2 py-1 rounded ${
              f.match ? 'bg-green-500/10' : 'bg-red-500/10'
            }`}
          >
            {f.match ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
            ) : (
              <XCircle className="w-3.5 h-3.5 text-red-500 shrink-0" />
            )}
            <code className="shrink-0 w-44">{f.field_path}</code>
            <span className="text-muted-foreground shrink-0">gold:</span>
            <code className="truncate">{JSON.stringify(f.gold)}</code>
            <span className="text-muted-foreground shrink-0">vlm:</span>
            <code className="truncate">{JSON.stringify(f.vlm)}</code>
            {f.f1 !== null && f.f1 !== undefined && (
              <span className="text-muted-foreground ml-auto tabular-nums">
                F1 {f.f1.toFixed(2)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
