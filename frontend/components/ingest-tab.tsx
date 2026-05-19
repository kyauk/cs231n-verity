'use client'

import { useState } from 'react'
import { Rocket, FolderOpen, CheckCircle2, XCircle, Loader2, ExternalLink, ChevronDown, ChevronUp, Terminal, Info, AlertCircle, ScanSearch } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import type { BatchJob } from '@/lib/types'
import { probePath } from '@/lib/api'

interface IngestTabProps {
  batchJobs: BatchJob[]
  onLaunchBatch: (dataSourceUri: string, label: string, region: string, maxSegments: number) => Promise<void>
  onViewClusterSpace: (batchId: string) => void
}

const REGIONS = [
  { value: 'US-West', label: 'US West (Phoenix, SF, LA)' },
  { value: 'US-East', label: 'US East (Boston, NYC, Miami)' },
  { value: 'EU-West', label: 'EU West (Munich, London, Paris)' },
  { value: 'APAC', label: 'APAC (Tokyo, Singapore)' },
]

const DATASET_EXAMPLES = [
  {
    name: 'Waymo Open Dataset',
    uri: 'gs://waymo_open_dataset_v_2_0_1/validation/camera_image',
    auth: 'gcloud auth application-default login',
    note: 'Requires Waymo dataset access approval at waymo.com/open',
  },
  {
    name: 'nuScenes (GCS mirror)',
    uri: 'gs://your-bucket/nuscenes/v1.0-trainval/samples',
    auth: 'gcloud auth application-default login',
    note: 'Point at your org\'s GCS mirror of the nuScenes sample directory',
  },
  {
    name: 'Custom GCS Dataset',
    uri: 'gs://your-bucket/path/to/parquet-files',
    auth: 'gcloud auth application-default login',
    note: 'Any GCS path containing Parquet scene files',
  },
]

export function IngestTab({ batchJobs, onLaunchBatch, onViewClusterSpace }: IngestTabProps) {
  const [dataSourceUri, setDataSourceUri] = useState('')
  const [batchLabel, setBatchLabel] = useState('')
  const [region, setRegion] = useState('')
  const [setupOpen, setSetupOpen] = useState(false)
  const [pathError, setPathError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)
  const [probing, setProbing] = useState(false)
  const [segmentCount, setSegmentCount] = useState<number | null>(null)
  const [maxSegments, setMaxSegments] = useState<number | 'all'>(5)

  const isValidGcsUri = (uri: string) => /^gs:\/\/[^/]+\/.+/.test(uri.trim())
  const canLaunch = isValidGcsUri(dataSourceUri) && batchLabel.trim() !== '' && region !== '' && !launching && !probing && !pathError

  const handleUriChange = (val: string) => {
    setDataSourceUri(val)
    setPathError(null)
    setSegmentCount(null)
    setMaxSegments(5)
  }

  const handleUriBlur = async () => {
    if (!isValidGcsUri(dataSourceUri)) return
    setProbing(true)
    setPathError(null)
    setSegmentCount(null)
    try {
      const result = await probePath(dataSourceUri)
      if (result.valid) {
        setSegmentCount(result.segmentCount)
        setMaxSegments(result.segmentCount)
      } else {
        setPathError(result.detail)
      }
    } catch {
      setPathError('Could not reach backend to validate path.')
    } finally {
      setProbing(false)
    }
  }

  const handleLaunch = async () => {
    if (!canLaunch) return
    setLaunching(true)
    setPathError(null)
    try {
      await onLaunchBatch(dataSourceUri, batchLabel, region, maxSegments === 'all' ? 0 : maxSegments)
      setDataSourceUri('')
      setBatchLabel('')
      setRegion('')
      setMaxSegments(5)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to launch batch.'
      setPathError(msg)
    } finally {
      setLaunching(false)
    }
  }

  const getStatusBadge = (status: BatchJob['status']) => {
    switch (status) {
      case 'running':
        return (
          <Badge variant="secondary" className="bg-amber-100 text-amber-700">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Running
          </Badge>
        )
      case 'completed':
        return (
          <Badge className="bg-primary/10 text-primary border-primary/20">
            <CheckCircle2 className="w-3 h-3 mr-1" />
            Completed
          </Badge>
        )
      case 'failed':
        return (
          <Badge variant="destructive">
            <XCircle className="w-3 h-3 mr-1" />
            Failed
          </Badge>
        )
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    const month = months[date.getUTCMonth()]
    const day = date.getUTCDate()
    const hours = date.getUTCHours().toString().padStart(2, '0')
    const minutes = date.getUTCMinutes().toString().padStart(2, '0')
    return `${month} ${day}, ${hours}:${minutes} UTC`
  }

  const formatSceneCount = (job: BatchJob) => {
    if (job.status === 'running') {
      return `${job.scenesProcessed.toLocaleString()}...`
    }
    if (job.status === 'failed') {
      return `${job.scenesProcessed.toLocaleString()} / ${job.totalScenes?.toLocaleString() ?? '?'}`
    }
    return job.scenesProcessed.toLocaleString()
  }

  return (
    <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-foreground">Launch Embedding Batch</h2>
        <p className="text-sm text-muted-foreground">
          Point Verity at your processed fleet data to begin the embedding pipeline
        </p>
      </div>

      {/* Setup Instructions */}
      <Card className="border-muted">
        <CardHeader className="pb-0 pt-4 px-4">
          <button
            onClick={() => setSetupOpen(v => !v)}
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Info className="w-4 h-4 text-muted-foreground" />
              How to connect your dataset
            </div>
            {setupOpen ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
          </button>
        </CardHeader>
        {setupOpen && (
          <CardContent className="pt-4 space-y-4">
            <div className="text-sm text-muted-foreground space-y-1">
              <p>Verity reads scene data directly from GCS. The machine running the backend needs Google Cloud credentials — no credentials are entered here.</p>
              <p className="font-medium text-foreground mt-2">One-time setup (run on the backend machine):</p>
            </div>
            <div className="bg-terminal-bg rounded-lg px-4 py-3 flex items-start gap-3">
              <Terminal className="w-4 h-4 text-primary mt-0.5 shrink-0" />
              <pre className="text-terminal-text text-xs font-mono whitespace-pre-wrap">gcloud auth application-default login</pre>
            </div>
            <p className="text-xs text-muted-foreground">This writes credentials that the pipeline picks up automatically. Re-run every 60 days or use a service account key for production.</p>

            <div className="border-t pt-4">
              <p className="text-sm font-medium text-foreground mb-3">Supported datasets &amp; example URIs</p>
              <div className="space-y-3">
                {DATASET_EXAMPLES.map(ds => (
                  <div key={ds.name} className="rounded-lg border bg-muted/30 p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">{ds.name}</span>
                      <button
                        onClick={() => setDataSourceUri(ds.uri)}
                        className="text-xs text-primary hover:underline"
                      >
                        Use this
                      </button>
                    </div>
                    <p className="text-xs font-mono text-muted-foreground">{ds.uri}</p>
                    <p className="text-xs text-muted-foreground">{ds.note}</p>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Launch Form */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid gap-6">
            {/* Data Source URI */}
            <div className="space-y-2">
              <Label htmlFor="dataSource" className="text-sm font-medium">
                Data Source Path
              </Label>
              <div className="relative">
                <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="dataSource"
                  placeholder="gs://your-bucket/path/to/parquet-files"
                  value={dataSourceUri}
                  onChange={(e) => handleUriChange(e.target.value)}
                  onBlur={handleUriBlur}
                  className={`pl-10 pr-10 font-mono text-sm ${pathError ? 'border-destructive focus-visible:ring-destructive' : segmentCount !== null ? 'border-primary/50' : ''}`}
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  {probing && <Loader2 className="w-4 h-4 text-muted-foreground animate-spin" />}
                  {!probing && segmentCount !== null && <CheckCircle2 className="w-4 h-4 text-primary" />}
                  {!probing && pathError && <AlertCircle className="w-4 h-4 text-destructive" />}
                </div>
              </div>
              {pathError ? (
                <div className="flex items-start gap-2 text-destructive text-xs">
                  <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                  <span>{pathError}</span>
                </div>
              ) : segmentCount !== null ? (
                <p className="text-xs text-primary font-medium flex items-center gap-1">
                  <ScanSearch className="w-3.5 h-3.5" />
                  {segmentCount} segments found
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">
                  GCS path to the directory containing your dataset&apos;s Parquet scene files. Tab out of the field to scan available segments.
                </p>
              )}
            </div>

            {/* Batch Label and Region */}
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="batchLabel" className="text-sm font-medium">
                  Batch Label
                </Label>
                <Input
                  id="batchLabel"
                  placeholder="e.g., Phoenix Q4 Highway Collection"
                  value={batchLabel}
                  onChange={(e) => setBatchLabel(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="region" className="text-sm font-medium">
                  Deployment Region
                </Label>
                <Select value={region} onValueChange={setRegion}>
                  <SelectTrigger id="region">
                    <SelectValue placeholder="Select region" />
                  </SelectTrigger>
                  <SelectContent>
                    {REGIONS.map((r) => (
                      <SelectItem key={r.value} value={r.value}>
                        {r.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Segment Count — only shown after path is probed */}
            {segmentCount !== null && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">
                    Segments to Process
                  </Label>
                  <button
                    type="button"
                    onClick={() => setMaxSegments(maxSegments === 'all' ? Math.min(5, segmentCount) : 'all')}
                    className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                      maxSegments === 'all'
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'text-muted-foreground border-muted-foreground/30 hover:border-primary hover:text-primary'
                    }`}
                  >
                    All
                  </button>
                </div>
                {maxSegments !== 'all' ? (
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={1}
                      max={segmentCount}
                      value={maxSegments}
                      onChange={e => setMaxSegments(Number(e.target.value))}
                      className="flex-1 accent-primary"
                    />
                    <Input
                      type="number"
                      min={1}
                      max={segmentCount}
                      value={maxSegments}
                      onChange={e => {
                        const v = Math.min(segmentCount, Math.max(1, Number(e.target.value)))
                        setMaxSegments(v)
                      }}
                      className="w-20 text-center font-mono"
                    />
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">All {segmentCount} segments will be processed.</p>
                )}
              </div>
            )}

            {/* Launch Button */}
            <Button
              size="lg"
              className="w-full sm:w-auto sm:self-end"
              onClick={handleLaunch}
              disabled={!canLaunch}
            >
              {launching ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Rocket className="w-4 h-4 mr-2" />
              )}
              {launching ? 'Validating path...' : 'Launch Batch'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Batch History */}
      <div className="flex-1 min-h-0">
        <Card className="h-full flex flex-col">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium">Batch History</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Label</TableHead>
                  <TableHead>Region</TableHead>
                  <TableHead>Scenes</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Started</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {batchJobs.map((job) => (
                  <TableRow key={job.id}>
                    <TableCell>
                      <div>
                        <p className="font-medium text-sm">{job.label}</p>
                        <p className="text-xs text-muted-foreground font-mono truncate max-w-[200px]">
                          {job.dataSourceUri}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-normal">
                        {job.region}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {formatSceneCount(job)}
                    </TableCell>
                    <TableCell>{getStatusBadge(job.status)}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(job.startedAt)}
                    </TableCell>
                    <TableCell className="text-right">
                      {job.status === 'completed' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onViewClusterSpace(job.id)}
                          className="text-primary hover:text-primary"
                        >
                          View Clusters
                          <ExternalLink className="w-3 h-3 ml-1" />
                        </Button>
                      )}
                      {job.status === 'running' && (
                        <span className="text-xs text-muted-foreground">In progress...</span>
                      )}
                      {job.status === 'failed' && (
                        <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive">
                          View Logs
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
