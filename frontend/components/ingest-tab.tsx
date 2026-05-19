'use client'

import { useState } from 'react'
import { Rocket, FolderOpen, CheckCircle2, XCircle, Loader2, ExternalLink } from 'lucide-react'
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

interface IngestTabProps {
  batchJobs: BatchJob[]
  onLaunchBatch: (dataSourceUri: string, label: string, region: string) => void
  onViewClusterSpace: (batchId: string) => void
}

const REGIONS = [
  { value: 'US-West', label: 'US West (Phoenix, SF, LA)' },
  { value: 'US-East', label: 'US East (Boston, NYC, Miami)' },
  { value: 'EU-West', label: 'EU West (Munich, London, Paris)' },
  { value: 'APAC', label: 'APAC (Tokyo, Singapore)' },
]

export function IngestTab({ batchJobs, onLaunchBatch, onViewClusterSpace }: IngestTabProps) {
  const [dataSourceUri, setDataSourceUri] = useState('')
  const [batchLabel, setBatchLabel] = useState('')
  const [region, setRegion] = useState('')

  const canLaunch = dataSourceUri.trim() !== '' && batchLabel.trim() !== '' && region !== ''

  const handleLaunch = () => {
    if (canLaunch) {
      onLaunchBatch(dataSourceUri, batchLabel, region)
      setDataSourceUri('')
      setBatchLabel('')
      setRegion('')
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
                  placeholder="s3://bucket/path/to/scenes/ or gs://bucket/path/"
                  value={dataSourceUri}
                  onChange={(e) => setDataSourceUri(e.target.value)}
                  className="pl-10 font-mono text-sm"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                S3, GCS, or Azure Blob URI containing processed scene data
              </p>
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

            {/* Launch Button */}
            <Button 
              size="lg" 
              className="w-full sm:w-auto sm:self-end"
              onClick={handleLaunch}
              disabled={!canLaunch}
            >
              <Rocket className="w-4 h-4 mr-2" />
              Launch Batch
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
