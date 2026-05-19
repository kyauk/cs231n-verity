'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { FileSearch, AlertTriangle, FileCode, Flag, Eye } from 'lucide-react'
import type { FlaggedScenario } from '@/lib/types'

interface DashboardTabProps {
  scenarios: FlaggedScenario[]
  onViewScenario: (scenarioId: string) => void
}

export function DashboardTab({ scenarios, onViewScenario }: DashboardTabProps) {
  const [regionFilter, setRegionFilter] = useState<string>('all')

  const filteredScenarios = regionFilter === 'all' 
    ? scenarios 
    : scenarios.filter(s => s.region === regionFilter)

  const regions = ['all', ...new Set(scenarios.map(s => s.region))]

  const totalAnalyzed = scenarios.length
  const coverageGaps = scenarios.filter(s => s.priorityScore > 80).length
  const simulationSpecs = scenarios.filter(s => s.hasSimulationSpec).length
  const highPriority = scenarios.filter(s => s.priorityScore >= 90).length

  const getPriorityColor = (score: number) => {
    if (score >= 90) return 'bg-destructive text-destructive-foreground'
    if (score >= 80) return 'bg-amber-500 text-white'
    if (score >= 70) return 'bg-blue-500 text-white'
    return 'bg-muted text-muted-foreground'
  }

  return (
    <div className="flex flex-col gap-6 p-6 h-full overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Results Dashboard</h2>
          <p className="text-sm text-muted-foreground">Overview of analyzed scenarios and generated specifications</p>
        </div>
        <Select value={regionFilter} onValueChange={setRegionFilter}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by region" />
          </SelectTrigger>
          <SelectContent>
            {regions.map((region) => (
              <SelectItem key={region} value={region}>
                {region === 'all' ? 'All Regions' : region}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                <FileSearch className="w-6 h-6 text-primary" />
              </div>
              <div>
                <p className="text-3xl font-bold text-foreground">{totalAnalyzed}</p>
                <p className="text-sm text-muted-foreground">Scenes Analyzed</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-amber-100 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-amber-600" />
              </div>
              <div>
                <p className="text-3xl font-bold text-foreground">{coverageGaps}</p>
                <p className="text-sm text-muted-foreground">Coverage Gaps</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-blue-100 flex items-center justify-center">
                <FileCode className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <p className="text-3xl font-bold text-foreground">{simulationSpecs}</p>
                <p className="text-sm text-muted-foreground">Simulation Specs</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-destructive/10 flex items-center justify-center">
                <Flag className="w-6 h-6 text-destructive" />
              </div>
              <div>
                <p className="text-3xl font-bold text-destructive">{highPriority}</p>
                <p className="text-sm text-muted-foreground">High Priority</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Scenarios Table */}
      <Card className="flex-1 min-h-0 flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium">Flagged Scenarios</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0">
          <div className="overflow-auto h-full">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50">
                  <TableHead className="w-[250px]">Scenario Name</TableHead>
                  <TableHead className="w-[100px]">Cluster</TableHead>
                  <TableHead className="w-[100px]">Priority</TableHead>
                  <TableHead>Defining Conditions</TableHead>
                  <TableHead className="w-[120px]">Spec Status</TableHead>
                  <TableHead className="w-[80px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredScenarios.map((scenario) => (
                  <TableRow key={scenario.id}>
                    <TableCell className="font-medium">{scenario.scenarioName}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono">
                        C-{scenario.clusterId}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge className={getPriorityColor(scenario.priorityScore)}>
                        {scenario.priorityScore}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground max-w-xs truncate">
                      {scenario.definingConditions}
                    </TableCell>
                    <TableCell>
                      {scenario.hasSimulationSpec ? (
                        <Badge variant="secondary" className="bg-primary/10 text-primary">
                          Generated
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-muted text-muted-foreground">
                          Pending
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => onViewScenario(scenario.id)}
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
