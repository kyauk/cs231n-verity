'use client'

import { useRef, useState, useMemo, useEffect } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Play, Maximize2, Loader2, ScatterChart } from 'lucide-react'
import type { ClusterPoint, ClusterStats, Scene } from '@/lib/types'
import { fetchScene } from '@/lib/api'

const CLUSTER_COLORS = [
  '#76B900', // NVIDIA green
  '#38BDF8', // sky blue
  '#A78BFA', // violet
  '#FB923C', // orange
  '#34D399', // emerald
  '#F472B6', // pink
  '#FACC15', // yellow
  '#60A5FA', // blue
]
const NOISE_COLOR = '#F87171' // soft red

// One sphere per point — R3F onClick on individual meshes is rock-solid.
function Point({
  point,
  isSelected,
  onClick,
}: {
  point: ClusterPoint
  isSelected: boolean
  onClick: (p: ClusterPoint) => void
}) {
  const color = point.isNoise ? NOISE_COLOR : CLUSTER_COLORS[point.clusterId % CLUSTER_COLORS.length]
  const baseScale = point.isNoise ? 0.11 : 0.09
  const scale = isSelected ? baseScale * 2.0 : baseScale

  return (
    <mesh
      position={[point.x, point.y, point.z]}
      scale={scale}
      onClick={(e) => { e.stopPropagation(); onClick(point) }}
    >
      <sphereGeometry args={[1, 10, 10]} />
      <meshBasicMaterial color={color} toneMapped={false} />
    </mesh>
  )
}

function AxisLabel({ position, label, color }: { position: [number, number, number]; label: string; color: string }) {
  return (
    <Html position={position} center style={{ pointerEvents: 'none' }}>
      <span style={{ color, fontSize: 11, fontWeight: 600, fontFamily: 'monospace', textShadow: '0 0 4px white' }}>
        {label}
      </span>
    </Html>
  )
}

function Grid({ axisLen }: { axisLen: number }) {
  return (
    <group>
      <gridHelper args={[axisLen * 2, 20, '#e5e7eb', '#f3f4f6']} />
      <axesHelper args={[axisLen]} />
      <AxisLabel position={[axisLen + 0.5, 0, 0]} label="UMAP-1" color="#ef4444" />
      <AxisLabel position={[0, axisLen + 0.5, 0]} label="UMAP-2" color="#22c55e" />
      <AxisLabel position={[0, 0, axisLen + 0.5]} label="UMAP-3" color="#3b82f6" />
    </group>
  )
}

function CameraFit({ points }: { points: ClusterPoint[] }) {
  const { camera, controls } = useThree() as any
  const fitted = useRef(false)

  useEffect(() => {
    if (fitted.current || points.length === 0 || !controls) return
    fitted.current = true

    const box = new THREE.Box3()
    points.forEach(p => box.expandByPoint(new THREE.Vector3(p.x, p.y, p.z)))
    const center = new THREE.Vector3()
    box.getCenter(center)
    const size = new THREE.Vector3()
    box.getSize(size)
    const maxDim = Math.max(size.x, size.y, size.z, 1)
    const fov = (camera as THREE.PerspectiveCamera).fov * (Math.PI / 180)
    const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.8

    camera.position.set(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.6)
    camera.near = dist * 0.01
    camera.far = dist * 10
    camera.updateProjectionMatrix()

    controls.target.copy(center)
    controls.minDistance = dist * 0.1
    controls.maxDistance = dist * 5
    controls.update()
  }, [points, camera, controls])

  return null
}

function SceneModal({ scene, loading, open, onClose, onAnalyze }: {
  scene: Scene | null
  loading: boolean
  open: boolean
  onClose: () => void
  onAnalyze: () => void
}) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Scene Details
            <Badge variant="outline" className="font-mono text-xs">{scene?.id ?? '—'}</Badge>
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
            {scene?.videoUrl ? (
              <video
                key={scene.videoUrl}
                src={scene.videoUrl}
                controls
                playsInline
                className="absolute inset-0 w-full h-full object-contain bg-black"
              />
            ) : (
              <>
                <div className="absolute inset-0 bg-gradient-to-br from-muted to-muted/50" />
                <div className="relative z-10 text-center">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                    {loading ? <Loader2 className="w-8 h-8 text-primary animate-spin" /> : <Play className="w-8 h-8 text-primary" />}
                  </div>
                  <p className="text-sm text-muted-foreground">{loading ? 'Loading scene…' : 'No video available'}</p>
                </div>
              </>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Environment</h4>
              <div className="space-y-1.5">
                {(['weather', 'timeOfDay', 'roadType'] as const).map(k => (
                  <div key={k} className="flex justify-between text-sm">
                    <span className="text-muted-foreground capitalize">{k === 'timeOfDay' ? 'Time of Day' : k === 'roadType' ? 'Road Type' : 'Weather'}</span>
                    <span className="font-medium">{scene?.annotations[k] ?? '—'}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Events</h4>
              <div className="flex flex-wrap gap-1">
                {(scene?.annotations.events ?? []).map(ev => (
                  <Badge key={ev} variant="outline" className="text-xs">{ev}</Badge>
                ))}
              </div>
            </div>
          </div>
          <Button className="w-full" onClick={onAnalyze} disabled={loading || !scene}>
            <Play className="w-4 h-4 mr-2" />
            Analyze this scene
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

interface ClusterSpaceTabProps {
  points: ClusterPoint[]
  clusterStats: ClusterStats[]
  onAnalyzeScene: (scene: Scene) => void
}

export function ClusterSpaceTab({ points, clusterStats, onAnalyzeScene }: ClusterSpaceTabProps) {
  const [selectedPoint, setSelectedPoint] = useState<ClusterPoint | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [activeScene, setActiveScene] = useState<Scene | null>(null)
  const [sceneLoading, setSceneLoading] = useState(false)

  const axisLen = useMemo(() => {
    if (points.length === 0) return 6
    const vals = points.flatMap(p => [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)])
    return Math.ceil(Math.max(...vals) * 1.2) || 6
  }, [points])

  const totalScenes = points.length
  const noiseCount = points.filter(p => p.isNoise).length
  const clusterCount = clusterStats.length

  const handlePointClick = async (point: ClusterPoint) => {
    setSelectedPoint(point)
    setActiveScene(null)
    setModalOpen(true)
    setSceneLoading(true)
    try {
      setActiveScene(await fetchScene(point.sceneId))
    } catch {
      setActiveScene({
        id: point.sceneId,
        videoUrl: '',
        thumbnail: '',
        annotations: { weather: 'Unknown', timeOfDay: 'Unknown', roadType: 'Unknown', actors: [], events: [] },
      })
    } finally {
      setSceneLoading(false)
    }
  }

  if (points.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-12">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
          <ScatterChart className="w-8 h-8 text-muted-foreground" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">No Cluster Data Yet</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Ingest a Waymo batch and wait for it to complete — clusters will appear here once embeddings are ready.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      <div className="flex-1 relative bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg m-4 mr-0">
        <Canvas camera={{ position: [8, 6, 8], fov: 50 }} gl={{ antialias: true }}>
          <ambientLight intensity={0.7} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          <directionalLight position={[-5, 5, 5]} intensity={0.4} />

          {points.map(p => (
            <Point
              key={p.id}
              point={p}
              isSelected={selectedPoint?.id === p.id}
              onClick={handlePointClick}
            />
          ))}

          <Grid axisLen={axisLen} />
          <CameraFit points={points} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
        </Canvas>

        <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border">
          <p className="text-xs font-medium text-foreground mb-2">Cluster Legend</p>
          <div className="space-y-1">
            {clusterStats.map(c => (
              <div key={c.id} className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: CLUSTER_COLORS[c.id % CLUSTER_COLORS.length] }} />
                <span className="text-muted-foreground">Cluster {c.id}</span>
                <span className="text-muted-foreground/60 ml-auto pl-2">{c.sceneCount}</span>
              </div>
            ))}
            {noiseCount > 0 && (
              <div className="flex items-center gap-2 text-xs border-t pt-1 mt-1">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: NOISE_COLOR }} />
                <span className="text-muted-foreground">Noise</span>
                <span className="text-muted-foreground/60 ml-auto pl-2">{noiseCount}</span>
              </div>
            )}
          </div>
        </div>

        <div className="absolute top-4 left-4">
          <Badge variant="secondary" className="bg-card/90 backdrop-blur-sm">
            <Maximize2 className="w-3 h-3 mr-1" />
            Drag to rotate · Scroll to zoom · Click a point
          </Badge>
        </div>
      </div>

      <div className="w-80 p-4 flex flex-col gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Cluster Statistics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-muted rounded-lg p-2">
                <p className="text-2xl font-bold text-primary">{totalScenes}</p>
                <p className="text-xs text-muted-foreground">Total Scenes</p>
              </div>
              <div className="bg-muted rounded-lg p-2">
                <p className="text-2xl font-bold text-foreground">{clusterCount}</p>
                <p className="text-xs text-muted-foreground">Clusters</p>
              </div>
              <div className="bg-muted rounded-lg p-2">
                <p className="text-2xl font-bold text-destructive">{noiseCount}</p>
                <p className="text-xs text-muted-foreground">Noise</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="flex-1 min-h-0">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Clusters</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[calc(100%-3rem)] px-4 pb-4">
              <div className="space-y-2">
                {clusterStats.map(cluster => (
                  <div key={cluster.id} className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length] }} />
                        <span className="text-sm font-medium">Cluster {cluster.id}</span>
                      </div>
                      <Badge variant="secondary" className="text-xs">{cluster.sceneCount} scenes</Badge>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Density</span>
                        <span className="font-medium">{Math.round(cluster.density * 100)}%</span>
                      </div>
                      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.max(cluster.density * 100, 4)}%`, backgroundColor: CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length] }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <SceneModal
        scene={activeScene}
        loading={sceneLoading}
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onAnalyze={() => { if (activeScene) { onAnalyzeScene(activeScene); setModalOpen(false) } }}
      />
    </div>
  )
}
