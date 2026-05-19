'use client'

import { useRef, useState, useMemo, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Play, X, Maximize2 } from 'lucide-react'
import type { ClusterPoint, ClusterStats, Scene } from '@/lib/types'

// Cluster colors
const CLUSTER_COLORS = [
  '#76B900', // NVIDIA Green
  '#0077B6', // Blue
  '#7B2CBF', // Purple
  '#F77F00', // Orange
  '#2D6A4F', // Teal
]
const NOISE_COLOR = '#DC2626' // Red for noise

interface PointCloudProps {
  points: ClusterPoint[]
  onPointClick: (point: ClusterPoint) => void
  selectedPoint: ClusterPoint | null
}

function PointCloud({ points, onPointClick, selectedPoint }: PointCloudProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  
  const tempObject = useMemo(() => new THREE.Object3D(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])
  
  // Set positions and colors after mesh is mounted
  useEffect(() => {
    if (!meshRef.current) return
    
    points.forEach((point, i) => {
      tempObject.position.set(point.x, point.y, point.z)
      tempObject.scale.setScalar(point.isNoise ? 0.12 : 0.1)
      tempObject.updateMatrix()
      meshRef.current!.setMatrixAt(i, tempObject.matrix)
      
      const color = point.isNoise 
        ? NOISE_COLOR 
        : CLUSTER_COLORS[point.clusterId % CLUSTER_COLORS.length]
      tempColor.set(color)
      meshRef.current!.setColorAt(i, tempColor)
    })
    
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true
    }
  }, [points, tempObject, tempColor])

  // Pulse animation for selected point
  useFrame((state) => {
    if (!meshRef.current || !selectedPoint) return
    
    const selectedIndex = points.findIndex(p => p.id === selectedPoint.id)
    if (selectedIndex === -1) return
    
    const scale = 0.15 + Math.sin(state.clock.elapsedTime * 4) * 0.05
    tempObject.position.set(selectedPoint.x, selectedPoint.y, selectedPoint.z)
    tempObject.scale.setScalar(scale)
    tempObject.updateMatrix()
    meshRef.current.setMatrixAt(selectedIndex, tempObject.matrix)
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  const handleClick = (e: any) => {
    e.stopPropagation()
    if (e.instanceId !== undefined) {
      onPointClick(points[e.instanceId])
    }
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}

function Grid() {
  return (
    <group>
      <gridHelper args={[20, 20, '#e5e7eb', '#f3f4f6']} position={[0, -5, 0]} />
      <axesHelper args={[6]} />
    </group>
  )
}

interface SceneModalProps {
  scene: Scene
  open: boolean
  onClose: () => void
  onAnalyze: () => void
}

function SceneModal({ scene, open, onClose, onAnalyze }: SceneModalProps) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Scene Details
            <Badge variant="outline" className="font-mono text-xs">{scene.id}</Badge>
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {/* Video Preview */}
          <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-muted to-muted/50" />
            <div className="relative z-10 text-center">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                <Play className="w-8 h-8 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground">Video Preview</p>
            </div>
          </div>

          {/* Annotations */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Environment</h4>
              <div className="space-y-1.5">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Weather</span>
                  <span className="font-medium">{scene.annotations.weather}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Time of Day</span>
                  <span className="font-medium">{scene.annotations.timeOfDay}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Road Type</span>
                  <span className="font-medium">{scene.annotations.roadType}</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Actors</h4>
              <div className="flex flex-wrap gap-1">
                {scene.annotations.actors.map((actor) => (
                  <Badge key={actor} variant="secondary" className="text-xs">{actor}</Badge>
                ))}
              </div>
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2 mt-3">Events</h4>
              <div className="flex flex-wrap gap-1">
                {scene.annotations.events.map((event) => (
                  <Badge key={event} variant="outline" className="text-xs">{event}</Badge>
                ))}
              </div>
            </div>
          </div>

          <Button className="w-full" onClick={onAnalyze}>
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
  scene: Scene
  onAnalyzeScene: (sceneId: string) => void
}

export function ClusterSpaceTab({ points, clusterStats, scene, onAnalyzeScene }: ClusterSpaceTabProps) {
  const [selectedPoint, setSelectedPoint] = useState<ClusterPoint | null>(null)
  const [modalOpen, setModalOpen] = useState(false)

  const totalScenes = points.length
  const noiseCount = points.filter(p => p.isNoise).length
  const clusterCount = clusterStats.length

  const handlePointClick = (point: ClusterPoint) => {
    setSelectedPoint(point)
    setModalOpen(true)
  }

  const handleAnalyze = () => {
    if (selectedPoint) {
      onAnalyzeScene(selectedPoint.sceneId)
      setModalOpen(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* 3D Visualization */}
      <div className="flex-1 relative bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg m-4 mr-0">
        <Canvas
          camera={{ position: [8, 6, 8], fov: 50 }}
          gl={{ antialias: true }}
        >
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          <PointCloud 
            points={points} 
            onPointClick={handlePointClick}
            selectedPoint={selectedPoint}
          />
          <Grid />
          <OrbitControls 
            enableDamping 
            dampingFactor={0.05}
            minDistance={5}
            maxDistance={25}
          />
        </Canvas>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border">
          <p className="text-xs font-medium text-foreground mb-2">Cluster Legend</p>
          <div className="space-y-1">
            {CLUSTER_COLORS.map((color, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-muted-foreground">Cluster {i}</span>
              </div>
            ))}
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NOISE_COLOR }} />
              <span className="text-muted-foreground">Noise</span>
            </div>
          </div>
        </div>

        <div className="absolute top-4 left-4 flex items-center gap-2">
          <Badge variant="secondary" className="bg-card/90 backdrop-blur-sm">
            <Maximize2 className="w-3 h-3 mr-1" />
            Drag to rotate / Scroll to zoom
          </Badge>
        </div>
      </div>

      {/* Sidebar */}
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
                {clusterStats.map((cluster) => (
                  <div 
                    key={cluster.id} 
                    className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length] }}
                        />
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
                          style={{ 
                            width: `${cluster.density * 100}%`,
                            backgroundColor: CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length]
                          }}
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

      {/* Scene Modal */}
      <SceneModal 
        scene={scene}
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onAnalyze={handleAnalyze}
      />
    </div>
  )
}
