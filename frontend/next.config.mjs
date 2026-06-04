/** @type {import('next').NextConfig} */

// Backend origins are resolved server-side (these are NOT NEXT_PUBLIC — the
// browser never sees them). Defaults target the local dev stack; override per
// deployment via env. Single-origin design: the browser only ever talks to the
// Next.js origin, which proxies to the right backend. So a customer forwards /
// exposes ONE port (3000) and never has to know these services exist.
const API_ORIGIN = process.env.VERITY_API_ORIGIN ?? 'http://localhost:8000'   // waymo_runner: ingest/cluster/analysis
const JUDGE_ORIGIN = process.env.VERITY_JUDGE_ORIGIN ?? 'http://localhost:8001' // judge_ui
const DEV_ORIGIN = process.env.VERITY_DEV_ORIGIN ?? 'http://localhost:8002'     // dev_dashboard (gated)

const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      // --- :8000 waymo_runner (Ingest / Cluster Space / Analysis tabs) ---
      { source: '/probe-path', destination: `${API_ORIGIN}/probe-path` },
      { source: '/batches', destination: `${API_ORIGIN}/batches` },
      { source: '/batches/:path*', destination: `${API_ORIGIN}/batches/:path*` },
      { source: '/cluster-space', destination: `${API_ORIGIN}/cluster-space` },
      { source: '/scenarios', destination: `${API_ORIGIN}/scenarios` },
      { source: '/scenes/:path*', destination: `${API_ORIGIN}/scenes/:path*` },
      { source: '/video/:path*', destination: `${API_ORIGIN}/video/:path*` },
      { source: '/segment-video/:path*', destination: `${API_ORIGIN}/segment-video/:path*` },
      { source: '/analysis/:path*', destination: `${API_ORIGIN}/analysis/:path*` },
      // --- :8001 judge_ui (Judge tab) ---
      { source: '/judge/:path*', destination: `${JUDGE_ORIGIN}/judge/:path*` },
      // --- :8002 dev_dashboard (Dev tabs — only reachable when that service is up) ---
      { source: '/dev/:path*', destination: `${DEV_ORIGIN}/dev/:path*` },
    ]
  },
}

export default nextConfig
