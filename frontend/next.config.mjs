/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Proxy the Judge API through the Next.js origin so the browser only needs
  // port 3000 forwarded (the Judge tab otherwise fetches :8001 directly from
  // the browser, which requires a second forwarded port). With
  // NEXT_PUBLIC_JUDGE_API_URL='' the client calls same-origin /judge/* and
  // these rewrites forward them to the backend server-side.
  async rewrites() {
    return [
      {
        source: '/judge/:path*',
        destination: 'http://localhost:8001/judge/:path*',
      },
    ]
  },
}

export default nextConfig
