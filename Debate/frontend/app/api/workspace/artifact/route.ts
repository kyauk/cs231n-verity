import { promises as fs } from "fs";
import path from "path";

import { NextRequest, NextResponse } from "next/server";

const OUTPUTS_ROOT = path.resolve(process.cwd(), "..", "outputs");

function is_safe_relative_path(candidate: string): boolean {
  return !candidate.includes("..") && !path.isAbsolute(candidate);
}

export async function GET(request: NextRequest): Promise<Response> {
  const raw_path = request.nextUrl.searchParams.get("path");
  if (!raw_path || !is_safe_relative_path(raw_path)) {
    return NextResponse.json({ detail: "Invalid artifact path." }, { status: 400 });
  }

  const absolute_path = path.resolve(OUTPUTS_ROOT, raw_path);
  if (!absolute_path.startsWith(OUTPUTS_ROOT)) {
    return NextResponse.json({ detail: "Artifact path escapes outputs directory." }, { status: 400 });
  }

  try {
    const bytes = await fs.readFile(absolute_path);
    const extension = path.extname(absolute_path).toLowerCase();
    const content_type =
      extension === ".mp4"
        ? "video/mp4"
        : extension === ".jpg" || extension === ".jpeg"
          ? "image/jpeg"
          : extension === ".png"
            ? "image/png"
            : "application/octet-stream";
    return new Response(bytes, {
      status: 200,
      headers: {
        "Content-Type": content_type,
        "Cache-Control": "no-store"
      }
    });
  } catch {
    return NextResponse.json({ detail: "Artifact not found." }, { status: 404 });
  }
}
