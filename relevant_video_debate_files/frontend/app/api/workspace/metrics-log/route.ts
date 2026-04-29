import { promises as fs } from "fs";
import path from "path";

import { NextResponse } from "next/server";

export const runtime = "nodejs";

const OUTPUTS_ROOT = path.resolve(process.cwd(), "..", "outputs");
const METRICS_LOG_PATH = path.resolve(OUTPUTS_ROOT, "metrics_log.jsonl");

type JsonRecord = Record<string, unknown>;

export async function POST(request: Request): Promise<Response> {
  let body: JsonRecord;
  try {
    body = (await request.json()) as JsonRecord;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  if (typeof body.runId !== "string" || typeof body.videoName !== "string") {
    return NextResponse.json({ detail: "Missing required metrics fields." }, { status: 400 });
  }

  await fs.mkdir(OUTPUTS_ROOT, { recursive: true });
  await fs.appendFile(METRICS_LOG_PATH, `${JSON.stringify(body)}\n`, "utf-8");
  return NextResponse.json({ ok: true });
}
