import { NextRequest, NextResponse } from "next/server";
import { API_BASE, API_HEADERS } from "@/lib/server-config";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ trace_id: string }> },
) {
  const { trace_id } = await params;
  const res = await fetch(`${API_BASE}/v1/traces/${trace_id}`, {
    headers: API_HEADERS,
    cache: "no-store",
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
