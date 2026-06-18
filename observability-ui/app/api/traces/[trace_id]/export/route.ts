import { NextRequest } from "next/server";
import { API_BASE, API_HEADERS } from "@/lib/server-config";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ trace_id: string }> },
) {
  const { trace_id } = await params;
  const res = await fetch(`${API_BASE}/v1/traces/${trace_id}/export`, {
    headers: API_HEADERS,
    cache: "no-store",
  });
  const text = await res.text();
  return new Response(text, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") ?? "text/plain; charset=utf-8" },
  });
}
