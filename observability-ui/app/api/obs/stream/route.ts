import { NextRequest } from "next/server";
import { API_BASE, API_HEADERS } from "@/lib/server-config";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.toString();
  const upstream = await fetch(
    `${API_BASE}/v1/observability/stream${q ? `?${q}` : ""}`,
    { headers: API_HEADERS, cache: "no-store" },
  );

  // Pipe the SSE stream from FastAPI through to the browser
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
      Connection: "keep-alive",
    },
  });
}
