import { NextRequest, NextResponse } from "next/server";
import { API_BASE, API_HEADERS } from "@/lib/server-config";

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.toString();
  const res = await fetch(`${API_BASE}/v1/traces${q ? `?${q}` : ""}`, {
    headers: API_HEADERS,
    cache: "no-store",
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
