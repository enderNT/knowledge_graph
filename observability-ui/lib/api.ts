// All calls go to Next.js API routes (/api/obs/*) which proxy to FastAPI on the Docker network.
// No NEXT_PUBLIC env vars needed — works at build time and runtime regardless of domain.

export async function listRuns(params: {
  sort?: string;
  limit?: number;
  skip?: number;
  job_id?: string;
  session_id?: string;
  episode_id?: string;
  since?: string;
  until?: string;
}) {
  const q = new URLSearchParams();
  if (params.sort) q.set("sort", params.sort);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.skip !== undefined) q.set("skip", String(params.skip));
  if (params.job_id) q.set("job_id", params.job_id);
  if (params.session_id) q.set("session_id", params.session_id);
  if (params.episode_id) q.set("episode_id", params.episode_id);
  if (params.since) q.set("since", params.since);
  if (params.until) q.set("until", params.until);
  const res = await fetch(`/api/obs/runs?${q}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ runs: LogRun[]; limit: number; skip: number }>;
}

export async function getRunDetail(run_id: string) {
  const res = await fetch(`/api/obs/runs/${run_id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ run: LogRun | null; events: LogEvent[] }>;
}

export async function listTraces(params: {
  sort?: string;
  limit?: number;
  skip?: number;
  execution_id?: string;
  episode_id?: string;
  status?: string;
  domain?: string;
}) {
  const q = new URLSearchParams();
  if (params.sort) q.set("sort", params.sort);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.skip !== undefined) q.set("skip", String(params.skip));
  if (params.execution_id) q.set("execution_id", params.execution_id);
  if (params.episode_id) q.set("episode_id", params.episode_id);
  if (params.status) q.set("status", params.status);
  if (params.domain) q.set("domain", params.domain);
  const res = await fetch(`/api/traces?${q}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<TraceSummary[]>;
}

export async function getTraceDetail(trace_id: string) {
  const res = await fetch(`/api/traces/${trace_id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<CanonicalTrace>;
}

export function openStream(params: { run_id?: string; job_id?: string; session_id?: string }) {
  const q = new URLSearchParams();
  if (params.run_id) q.set("run_id", params.run_id);
  if (params.job_id) q.set("job_id", params.job_id);
  if (params.session_id) q.set("session_id", params.session_id);
  return new EventSource(`/api/obs/stream?${q}`);
}

export interface LogRun {
  run_id: string;
  run_type: string;
  status: string;
  job_id?: string;
  session_id?: string;
  episode_id?: string;
  start_ts?: string;
  end_ts?: string;
  duration_ms?: number;
}

export interface LogEvent {
  ts: string;
  ts_epoch_ms: number;
  level: string;
  logger_name?: string;
  event: string;
  run_id?: string;
  step?: string;
  job_id?: string;
  session_id?: string;
  episode_id?: string;
  block_id?: string;
  tool_name?: string;
  path?: string;
  status?: string;
  duration_ms?: number;
  input_shape?: unknown;
  output_shape?: unknown;
  counts?: unknown;
  error_type?: string;
  error_message?: string;
}

export interface TraceSummary {
  trace_id: string;
  execution_type: string;
  execution_id: string;
  episode_id?: string | null;
  status: string;
  started_at: string;
  ended_at?: string | null;
  duration_ms?: number | null;
  total_steps: number;
  total_decisions: number;
  error_count: number;
  status_counts: Record<string, number>;
  semantic_counts: Record<string, number>;
  domain: string;
}

export interface TraceBoundaryPayload {
  kind: string;
  provider?: string;
  model?: string;
  request_text?: string | null;
  response_text?: string | null;
  request_json?: unknown;
  response_json?: unknown;
  metadata?: Record<string, unknown>;
}

export interface TraceEvent {
  event_id: string;
  trace_id: string;
  parent_event_id?: string | null;
  sequence: number;
  type: string;
  role: string;
  status: string;
  title: string;
  summary?: string;
  input?: unknown;
  output?: unknown;
  detail?: unknown;
  boundary_payload?: TraceBoundaryPayload | null;
  created_at: string;
}

export interface CanonicalTrace {
  summary: TraceSummary;
  events: TraceEvent[];
}
