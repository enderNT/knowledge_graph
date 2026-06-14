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
