"use client";
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Container from "@/components/Container";
import StatusBadge from "@/components/StatusBadge";
import { listRuns, type LogRun } from "@/lib/api";

function fmt(ms?: number) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtTs(ts?: string) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function shortId(id: string) {
  return id.length > 18 ? id.slice(0, 18) + "…" : id;
}

const LIMIT = 10;

export default function RunsPage() {
  const [runs, setRuns] = useState<LogRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  const [sort, setSort] = useState("desc");
  const [jobId, setJobId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [episodeId, setEpisodeId] = useState("");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");

  const load = useCallback(
    async (s: number) => {
      setLoading(true);
      try {
        const data = await listRuns({
          sort,
          limit: LIMIT,
          skip: s,
          job_id: jobId || undefined,
          session_id: sessionId || undefined,
          episode_id: episodeId || undefined,
          since: since || undefined,
          until: until || undefined,
        });
        setRuns(data.runs ?? []);
        setHasMore((data.runs ?? []).length === LIMIT);
      } catch {
        setRuns([]);
      } finally {
        setLoading(false);
      }
    },
    [sort, jobId, sessionId, episodeId, since, until],
  );

  useEffect(() => {
    setSkip(0);
    load(0);
  }, [load]);

  function prev() {
    const n = Math.max(0, skip - LIMIT);
    setSkip(n);
    load(n);
  }

  function next() {
    const n = skip + LIMIT;
    setSkip(n);
    load(n);
  }

  return (
    <Container>
      <div className="animate-enter" style={{ marginBottom: 32, display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 600, letterSpacing: "-0.02em", marginBottom: 4 }}>Executions</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>Ingestion jobs, adaptive sessions, and HTTP requests</p>
        </div>
        <Link href="/realtime" className="btn-ghost" style={{ textDecoration: "none" }}>Live stream</Link>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 20 }}>
        {/* Filters sidebar */}
        <div className="card animate-enter animate-enter-delay-1" style={{ padding: 20, alignSelf: "start" }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: 16 }}>Filters</div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Sort</label>
              <select value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="desc">Newest first</option>
                <option value="asc">Oldest first</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Job ID</label>
              <input type="text" value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="job-…" />
            </div>

            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Session ID</label>
              <input type="text" value={sessionId} onChange={(e) => setSessionId(e.target.value)} placeholder="ads-…" />
            </div>

            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Episode ID</label>
              <input type="text" value={episodeId} onChange={(e) => setEpisodeId(e.target.value)} placeholder="ep-…" />
            </div>

            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Since</label>
              <input type="text" value={since} onChange={(e) => setSince(e.target.value)} placeholder="2025-01-01T00:00" />
            </div>

            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Until</label>
              <input type="text" value={until} onChange={(e) => setUntil(e.target.value)} placeholder="2025-12-31T23:59" />
            </div>

            <button
              className="btn-ghost"
              onClick={() => {
                setJobId(""); setSessionId(""); setEpisodeId(""); setSince(""); setUntil(""); setSort("desc");
              }}
            >
              Clear
            </button>
          </div>
        </div>

        {/* Results table */}
        <div className="animate-enter animate-enter-delay-2">
          <div className="card" style={{ overflow: "hidden" }}>
            <table>
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Run ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Duration</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", color: "var(--text-secondary)", padding: 32 }}>
                      Loading…
                    </td>
                  </tr>
                )}
                {!loading && runs.length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", color: "var(--text-secondary)", padding: 32 }}>
                      No executions match the current filters
                    </td>
                  </tr>
                )}
                {!loading &&
                  runs.map((r) => (
                    <tr key={r.run_id}>
                      <td style={{ color: "var(--text-secondary)", fontFamily: "var(--font-geist-mono, monospace)", fontSize: 12, whiteSpace: "nowrap" }}>
                        {fmtTs(r.start_ts)}
                      </td>
                      <td>
                        <span
                          style={{ fontFamily: "var(--font-geist-mono, monospace)", fontSize: 12 }}
                          title={r.run_id}
                        >
                          {shortId(r.run_id)}
                        </span>
                      </td>
                      <td><StatusBadge status={r.run_type || "request"} size="sm" /></td>
                      <td><StatusBadge status={r.status} size="sm" /></td>
                      <td style={{ fontFamily: "var(--font-geist-mono, monospace)", fontSize: 12, color: "var(--text-secondary)" }}>
                        {fmt(r.duration_ms)}
                      </td>
                      <td>
                        <Link
                          href={`/runs/${r.run_id}`}
                          style={{ fontSize: 12, color: "var(--text-secondary)", textDecoration: "none" }}
                        >
                          View →
                        </Link>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16 }}>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              {runs.length > 0 ? `${skip + 1}–${skip + runs.length}` : "0 results"}
            </span>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-ghost" onClick={prev} disabled={skip === 0} style={{ opacity: skip === 0 ? 0.4 : 1 }}>
                ← Prev
              </button>
              <button className="btn-ghost" onClick={next} disabled={!hasMore} style={{ opacity: !hasMore ? 0.4 : 1 }}>
                Next →
              </button>
            </div>
          </div>
        </div>
      </div>
    </Container>
  );
}
