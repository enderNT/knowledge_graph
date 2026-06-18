"use client";
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Container from "@/components/Container";
import StatusBadge from "@/components/StatusBadge";
import { listTraces, type TraceSummary } from "@/lib/api";

const LIMIT = 20;

function fmt(ms?: number | null) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtTs(ts?: string) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function shortId(id: string) {
  return id.length > 20 ? `${id.slice(0, 20)}…` : id;
}

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState("desc");
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [draftExecutionId, setDraftExecutionId] = useState("");
  const [draftEpisodeId, setDraftEpisodeId] = useState("");
  const [draftStatus, setDraftStatus] = useState("");
  const [draftDomain, setDraftDomain] = useState("");
  const [executionId, setExecutionId] = useState("");
  const [episodeId, setEpisodeId] = useState("");
  const [status, setStatus] = useState("");
  const [domain, setDomain] = useState("");

  const load = useCallback(async (nextSkip: number) => {
    setLoading(true);
    try {
      const data = await listTraces({
        sort,
        limit: LIMIT,
        skip: nextSkip,
        execution_id: executionId || undefined,
        episode_id: episodeId || undefined,
        status: status || undefined,
        domain: domain || undefined,
      });
      setTraces(data);
      setHasMore(data.length === LIMIT);
    } catch {
      setTraces([]);
    } finally {
      setLoading(false);
    }
  }, [sort, executionId, episodeId, status, domain]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load(0);
  }, [load]);

  function applyFilters() {
    setExecutionId(draftExecutionId);
    setEpisodeId(draftEpisodeId);
    setStatus(draftStatus);
    setDomain(draftDomain);
    setSkip(0);
  }

  function clearFilters() {
    setDraftExecutionId(""); setDraftEpisodeId(""); setDraftStatus(""); setDraftDomain("");
    setExecutionId(""); setEpisodeId(""); setStatus(""); setDomain("");
    setSort("desc"); setSkip(0);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") applyFilters();
  }

  return (
    <Container>
      <div className="animate-enter" style={{ marginBottom: 32, display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>Canonical Traces</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>Readable ingestion executions recorded by the trace contract</p>
        </div>
        <Link href="/runs" className="btn-ghost" style={{ textDecoration: "none" }}>Operational logs</Link>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 20 }}>
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
            <TraceInput label="Execution ID" value={draftExecutionId} onChange={setDraftExecutionId} onKeyDown={handleKeyDown} />
            <TraceInput label="Episode ID" value={draftEpisodeId} onChange={setDraftEpisodeId} onKeyDown={handleKeyDown} />
            <TraceInput label="Status" value={draftStatus} onChange={setDraftStatus} onKeyDown={handleKeyDown} />
            <TraceInput label="Domain" value={draftDomain} onChange={setDraftDomain} onKeyDown={handleKeyDown} />
            <button className="btn-primary" onClick={applyFilters}>Search</button>
            <button className="btn-ghost" onClick={clearFilters}>Clear</button>
          </div>
        </div>

        <div className="animate-enter animate-enter-delay-2">
          {loading && <div style={{ color: "var(--text-secondary)", padding: 32, textAlign: "center" }}>Loading traces…</div>}
          {!loading && traces.length === 0 && <div style={{ color: "var(--text-secondary)", padding: 48, textAlign: "center" }}>No canonical traces found</div>}
          {!loading && traces.length > 0 && (
            <div className="card" style={{ overflow: "hidden" }}>
              <table>
                <thead>
                  <tr>
                    <th>Trace</th>
                    <th>Execution</th>
                    <th>Status</th>
                    <th>Domain</th>
                    <th>Steps</th>
                    <th>Duration</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {traces.map((trace) => (
                    <tr key={trace.trace_id}>
                      <td><Link href={`/traces/${trace.trace_id}`} className="mono" style={{ color: "var(--text-primary)" }}>{shortId(trace.trace_id)}</Link></td>
                      <td className="mono" style={{ color: "var(--text-secondary)" }}>{shortId(trace.execution_id)}</td>
                      <td><StatusBadge status={trace.status} size="sm" /></td>
                      <td>{trace.domain || "—"}</td>
                      <td>{trace.total_steps}</td>
                      <td className="mono">{fmt(trace.duration_ms)}</td>
                      <td className="mono" style={{ color: "var(--text-secondary)" }}>{fmtTs(trace.started_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
            <button className="btn-ghost" disabled={skip === 0} onClick={() => { const next = Math.max(0, skip - LIMIT); setSkip(next); load(next); }}>Previous</button>
            <button className="btn-ghost" disabled={!hasMore} onClick={() => { const next = skip + LIMIT; setSkip(next); load(next); }}>Next</button>
          </div>
        </div>
      </div>
    </Container>
  );
}

function TraceInput({ label, value, onChange, onKeyDown }: { label: string; value: string; onChange: (value: string) => void; onKeyDown: (e: React.KeyboardEvent) => void }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>{label}</label>
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)} onKeyDown={onKeyDown} />
    </div>
  );
}
