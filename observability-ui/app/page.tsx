import Link from "next/link";
import Container from "@/components/Container";
import StatusBadge from "@/components/StatusBadge";
import { listRuns } from "@/lib/api";

function fmt(ms?: number) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function relativeTime(ts?: string) {
  if (!ts) return "—";
  const diff = Date.now() - new Date(ts).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default async function DashboardPage() {
  let recentRuns: Awaited<ReturnType<typeof listRuns>>["runs"] = [];
  let failedRuns: typeof recentRuns = [];
  let totalToday = 0;
  let errorRate = "—";

  try {
    const data = await listRuns({ sort: "desc", limit: 10 });
    recentRuns = data.runs ?? [];
    failedRuns = recentRuns.filter((r) => r.status === "failed").slice(0, 5);
    totalToday = recentRuns.length;
    const errors = recentRuns.filter((r) => r.status === "failed").length;
    errorRate = totalToday > 0 ? `${Math.round((errors / totalToday) * 100)}%` : "0%";
  } catch {
    // backend not reachable yet
  }

  const ingestionRuns = recentRuns.filter((r) => r.run_type === "ingestion");
  const adaptiveRuns = recentRuns.filter((r) => r.run_type === "adaptive");

  function avg(runs: typeof recentRuns) {
    const withMs = runs.filter((r) => r.duration_ms != null);
    if (!withMs.length) return "—";
    return fmt(Math.round(withMs.reduce((s, r) => s + (r.duration_ms ?? 0), 0) / withMs.length));
  }

  return (
    <Container>
      <div className="animate-enter" style={{ marginBottom: 40 }}>
        <h1
          style={{
            fontSize: 28,
            fontWeight: 600,
            letterSpacing: "-0.02em",
            color: "var(--text-primary)",
            marginBottom: 6,
          }}
        >
          Overview
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>
          Execution telemetry for the knowledge graph pipeline
        </p>
      </div>

      {/* Stats grid */}
      <div
        className="animate-enter animate-enter-delay-1"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 12,
          marginBottom: 32,
        }}
      >
        {[
          { label: "Recent runs", value: String(totalToday) },
          { label: "Error rate", value: errorRate },
          { label: "Avg ingestion", value: avg(ingestionRuns) },
          { label: "Avg session", value: avg(adaptiveRuns) },
        ].map((s) => (
          <div
            key={s.label}
            className="card"
            style={{ padding: "20px 24px" }}
          >
            <div style={{ fontSize: 11, color: "var(--text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>
              {s.label}
            </div>
            <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: "-0.02em", color: "var(--text-primary)", fontFamily: "var(--font-geist-mono, monospace)" }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        {/* Recent runs */}
        <div className="card animate-enter animate-enter-delay-2" style={{ overflow: "hidden" }}>
          <div
            style={{
              padding: "16px 20px",
              borderBottom: "1px solid var(--border)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 600 }}>Recent executions</span>
            <Link href="/runs" style={{ fontSize: 12, color: "var(--text-secondary)", textDecoration: "none" }}>
              View all →
            </Link>
          </div>
          <table>
            <tbody>
              {recentRuns.slice(0, 6).map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <Link
                      href={`/runs/${r.run_id}`}
                      style={{ fontFamily: "var(--font-geist-mono, monospace)", fontSize: 12, color: "var(--text-primary)", textDecoration: "none" }}
                    >
                      {r.run_id.slice(0, 14)}…
                    </Link>
                  </td>
                  <td><StatusBadge status={r.run_type} size="sm" /></td>
                  <td><StatusBadge status={r.status} size="sm" /></td>
                  <td style={{ color: "var(--text-secondary)", fontSize: 12 }}>{relativeTime(r.start_ts)}</td>
                </tr>
              ))}
              {recentRuns.length === 0 && (
                <tr><td colSpan={4} style={{ color: "var(--text-secondary)", textAlign: "center", padding: 24 }}>No executions recorded yet</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Recent failures */}
        <div className="card animate-enter animate-enter-delay-3" style={{ overflow: "hidden" }}>
          <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Recent failures</span>
            <Link href="/runs?status=failed" style={{ fontSize: 12, color: "var(--text-secondary)", textDecoration: "none" }}>
              Filter →
            </Link>
          </div>
          <table>
            <tbody>
              {failedRuns.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <Link
                      href={`/runs/${r.run_id}`}
                      style={{ fontFamily: "var(--font-geist-mono, monospace)", fontSize: 12, color: "var(--accent-red-text)", textDecoration: "none" }}
                    >
                      {r.run_id.slice(0, 14)}…
                    </Link>
                  </td>
                  <td><StatusBadge status={r.run_type} size="sm" /></td>
                  <td style={{ color: "var(--text-secondary)", fontSize: 12 }}>{fmt(r.duration_ms)}</td>
                  <td style={{ color: "var(--text-secondary)", fontSize: 12 }}>{relativeTime(r.start_ts)}</td>
                </tr>
              ))}
              {failedRuns.length === 0 && (
                <tr><td colSpan={4} style={{ color: "var(--text-secondary)", textAlign: "center", padding: 24 }}>No failures</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick access */}
      <div
        className="animate-enter animate-enter-delay-4"
        style={{ marginTop: 32, display: "flex", gap: 12 }}
      >
        <Link href="/realtime" className="btn-primary" style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent-red-bg)", display: "inline-block" }} />
          Open Live Stream
        </Link>
        <Link href="/runs" className="btn-ghost" style={{ textDecoration: "none", display: "inline-flex", alignItems: "center" }}>
          Browse executions
        </Link>
      </div>
    </Container>
  );
}
