"use client";
import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import Container from "@/components/Container";
import PayloadTree from "@/components/PayloadTree";
import StatusBadge from "@/components/StatusBadge";
import { getTraceDetail, type CanonicalTrace, type TraceEvent } from "@/lib/api";

function fmt(ms?: number | null) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtTs(ts?: string) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3,
  } as Intl.DateTimeFormatOptions);
}

function hasData(value: unknown) {
  if (value == null) return false;
  if (typeof value === "object") return Object.keys(value as Record<string, unknown>).length > 0;
  return true;
}

function readingOrder(events: TraceEvent[]) {
  const byParent = new Map<string | null, TraceEvent[]>();
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    const key = event.parent_event_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), event]);
  }
  const rows: { event: TraceEvent; depth: number; index: number }[] = [];
  let index = 1;
  function visit(parent: string | null, depth: number) {
    for (const event of byParent.get(parent) ?? []) {
      rows.push({ event, depth, index });
      index += 1;
      visit(event.event_id, depth + 1);
    }
  }
  visit(null, 0);
  return rows;
}

export default function TraceDetailPage({ params }: { params: Promise<{ trace_id: string }> }) {
  const { trace_id } = use(params);
  const [trace, setTrace] = useState<CanonicalTrace | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTraceDetail(trace_id)
      .then(setTrace)
      .catch(() => setTrace(null))
      .finally(() => setLoading(false));
  }, [trace_id]);

  const rows = useMemo(() => readingOrder(trace?.events ?? []), [trace]);

  return (
    <Container maxWidth={1040}>
      <div className="animate-enter" style={{ marginBottom: 24 }}>
        <Link href="/traces" style={{ fontSize: 13, color: "var(--text-secondary)", textDecoration: "none" }}>← Traces</Link>
      </div>

      {loading && <div style={{ color: "var(--text-secondary)", textAlign: "center", padding: 48 }}>Loading trace…</div>}
      {!loading && !trace && <div style={{ color: "var(--text-secondary)", textAlign: "center", padding: 48 }}>Trace not found</div>}

      {trace && (
        <>
          <div className="card animate-enter" style={{ padding: "20px 24px", marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <div className="mono" style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>{trace.summary.trace_id}</div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <StatusBadge status={trace.summary.execution_type} />
                  <StatusBadge status={trace.summary.status} />
                </div>
              </div>
              <a className="btn-ghost" href={`/api/traces/${trace.summary.trace_id}/export`} style={{ textDecoration: "none" }}>Export text</a>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, auto)", gap: "4px 24px", fontSize: 12, textAlign: "right" }}>
                <span style={{ color: "var(--text-secondary)" }}>Started</span>
                <span style={{ color: "var(--text-secondary)" }}>Ended</span>
                <span style={{ color: "var(--text-secondary)" }}>Duration</span>
                <span className="mono">{fmtTs(trace.summary.started_at)}</span>
                <span className="mono">{fmtTs(trace.summary.ended_at ?? undefined)}</span>
                <span className="mono">{fmt(trace.summary.duration_ms)}</span>
              </div>
            </div>
            <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap", paddingTop: 14, borderTop: "1px solid var(--border)" }}>
              <Chip label="Execution" value={trace.summary.execution_id} />
              {trace.summary.episode_id && <Chip label="Episode" value={trace.summary.episode_id} />}
              {trace.summary.domain && <Chip label="Domain" value={trace.summary.domain} />}
            </div>
            <div style={{ marginTop: 14, display: "flex", gap: 20, paddingTop: 14, borderTop: "1px solid var(--border)", fontSize: 12 }}>
              <span><strong>{trace.summary.total_steps}</strong> <span style={{ color: "var(--text-secondary)" }}>steps</span></span>
              <span><strong>{trace.summary.total_decisions}</strong> <span style={{ color: "var(--text-secondary)" }}>decisions</span></span>
              {trace.summary.error_count > 0 && <span style={{ color: "var(--accent-red-text)" }}><strong>{trace.summary.error_count}</strong> errors</span>}
            </div>
          </div>

          <div className="animate-enter animate-enter-delay-1">
            {rows.map(({ event, depth, index }) => (
              <TraceEventRow key={event.event_id} event={event} depth={depth} index={index} />
            ))}
          </div>
        </>
      )}
    </Container>
  );
}

function TraceEventRow({ event, depth, index }: { event: TraceEvent; depth: number; index: number }) {
  const [open, setOpen] = useState(false);
  const hasPayload = hasData(event.input) || hasData(event.output) || hasData(event.detail) || !!event.boundary_payload;
  const borderColor = event.status === "failed" ? "var(--accent-red-text)" : event.status === "needs_review" ? "var(--accent-yellow-text)" : "var(--border)";

  return (
    <div style={{ marginLeft: depth * 28, borderLeft: `2px solid ${borderColor}`, paddingLeft: 16, paddingBottom: 18, position: "relative" }}>
      <div style={{ position: "absolute", left: -5, top: 5, width: 8, height: 8, borderRadius: "50%", background: borderColor, border: "2px solid var(--canvas)" }} />
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start", flexWrap: "wrap" }}>
        <span className="mono" style={{ fontSize: 11, minWidth: 32, color: "var(--text-tertiary)", paddingTop: 2 }}>{index}</span>
        <StatusBadge status={event.status} size="sm" />
        <button
          onClick={() => hasPayload && setOpen((v) => !v)}
          style={{ background: "none", border: "none", padding: 0, cursor: hasPayload ? "pointer" : "default", textAlign: "left", flex: 1 }}
        >
          <span style={{ fontSize: 14, fontWeight: 500 }}>{event.title}</span>
          <span className="mono" style={{ marginLeft: 8, fontSize: 10, color: "var(--text-tertiary)" }}>{event.type}</span>
          {hasPayload && <span style={{ marginLeft: 6, fontSize: 10, color: "var(--text-tertiary)" }}>{open ? "▼" : "▶"}</span>}
          {event.summary && <div style={{ marginTop: 3, color: "var(--text-secondary)", fontSize: 12 }}>{event.summary}</div>}
        </button>
      </div>
      {open && (
        <div style={{ marginTop: 10, marginLeft: 42, display: "grid", gap: 10 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {hasData(event.input) && <PayloadTree label="Input" data={event.input} side="input" />}
            {hasData(event.output) && <PayloadTree label="Output" data={event.output} side="output" />}
            {hasData(event.detail) && <PayloadTree label="Detail" data={event.detail} side="output" />}
          </div>
          {event.boundary_payload && <BoundaryPayload payload={event.boundary_payload} />}
        </div>
      )}
    </div>
  );
}

function BoundaryPayload({ payload }: { payload: NonNullable<TraceEvent["boundary_payload"]> }) {
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Chip label="Boundary" value={payload.kind} />
        {payload.provider && <Chip label="Provider" value={payload.provider} />}
        {payload.model && <Chip label="Model" value={payload.model} />}
      </div>
      {payload.request_text && <TextBlock label="Input enviado" value={payload.request_text} />}
      {payload.response_text && <TextBlock label="Output recibido" value={payload.response_text} />}
    </div>
  );
}

function TextBlock({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>{label}</div>
      <pre className="scrollbar-thin" style={{ margin: 0, maxHeight: 260, overflow: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", background: "var(--canvas)", border: "1px solid var(--border)", borderRadius: 6, padding: 12, fontSize: 11, lineHeight: 1.5 }}>{value}</pre>
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="mono" style={{ display: "inline-flex", gap: 6, padding: "3px 8px", background: "var(--canvas)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }}>
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span>{value}</span>
    </span>
  );
}
