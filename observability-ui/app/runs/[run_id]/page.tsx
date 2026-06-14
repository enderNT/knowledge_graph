"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { use } from "react";
import Container from "@/components/Container";
import StatusBadge from "@/components/StatusBadge";
import PayloadTree from "@/components/PayloadTree";
import { getRunDetail, type LogEvent, type LogRun } from "@/lib/api";

function fmt(ms?: number | null) {
  if (ms == null) return null;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function offset(baseMs: number, eventMs: number) {
  const d = eventMs - baseMs;
  if (d < 0) return "+0ms";
  if (d < 1000) return `+${d}ms`;
  return `+${(d / 1000).toFixed(2)}s`;
}

function fmtTs(ts?: string) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3,
  } as Intl.DateTimeFormatOptions);
}

function levelColor(level: string) {
  if (level === "ERROR" || level === "CRITICAL") return "var(--accent-red-text)";
  if (level === "WARNING") return "var(--accent-yellow-text)";
  if (level === "DEBUG") return "var(--text-tertiary)";
  return "var(--text-primary)";
}

function levelDot(level: string) {
  if (level === "ERROR" || level === "CRITICAL") return "var(--accent-red-text)";
  if (level === "WARNING") return "var(--accent-yellow-text)";
  if (level === "DEBUG") return "var(--text-tertiary)";
  return "#c8c5bf";
}

function EventRow({ event, baseMs }: { event: LogEvent; baseMs: number }) {
  const [open, setOpen] = useState(false);
  const hasPayload = event.input_shape != null || event.output_shape != null || event.counts != null || !!event.error_message;
  const isError = event.level === "ERROR" || event.level === "CRITICAL";

  return (
    <div
      style={{
        borderLeft: `2px solid ${isError ? "var(--accent-red-text)" : "var(--border)"}`,
        paddingLeft: 16,
        paddingBottom: 20,
        position: "relative",
      }}
    >
      {/* Timeline dot */}
      <div
        style={{
          position: "absolute",
          left: -5,
          top: 5,
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: levelDot(event.level),
          border: "2px solid var(--canvas)",
        }}
      />

      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, flexWrap: "wrap" }}>
        {/* Offset */}
        <span
          style={{
            fontFamily: "var(--font-geist-mono, monospace)",
            fontSize: 11,
            color: "var(--text-tertiary)",
            minWidth: 64,
            paddingTop: 1,
          }}
        >
          {offset(baseMs, event.ts_epoch_ms)}
        </span>

        {/* Step tag */}
        {event.step && (
          <span
            style={{
              fontFamily: "var(--font-geist-mono, monospace)",
              fontSize: 10,
              background: "var(--canvas)",
              border: "1px solid var(--border)",
              borderRadius: 3,
              padding: "1px 5px",
              color: "var(--text-secondary)",
            }}
          >
            {event.step}
          </span>
        )}

        {/* Event message */}
        <button
          onClick={() => hasPayload && setOpen((v) => !v)}
          style={{
            background: "none",
            border: "none",
            cursor: hasPayload ? "pointer" : "default",
            padding: 0,
            textAlign: "left",
            flex: 1,
          }}
        >
          <span
            style={{
              fontSize: 13,
              color: levelColor(event.level),
              fontWeight: isError ? 500 : 400,
            }}
          >
            {event.event}
          </span>
          {event.duration_ms != null && (
            <span style={{ marginLeft: 8, fontSize: 11, color: "var(--text-tertiary)", fontFamily: "var(--font-geist-mono, monospace)" }}>
              {fmt(event.duration_ms)}
            </span>
          )}
          {hasPayload && (
            <span style={{ marginLeft: 6, fontSize: 10, color: "var(--text-tertiary)" }}>
              {open ? "▼" : "▶"}
            </span>
          )}
        </button>

        {/* Logger */}
        <span style={{ fontSize: 11, color: "var(--text-tertiary)", fontFamily: "var(--font-geist-mono, monospace)", paddingTop: 1 }}>
          {event.logger_name}
        </span>
      </div>

      {/* Expanded payload */}
      {open && (
        <div style={{ marginTop: 8, marginLeft: 74 }}>
          {event.error_message && (
            <div
              style={{
                padding: "8px 12px",
                background: "var(--accent-red-bg)",
                borderRadius: 6,
                fontSize: 12,
                fontFamily: "var(--font-geist-mono, monospace)",
                color: "var(--accent-red-text)",
                marginBottom: 6,
              }}
            >
              <strong>{event.error_type}:</strong> {event.error_message}
            </div>
          )}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <PayloadTree label="Input" data={event.input_shape} side="input" />
            <PayloadTree label="Output" data={event.output_shape} side="output" />
            <PayloadTree label="Counts" data={event.counts} side="output" />
          </div>
        </div>
      )}
    </div>
  );
}

const DB_INTERNAL_LOGGERS = new Set(["app.arcadedb_client"]);
const DB_INTERNAL_EVENTS = new Set(["arcadedb error", "safe query failed, returning empty result"]);

function isDbInternal(event: LogEvent) {
  return DB_INTERNAL_LOGGERS.has(event.logger_name ?? "") || DB_INTERNAL_EVENTS.has(event.event ?? "");
}

function groupByStep(events: LogEvent[]) {
  const groups: { step: string; events: LogEvent[] }[] = [];
  let current: { step: string; events: LogEvent[] } | null = null;
  for (const e of events) {
    const s = e.step || "_";
    if (!current || current.step !== s) {
      current = { step: s, events: [] };
      groups.push(current);
    }
    current.events.push(e);
  }
  return groups;
}

export default function RunDetailPage({ params }: { params: Promise<{ run_id: string }> }) {
  const { run_id } = use(params);
  const [run, setRun] = useState<LogRun | null>(null);
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  useEffect(() => {
    getRunDetail(run_id)
      .then((d) => {
        setRun(d.run);
        setEvents(d.events);
        const steps = new Set(d.events.map((e) => e.step || "_"));
        setExpandedSteps(steps);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [run_id]);

  const baseMs = events[0]?.ts_epoch_ms ?? Date.now();
  const businessEvents = events.filter((e) => !isDbInternal(e));
  const dbEvents = events.filter((e) => isDbInternal(e));
  const groups = groupByStep(businessEvents);
  const errorCount = businessEvents.filter((e) => e.level === "ERROR" || e.level === "CRITICAL").length;
  const [dbOpen, setDbOpen] = useState(false);

  function toggleStep(step: string) {
    setExpandedSteps((prev) => {
      const n = new Set(prev);
      if (n.has(step)) n.delete(step);
      else n.add(step);
      return n;
    });
  }

  return (
    <Container maxWidth={960}>
      {/* Breadcrumb */}
      <div className="animate-enter" style={{ marginBottom: 24 }}>
        <Link href="/runs" style={{ fontSize: 13, color: "var(--text-secondary)", textDecoration: "none" }}>
          ← Executions
        </Link>
      </div>

      {/* Header */}
      <div className="card animate-enter" style={{ padding: "20px 24px", marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontFamily: "var(--font-geist-mono, monospace)", fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
              {run_id}
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {run && <StatusBadge status={run.run_type || "request"} />}
              {run && <StatusBadge status={run.status} />}
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, auto)", gap: "4px 24px", fontSize: 12, textAlign: "right" }}>
            <span style={{ color: "var(--text-secondary)" }}>Start</span>
            <span style={{ color: "var(--text-secondary)" }}>End</span>
            <span style={{ color: "var(--text-secondary)" }}>Duration</span>
            <span style={{ fontFamily: "var(--font-geist-mono, monospace)" }}>{fmtTs(run?.start_ts)}</span>
            <span style={{ fontFamily: "var(--font-geist-mono, monospace)" }}>{fmtTs(run?.end_ts)}</span>
            <span style={{ fontFamily: "var(--font-geist-mono, monospace)" }}>{fmt(run?.duration_ms) ?? "—"}</span>
          </div>
        </div>

        {/* Meta chips */}
        {(run?.job_id || run?.session_id || run?.episode_id) && (
          <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap", paddingTop: 14, borderTop: "1px solid var(--border)" }}>
            {run.job_id && <Chip label="Job" value={run.job_id} />}
            {run.session_id && <Chip label="Session" value={run.session_id} />}
            {run.episode_id && <Chip label="Episode" value={run.episode_id} />}
          </div>
        )}

        {/* Stats row */}
        <div style={{ marginTop: 14, display: "flex", gap: 20, paddingTop: 14, borderTop: "1px solid var(--border)", fontSize: 12 }}>
          <span><strong>{businessEvents.length}</strong> <span style={{ color: "var(--text-secondary)" }}>events</span></span>
          {errorCount > 0 && (
            <span style={{ color: "var(--accent-red-text)" }}><strong>{errorCount}</strong> errors</span>
          )}
          {dbEvents.length > 0 && (
            <span style={{ color: "var(--text-tertiary)" }}>
              <strong>{dbEvents.length}</strong> db internal{dbEvents.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      {/* Timeline */}
      {loading && <div style={{ color: "var(--text-secondary)", textAlign: "center", padding: 48 }}>Loading events…</div>}

      {!loading && events.length === 0 && (
        <div style={{ color: "var(--text-secondary)", textAlign: "center", padding: 48 }}>No events found for this run</div>
      )}

      {!loading && dbEvents.length > 0 && (
        <div className="animate-enter animate-enter-delay-2" style={{ marginTop: 24 }}>
          <button
            onClick={() => setDbOpen((v) => !v)}
            style={{
              width: "100%",
              background: "none",
              border: "none",
              borderTop: "1px solid var(--border)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "12px 0",
            }}
          >
            <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{dbOpen ? "▼" : "▶"}</span>
            <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-tertiary)", letterSpacing: "0.04em", textTransform: "uppercase" }}>
              DB Internals
            </span>
            <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
              {dbEvents.length} event{dbEvents.length !== 1 ? "s" : ""}
            </span>
            {dbEvents.some((e) => e.level === "ERROR") && (
              <span
                style={{
                  fontSize: 10,
                  padding: "1px 6px",
                  background: "var(--accent-red-bg)",
                  color: "var(--accent-red-text)",
                  borderRadius: 3,
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  fontWeight: 600,
                }}
              >
                {dbEvents.filter((e) => e.level === "ERROR").length} error{dbEvents.filter((e) => e.level === "ERROR").length !== 1 ? "s" : ""}
              </span>
            )}
          </button>
          {dbOpen && (
            <div style={{ paddingLeft: 8, opacity: 0.8 }}>
              {dbEvents.map((event, i) => (
                <EventRow key={`db-${event.ts_epoch_ms}-${i}`} event={event} baseMs={baseMs} />
              ))}
            </div>
          )}
        </div>
      )}

      {!loading && businessEvents.length === 0 && events.length > 0 && (
        <div style={{ color: "var(--text-secondary)", textAlign: "center", padding: 48 }}>
          No business events — all {events.length} events are DB internals
        </div>
      )}

      {!loading && businessEvents.length > 0 && (
        <div className="animate-enter animate-enter-delay-1">
          {groups.map((group) => {
            const isExpanded = expandedSteps.has(group.step);
            const hasErrors = group.events.some((e) => e.level === "ERROR");
            return (
              <div key={group.step} style={{ marginBottom: 4 }}>
                {/* Step header */}
                <button
                  onClick={() => toggleStep(group.step)}
                  style={{
                    width: "100%",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 0",
                    marginBottom: 2,
                  }}
                >
                  <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{isExpanded ? "▼" : "▶"}</span>
                  <span
                    style={{
                      fontFamily: "var(--font-geist-mono, monospace)",
                      fontSize: 11,
                      fontWeight: 600,
                      color: hasErrors ? "var(--accent-red-text)" : "var(--text-secondary)",
                      letterSpacing: "0.04em",
                      textTransform: "uppercase",
                    }}
                  >
                    {group.step === "_" ? "unscoped" : group.step}
                  </span>
                  <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
                    {group.events.length} event{group.events.length !== 1 ? "s" : ""}
                  </span>
                  {hasErrors && <span className="badge badge-failed" style={{ fontSize: 9, padding: "1px 5px" }}>error</span>}
                </button>

                {isExpanded && (
                  <div style={{ paddingLeft: 8 }}>
                    {group.events.map((event, i) => (
                      <EventRow key={`${event.ts_epoch_ms}-${i}`} event={event} baseMs={baseMs} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Container>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 8px",
        background: "var(--canvas)",
        border: "1px solid var(--border)",
        borderRadius: 4,
        fontSize: 11,
        fontFamily: "var(--font-geist-mono, monospace)",
      }}
    >
      <span style={{ color: "var(--text-secondary)", fontFamily: "inherit" }}>{label}</span>
      <span style={{ color: "var(--text-primary)" }}>{value}</span>
    </span>
  );
}
