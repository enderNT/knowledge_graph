"use client";
import { useEffect, useRef, useState } from "react";
import Container from "@/components/Container";
import StatusBadge from "@/components/StatusBadge";
import PayloadTree from "@/components/PayloadTree";
import { openStream, type LogEvent } from "@/lib/api";

const MAX_ROWS = 500;

function levelColor(level: string) {
  if (level === "ERROR" || level === "CRITICAL") return "var(--accent-red-text)";
  if (level === "WARNING") return "var(--accent-yellow-text)";
  if (level === "DEBUG") return "var(--text-tertiary)";
  return "var(--text-primary)";
}

function levelBg(level: string) {
  if (level === "ERROR" || level === "CRITICAL") return "rgba(253, 235, 236, 0.5)";
  if (level === "WARNING") return "rgba(251, 243, 219, 0.4)";
  return "transparent";
}

function fmtTs(epochMs: number) {
  return new Date(epochMs).toLocaleTimeString("en-US", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3,
  } as Intl.DateTimeFormatOptions);
}

export default function RealtimePage() {
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [paused, setPaused] = useState(false);
  const [connected, setConnected] = useState(false);
  const [filterLevel, setFilterLevel] = useState("ALL");
  const [filterService, setFilterService] = useState("");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pausedRef = useRef(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const es = openStream({});
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e) => {
      if (pausedRef.current) return;
      try {
        const event: LogEvent = JSON.parse(e.data);
        setEvents((prev) => {
          const next = [...prev, event];
          return next.length > MAX_ROWS ? next.slice(next.length - MAX_ROWS) : next;
        });
      } catch {}
    };

    return () => { es.close(); };
  }, []);

  useEffect(() => {
    if (!paused) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [events, paused]);

  function clear() {
    setEvents([]);
    setExpandedIdx(null);
  }

  const filtered = events.filter((e) => {
    if (filterLevel !== "ALL" && e.level !== filterLevel) return false;
    if (filterService && !(e.logger_name ?? "").includes(filterService)) return false;
    return true;
  });

  return (
    <Container maxWidth={1100}>
      <div className="animate-enter" style={{ marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ fontSize: 24, fontWeight: 600, letterSpacing: "-0.02em" }}>Live Stream</h1>
            <div
              style={{
                width: 8, height: 8, borderRadius: "50%",
                background: connected ? "var(--accent-green-text)" : "var(--accent-red-text)",
                boxShadow: connected ? "0 0 0 2px rgba(52, 101, 56, 0.2)" : "none",
              }}
            />
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>
            {connected ? `Streaming · ${filtered.length} events` : "Connecting…"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn-ghost" onClick={() => setPaused((v) => !v)}>
            {paused ? "Resume" : "Pause"}
          </button>
          <button className="btn-ghost" onClick={clear}>Clear</button>
        </div>
      </div>

      {/* Filters */}
      <div className="animate-enter animate-enter-delay-1" style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        {["ALL", "DEBUG", "INFO", "WARNING", "ERROR"].map((l) => (
          <button
            key={l}
            onClick={() => setFilterLevel(l)}
            style={{
              padding: "4px 10px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: filterLevel === l ? "#111" : "var(--surface)",
              color: filterLevel === l ? "#fff" : "var(--text-secondary)",
              fontSize: 12,
              cursor: "pointer",
              fontWeight: filterLevel === l ? 500 : 400,
            }}
          >
            {l}
          </button>
        ))}
        <input
          type="text"
          placeholder="Filter by logger…"
          value={filterService}
          onChange={(e) => setFilterService(e.target.value)}
          style={{ width: 180 }}
        />
      </div>

      {/* Feed */}
      <div
        className="card scrollbar-thin animate-enter animate-enter-delay-2"
        style={{
          height: "calc(100vh - 260px)",
          minHeight: 400,
          overflowY: "auto",
          fontFamily: "var(--font-geist-mono, monospace)",
        }}
      >
        {filtered.length === 0 && (
          <div style={{ padding: 48, textAlign: "center", color: "var(--text-secondary)" }}>
            {connected ? "Waiting for events…" : "Connecting to stream…"}
          </div>
        )}

        {filtered.map((event, i) => {
          const isOpen = expandedIdx === i;
          const hasPayload = event.input_shape != null || event.output_shape != null || event.counts != null || !!event.error_message;
          return (
            <div
              key={i}
              style={{
                borderBottom: "1px solid var(--border-subtle)",
                background: levelBg(event.level),
                padding: "7px 14px",
                cursor: hasPayload ? "pointer" : "default",
              }}
              onClick={() => hasPayload && setExpandedIdx(isOpen ? null : i)}
            >
              <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{ fontSize: 11, color: "var(--text-tertiary)", whiteSpace: "nowrap", paddingTop: 1 }}>
                  {fmtTs(event.ts_epoch_ms)}
                </span>
                <StatusBadge status={event.level} size="sm" />
                {event.step && (
                  <span style={{ fontSize: 11, background: "var(--canvas)", border: "1px solid var(--border)", borderRadius: 3, padding: "1px 5px", color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                    {event.step}
                  </span>
                )}
                <span style={{ flex: 1, fontSize: 12, color: levelColor(event.level) }}>
                  {event.event}
                </span>
                {event.duration_ms != null && (
                  <span style={{ fontSize: 11, color: "var(--text-tertiary)", whiteSpace: "nowrap" }}>
                    {event.duration_ms}ms
                  </span>
                )}
                <span style={{ fontSize: 10, color: "var(--text-tertiary)", whiteSpace: "nowrap" }}>
                  {event.logger_name}
                </span>
                {hasPayload && <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{isOpen ? "▼" : "▶"}</span>}
              </div>
              {isOpen && (
                <div style={{ paddingTop: 8, paddingLeft: 72 }}>
                  {event.error_message && (
                    <div style={{ padding: "6px 10px", background: "var(--accent-red-bg)", borderRadius: 4, fontSize: 11, color: "var(--accent-red-text)", marginBottom: 6 }}>
                      <strong>{event.error_type}:</strong> {event.error_message}
                    </div>
                  )}
                  <div style={{ display: "flex", gap: 16 }}>
                    <PayloadTree label="Input" data={event.input_shape} side="input" />
                    <PayloadTree label="Output" data={event.output_shape} side="output" />
                    <PayloadTree label="Counts" data={event.counts} side="output" />
                  </div>
                </div>
              )}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </Container>
  );
}
