"use client";
import { useState } from "react";

interface Props { label: string; data: unknown; side?: "input" | "output" }

function isContainer(v: unknown): v is Record<string, unknown> | unknown[] {
  return v !== null && typeof v === "object";
}

function valueColor(v: unknown): string {
  if (typeof v === "string") return "var(--accent-green-text)";
  if (typeof v === "number") return "var(--accent-blue-text)";
  if (typeof v === "boolean") return "var(--accent-yellow-text)";
  if (v === null) return "var(--text-tertiary)";
  return "var(--text-primary)";
}

function fmtPrimitive(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "string") return v.length ? `"${v}"` : '""';
  return String(v);
}

function Node({ k, value, depth }: { k: string | number | null; value: unknown; depth: number }) {
  // Auto-expand the first two levels so the most useful data is visible at a glance.
  const [open, setOpen] = useState(depth < 2);
  const indent = depth * 14;

  const keyNode =
    k === null ? null : (
      <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>{k}</span>
    );

  if (!isContainer(value)) {
    return (
      <div style={{ paddingLeft: indent, display: "flex", gap: 6, lineHeight: 1.7 }}>
        {keyNode}
        {k !== null && <span style={{ color: "var(--text-tertiary)" }}>:</span>}
        <span style={{ color: valueColor(value), wordBreak: "break-word" }}>{fmtPrimitive(value)}</span>
      </div>
    );
  }

  const entries: [string | number, unknown][] = Array.isArray(value)
    ? value.map((v, i) => [i, v])
    : Object.entries(value);
  const summary = Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`;

  if (entries.length === 0) {
    return (
      <div style={{ paddingLeft: indent, display: "flex", gap: 6, lineHeight: 1.7 }}>
        {keyNode}
        {k !== null && <span style={{ color: "var(--text-tertiary)" }}>:</span>}
        <span style={{ color: "var(--text-tertiary)" }}>{summary}</span>
      </div>
    );
  }

  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          paddingLeft: indent,
          background: "none",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 6,
          lineHeight: 1.7,
          font: "inherit",
          color: "inherit",
        }}
      >
        <span style={{ fontSize: 8, color: "var(--text-tertiary)", width: 8 }}>{open ? "▼" : "▶"}</span>
        {keyNode}
        {k !== null && <span style={{ color: "var(--text-tertiary)" }}>:</span>}
        <span style={{ color: "var(--text-tertiary)" }}>{summary}</span>
      </button>
      {open && entries.map(([ck, cv]) => <Node key={String(ck)} k={ck} value={cv} depth={depth + 1} />)}
    </div>
  );
}

export default function PayloadTree({ label, data, side = "input" }: Props) {
  const [open, setOpen] = useState(false);
  if (data === undefined || data === null) return null;
  if (isContainer(data) && (Array.isArray(data) ? data.length === 0 : Object.keys(data).length === 0)) return null;

  const accent = side === "input" ? "var(--accent-blue-text)" : "var(--accent-green-text)";
  const bg = side === "input" ? "var(--accent-blue-bg)" : "var(--accent-green-bg)";

  const rootEntries: [string | number, unknown][] | null = isContainer(data)
    ? Array.isArray(data)
      ? data.map((v, i) => [i, v])
      : Object.entries(data)
    : null;

  return (
    <div style={{ marginTop: 6, minWidth: 220 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: 0,
          color: accent,
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        }}
      >
        <span style={{ fontSize: 9 }}>{open ? "▼" : "▶"}</span>
        {label}
      </button>
      {open && (
        <div
          style={{
            marginTop: 6,
            padding: "10px 12px",
            background: bg,
            borderRadius: 6,
            fontSize: 11.5,
            fontFamily: "var(--font-geist-mono, monospace)",
            color: "var(--text-primary)",
            overflowX: "auto",
          }}
        >
          {rootEntries
            ? rootEntries.map(([k, v]) => <Node key={String(k)} k={k} value={v} depth={0} />)
            : <Node k={null} value={data} depth={0} />}
        </div>
      )}
    </div>
  );
}
