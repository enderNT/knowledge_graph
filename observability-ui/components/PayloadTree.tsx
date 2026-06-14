"use client";
import { useState } from "react";

interface Props { label: string; json: string | undefined; side?: "input" | "output" }

export default function PayloadTree({ label, json, side = "input" }: Props) {
  const [open, setOpen] = useState(false);
  if (!json) return null;
  let parsed: unknown;
  try { parsed = JSON.parse(json); } catch { return null; }

  const accent = side === "input" ? "var(--accent-blue-text)" : "var(--accent-green-text)";
  const bg = side === "input" ? "var(--accent-blue-bg)" : "var(--accent-green-bg)";

  return (
    <div style={{ marginTop: 6 }}>
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
        <pre
          style={{
            margin: "6px 0 0",
            padding: "10px 12px",
            background: bg,
            borderRadius: 6,
            fontSize: 11,
            fontFamily: "var(--font-geist-mono, monospace)",
            color: "var(--text-primary)",
            overflowX: "auto",
            lineHeight: 1.5,
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {JSON.stringify(parsed, null, 2)}
        </pre>
      )}
    </div>
  );
}
