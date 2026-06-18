"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Overview" },
  { href: "/runs", label: "Executions" },
  { href: "/traces", label: "Traces" },
  { href: "/realtime", label: "Live" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "0 32px",
          height: 52,
          display: "flex",
          alignItems: "center",
          gap: 32,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-geist-mono, monospace)",
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text-primary)",
            letterSpacing: "-0.01em",
          }}
        >
          kg / obs
        </span>
        <nav style={{ display: "flex", gap: 4, flex: 1 }}>
          {links.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                style={{
                  padding: "4px 10px",
                  borderRadius: 4,
                  fontSize: 13,
                  fontWeight: active ? 500 : 400,
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                  background: active ? "var(--canvas)" : "transparent",
                  textDecoration: "none",
                  transition: "color 150ms ease, background 150ms ease",
                }}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: "var(--accent-green-text)",
          }}
          title="Connected"
        />
      </div>
    </header>
  );
}
