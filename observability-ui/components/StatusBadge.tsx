interface Props { status: string; size?: "sm" | "md" }

const MAP: Record<string, string> = {
  running: "badge-running",
  completed: "badge-completed",
  failed: "badge-failed",
  ingestion: "badge-ingestion",
  adaptive: "badge-adaptive",
  request: "badge-request",
  INFO: "badge-request",
  WARNING: "badge-ingestion",
  ERROR: "badge-failed",
  DEBUG: "badge-request",
};

export default function StatusBadge({ status, size = "md" }: Props) {
  const cls = MAP[status] ?? "badge-request";
  return (
    <span
      className={`badge ${cls}`}
      style={size === "sm" ? { fontSize: 10, padding: "1px 6px" } : undefined}
    >
      {status}
    </span>
  );
}
