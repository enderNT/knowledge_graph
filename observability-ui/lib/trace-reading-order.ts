import type { TraceEvent } from "@/lib/api";

export type TraceReadingOrderRow = {
  event: TraceEvent;
  depth: number;
  index: number;
};

export function traceReadingOrder(events: TraceEvent[]): TraceReadingOrderRow[] {
  const byParent = new Map<string | null, TraceEvent[]>();
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    const key = event.parent_event_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), event]);
  }

  const rows: TraceReadingOrderRow[] = [];
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
