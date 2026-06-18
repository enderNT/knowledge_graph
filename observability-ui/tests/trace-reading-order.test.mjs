import assert from "node:assert/strict";
import test from "node:test";

import { traceReadingOrder } from "../lib/trace-reading-order.ts";

const baseEvent = {
  trace_id: "tr_ui",
  type: "concepts_resolved",
  role: "step",
  status: "succeeded",
  title: "Conceptos resueltos",
  summary: "",
  input: {},
  output: {},
  detail: {},
  created_at: "2026-06-17T00:00:00+00:00",
};

test("traceReadingOrder renders parents before children without mutating forensic sequence", () => {
  const events = [
    { ...baseEvent, event_id: "te_child", parent_event_id: "te_parent", sequence: 1, role: "decision" },
    { ...baseEvent, event_id: "te_parent", parent_event_id: null, sequence: 2 },
    { ...baseEvent, event_id: "te_sibling", parent_event_id: null, sequence: 3 },
  ];

  const rows = traceReadingOrder(events);

  assert.deepEqual(rows.map((row) => row.event.event_id), ["te_parent", "te_child", "te_sibling"]);
  assert.deepEqual(rows.map((row) => row.depth), [0, 1, 0]);
  assert.deepEqual(rows.map((row) => row.index), [1, 2, 3]);
  assert.deepEqual(events.map((event) => event.sequence), [1, 2, 3]);
});
