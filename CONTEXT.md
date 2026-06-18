# Knowledge Graph Learning Context

This context describes the language of a knowledge graph system that ingests source material, grounds learning evidence, and drives adaptive study sessions.

## Language

**Trace**:
A bounded reconstruction of one execution that lets a person follow what happened and why from start to finish.
_Avoid_: Log stream, request log, session transcript

**Canonical Trace**:
A trace recorded by the current trace contract for a new execution.
_Avoid_: Reconstructed log, inferred trace, migrated trace

**Trace API**:
The dedicated API surface for listing, reading, and exporting canonical traces.
_Avoid_: Observability API, operational log endpoint

**Trace Contract**:
The backend-owned structure and semantics that make canonical traces deterministic, readable, and testable.
_Avoid_: UI convention, log format

**Trace Slice**:
A narrow end-to-end implementation of canonical tracing that proves the contract across backend, API, and UI before every ingestion step is instrumented.
_Avoid_: Partial backend-only change, full instrumentation pass

**Trace Step**:
A complete domain decision or transformation inside a trace, with its relevant input and outcome understood together.
_Avoid_: Start log, end log, raw operation

**Trace Decision**:
A relevant item-level outcome nested under a trace step, such as a concept being rejected or a piece of evidence being approved.
_Avoid_: Query, helper call, validation detail

**Forensic Sequence**:
The immutable order in which trace material was recorded during an execution.
_Avoid_: Display order, step number

**Reading Order**:
The presentation order that helps a person follow a trace while preserving the truth of the forensic sequence.
_Avoid_: Stored sequence, rewritten history

**Trace Buffer**:
An in-memory collection of trace material for one execution that is persisted when the execution closes.
_Avoid_: Streaming trace, operational log

**Trace Blind Spot**:
The accepted loss of a buffered trace when a process dies before the execution can close and persist its trace.
_Avoid_: Silent failure, ignored error

**Trace Failure**:
A failure to record or persist canonical trace material that does not change the domain result of the execution.
_Avoid_: Ingestion failure, job failure

**Trace Status**:
The semantic outcome of a trace step, such as success, emptiness, fallback, review need, or failure.
_Avoid_: Log level, severity

**Trace Copy**:
Spanish human-facing wording used to label and summarize trace steps and decisions, produced deterministically by the system.
_Avoid_: LLM-written summary, raw event name, English debug message

**Operational Log**:
A low-level operational record used to inspect runtime behavior, failures, or service health.
_Avoid_: Trace, execution narrative

**Semantic Effect**:
A domain-meaningful outcome of an execution step, such as creating concepts, rejecting weak evidence, or linking graph material.
_Avoid_: Query result, raw database call

**Boundary Payload**:
The exact material that crosses a boundary with an external or non-deterministic system. For LLM boundaries this means the full request input and full response output.
_Avoid_: Payload shape, summary, inferred prompt

**Silent Truncation**:
Losing part of a boundary payload without making that loss explicit to the trace reader.
_Avoid_: Trace detail, explicit size failure

**Trace Detail**:
Expandable forensic material that supports a trace step or decision without overwhelming the main reading path.
_Avoid_: Primary copy, raw main view

**Trace Summary**:
The closing aggregate for a trace, used to scan the execution outcome without rereading every step.
_Avoid_: Live rollup, frontend recount

**Trace Export**:
A plain-text representation of a trace that mirrors the on-screen reading order, indentation, copy, and details.
_Avoid_: Raw JSON dump, log download

**Operational Detail**:
A technical runtime detail that helps diagnose infrastructure but does not explain the domain outcome of an execution.
_Avoid_: Semantic effect, trace step

**Execution**:
A finite unit of work with a clear beginning and ending, such as processing an ingestion job or handling one adaptive learning action.
_Avoid_: Session, lifecycle, workflow

**Trace Identity**:
The identity of the canonical trace itself, distinct from the domain execution it describes.
_Avoid_: Job identity, episode identity

**Adaptive Session**:
A learner-facing study continuity that can span several executions across multiple blocks or submissions.
_Avoid_: Trace, execution

**Ingestion Job**:
An execution that turns one submitted knowledge fragment into graph material and grounded learning evidence.
_Avoid_: Fragment, episode, import
