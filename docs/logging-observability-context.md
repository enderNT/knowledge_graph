# Logging Observability Context

## Objetivo

Contexto logs para volver viaje de datos en visualizacion operativa. Foco: datos entran por MCP, cruzan servicios, cambian estado, vuelven cliente. Logs deben permitir realtime + historico por `job_id`, `session_id`, `request_id`.

No cambio codigo. Propuesta define logs existentes (`E`) + logs faltantes/propuestos (`P`) para UI futura.

## Base existente

App ya tiene logging estructurado:

- `app/logging_config.py`: `JsonFormatter` emite `ts`, `level`, `logger`, `msg`, `run_id`, `step`, extras.
- `app/trace.py`: `bind(run_id=..., step=...)` propaga contexto async.
- Logs fuertes ya viven en ingesta, AI provider, worker, MCP clients, ArcadeDB client, auth/dependencies, adaptive learning.

Problema no es falta total logs. Falta logging en boundaries + cambios estado: entrada/salida tool, route latency, payload shape, status transition, response shape.

## Modelo mental

- Ingesta: `job_id` = `run_id`.
- Adaptive: `session_id` = `run_id`.
- HTTP general: `x-request-id` / `req_id` = `run_id`.
- MCP tools: loggear `tool_name`, `input_shape`, `output_shape`, `status`, `duration_ms`.
- Store/DB: loggear `entity`, `operation`, `old_status`, `new_status`, `duration_ms`.

UI agrupa por `run_id`. Logs crudos pasan a eventos normalizados.

## Flujo 1: ingesta por MCP

Entrada: `kg_add_knowledge_fragment` con `text`, `source_type`, `tags`, `language`, `temporal`, `expires_at`.

Viaje:

1. Cliente MCP llama `kg_add_knowledge_fragment`.
2. `agent-mcp` recibe tool.
3. `AgentMCPUpstreamClient` llama `add_knowledge_fragment`.
4. `knowledge-mcp` recibe tool interno.
5. `MCPBackendClient` llama `POST /v1/knowledge/fragments`.
6. REST valida payload.
7. `IngestionService.submit_fragment` crea `Episode` + `IngestionJob`.
8. Queue recibe `job_id`.
9. Worker hace dequeue.
10. `process_job` ejecuta embed -> extract -> vet -> resolve concepts -> claims -> evidence -> relations.
11. Store actualiza episode/job a `processed` o `failed`.
12. MCP backend poll `GET /v1/jobs/{job_id}`.
13. Cliente recibe `{ status, episode_id, job_id, result | error }`.

Logs existentes buenos:

- `fragment queued`
- `job started`
- `job completed`
- `job failed`
- `episode embedded`
- `extraction complete`
- `extraction vetted`
- `concept resolved`
- `claim created`
- `evidence vetted`
- `llm extract done/failed`
- `llm vet done/failed`
- `embed timeout/http error`
- `backend request failed`
- `mcp tool backend error`
- `agent mcp tool upstream error`

Huecos mas valiosos:

- `kg_add_knowledge_fragment.start` + `.success`, sin `text` completo.
- REST accepted/rejected con `input_shape`.
- `job_dequeued`, `queue_wait_ms`.
- `job.status queued -> processing -> completed|failed`.
- Poll status change en `ingest_fragment_and_wait`.
- Graph write summary: concepts, claims, evidence, relations, skips.
- Response boundary antes de volver cliente MCP.

## Flujo 2: sesion adaptativa por MCP

Entrada tools: `start_adaptive_session`, `submit_adaptive_block`, `get_adaptive_session`.

`start_adaptive_session` recibe `user_id`, anchor (`query`, `episode_id`, `job_id`), `study_mode`, `domain_hint`, `language`, `constraints`.

Start viaje:

1. Cliente MCP llama `start_adaptive_session`.
2. `agent-mcp` recibe tool.
3. `AgentMCPUpstreamClient` llama knowledge MCP.
4. `knowledge-mcp` recibe tool.
5. `MCPBackendClient` llama `POST /v1/adaptive/sessions/start`.
6. Router crea `AdaptiveLearningService`.
7. Service resuelve tutor context.
8. Service carga pedagogical context.
9. Service consulta SR candidates.
10. Planner elige concept, dimension, difficulty, question types, scaffolding.
11. Block generation usa LLM o deterministic fallback.
12. Store persiste `AdaptiveSession` + `AdaptiveBlockAttempt`.
13. Cliente recibe `{ session, current_block, planner_explanation, grounding_status }`.

Submit viaje:

1. Cliente manda `session_id`, `block_id`, `submissions`, `interaction_events`.
2. Service carga session.
3. Service valida `block_id`.
4. Service evalua items.
5. SR actualiza quality/ease/interval/due dates.
6. Prereq relief propaga mejoras.
7. Pedagogical state actualiza dimensions.
8. Service decide `next_action`.
9. Service genera `next_block` o cierra session.
10. Store persiste attempt + session update.
11. Cliente recibe `{ session, block_result, updated_context, next_action, next_block, session_closed }`.

Logs existentes buenos:

- `adaptive session starting`
- `adaptive session created`
- `block submission received`
- `block mismatch`
- `llm block generation`
- `llm mcq validation failed, falling back to deterministic`
- `block submission complete`
- `generate_adaptive_block` done/failure fallback
- missing submission field warnings
- bad choice text warnings

Huecos mas valiosos:

- Tool start/success para `start_adaptive_session`, `submit_adaptive_block`, `get_adaptive_session`.
- Route latency/status en `/v1/adaptive/*`.
- Tutor context: anchor, resolved reference, evidence count, failure reason.
- Auto mode: due candidates count, fallback candidates count.
- Planner: target concept, dimension, difficulty, block purpose, question types.
- LLM vs fallback: provider, item count, fallback reason.
- Store: session upsert, attempt upsert, block history length, old/new status.
- Submit scoring: item summary, block score, verdict, next action.
- SR update: `quality_q`, interval old/new, ease old/new, due date old/new.
- Propagation: prerequisite/related concept update count.
- Response boundary: next block vs closed session.

## Event schema recomendado

`LogEvent` minimo:

- `ts`
- `level`
- `service`
- `logger`
- `event`
- `run_id`
- `step`
- `tool_name`
- `path`
- `status`
- `duration_ms`
- `job_id`
- `session_id`
- `episode_id`
- `block_id`
- `input_shape`
- `output_shape`
- `counts`
- `error_type`
- `error_message`

Regla: loggear shape + metadata, no payload completo. Ingesta: no guardar `text` completo. Adaptive: no guardar respuestas completas si pueden ser sensibles; guardar counts, item ids, scores.

## UI propuesta

Dos modos:

- Realtime: `GET /v1/logs/stream`.
- Historico: `GET /v1/logs`.

Filtros:

- `run_id`
- `job_id`
- `session_id`
- `service`
- `level`
- `event`
- `step`
- `since`
- `limit`
- `cursor`

Vistas:

- Live tail.
- Timeline por job.
- Timeline por session.
- Error grouping.
- LLM latency/tokens.
- Store transitions.
- MCP tool latency.
- Adaptive decision trace.

## Prioridad futura

1. Capturar logs existentes como eventos normalizados.
2. Agregar boundary logs en MCP tool start/success/error.
3. Agregar route middleware con request start/end/status/duration.
4. Agregar status transitions en jobs + adaptive sessions.
5. Agregar planner/SR decision logs.
6. Agregar persistence/query duration logs en ArcadeDB client.
7. Persistir `LogEvent`.
8. Exponer `GET /v1/logs` + `GET /v1/logs/stream`.

## Principio de diseno

Logs no son texto debug. Logs son eventos dominio observables. Cada evento responde: que entro, que decision tomo app, que estado cambio, que salio, cuanto tardo, bajo que `run_id`.

Agregar logs en boundaries + cambios estado. Evitar ruido interno. Max visibilidad, minimo volumen.
