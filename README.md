# Knowledge Graph API + MCP

Stack semántico en Python con tres servicios: una API REST privada en `FastAPI`, un `knowledge MCP` upstream y un `agent MCP` público, todos respaldados por `ArcadeDB`. Incluye un motor de aprendizaje adaptativo con spaced repetition (SM-2), contexto pedagógico por usuario y sesiones de estudio generadas on-demand.

## Qué incluye

- API privada con `X-API-Key`
- `knowledge MCP` upstream con bearer token propio en `/mcp`
- `agent MCP` público con bearer token propio en `/mcp`
- Ingesta asíncrona de fragmentos con `Episode` + `IngestionJob`
- Bootstrap idempotente del esquema en ArcadeDB
- Routing por dominio, deduplicación conservadora y `needs_review`
- Motor de aprendizaje adaptativo con 5 modos de estudio (hybrid, backlog, recovery, isolated, auto)
- Spaced repetition SM-2 por concepto y dimensión pedagógica
- Contexto pedagógico global por usuario (4 dimensiones: recognition, recall, explanation, application)
- 28 tools en el agent MCP: 3 pedagógicas grounded + 3 de sesión adaptativa + 22 de inspección y curación

## Topología

```
Claude / Manus / agente externo
  └─→ agent-mcp público (puerto 9100)  Authorization: Bearer <AGENT_MCP_BEARER_TOKEN>
        └─→ knowledge-mcp interno (puerto 9000)  Authorization: Bearer <KNOWLEDGE_MCP_BEARER_TOKEN>
              └─→ api privada (puerto 8000)  X-API-Key: <KG_API_KEY>
                    ├─→ arcadedb (red interna)
                    └─→ anthropic-gateway interno  (cuando AI_PROVIDER=anthropic)
```

Por defecto, en Docker Compose solo `agent-mcp` se publica; `mcp`, `api`, `arcadedb` y `anthropic-gateway` quedan en red interna.

## Endpoints REST

### Salud
- `GET /health/live`
- `GET /health/ready`

### Ingesta y jobs
- `POST /v1/knowledge/fragments`
- `GET /v1/jobs/{job_id}`
- `GET /v1/episodes` — lista paginada con ordenamiento y resumen de conceptos
- `GET /v1/episodes/{episode_id}`

### Búsqueda y contexto
- `POST /v1/search/candidates`
- `POST /v1/search/learning-context`
- `POST /v1/search/tutor-context`

### Grafo
- `PUT /v1/concepts/upsert`
- `POST /v1/concepts/link`
- `GET /v1/concepts/{concept_ref}/neighborhood?depth=1|2`

### Sesiones adaptativas
- `POST /v1/adaptive/sessions/start`
- `POST /v1/adaptive/sessions/{session_id}/submit`
- `GET /v1/adaptive/sessions/{session_id}`

### Contexto pedagógico
- `GET /v1/pedagogical/{user_id}`
- `POST /v1/pedagogical/{user_id}/update`
- `GET /v1/pedagogical/{user_id}/session-view`

### Spaced repetition
- `GET /v1/sr/{user_id}/{concept_uid}/{dimension}`
- `GET /v1/sr/{user_id}/due`
- `POST /v1/sr/update`
- `POST /v1/sr/apply-relief`
- `GET /v1/sr/{user_id}/stats`

## Knowledge MCP Upstream

Endpoint: `GET/POST /mcp`

Tools expuestas:
- `add_knowledge_fragment`
- `reset_knowledge_base`
- `search_candidates`
- `get_learning_context`
- `get_tutor_context`
- `upsert_concept`
- `create_concept`
- `attach_concept_evidence`
- `link_concepts`
- `preview_delete_episode_content` / `delete_episode_content`
- `preview_delete_relation` / `delete_relation`
- `get_neighborhood`
- `get_pedagogical_context` / `update_pedagogical_context` / `get_pedagogical_session_view`
- `get_sr_state` / `get_due_sr_items` / `update_sr_from_block` / `apply_prereq_relief` / `get_sr_stats`
- `start_adaptive_session` / `submit_adaptive_block` / `get_adaptive_session`

`add_knowledge_fragment` encapsula la asincronía del backend: crea el job, hace polling y devuelve `completed`, `failed` o `processing` con `episode_id` y `job_id`.

Parámetros opcionales de temporalidad: `temporal: bool = false` y `expires_at: str | None = null` (ISO-8601). Úsalos solo para contenido que puede quedar obsoleto (docs de APIs, versiones de frameworks, precios). Cuando `expires_at` ha pasado, los ítems SR vinculados a ese episodio aparecen con `is_stale: true` en `get_due_sr_items`.

`get_tutor_context` acepta exactamente una referencia entre `query`, `episode_id` o `job_id`. Devuelve un paquete estructurado y trazable, y falla con `status=failed` + `failure_reason` cuando no hay evidencia suficiente.

## Agent MCP Público

Endpoint: `GET/POST /mcp` — 28 tools en total.

### Tools pedagógicas de alto nivel
- `explain_topic` — Explicación grounded desde el grafo
- `generate_quiz` — Quiz con evidencia trazable
- `evaluate_answer` — Evaluación de respuesta contra el grafo

### Tools de sesión adaptativa
- `start_adaptive_session` — Inicia sesión con primer bloque listo. Acepta `query`, `episode_id`, `job_id`, o ninguno cuando `study_mode="auto"`
- `submit_adaptive_block` — Envía respuestas de un bloque y recibe el siguiente
- `get_adaptive_session` — Consulta estado actual de sesión sin avanzarla

### Tools de inspección (solo lectura)
- `kg_list_episodes` — lista paginada de episodes con conceptos por episode
- `kg_search_candidates`, `kg_get_learning_context`, `kg_get_tutor_context`
- `kg_get_neighborhood`
- `kg_get_pedagogical_context`, `kg_get_pedagogical_session_view`
- `kg_sr_get_state`, `kg_sr_get_due_items`, `kg_sr_get_stats`

### Tools de curación y modificación
- `kg_add_knowledge_fragment`, `kg_upsert_concept`, `kg_create_concept`
- `kg_attach_concept_evidence`, `kg_link_concepts`
- `kg_update_pedagogical_context`
- `kg_sr_update_from_block`, `kg_sr_apply_relief`

### Tools de eliminación
- `kg_preview_delete_episode_content` / `kg_delete_episode_content`
- `kg_preview_delete_relation` / `kg_delete_relation`

### Reset total
- `kg_reset_knowledge_base` — **DESTRUCTIVA.** Borra grafo, estado pedagógico, sesiones y cola de ingesta.

## Modos de estudio

| Modo | Anchor requerido | Comportamiento |
|------|-----------------|----------------|
| `hybrid` | Sí | Mezcla SR pendientes + contenido nuevo del anchor |
| `backlog` | Sí | 100% revisión SR, falla si no hay pendientes |
| `recovery` | Sí | 100% revisión filtrada a items débiles/vencidos |
| `isolated` | Sí + `domain_hint` | Solo el dominio especificado, sin mezcla SR global |
| `auto` | No (opcional) | El motor elige el concepto de mayor prioridad automáticamente |

En modo `auto`, si se pasa un anchor, se respeta y la selección automática no aplica. Si no hay anchor, el motor consulta los SR due items primero; si no hay due, elige el concepto de menor mastery del contexto pedagógico del usuario.

## Motor adaptativo

La sesión se cierra cuando ocurre cualquiera de estas dos condiciones:
1. Se completaron `max_blocks` bloques (default 4, máximo recomendado 12)
2. Resultado correcto y la dimensión más débil del concepto ≥ 70

Las actualizaciones al contexto pedagógico son globales por `user_id` — no están atadas a un episode ni a una sesión. Cada `submit_adaptive_block` aplica la fórmula de actualización de score y SM-2 inmediatamente:

```
nuevo_score = (score_anterior × 0.35) + (resultado_actual × 0.65)
```

SM-2 determina cuándo vuelve a aparecer cada concepto como "due":
- Falla → próxima revisión mañana
- Aprueba parcial → pocos días
- Aprueba bien → intervalo × ease_factor (~2.5x), compoundea con cada repetición exitosa

Para listar todos los episodes disponibles usar `kg_list_episodes`. Ver [STUDY_GUIDE.md](STUDY_GUIDE.md) para la referencia completa de parámetros.

## Desarrollo local

1. Crear `.env` a partir de `.env.example`, o usar `ENV_FILE=.env.example` para una prueba rápida.
2. Levantar con `docker compose up --build`.
3. El `agent MCP` queda en `http://localhost:9100/mcp`.
4. El `knowledge MCP` queda interno en `http://mcp:9000/mcp`.
5. La API REST queda solo dentro de la red interna del compose.

Mientras `AI_PROVIDER=stub`, no hace falta proveedor externo.

Los logs salen en JSON estructurado (un objeto por línea) con `run_id` y `step` en cada entrada. Para rastrear un request concreto, pasa `X-Request-ID: <id>` — ese valor aparece como `run_id` en todos los logs del ciclo de vida de esa petición.

Matriz de proveedores:

| `AI_PROVIDER` | Extracción LLM | Embeddings |
|--------------|---------------|------------|
| `stub` | Heurística local | Locales |
| `openai_compatible` | LLM OpenAI-compatible | OpenAI-compatible |
| `anthropic` | anthropic-gateway | `EMBEDDING_PROVIDER` |

Para usar un proveedor OpenAI-compatible:
- Cambia `AI_PROVIDER=openai_compatible`
- Define `OPENAI_API_KEY`
- Ajusta `EMBEDDING_DIMENSIONS` para que coincida con tu modelo de embeddings

Para usar Anthropic vía gateway Go:
- Cambia `AI_PROVIDER=anthropic`
- Define `EMBEDDING_PROVIDER=stub` o `EMBEDDING_PROVIDER=openai_compatible`
- Define `ANTHROPIC_GATEWAY_BEARER_TOKEN`, `ANTHROPIC_API_KEY`, `ANTHROPIC_CHAT_MODEL`
- Opcionalmente define `ANTHROPIC_THINKING_TYPE`, `ANTHROPIC_THINKING_BUDGET_TOKENS`, `ANTHROPIC_EFFORT`

Ejemplo recomendado para Claude Sonnet 4.6 con esfuerzo medio:

```env
AI_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai_compatible
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
ANTHROPIC_GATEWAY_BEARER_TOKEN=replace-with-an-internal-bearer-token
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6
ANTHROPIC_THINKING_TYPE=adaptive
ANTHROPIC_EFFORT=medium
```

Ejemplo mínimo con stub:

```env
AI_PROVIDER=stub
```

## Variables de entorno

### API (`api`)
- `KG_API_KEY`
- `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` (default `INFO`) — nivel de logging JSON estructurado
- `AI_PROVIDER=stub|openai_compatible|anthropic`
- `EMBEDDING_PROVIDER=stub|openai_compatible`
- `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDINGS_MODEL`, `OPENAI_BASE_URL`
- `EMBEDDING_DIMENSIONS`
- `ANTHROPIC_GATEWAY_BASE_URL=http://anthropic-gateway:8081`

### Knowledge MCP (`mcp`)
- `MCP_PORT=9000`
- `MCP_BEARER_TOKEN`
- `KNOWLEDGE_MCP_BASE_URL=http://mcp:9000`
- `KNOWLEDGE_MCP_BEARER_TOKEN`
- `KG_API_BASE_URL=http://api:8000`
- `KG_API_KEY`
- `MCP_POLL_INTERVAL_SECONDS=1.0`
- `MCP_INGESTION_TIMEOUT_SECONDS=90.0`

### Agent MCP (`agent-mcp`)
- `AGENT_MCP_PORT=9100`
- `AGENT_MCP_BEARER_TOKEN`
- `AGENT_OPENAI_BASE_URL` opcional; si no existe usa `OPENAI_BASE_URL`
- `AGENT_OPENAI_API_KEY` opcional; si no existe usa `OPENAI_API_KEY`
- `AGENT_OPENAI_CHAT_MODEL` opcional; si no existe usa `OPENAI_CHAT_MODEL`

### Anthropic Gateway (`anthropic-gateway`)
- `ANTHROPIC_GATEWAY_BASE_URL=http://anthropic-gateway:8081`
- `ANTHROPIC_GATEWAY_BEARER_TOKEN`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_CHAT_MODEL`
- `ANTHROPIC_THINKING_TYPE=adaptive|enabled`
- `ANTHROPIC_THINKING_BUDGET_TOKENS` obligatorio si `ANTHROPIC_THINKING_TYPE=enabled`
- `ANTHROPIC_EFFORT=low|medium|high|max|xhigh`
- `ANTHROPIC_TIMEOUT_SECONDS=180`

## Despliegue en Coolify

- Crear un servicio Docker Compose apuntando a este repo.
- Configurar las variables del `.env.example` en Coolify.
- Exponer solo el servicio `agent-mcp` por HTTPS.
- Mantener `mcp`, `api` y `arcadedb` como servicios internos.
- Usar `GET /health/ready` del servicio `agent-mcp` como health check HTTP público.

## Integración con Claude Code

```bash
claude mcp add --transport http knowledge-agent https://tu-dominio-agent-mcp/mcp \
  -e Authorization="Bearer ${AGENT_MCP_BEARER_TOKEN}"
```

Ejemplo `.mcp.json`:

```json
{
  "mcpServers": {
    "knowledge-agent": {
      "type": "http",
      "url": "https://tu-dominio-agent-mcp/mcp",
      "headers": {
        "Authorization": "Bearer ${AGENT_MCP_BEARER_TOKEN}"
      }
    }
  }
}
```

## Smoke Tests

- `curl -i https://tu-dominio-agent-mcp/health/ready`
- Conectar cliente MCP a `https://tu-dominio-agent-mcp/mcp` y verificar `tools/list` (debe listar 28 tools)
- Flujo mínimo: `kg_add_knowledge_fragment` → `start_adaptive_session` con el `episode_id` retornado → `submit_adaptive_block`
- Flujo auto: `start_adaptive_session` con `study_mode="auto"` sin anchor (requiere contenido previamente ingestado)

Ver [STUDY_GUIDE.md](STUDY_GUIDE.md) para el flujo completo de estudio y referencia de todas las tools.
