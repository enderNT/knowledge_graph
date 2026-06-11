# Knowledge Graph API + MCP

Stack semántico en Python con tres servicios: una API REST privada en `FastAPI`, un `knowledge MCP` upstream y un `agent MCP` público, todos respaldados por `ArcadeDB`.

## Qué incluye

- API privada con `X-API-Key`
- `knowledge MCP` upstream con bearer token propio en `/mcp`
- `agent MCP` público con bearer token propio en `/mcp`
- Ingesta asíncrona de fragmentos con `Episode` + `IngestionJob`
- Bootstrap idempotente del esquema en ArcadeDB
- Routing por dominio, deduplicación conservadora y `needs_review`
- MCP upstream opinionado con 6 tools semánticas, sin SQL libre, resources ni prompts
- MCP agente con 9 tools: 3 pedagógicas grounded + 6 passthrough `kg_*`

## Topología

- Claude Code/Codex/Copilot CLI/etc → `agent-mcp` público (`Authorization: Bearer <AGENT_MCP_BEARER_TOKEN>`)
- `agent-mcp` → `knowledge mcp` interno (`Authorization: Bearer <KNOWLEDGE_MCP_BEARER_TOKEN>`)
- `knowledge mcp` → `api` privada (`X-API-Key: <KG_API_KEY>`)
- `api` → `arcadedb` interna
- `api` → `anthropic-gateway` interno cuando `AI_PROVIDER=anthropic`

Por defecto, en Docker Compose solo `agent-mcp` se publica; `mcp`, `api`, `arcadedb` y `anthropic-gateway` quedan en red interna.

## Endpoints REST

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/knowledge/fragments`
- `GET /v1/jobs/{job_id}`
- `GET /v1/episodes/{episode_id}`
- `POST /v1/search/candidates`
- `POST /v1/search/learning-context`
- `POST /v1/search/tutor-context`
- `PUT /v1/concepts/upsert`
- `POST /v1/concepts/link`
- `GET /v1/concepts/{concept_ref}/neighborhood?depth=1|2`

## Knowledge MCP Upstream

Endpoint MCP upstream:

- `GET/POST /mcp`
- `/mcp/` tambien responde, pero la URL canónica documentada para clientes es `/mcp`

Health checks del servicio `knowledge MCP`:

- `GET /health/live`
- `GET /health/ready`

Tools expuestas:

- `add_knowledge_fragment`
- `search_candidates`
- `get_learning_context`
- `get_tutor_context`
- `upsert_concept`
- `link_concepts`
- `get_neighborhood`

`add_knowledge_fragment` encapsula la asincronía del backend: crea el job, hace polling y devuelve `completed`, `failed` o `processing` con `episode_id` y `job_id`.

`search_candidates` sigue siendo la búsqueda cruda para compatibilidad y debugging. `get_learning_context` conserva la recuperación amplia actual. `get_tutor_context` agrega el contrato estricto para tutor: acepta exactamente una referencia entre `query`, `episode_id` o `job_id`, devuelve un paquete estructurado y trazable, y falla con `status=failed` + `failure_reason` cuando no hay evidencia suficiente.

## Agent MCP Público

Endpoint MCP público:

- `GET/POST /mcp`

Health checks:

- `GET /health/live`
- `GET /health/ready`

Tools expuestas:

- `explain_topic`
- `generate_quiz`
- `evaluate_answer`
- `kg_add_knowledge_fragment`
- `kg_search_candidates`
- `kg_get_learning_context`
- `kg_get_tutor_context`
- `kg_upsert_concept`
- `kg_link_concepts`
- `kg_get_neighborhood`

Las tools de alto nivel hacen retrieval primero sobre `kg_get_learning_context` y solo generan contenido pedagógico grounded con lo recuperado.

## Contrato estricto para tutor

`POST /v1/search/tutor-context` y la tool MCP `get_tutor_context` comparten el mismo contrato:

- Entrada: exactamente una referencia entre `query`, `episode_id` o `job_id`
- Controles: `depth=1` e `include_evidence=true|false`
- Salida: `resolved_reference`, `status`, `concepts`, `claims`, `relations`, `source_fragments`, `evidence`, `warnings`, `failure_reason`

Comportamiento de esta iteración:

- Incluye solo soporte directo y trazable del tema, episodio o job pedido
- Reutiliza `job_id -> episode_id` sin una ruta paralela distinta
- Falla de forma dura cuando no hay evidencia trazable suficiente
- No incluye personalización por nivel de alumno, dificultad adaptativa, memoria histórica ni expansión pedagógica variable

## Desarrollo local

1. Crear `.env` a partir de `.env.example`, o usar `ENV_FILE=.env.example` para una prueba rápida.
2. Levantar con `docker compose up --build`.
3. El `agent MCP` queda en `http://localhost:9100/mcp`.
4. El `knowledge MCP` queda interno en `http://mcp:9000/mcp`.
5. La API REST queda solo dentro de la red interna del compose.

Mientras `AI_PROVIDER=stub`, no hace falta proveedor externo.

Matriz de proveedores:

- `AI_PROVIDER=stub`: extracción heurística local + embeddings locales
- `AI_PROVIDER=openai_compatible`: extracción LLM y embeddings por OpenAI-compatible
- `AI_PROVIDER=anthropic`: extracción LLM por `anthropic-gateway` + embeddings por `EMBEDDING_PROVIDER`

Para usar un proveedor OpenAI-compatible:

- Cambia `AI_PROVIDER=openai_compatible`
- Define `OPENAI_API_KEY`
- Ajusta `EMBEDDING_DIMENSIONS` para que coincida con tu modelo de embeddings

Para usar Anthropic vía gateway Go:

- Cambia `AI_PROVIDER=anthropic`
- Define `EMBEDDING_PROVIDER=stub` o `EMBEDDING_PROVIDER=openai_compatible`
- Define `ANTHROPIC_GATEWAY_BEARER_TOKEN`
- Define `ANTHROPIC_API_KEY`
- Define `ANTHROPIC_CHAT_MODEL`
- Opcionalmente define `ANTHROPIC_THINKING_TYPE`
- Si `ANTHROPIC_THINKING_TYPE=enabled`, define también `ANTHROPIC_THINKING_BUDGET_TOKENS`
- Opcionalmente define `ANTHROPIC_EFFORT`
- Opcionalmente ajusta `ANTHROPIC_GATEWAY_BASE_URL` y `ANTHROPIC_TIMEOUT_SECONDS`

Ejemplo mínimo:

```env
AI_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai_compatible
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
ANTHROPIC_GATEWAY_BEARER_TOKEN=replace-with-an-internal-bearer-token
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6
EMBEDDING_DIMENSIONS=16
```

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

Ejemplo para Claude Haiku 4.5 con pensamiento extendido:

```env
AI_PROVIDER=anthropic
EMBEDDING_PROVIDER=stub
ANTHROPIC_GATEWAY_BEARER_TOKEN=replace-with-an-internal-bearer-token
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_CHAT_MODEL=claude-haiku-4-5
ANTHROPIC_THINKING_TYPE=enabled
ANTHROPIC_THINKING_BUDGET_TOKENS=8000
```

Variables del `knowledge MCP`:

- `MCP_PORT=9000`
- `MCP_BEARER_TOKEN`
- `KNOWLEDGE_MCP_BASE_URL=http://mcp:9000`
- `KNOWLEDGE_MCP_BEARER_TOKEN`
- `KG_API_BASE_URL=http://api:8000`
- `KG_API_KEY`
- `MCP_POLL_INTERVAL_SECONDS=1.0`
- `MCP_INGESTION_TIMEOUT_SECONDS=90.0`

Variables del `agent MCP`:

- `AGENT_MCP_PORT=9100`
- `AGENT_MCP_BEARER_TOKEN`
- `AGENT_OPENAI_BASE_URL` opcional; si no existe usa `OPENAI_BASE_URL`
- `AGENT_OPENAI_API_KEY` opcional; si no existe usa `OPENAI_API_KEY`
- `AGENT_OPENAI_CHAT_MODEL` opcional; si no existe usa `OPENAI_CHAT_MODEL`

Variables del `anthropic-gateway`:

- `ANTHROPIC_GATEWAY_BASE_URL=http://anthropic-gateway:8081` para la API Python
- `ANTHROPIC_GATEWAY_BEARER_TOKEN`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_CHAT_MODEL`
- `ANTHROPIC_THINKING_TYPE=adaptive|enabled`
- `ANTHROPIC_THINKING_BUDGET_TOKENS` obligatorio si `ANTHROPIC_THINKING_TYPE=enabled`
- `ANTHROPIC_EFFORT=low|medium|high|max|xhigh`
- `ANTHROPIC_TIMEOUT_SECONDS=180`

Health checks del `anthropic-gateway`:

- `GET /health/live`
- `GET /health/ready`

## Despliegue en Coolify

- Crear un servicio Docker Compose apuntando a este repo.
- Configurar las variables del `.env.example` en Coolify.
- Exponer solo el servicio `agent-mcp` por HTTPS.
- Mantener `mcp`, `api` y `arcadedb` como servicios internos.
- Usar `GET /health/ready` del servicio `agent-mcp` como health check HTTP público.

## Smoke Tests

- Salud del `agent MCP`: `curl -i https://tu-dominio-agent-mcp/health/ready`
- URL MCP canónica sin redirect manual: conectar el cliente MCP a `https://tu-dominio-agent-mcp/mcp`
- Verificación mínima esperada con cliente MCP: `initialize`, `tools/list`, `explain_topic`, `generate_quiz`, `evaluate_answer` y un passthrough `kg_get_learning_context`

## Integración con Claude Code

Snippet oficial recomendado con CLI:

```bash
claude mcp add --transport http knowledge-agent https://tu-dominio-agent-mcp/mcp \
  -e Authorization="Bearer ${AGENT_MCP_BEARER_TOKEN}"
```

Ejemplo `.mcp.json` para SDKs o clientes compatibles con configuración HTTP:

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

Playbook de uso:

- Usa `explain_topic` para explicaciones grounded rápidas.
- Usa `generate_quiz` para evaluaciones estructuradas.
- Usa `evaluate_answer` para feedback sobre una respuesta del alumno.
- Usa `kg_*` cuando necesites curar, depurar o inspeccionar el grafo directamente.

## Diseño operativo

- La API no expone SQL libre.
- ArcadeDB queda accesible solo dentro de la red interna del compose.
- El worker asíncrono corre dentro del mismo contenedor de la API para este MVP.
- El `knowledge MCP` no reimplementa lógica de dominio; solo envuelve los endpoints semánticos ya existentes.
- El `agent MCP` orquesta retrieval + generación grounded, y deja `kg_*` disponibles para operación interna.

=== NOTAS PARA ACLARAR ===
### get_learning_context VS get_tutor_context
get_learning_context es una recuperación más abierta y de apoyo, mientras que get_tutor_context es el paquete estricto, trazable y listo para que el agente actúe; además, get_tutor_context falla explícitamente si no hay evidencia suficiente, y get_learning_context puede devolver contexto útil pero más “exploratorio” o parcial.
Get TUTOR CONTEXT ES el predominante.
