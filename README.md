# Knowledge Graph API + MCP

Stack semántico en Python con dos servicios: una API REST privada en `FastAPI` y un servidor MCP público sobre `Streamable HTTP`, ambos respaldados por `ArcadeDB`.

## Qué incluye

- API privada con `X-API-Key`
- Servidor MCP público con bearer token propio en `/mcp`
- Ingesta asíncrona de fragmentos con `Episode` + `IngestionJob`
- Bootstrap idempotente del esquema en ArcadeDB
- Routing por dominio, deduplicación conservadora y `needs_review`
- MCP opinionado con 5 tools semánticas, sin SQL libre, resources ni prompts

## Topología

- Cliente/agente → `mcp` público (`Authorization: Bearer <MCP_BEARER_TOKEN>`)
- `mcp` → `api` privada (`X-API-Key: <KG_API_KEY>`)
- `api` → `arcadedb` interna

Por defecto, en Docker Compose solo `mcp` se publica; `api` y `arcadedb` quedan en red interna.

## Endpoints REST

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/knowledge/fragments`
- `GET /v1/jobs/{job_id}`
- `GET /v1/episodes/{episode_id}`
- `POST /v1/search/candidates`
- `PUT /v1/concepts/upsert`
- `POST /v1/concepts/link`
- `GET /v1/concepts/{concept_ref}/neighborhood?depth=1|2`

## MCP

Endpoint MCP:

- `GET/POST /mcp`
- `/mcp/` tambien responde, pero la URL canónica documentada para clientes es `/mcp`

Health checks del servicio MCP:

- `GET /health/live`
- `GET /health/ready`

Tools expuestas:

- `add_knowledge_fragment`
- `search_candidates`
- `upsert_concept`
- `link_concepts`
- `get_neighborhood`

`add_knowledge_fragment` encapsula la asincronía del backend: crea el job, hace polling y devuelve `completed`, `failed` o `processing` con `episode_id` y `job_id`.

## Desarrollo local

1. Crear `.env` a partir de `.env.example`, o usar `ENV_FILE=.env.example` para una prueba rápida.
2. Levantar con `docker compose up --build`.
3. El MCP queda en `http://localhost:9000/mcp`.
4. La API REST queda solo dentro de la red interna del compose.

Mientras `AI_PROVIDER=stub`, no hace falta proveedor externo. Para usar un proveedor OpenAI-compatible:

- Cambia `AI_PROVIDER=openai_compatible`
- Define `OPENAI_API_KEY`
- Ajusta `EMBEDDING_DIMENSIONS` para que coincida con tu modelo de embeddings

Variables nuevas para MCP:

- `MCP_PORT=9000`
- `MCP_BEARER_TOKEN`
- `KG_API_BASE_URL=http://api:8000`
- `KG_API_KEY`
- `MCP_POLL_INTERVAL_SECONDS=1.0`
- `MCP_INGESTION_TIMEOUT_SECONDS=90.0`

## Despliegue en Coolify

- Crear un servicio Docker Compose apuntando a este repo.
- Configurar las variables del `.env.example` en Coolify.
- Exponer solo el servicio `mcp` por HTTPS.
- Mantener `api` y `arcadedb` como servicios internos.
- Usar `GET /health/ready` del servicio `mcp` como health check HTTP público.

## Smoke Tests

- Salud del servicio MCP: `curl -i https://tu-dominio-mcp/health/ready`
- URL MCP canónica sin redirect manual: conectar el cliente MCP a `https://tu-dominio-mcp/mcp`
- Verificación mínima esperada con cliente MCP: `initialize`, `tools/list`, `add_knowledge_fragment`, `search_candidates` y `get_neighborhood`

## Diseño operativo

- La API no expone SQL libre.
- ArcadeDB queda accesible solo dentro de la red interna del compose.
- El worker asíncrono corre dentro del mismo contenedor de la API para este MVP.
- El servidor MCP no reimplementa lógica de dominio; solo envuelve los endpoints semánticos ya existentes.
