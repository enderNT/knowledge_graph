# Knowledge Graph API

Servicio privado en `FastAPI` para ingesta y consulta semántica de un knowledge graph respaldado por `ArcadeDB`, pensado para desplegarse en `Coolify` con Docker y quedar listo para una futura envoltura MCP.

## Qué incluye

- API privada con `X-API-Key`
- Ingesta asíncrona de fragmentos con `Episode` + `IngestionJob`
- Bootstrap idempotente del esquema en ArcadeDB
- Routing por dominio, deduplicación conservadora y `needs_review`
- Endpoints semánticos alineados con una futura capa MCP

## Endpoints

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/knowledge/fragments`
- `GET /v1/jobs/{job_id}`
- `GET /v1/episodes/{episode_id}`
- `POST /v1/search/candidates`
- `PUT /v1/concepts/upsert`
- `POST /v1/concepts/link`
- `GET /v1/concepts/{concept_ref}/neighborhood?depth=1|2`

## Desarrollo local

1. Crear `.env` a partir de `.env.example`, o usar `ENV_FILE=.env.example` para una prueba rápida.
2. Levantar con `docker compose up --build`.
3. La API queda en `http://localhost:8000`.

Mientras `AI_PROVIDER=stub`, no hace falta proveedor externo. Para usar un proveedor OpenAI-compatible:

- Cambia `AI_PROVIDER=openai_compatible`
- Define `OPENAI_API_KEY`
- Ajusta `EMBEDDING_DIMENSIONS` para que coincida con tu modelo de embeddings

## Despliegue en Coolify

- Crear un servicio Docker Compose apuntando a este repo.
- Configurar las variables del `.env.example` en Coolify.
- Exponer solo el servicio `api`.
- Mantener `arcadedb` como servicio interno con volumen persistente `arcadedb_data`.
- Usar `GET /health/ready` como health check HTTP del servicio público.

## Diseño operativo

- La API no expone SQL libre.
- ArcadeDB queda accesible solo dentro de la red interna del compose.
- El worker asíncrono corre dentro del mismo contenedor de la API para este MVP.
- La compatibilidad futura con MCP se logra envolviendo los endpoints semánticos ya existentes.
