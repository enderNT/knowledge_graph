# MCP Server Remoto para el Knowledge Graph

## Resumen
- Añadir un servicio `mcp` separado, en Python, usando el SDK oficial de MCP con `Streamable HTTP` como transporte principal.
- Exponer `mcp` públicamente por HTTPS en Coolify y dejar `api` + `arcadedb` en red interna por defecto.
- Mantener un MCP opinionado y pequeño: solo 5 tools semánticas, sin SQL libre, sin resources ni prompts en v1.
- Hacer que `add_knowledge_fragment` oculte la asincronía del backend: envía el fragmento, espera el job y devuelve un resumen útil al agente.

## Cambios de implementación
- Crear un app MCP dedicada dentro del repo, por ejemplo en `app/mcp_server.py`, con:
  - `FastAPI`/`Starlette` mínimo para `GET /health/live` y `GET /health/ready`
  - instancia `FastMCP` montada en `/mcp`
  - middleware o dependency que exija `Authorization: Bearer <MCP_BEARER_TOKEN>` solo en `/mcp`
- Crear un cliente interno hacia la API actual, por ejemplo en `app/mcp_backend_client.py`, con:
  - `KG_API_BASE_URL`
  - `KG_API_KEY`
  - timeout HTTP, polling interval y timeout total de ingesta
  - traducción de errores HTTP a errores MCP consistentes
- Extender `app/config.py` con settings MCP:
  - `MCP_PORT=9000`
  - `MCP_BEARER_TOKEN`
  - `KG_API_BASE_URL=http://api:8000`
  - `KG_API_KEY`
  - `MCP_POLL_INTERVAL_SECONDS=1.0`
  - `MCP_INGESTION_TIMEOUT_SECONDS=90.0`
- Reusar la imagen Python actual y agregar un servicio `mcp` en `docker-compose.yml` con comando separado, por ejemplo `uvicorn app.mcp_server:app --host 0.0.0.0 --port 9000`.
- Ajustar `README.md` y `.env.example` para documentar:
  - nuevo servicio `mcp`
  - variables de entorno MCP
  - flujo recomendado: agente → `mcp` → `api` → `arcadedb`
  - healthcheck HTTP del servicio público MCP

## Interfaces MCP
- Endpoint MCP:
  - `POST/GET /mcp` con `Streamable HTTP`
  - healthchecks públicos fuera de `/mcp`
- Tools expuestas, exactamente estas 5:
  1. `add_knowledge_fragment`
  2. `search_candidates`
  3. `upsert_concept`
  4. `link_concepts`
  5. `get_neighborhood`
- Mapeo de tools a la API existente:
  - `add_knowledge_fragment`
    - input: `text`, `source_type="manual_input"`, `tags=[]`, `language="es"`
    - implementación: `POST /v1/knowledge/fragments` + polling a `GET /v1/jobs/{job_id}`
    - comportamiento:
      - si completa antes de `90s`, devuelve el `result` del job más `status`, `episode_id`, `job_id`
      - si falla, devuelve `status="failed"` con `error`, `episode_id`, `job_id`
      - si expira el timeout, devuelve `status="processing"` con `episode_id`, `job_id`
  - `search_candidates`
    - input: `query`, `domain_hint?`, `limit=10`
    - implementación: `POST /v1/search/candidates`
    - salida: passthrough del backend
  - `upsert_concept`
    - input: `canonical_name`, `aliases=[]`, `domain`, `description=""`
    - implementación: `PUT /v1/concepts/upsert`
    - salida: passthrough del backend
  - `link_concepts`
    - input: `from`, `relation`, `to`, `evidence_episode_id?`
    - implementación: `POST /v1/concepts/link`
    - salida: `{"status":"linked"}` o error MCP si el backend devuelve fallo lógico
  - `get_neighborhood`
    - input: `concept`, `depth=1`
    - implementación: `GET /v1/concepts/{concept}/neighborhood?depth=...`
    - salida: passthrough del backend
- Política de errores:
  - errores de red/auth/backend no semánticos: error MCP
  - errores semánticos del dominio de ingesta: resultado estructurado con `status="failed"` o `status="processing"`, no excepción genérica

## Despliegue y seguridad
- Topología final:
  - `mcp` público
  - `api` interno
  - `arcadedb` interno
- Autenticación:
  - cliente/agente → `mcp`: bearer token propio
  - `mcp` → `api`: `X-API-Key` privada existente
- No agregar CORS específico en v1; asumir consumo server-to-server o cliente MCP dedicado.
- No exponer SQL readonly ni herramientas de debug en este primer corte.

## Pruebas y aceptación
- Unit tests del cliente MCP interno:
  - auth bearer faltante/incorrecta
  - timeout de polling
  - job `completed`
  - job `failed`
  - backend `404/401/500`
- Unit tests de tools:
  - schema de entrada esperado
  - nombres exactos de las 5 tools
  - traducción correcta de respuestas y errores
- Smoke test HTTP:
  - `GET /health/ready` del servicio `mcp`
  - conexión desde MCP Inspector a `/mcp`
  - listado de exactamente 5 tools
- End-to-end en compose:
  - stack limpio
  - `add_knowledge_fragment` desde MCP devuelve resumen útil
  - `search_candidates` y `get_neighborhood` responden después de una ingesta
- Acceptance criteria:
  - un cliente MCP remoto puede conectarse por HTTPS
  - puede descubrir y llamar las 5 tools sin conocer la API REST
  - el backend REST queda oculto para uso normal del agente

## Suposiciones y defaults
- Lenguaje: Python, no Node, para maximizar reuse del repo actual.
- Transporte inicial: `Streamable HTTP` únicamente; `stdio` queda fuera de v1.
- Superficie MCP: solo tools, sin resources/prompts en esta fase.
- La API REST existente se conserva como backend semántico interno; el MCP no reimplementa lógica de dominio.
- La base técnica del servidor MCP será el SDK oficial de Python y la documentación oficial de SDK/transports:
  - https://py.sdk.modelcontextprotocol.io/
  - https://modelcontextprotocol.io/docs/sdk
