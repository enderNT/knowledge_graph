from __future__ import annotations

from app.trace_models import TraceEventType, TraceStatus


_TITLE_TEMPLATES: dict[TraceEventType, dict[TraceStatus, str]] = {
    "fragment_received": {
        "succeeded": "Fragmento recibido",
        "failed": "No se pudo recibir el fragmento",
    },
    "episode_embedded": {
        "succeeded": "Episodio vectorizado",
        "failed": "No se pudo vectorizar el episodio",
    },
    "knowledge_extracted": {
        "succeeded": "Conocimiento extraido",
        "empty": "Extraccion sin conocimiento util",
        "partial": "Conocimiento extraido parcialmente",
        "failed": "No se pudo extraer conocimiento",
    },
    "extraction_vetted": {
        "succeeded": "Extraccion revisada",
        "partial": "Extraccion revisada con descartes",
        "empty": "Revision sin elementos conservados",
        "failed": "No se pudo revisar la extraccion",
    },
    "concepts_resolved": {
        "succeeded": "Conceptos resueltos",
        "partial": "Conceptos resueltos parcialmente",
        "needs_review": "Conceptos requieren revision",
        "failed": "No se pudieron resolver conceptos",
    },
    "claims_created": {
        "succeeded": "Claims creados",
        "partial": "Claims creados parcialmente",
        "empty": "Sin claims creados",
        "failed": "No se pudieron crear claims",
    },
    "pedagogical_evidence_vetted": {
        "succeeded": "Evidencia pedagogica aprobada",
        "partial": "Evidencia pedagogica revisada con rechazos",
        "empty": "Sin evidencia pedagogica aprobada",
        "failed": "No se pudo revisar evidencia pedagogica",
    },
    "relations_created": {
        "succeeded": "Relaciones creadas",
        "partial": "Relaciones creadas parcialmente",
        "empty": "Sin relaciones creadas",
        "failed": "No se pudieron crear relaciones",
    },
    "ingestion_finalized": {
        "succeeded": "Ingesta finalizada",
        "partial": "Ingesta finalizada con pendientes",
        "needs_review": "Ingesta finalizada con revision pendiente",
    },
    "ingestion_failed": {
        "failed": "Ingesta fallida",
    },
}

_SUMMARY_TEMPLATES: dict[TraceEventType, str] = {
    "fragment_received": "El sistema acepto el fragmento y preparo su procesamiento.",
    "episode_embedded": "El texto del episodio se convirtio en metadatos vectoriales.",
    "knowledge_extracted": "El proveedor LLM devolvio una propuesta estructurada de conocimiento.",
    "extraction_vetted": "El juez LLM reviso la extraccion antes de persistirla.",
    "concepts_resolved": "Los conceptos extraidos se compararon con el grafo existente.",
    "claims_created": "Los claims grounded se persistieron y enlazaron.",
    "pedagogical_evidence_vetted": "La evidencia de aprendizaje se reviso antes de exponerla al estudiante.",
    "relations_created": "Las relaciones validas entre conceptos se persistieron.",
    "ingestion_finalized": "El job y el episodio quedaron cerrados con resumen final.",
    "ingestion_failed": "La ejecucion de ingesta termino con error.",
}


def trace_title(event_type: TraceEventType, status: TraceStatus, *, subject: str = "") -> str:
    title = _TITLE_TEMPLATES.get(event_type, {}).get(status)
    if title is None:
        title = _TITLE_TEMPLATES.get(event_type, {}).get("succeeded", "Paso de traza registrado")
    if subject:
        return f"{title}: {subject}"
    return title


def trace_summary(event_type: TraceEventType) -> str:
    return _SUMMARY_TEMPLATES.get(event_type, "")
