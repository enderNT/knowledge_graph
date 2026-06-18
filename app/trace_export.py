from __future__ import annotations

import json

from app.trace_models import CanonicalTrace, TraceEvent


def render_trace_text(trace: CanonicalTrace) -> str:
    lines = [
        f"Traza: {trace.summary.trace_id}",
        f"Ejecucion: {trace.summary.execution_type} {trace.summary.execution_id}",
        f"Estado: {trace.summary.status}",
    ]
    if trace.summary.episode_id:
        lines.append(f"Episodio: {trace.summary.episode_id}")
    if trace.summary.domain:
        lines.append(f"Dominio: {trace.summary.domain}")
    if trace.summary.duration_ms is not None:
        lines.append(f"Duracion: {trace.summary.duration_ms}ms")
    lines.append("")
    lines.extend(_render_events(trace.events))
    return "\n".join(lines).rstrip() + "\n"


def _render_events(events: list[TraceEvent]) -> list[str]:
    by_parent: dict[str | None, list[TraceEvent]] = {}
    for event in sorted(events, key=lambda item: item.sequence):
        by_parent.setdefault(event.parent_event_id, []).append(event)

    lines: list[str] = []
    visible_index = 1
    for event in by_parent.get(None, []):
        rendered, visible_index = _render_event(event, by_parent, visible_index, depth=0)
        lines.extend(rendered)
    return lines


def _render_event(
    event: TraceEvent,
    by_parent: dict[str | None, list[TraceEvent]],
    visible_index: int,
    *,
    depth: int,
) -> tuple[list[str], int]:
    indent = "  " * depth
    lines = [f"{indent}{visible_index}. {event.title} [{event.status}]"]
    if event.summary:
        lines.append(f"{indent}   {event.summary}")
    for label, payload in (("Entrada", event.input), ("Salida", event.output), ("Detalle", event.detail)):
        if payload:
            lines.append(f"{indent}   {label}: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")
    if event.boundary_payload:
        if event.boundary_payload.request_text:
            lines.append(f"{indent}   Input enviado:")
            lines.append(_indent_block(event.boundary_payload.request_text, indent + "     "))
        if event.boundary_payload.response_text:
            lines.append(f"{indent}   Output recibido:")
            lines.append(_indent_block(event.boundary_payload.response_text, indent + "     "))
    visible_index += 1
    for child in by_parent.get(event.event_id, []):
        child_lines, visible_index = _render_event(child, by_parent, visible_index, depth=depth + 1)
        lines.extend(child_lines)
    return lines, visible_index


def _indent_block(value: str, indent: str) -> str:
    return "\n".join(f"{indent}{line}" for line in value.splitlines() or [""])
