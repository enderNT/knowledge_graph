# Guía de estudio — Knowledge Agent MCP

Este documento explica cómo usar el MCP para estudiar, qué hace cada tool, en qué orden usarlas y cómo funciona el motor por dentro.

---

## Conceptos clave (entidades del sistema)

| Entidad | Qué es |
|---------|--------|
| **Episode** | Un texto que ingestaste. Cada ingesta genera un `ep_xxx` ID. |
| **Concepto** | Una idea que el sistema extrajo de uno o más episodes. Un episode puede generar varios conceptos. Los conceptos son lo que se estudia. |
| **Bloque** | Un conjunto de preguntas (por defecto 3) sobre un concepto. Se genera on-demand al arrancar la sesión y al enviar cada bloque anterior. |
| **Sesión** | Contenedor con un `ads_xxx` ID que agrupa bloques. Tiene un anchor (fuente de conceptos) y lleva historial. |
| **SR item / due** | Cuando fallas o pasa suficiente tiempo, el sistema marca un concepto como pendiente de repasar. Aparece automáticamente en la siguiente sesión. |
| **Contexto tutor** | Lo que hay en el grafo sobre un tema: conceptos, relaciones, evidencia aprobada. Se construye desde el anchor al iniciar sesión. |
| **Contexto pedagógico** | Tu historial como estudiante: scores por dimensión, mastery, tendencia. Es **global por `user_id`** — no está atado a un episode ni a una sesión. |

### Las 4 dimensiones de aprendizaje

Cada concepto se mide en 4 dimensiones (score 0–100). El motor siempre ataca la dimensión más débil primero.

| Dimensión | Qué mide | Tipo de preguntas |
|-----------|----------|------------------|
| `recognition` | ¿Reconoces la idea cuando la ves? | Opción múltiple, verdadero/falso |
| `recall` | ¿La recuerdas sin pistas? | Completar, definición corta |
| `explanation` | ¿Puedes explicarla con tus palabras? | Respuesta abierta, parafraseo |
| `application` | ¿Puedes aplicarla en contextos nuevos? | Casos, ejemplos propios, analogías |

---

## Modos de estudio

### Matriz comparativa

| Modo | Anchor | Cuando usarlo | SR due | Contenido nuevo | Falla sin contenido |
|------|--------|--------------|--------|-----------------|---------------------|
| `hybrid` | Obligatorio | Estudio normal balanceado | Sí, hasta 75% | Sí | No |
| `backlog` | Obligatorio | Solo repasar pendientes | 100% | No | Sí, si no hay due |
| `recovery` | Obligatorio | Repasar los más débiles/vencidos | 100% filtrado | No | Sí, si no hay due |
| `isolated` | Obligatorio + `domain_hint` | Profundizar en un dominio sin mezcla | Solo del dominio | Solo del dominio | No |
| `auto` | Opcional | Estudiar sin decidir qué | Sí, primero | Sí | No |

### Cómo elige el concepto cada modo

```
hybrid / isolated
  └─ anchor (query / episode_id / job_id)
       └─ TutorContextBuilder resuelve → lista de conceptos
            └─ _select_target_concept() elige por prioridad pedagógica

backlog / recovery
  └─ anchor (mismo flujo que hybrid)
       └─ pero _build_next_block() va directo a SR due items
            └─ si se agotan → error (no hay nada que repasar)

auto (sin anchor)
  └─ _auto_resolve_tutor_context()
       ├─ 1. Consulta SR due items del usuario
       │    └─ Si hay → usa el concept_uid del primero como anchor
       └─ 2. Si no hay due → consulta contexto pedagógico del usuario
                └─ Ordena conceptos por prioridad_score desc, weakest_score asc
                     └─ Prueba cada uno hasta encontrar uno con evidencia trazable

auto (con anchor opcional)
  └─ Se comporta igual que hybrid pero sin exigir anchor al usuario
```

---

## Flujo de sesión — diagramas

### Diagrama general (todos los modos)

```
┌─────────────────────────────────────────────────────────────┐
│                      start_adaptive_session                  │
│                                                              │
│  user_id + [anchor] + study_mode + constraints               │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ ¿Tiene anchor?        │
         │ (query/episode/job)   │
         └──────┬───────┬───────┘
                │ Sí    │ No (solo auto)
                ▼       ▼
     TutorContextBuilder  _auto_resolve_tutor_context()
     .build(anchor)        │
                           ├─ SR due items disponibles?
                           │    Sí → usa concept_uid del 1º due
                           └─   No → concepto de menor mastery
                                       en contexto pedagógico
                    │
                    ▼
         ┌──────────────────────┐
         │ get_pedagogical_     │
         │ context(user_id)     │  ← siempre global
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ list_review_         │
         │ candidates(user_id)  │  ← SR due + forced items
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────────────────────┐
         │ _session_targets(study_mode, ...)    │
         │                                      │
         │  backlog/recovery → (max_blocks, 0)  │
         │  hybrid/auto/isolated → _session_mix │
         │    ≥4 due → 75% review               │
         │    <4 due → 50% review               │
         └──────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ _build_next_block()  │  ← genera bloque 1
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Crea sesión (ads_xxx)│
         │ Persiste bloque      │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Retorna session +   │
         │  current_block       │  ← listo para responder
         └──────────────────────┘
```

### Ciclo de un bloque (submit → siguiente)

```
┌───────────────────────────────────────────────────────────────┐
│                    submit_adaptive_block                       │
│              session_id + block_id + submissions              │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
              Evalúa cada item
              coverage + precision → score → verdict
                         │
                         ▼
              SR update (SM-2) automático
              ┌──────────────────────────┐
              │ correct → intervalo × ef │ ef ≈ 2.5x
              │ partial → intervalo corto│
              │ incorrect → mañana       │
              └──────────────────────────┘
                         │
                         ▼
              Actualiza contexto pedagógico (global)
              nuevo_score = old×0.35 + resultado×0.65
                         │
                         ▼
              ¿Cerrar sesión?
              ┌──────────────────────────────────────┐
              │ bloques completados ≥ max_blocks → Sí│
              │ correct AND weakest_dim ≥ 70     → Sí│
              │ otherwise                        → No│
              └──────────────────────────────────────┘
                    │ No                  │ Sí
                    ▼                    ▼
           _build_next_block()    session_closed=true
                    │              next_block=null
                    ▼
           Retorna next_block
```

### Lógica de `_build_next_block()` (qué bloque genera)

```
_build_next_block()
│
├─ ¿review_blocks_completed < review_target?
│   └─ Sí → intenta construir bloque SR
│           ├─ Recorre due items en orden
│           ├─ Resuelve tutor context para ese concept_uid
│           └─ Si hay evidencia → genera bloque SR (block_purpose=spaced_repetition_review)
│               Si no hay evidencia para ese item → intenta siguiente
│
│   └─ No (o se agotaron los due items)
│       ├─ backlog/recovery → error "no_review_candidates_for_mode"
│       └─ hybrid/isolated/auto → genera bloque de contenido nuevo
│             _select_target_concept()  ← elige por prioridad del tutor_context
│             _plan_block()             ← decide goal, difficulty, dimension
│             _generate_block()         ← genera items + answer keys
│                                         block_purpose=new_content
```

---

## Flujo completo para estudiar (paso a paso)

```
PASO 1 — Tener contenido ingestado
  Si aún no tienes episodes:
  → kg_add_knowledge_fragment(text="...", language="es")
  → Guarda el episode_id que retorna (ep_xxx)

PASO 2 — (Opcional) Ver qué tienes disponible
  → kg_get_pedagogical_context(user_id="tu_id")
    Retorna todos tus conceptos con scores actuales.

  → kg_sr_get_due_items(user_id="tu_id")
    Retorna conceptos marcados como pendientes hoy.

PASO 3 — Iniciar sesión adaptativa

  OPCIÓN A — Especificando qué estudiar:
  → start_adaptive_session(
       user_id="tu_id",
       episode_id="ep_xxx",        ← o bien: query="tema"
       study_mode="hybrid",
       constraints={"max_blocks": 12}
     )

  OPCIÓN B — Modo auto (el motor elige):
  → start_adaptive_session(
       user_id="tu_id",
       study_mode="auto",
       constraints={"max_blocks": 12}
     )

  Guarda el session_id (ads_xxx) y lee el current_block.
  El primer bloque ya viene listo — no hay que pedirlo por separado.

PASO 4 — Leer el bloque y responder
  El current_block contiene los items, cada uno con un item_id.

PASO 5 — Enviar respuestas
  → submit_adaptive_block(
       session_id="ads_xxx",
       block_id="blk_xxx",         ← viene en el current_block
       submissions=[
         {"item_id": "itm_xxx", "response": "tu respuesta"},
         {"item_id": "itm_yyy", "response": "tu respuesta"},
         {"item_id": "itm_zzz", "response": "tu respuesta"}
       ]
     )
  Retorna: block_result, updated_context, next_block, session_closed

PASO 6 — Repetir desde PASO 4 con el next_block
  Continúa hasta que session_closed=true o tú pares.

PASO 7 — Si session_closed=true y quieres seguir
  Vuelve al PASO 3.
  Si usaste auto, el motor elige el siguiente concepto de mayor prioridad.
  Si usaste anchor fijo, puedes pasarlo de nuevo.
  Nueva sesión — el estado pedagógico ya tiene todo lo que hiciste.
```

---

## Cómo se acumulan los resultados

### Score por dimensión

```
nuevo_score = (score_anterior × 0.35) + (resultado_actual × 0.65)
```

El peso del resultado reciente (0.65) es mayor que el histórico (0.35). En el mismo día, sesión tras sesión, el score sube notablemente si respondes bien.

### SM-2 (spaced repetition)

Se aplica automáticamente al cerrar cada bloque, sin intervención manual:

| Resultado | Próxima revisión | Intervalo |
|-----------|-----------------|-----------|
| `incorrect` | Mañana | Resetea a 1 día |
| `partial_low` | 1–2 días | Intervalo corto |
| `partial_high` | Pocos días | Intervalo moderado |
| `correct` | Intervalo × ease_factor | ~2.5x, compoundea |

Si hoy el intervalo era 4 días y apruebas bien → siguiente será ~10 días → luego ~25 → etc.

### Cuándo se cierra la sesión

La sesión se cierra cuando ocurre **cualquiera** de estas:

1. `bloques_completados >= max_blocks`
2. `resultado == "correct"` Y `dimensión_más_débil.score >= 70`

Para estudiar sin que se corte pronto: usa `constraints={"max_blocks": 12}`. Para continuar más allá: inicia nueva sesión — el motor retoma desde donde quedaste.

---

## Las 28 tools — referencia completa

### Grupo 1: Sesión adaptativa (para estudiar)

#### `start_adaptive_session`
Inicia una sesión. Retorna el primer bloque listo para responder.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `user_id` | string | **requerido** | ID del estudiante |
| `query` | string | null | Tema en lenguaje natural (anchor) |
| `episode_id` | string | null | ID de episode específico (anchor) |
| `job_id` | string | null | ID de job de ingesta (anchor) |
| `study_mode` | enum | `"hybrid"` | `hybrid`, `backlog`, `recovery`, `isolated`, `auto` |
| `domain_hint` | string | null | Filtrar por dominio (requerido si `isolated`) |
| `language` | string | `"es"` | Idioma de los bloques |
| `constraints` | dict | null | `{"max_blocks": 12, "max_items_per_block": 3, ...}` |

> En todos los modos excepto `auto` se requiere exactamente uno de: `query`, `episode_id`, o `job_id`.
> En modo `auto`, el anchor es opcional — si no se pasa, el motor elige automáticamente.

#### `submit_adaptive_block`
Envía las respuestas de un bloque. Retorna evaluación + siguiente bloque o cierre.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `session_id` | string | **requerido** | ID de la sesión activa |
| `block_id` | string | **requerido** | ID del bloque respondido (viene en `current_block`) |
| `submissions` | list | **requerido** | `[{"item_id": "...", "response": "..."}]` |
| `interaction_events` | list | null | Eventos opcionales (hints usados, retries, etc.) |

#### `get_adaptive_session`
Consulta estado de una sesión sin avanzarla.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `session_id` | string | ID de la sesión |

Retorna: historial de bloques, `current_block` si sigue abierta, contadores, `study_mode`, constraints.

---

### Grupo 2: Pedagógicas de alto nivel

No actualizan el estado pedagógico. Para explorar o practicar sin registro formal.

#### `explain_topic`
| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | **requerido** |
| `domain_hint` | string | null |
| `audience` | enum | `"intermediate"` (`beginner`\|`intermediate`\|`advanced`) |
| `focus` | string | null |
| `include_examples` | bool | `true` |

#### `generate_quiz`
| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | **requerido** |
| `domain_hint` | string | null |
| `difficulty` | enum | `"intermediate"` |
| `question_count` | int (1–10) | `5` |
| `question_type` | enum | `"mixed"` (`mixed`\|`multiple_choice`\|`open`) |

#### `evaluate_answer`
| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | **requerido** |
| `question` | string | **requerido** |
| `learner_answer` | string | **requerido** |
| `domain_hint` | string | null |
| `expected_answer` | string | null |

---

### Grupo 3: Inspección del grafo (solo lectura)

#### `kg_list_episodes`
Lista todos los episodes ingestados con paginación, ordenamiento y resumen de conceptos por episode.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `sort_by` | enum | `"alphabetical"` | `alphabetical` (por `uid`) \| `date` (por `created_at`) |
| `sort_order` | enum | `"asc"` | `asc` \| `desc` |
| `limit` | int (1–100) | `10` | Máximo de episodes por página |
| `page` | int (≥1) | `1` | Página a recuperar |
| `concept_sort_by` | enum | null | Si se pasa, ordena los conceptos dentro de cada episode: `alphabetical` \| `date` |
| `concept_sort_order` | enum | `"asc"` | Orden de los conceptos: `asc` \| `desc` |

**Respuesta:**
```json
{
  "episodes": [
    {
      "uid": "ep_xxx",
      "source_type": "manual_input",
      "tags": ["matemáticas"],
      "language": "es",
      "status": "completed",
      "created_at": "2026-06-11T...",
      "concepts": [
        {"uid": "cn_yyy", "canonical_name": "Logaritmo", "domain": "matemáticas", "created_at": "..."}
      ]
    }
  ],
  "total": 42,
  "page": 1,
  "limit": 10,
  "total_pages": 5,
  "has_next": true,
  "has_prev": false,
  "warnings": []
}
```

> Si `limit` supera el total disponible, retorna todos los resultados e incluye un `warning` explicativo.

#### `kg_search_candidates`
Búsqueda de conceptos relacionados. Útil para descubrir qué temas tienes ingestados.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | **requerido** |
| `domain_hint` | string | null |
| `limit` | int (1–50) | `10` |

#### `kg_get_learning_context`
Recupera contexto pedagógico amplio (candidatos, claims, episodes, vecindad). Para exploración.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | **requerido** |
| `domain_hint` | string | null |
| `candidate_limit` | int (1–50) | `8` |
| `concept_limit` | int (1–10) | `3` |
| `claim_limit` | int (1–20) | `6` |
| `episode_limit` | int (1–10) | `3` |
| `include_neighborhood` | bool | `true` |
| `depth` | int (1–2) | `1` |

#### `kg_get_tutor_context`
Contexto estricto y trazable desde exactamente una referencia. Falla si no hay evidencia suficiente.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `query` | string | null |
| `episode_id` | string | null |
| `job_id` | string | null |
| `depth` | int | `1` |
| `include_evidence` | bool | `true` |

#### `kg_get_neighborhood`
Conceptos relacionados alrededor de uno dado.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `concept` | string | **requerido** |
| `depth` | int (1–2) | `1` |

#### `kg_get_pedagogical_context`
Tu perfil completo: todos los conceptos, scores por dimensión, mastery, tendencia. **Para ver en qué estás.**

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `user_id` | string | **requerido** |
| `domain` | string | null |
| `concept_uids` | list | null |

#### `kg_get_pedagogical_session_view`
Vista operacional del estado pedagógico, optimizada para planear una sesión.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `user_id` | string | **requerido** |
| `domain_hint` | string | null |
| `concept_uids` | list | null |
| `query` | string | null |

#### `kg_sr_get_state`
Estado SR para un concepto y dimensión específicos.

| Parámetro | Tipo |
|-----------|------|
| `user_id` | string |
| `concept_uid` | string |
| `dimension` | enum (`recognition`\|`recall`\|`explanation`\|`application`) |

#### `kg_sr_get_due_items`
Lista todos los conceptos marcados como pendientes de repasar hoy.

| Parámetro | Tipo |
|-----------|------|
| `user_id` | string |

#### `kg_sr_get_stats`
Estadísticas agregadas de SR: totales, pendientes, intervalos promedio.

| Parámetro | Tipo |
|-----------|------|
| `user_id` | string |

---

### Grupo 4: Modificación del grafo

#### `kg_add_knowledge_fragment`
Ingesta un texto. Retorna `episode_id` y `job_id`. **Guarda el `episode_id` que retorna.**

| Parámetro | Tipo | Default | Notas |
|-----------|------|---------|-------|
| `text` | string | **requerido** | |
| `source_type` | string | `"manual_input"` | |
| `tags` | list | null | |
| `language` | string | `"es"` | |
| `temporal` | bool | `false` | Marca conocimiento que puede quedar obsoleto (APIs, versiones, precios) |
| `expires_at` | string | null | Fecha ISO-8601 en que expira el contenido (`"2026-12-31"`) |

Cuando `temporal=true` y la fecha de `expires_at` ya pasó, los ítems SR vinculados aparecerán con `is_stale: true` en `kg_get_due_spaced_repetition_items`. Omitir ambos campos si el conocimiento es estable (matemáticas, física, conceptos fundamentales).

#### `kg_patch_episode_temporality`
Modifica el flag de temporalidad de un episodio existente. Úsalo para corregir marcados incorrectos sin necesidad de re-ingestar.

| Parámetro | Tipo | Notas |
|-----------|------|-------|
| `episode_id` | string | **requerido** |
| `temporal` | bool | `false` desmarca, `true` marca como temporal |
| `expires_at` | string | Fecha ISO-8601; `null` elimina la fecha de expiración |

Flujo típico de corrección: el agente marcó un episodio como `temporal=true` cuando no aplica → ejecutar `kg_patch_episode_temporality(episode_id="ep_...", temporal=false)` → el ítem deja de aparecer como `is_stale` en SR due items.

#### `kg_upsert_concept`
Crea o actualiza un concepto por UID o nombre canónico.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `canonical_name` | string | **requerido** |
| `domain` | string | **requerido** |
| `aliases` | list | null |
| `description` | string | `""` |
| `uid` | string | null |

#### `kg_create_concept`
Crea un concepto estrictamente. Falla si el nombre canónico o alias ya existen.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `canonical_name` | string | **requerido** |
| `domain` | string | **requerido** |
| `aliases` | list | null |
| `description` | string | `""` |

#### `kg_attach_concept_evidence`
Adjunta un episode como evidencia explícita de un concepto.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `concept_ref` | string | **requerido** |
| `episode_id` | string | **requerido** |
| `link_episode_claims` | bool | `true` |

#### `kg_link_concepts`
Crea una relación semántica entre dos conceptos.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `from` | string | **requerido** |
| `relation` | string | **requerido** |
| `to` | string | **requerido** |
| `evidence_episode_id` | string | null |

#### `kg_update_pedagogical_context`
Persiste evaluaciones formales en el contexto pedagógico. El motor adaptativo lo hace automáticamente — solo para ajustes manuales.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `user_id` | string | **requerido** |
| `evaluations` | list | **requerido** |
| `domain_hint` | string | null |
| `session_closed_at` | string | null |

#### `kg_sr_update_from_block`
Actualiza estado SR desde resultado de bloque. El motor lo hace automáticamente — solo para correcciones manuales.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `user_id` | string | **requerido** |
| `concept_uid` | string | **requerido** |
| `dimension` | enum | **requerido** |
| `block_verdict` | enum | **requerido** (`correct`\|`partial_high`\|`partial_low`\|`incorrect`\|`unsupported`) |
| `block_difficulty` | enum | **requerido** (`introductory`\|`intermediate`\|`advanced`) |
| `hint_used` | bool | `false` |
| `retry_used` | bool | `false` |
| `coverage` | float (0–1) | `0.0` |
| `precision` | float (0–1) | `0.0` |
| `was_direct_evaluation` | bool | `true` |

#### `kg_sr_apply_relief`
Marca conceptos padre como pendientes de validación cuando un concepto hijo fue dominado. El motor lo aplica automáticamente.

| Parámetro | Tipo |
|-----------|------|
| `user_id` | string |
| `source_concept_uid` | string |
| `source_dimension` | enum |
| `quality_q` | int (0–5) |

---

### Grupo 5: Eliminación (destructivas)

Siempre usa primero el `preview_` para ver el impacto antes de confirmar.

#### `kg_preview_delete_episode_content`
Muestra qué se eliminaría. No borra nada.

| Parámetro | Tipo |
|-----------|------|
| `episode_id` | string |
| `job_id` | string |

#### `kg_delete_episode_content`
Elimina un episode y su contenido asociado.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `episode_id` | string | null |
| `job_id` | string | null |
| `confirm` | bool | `true` |

#### `kg_preview_delete_relation`
Muestra qué se eliminaría. No borra nada.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `from` | string | **requerido** |
| `relation` | string | **requerido** |
| `to` | string | **requerido** |
| `evidence_episode_id` | string | null |
| `delete_all_matching` | bool | `false` |

#### `kg_delete_relation`
Elimina una relación entre conceptos.

| Parámetro | Tipo | Default |
|-----------|------|---------|
| `from` | string | **requerido** |
| `relation` | string | **requerido** |
| `to` | string | **requerido** |
| `evidence_episode_id` | string | null |
| `delete_all_matching` | bool | `false` |
| `confirm` | bool | `true` |

---

### Grupo 6: Reset total

#### `kg_reset_knowledge_base`
**PELIGROSA. Sin parámetros. Sin confirmación adicional. No hay vuelta atrás.**
Borra absolutamente todo: grafo, estado pedagógico, sesiones adaptativas y cola de ingesta.

---

## Cómo descubrir qué tienes para estudiar

| Necesidad | Tool recomendada |
|-----------|-----------------|
| Ver todos los episodes con sus conceptos | `kg_list_episodes()` |
| Buscar si un tema específico existe | `kg_search_candidates(query="tema")` |
| Ver tu progreso por concepto | `kg_get_pedagogical_context(user_id="tu_id")` |
| Ver qué conceptos tienes pendientes hoy | `kg_sr_get_due_items(user_id="tu_id")` |

**Guarda siempre los `episode_id`** cuando ejecutas `kg_add_knowledge_fragment` — son la referencia principal para iniciar sesiones.

---

## Instrucciones para el agente (Manus u otro)

Para conducir una sesión de estudio, el agente debe:

1. Usar **solo** `start_adaptive_session` + `submit_adaptive_block` para el estudio real.
2. Pasar siempre `constraints={"max_blocks": 12}` para no cortar la sesión prematuramente.
3. Guardar el `session_id` del `start_adaptive_session` y usarlo en cada `submit_adaptive_block`.
4. Leer el `block_id` del `current_block` y los `item_id` de cada pregunta, pasarlos exactamente en el submit.
5. Cuando `session_closed=true`, iniciar nueva sesión con el mismo anchor (o sin anchor si es modo `auto`).
6. **No usar** tools `kg_*` para estudiar — son para inspección y mantenimiento del grafo.

### Para modo auto (sin decidir qué estudiar)

```json
{
  "user_id": "tu_id",
  "study_mode": "auto",
  "constraints": {"max_blocks": 12}
}
```

El motor elige el concepto automáticamente: primero revisa SR due items; si no hay, elige el concepto de menor mastery del perfil del usuario. Cada nueva sesión `auto` avanza al siguiente concepto de mayor prioridad.
