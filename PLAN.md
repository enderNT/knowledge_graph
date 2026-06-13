# Fase 2 — La extracción es lenguaje: que la juzgue el LLM, no regex en español

## Context

El pipeline de ingesta decide tres cosas que son **puro lenguaje** y hoy las resuelve (o las
re-filtra) con reglas deterministas y listas de palabras en español, lo que descarta resultados
válidos y se rompe entero en inglés/portugués:

- **Punto #4 — ¿qué es un "concepto" enseñable?** `_is_valid_concept`, `_clean_concept_name`,
  `_extract_capitalized_concepts` (`app/ai_provider.py:609`, `:496`) + wordlists
  `_GENERIC_CONCEPT_TOKENS`, `_GENERIC_HEADINGS`, `_QUALIFIER_TOKENS`, `_CONNECTOR_TOKENS`.
- **Punto #5 — ¿qué es un "claim" enseñable?** `_looks_like_claim` (`app/ai_provider.py:598`) +
  `_CLAIM_VERB_MARKERS` (verbos en español).
- **Punto #6 — ¿qué relación semántica hay entre dos conceptos?** `_infer_relation_type`
  (`app/ai_provider.py:559`), substrings en español → enum.

Matiz que define el alcance: el `extract()` con LLM **ya existe** (`StructuredLLMProvider.extract`,
`app/ai_provider.py:720`; llamado desde `app/ingestion.py:82`). El daño está repartido:
- **#4 afecta a ambos caminos**: en el camino LLM, `_sanitize_llm_extraction` →
  `_sanitize_concepts` aplica `_is_valid_concept`/`_clean_concept_name` y **descarta conceptos
  que el LLM extrajo bien**.
- **#5 y #6 son camino Stub** (`StubAIProvider` / `refine_extraction` / `_extract_*`), que es el
  `AI_PROVIDER=stub` por defecto (`app/config.py:26`).

Principio rector (acordado): lenguaje/juicio cognitivo = LLM; estructura/IDs/grounding-textual =
determinista. La extracción y el juicio de calidad son lenguaje; **lo único determinista que se
conserva es lo agnóstico al idioma**: dedupe, verificación de cita textual contra la fuente
(grounding), validación del enum de relación, integridad referencial.

### Decisiones de diseño (confirmadas con el usuario)
1. **Eliminar la heurística en español y EXIGIR un proveedor LLM** para la extracción. Si el
   sistema arranca con `AI_PROVIDER=stub`, la extracción **falla explícitamente** (no hay
   extracción por regex silenciosa). Los embeddings stub (`stable_embedding`) se conservan para
   offline.
2. **Extracción + juez LLM**: tras `extract()`, una segunda pasada LLM juzga la calidad
   (¿concepto/claim enseñable?, ¿relación correcta?), estilo `vet_pedagogical_evidence`
   (`app/ai_provider.py:768`). Para acotar costo, **un único `vet_extraction()` por episodio**
   (no por item), siguiendo la filosofía "una llamada por lote" de la Fase 1.

### Fundamento compartido por los 3 planes (se construye una vez)
- **Guard de proveedor**: `StubAIProvider.extract` (y el nuevo `vet_extraction`) lanzan un error
  claro ("extraction requires an LLM provider"). Validación temprana en el arranque/ingesta
  (`app/ingestion.py`, opcional en `app/dependencies.py`) para fallar rápido y con mensaje útil.
- **Nuevo juez `vet_extraction(*, extraction, text, language)` → `ExtractionVettingResult`** en
  `AIProvider`: recibe la `ExtractionResult` cruda + el texto fuente, devuelve la versión
  filtrada/reparada con `status` y `review_notes` por item (patrón de
  `PedagogicalEvidenceDecision`, `app/schemas.py:1213`). Implementado en `StructuredLLMProvider`;
  el Stub lo deja sin soporte (raise).
- **Sanitización reducida a lo estructural** en el camino LLM: se conserva `_sanitize_quote`
  (cita verbatim contra la fuente — grounding, `app/ai_provider.py:949`), el dedupe
  (`dedupe_preserve_order`/`normalize_text`) y la validación de enum de relación (ya en el
  `field_validator` de `ExtractedRelation`, `app/schemas.py:1181`). **Se elimina** todo gate por
  wordlist.
- **Dobles de prueba**: como los tests corrían con `ai_provider=stub`, se añade un
  `FakeExtractionProvider` (AIProvider de test que devuelve `ExtractionResult`/vetting prefijados)
  inyectado directamente, y/o el patrón `_CapturedClient` de `tests/test_ai_provider.py` para
  mockear la llamada HTTP. Ningún test depende ya de la regex en español.

---

## Plan 4 — "¿Qué es un concepto?" lo decide el LLM (Punto #4)

**Objetivo:** que la validez/canonicalización de conceptos sea juicio del LLM, no de wordlists.

**Cambios:**
- `app/ai_provider.py`:
  - Quitar del **camino LLM** el gating por idioma: en `_sanitize_concepts` (`:851`) eliminar las
    llamadas a `_is_valid_concept`/`_clean_concept_name` (trim de predicados, titleize forzado) y
    dejar solo limpieza de whitespace + dedupe + verificación de `evidence_quotes` verbatim.
  - Reforzar el contrato de `extract()` para que el LLM entregue `canonical_name` ya canónico y
    auto-suficiente (el system prompt actual ya lo pide; ajustarlo para no depender de limpieza
    posterior).
  - La decisión "¿es enseñable / es genérico / es ejemplo?" pasa a `vet_extraction` (el juez).
  - **Eliminar** del `StubAIProvider`: `_extract_concepts`, `_extract_table_concepts`,
    `_extract_section_concepts`, `_extract_pattern_concepts`, `_extract_capitalized_concepts(_from_chunk)`,
    `_is_valid_concept`, `_clean_concept_name`, `_trim_predicate_suffix`, y las wordlists
    `_GENERIC_CONCEPT_TOKENS`/`_GENERIC_HEADINGS`/`_QUALIFIER_TOKENS`/`_CONNECTOR_TOKENS`/
    `_CONCEPT_HEAD_PATTERNS`. `StubAIProvider.extract` → raise.

**Conserva:** `_sanitize_quote` (grounding), `dedupe_preserve_order`, `normalize_text`.

---

## Plan 5 — "¿Qué es un claim?" lo decide el LLM (Punto #5)

**Objetivo:** la "enseñabilidad" de un claim la juzga el LLM; lo determinista es solo el grounding.

**Cambios:**
- `app/ai_provider.py`:
  - **Eliminar** `_looks_like_claim` y `_CLAIM_VERB_MARKERS` (y sus usos en `_extract_claims`/
    `_refine_claims`, que desaparecen con el Stub).
  - En el camino LLM, `_sanitize_claims` (`:885`) conserva solo: texto no vacío, dedupe, y
    **`supporting_quote` verbatim** contra la fuente (grounding estructural). La decisión
    "¿este claim es enseñable / es meta-contenido?" pasa a `vet_extraction`.
- `vet_extraction` marca claims como `keep`/`drop`/`repair` con `review_notes`; los `repair`
  pueden reescribirse a una afirmación limpia (como ya hace `vet_pedagogical_evidence`).

**Conserva:** verificación verbatim de `supporting_quote`, dedupe.

---

## Plan 6 — La relación semántica la infiere el LLM (Punto #6)

**Objetivo:** inferir y validar relaciones con el LLM; lo determinista es enum + integridad
referencial.

**Cambios:**
- `app/ai_provider.py`:
  - **Eliminar** `_infer_relation_type` y `_extract_relations` (Stub).
  - En el camino LLM, `_sanitize_relations` (`:919`) conserva: validación del enum de relación y
    **resolución referencial tolerante** de `from_name`/`to_name` contra los conceptos extraídos
    (relajar el match exacto de `_find_matching_concept` para no perder relaciones por variantes
    de nombre; idealmente el contrato del LLM exige usar `canonical_name` ya emitidos).
  - `vet_extraction` valida que la relación afirmada es correcta y soportada por el texto, y puede
    descartar relaciones alucinadas.

**Conserva:** enum de `ExtractedRelation` (`app/schemas.py:1181`), resolución referencial.

---

## Archivos críticos a modificar
- `app/ai_provider.py` — borrar heurísticas/wordlists del Stub; reducir `_sanitize_*` a checks
  estructurales; añadir `vet_extraction` (abstracto + impl. en `StructuredLLMProvider`);
  `StubAIProvider.extract`/`vet_extraction` → raise.
- `app/schemas.py` — `ExtractionVettingResult` (decisiones keep/drop/repair por item, estilo
  `PedagogicalEvidenceDecision`).
- `app/ingestion.py` — tras `extract()` (`:82`), llamar `vet_extraction()`; guard de proveedor LLM.
- `app/config.py` / `app/dependencies.py` — guard/validación temprana cuando `ai_provider=stub`.
- `tests/` — `FakeExtractionProvider` y mocks (`_CapturedClient`); migrar tests que dependían de
  la extracción Stub en español.

## Riesgos y mitigaciones
- **Tests/CI sin API key** → dobles de prueba inyectados (`FakeExtractionProvider`/mocks); ningún
  test usa la regex. Embeddings siguen offline vía `stable_embedding`.
- **Romper el default `stub`** (decisión consciente) → guard con mensaje claro; documentar en
  README/.env.example que la ingesta exige `AI_PROVIDER=anthropic|openai_compatible`.
- **Alucinación de relaciones/claims** → `vet_extraction` como filtro + grounding verbatim de
  citas; descartar lo no soportado por el texto.
- **Latencia/costo** → un solo `vet_extraction` por episodio (no por item); logs de `duration_ms`.
- **Regresión de calidad de extracción** → comparar nº de conceptos/claims/relaciones y revisión
  manual sobre un corpus fijo (es/en/pt) antes y después.