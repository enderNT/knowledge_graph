```text
ArcadeDB = memoria estructural / knowledge graph / búsqueda híbrida
MCP = interfaz segura para que el agente lea, inserte y conecte conocimiento
LLM = extractor, normalizador y razonador, no dueño de la base
```

---

## 1. No lo diseñes como “un grafo universal”

Diseñaría ArcadeDB con **tres capas internas**:

```text
1. Episodic Layer
   Fragmentos crudos, fuentes, notas originales.

2. Semantic Graph Layer
   Conceptos, temas, entidades, relaciones, claims.

3. Routing / Index Layer
   Dominios, subdominios, aliases, embeddings, índices.
```

La capa importante para evitar el problema del grafo gigante es la tercera.

No preguntas:

```text
¿Dónde va esto en todo el grafo?
```

Preguntas:

```text
¿A qué dominio/subdominio pertenece?
¿Qué 20-50 nodos candidatos se parecen?
¿Esto crea algo nuevo o actualiza algo existente?
```

---

## 2. Modelo inicial de tipos en ArcadeDB

ArcadeDB requiere declarar tipos de documento, vértice o arista antes de insertar registros; luego puedes añadir propiedades, índices y validaciones. ([docs.arcadedb.com][2])

Yo empezaría con estos **vertex types**:

```sql
CREATE VERTEX TYPE Concept IF NOT EXISTS;
CREATE VERTEX TYPE Topic IF NOT EXISTS;
CREATE VERTEX TYPE Domain IF NOT EXISTS;
CREATE VERTEX TYPE Source IF NOT EXISTS;
CREATE VERTEX TYPE Episode IF NOT EXISTS;
CREATE VERTEX TYPE Claim IF NOT EXISTS;
CREATE VERTEX TYPE Alias IF NOT EXISTS;
```

Y estos **edge types**:

```sql
CREATE EDGE TYPE PART_OF IF NOT EXISTS;
CREATE EDGE TYPE IS_A IF NOT EXISTS;
CREATE EDGE TYPE RELATED_TO IF NOT EXISTS;
CREATE EDGE TYPE EXPLAINS IF NOT EXISTS;
CREATE EDGE TYPE CONTRASTS_WITH IF NOT EXISTS;
CREATE EDGE TYPE PREREQUISITE_FOR IF NOT EXISTS;
CREATE EDGE TYPE SUPPORTED_BY IF NOT EXISTS;
CREATE EDGE TYPE MENTIONED_IN IF NOT EXISTS;
CREATE EDGE TYPE ALIAS_OF IF NOT EXISTS;
```

En ArcadeDB puedes crear vértices y aristas desde SQL; `CREATE EDGE` conecta vértices existentes o resultados de queries. ([docs.arcadedb.com][3])

---

## 3. Nodos mínimos

### `Episode`

Es el texto original que tú mandas desde el celular.

```json
{
  "type": "Episode",
  "id": "ep_2026_06_02_001",
  "text": "Fragmento original...",
  "source_type": "manual_input",
  "created_at": "2026-06-02T10:00:00",
  "language": "es",
  "status": "processed"
}
```

Este nodo es importantísimo porque el grafo puede equivocarse, pero el episodio queda como fuente.

---

### `Concept`

Es una idea reusable.

```json
{
  "type": "Concept",
  "canonical_name": "Condicionamiento clásico",
  "normalized_name": "condicionamiento clasico",
  "description": "Tipo de aprendizaje asociativo...",
  "domain": "Psicología",
  "confidence": 0.91,
  "created_at": "...",
  "updated_at": "..."
}
```

---

### `Claim`

Esto es una afirmación puntual.

```json
{
  "type": "Claim",
  "text": "El estímulo neutro puede convertirse en estímulo condicionado tras repetidas asociaciones.",
  "confidence": 0.86,
  "status": "active"
}
```

No todo debe ser Concept. Muchas cosas deben ser **claims**.

---

### `Domain` / `Topic`

Sirven para routing.

```json
{
  "type": "Domain",
  "name": "Psicología"
}
```

```json
{
  "type": "Topic",
  "name": "Aprendizaje asociativo"
}
```

---

## 4. Relaciones mínimas

Ejemplo:

```text
Condicionamiento clásico
  PART_OF → Aprendizaje asociativo
  RELATED_TO → Pavlov
  CONTRASTS_WITH → Condicionamiento operante
  MENTIONED_IN → Episode_001

Claim_001
  SUPPORTED_BY → Episode_001
  EXPLAINS → Condicionamiento clásico
```

Separar `Concept` de `Claim` te evita ensuciar el grafo. Un concepto es una cosa estable; un claim es una afirmación específica extraída de una fuente.

---

## 5. El flujo de inserción MCP

Tu MCP no debería exponer una herramienta genérica tipo:

```text
run_any_sql()
```

Eso sería peligroso y caótico.

Mejor exponer herramientas semánticas:

```text
add_episode
extract_candidate_graph
find_similar_nodes
resolve_entity
create_concept
create_claim
create_relation
get_neighborhood
search_knowledge
```

---

## 6. Herramientas MCP mínimas

### 1. `add_knowledge_fragment`

Entrada:

```json
{
  "text": "Texto que mandas desde el celular",
  "source_type": "manual_input",
  "tags": ["psicología", "aprendizaje"]
}
```

Qué hace:

```text
1. Guarda Episode.
2. Genera embedding.
3. Extrae conceptos/claims/relaciones.
4. Clasifica dominio.
5. Busca candidatos locales.
6. Inserta o actualiza.
```

Esta sería la herramienta principal.

---

### 2. `search_candidates`

Entrada:

```json
{
  "query": "condicionamiento clásico",
  "domain_hint": "Psicología",
  "limit": 30
}
```

Busca por:

```text
nombre
alias
full-text
embedding
vecindario del dominio
```

ArcadeDB ya tiene índices, búsqueda full-text y vector search; su documentación menciona HNSW y LSMVectorIndex para similitud vectorial persistente. ([docs.arcadedb.com][4])

---

### 3. `upsert_concept`

Entrada:

```json
{
  "canonical_name": "Condicionamiento clásico",
  "aliases": ["classical conditioning", "condicionamiento pavloviano"],
  "domain": "Psicología",
  "description": "..."
}
```

Qué hace:

```text
si existe → actualiza aliases/descripción
si no existe → crea nodo nuevo
```

---

### 4. `link_concepts`

Entrada:

```json
{
  "from": "Condicionamiento clásico",
  "relation": "CONTRASTS_WITH",
  "to": "Condicionamiento operante",
  "evidence_episode_id": "ep_001"
}
```

---

### 5. `get_neighborhood`

Entrada:

```json
{
  "concept": "Condicionamiento clásico",
  "depth": 2
}
```

Devuelve:

```text
nodo central
vecinos
relaciones
claims relacionados
fuentes
```

Esta herramienta será clave para que el agente conteste sin cargar todo el grafo.

---

## 7. Diseño anti-regresión: routing antes de insertar

Tu problema principal se resuelve así:

```text
Fragmento
  ↓
clasificación de dominio
  ↓
búsqueda de candidatos solo en ese dominio
  ↓
entity resolution
  ↓
inserción local
```

Ejemplo:

```text
Texto nuevo:
"La extinción ocurre cuando el estímulo condicionado aparece repetidamente sin el estímulo incondicionado."

Routing:
Psicología → Aprendizaje → Condicionamiento clásico

Candidatos:
- Condicionamiento clásico
- Estímulo condicionado
- Estímulo incondicionado
- Extinción
- Aprendizaje asociativo

Inserción:
Extinción PART_OF Condicionamiento clásico
Claim SUPPORTED_BY Episode
```

No comparas contra todo.

---

## 8. Esquema de propiedades recomendado

### `Concept`

```sql
CREATE PROPERTY Concept.uid STRING;
CREATE PROPERTY Concept.canonical_name STRING;
CREATE PROPERTY Concept.normalized_name STRING;
CREATE PROPERTY Concept.description STRING;
CREATE PROPERTY Concept.domain STRING;
CREATE PROPERTY Concept.embedding LIST;
CREATE PROPERTY Concept.created_at DATETIME;
CREATE PROPERTY Concept.updated_at DATETIME;

CREATE INDEX ON Concept (uid) UNIQUE;
CREATE INDEX ON Concept (normalized_name) UNIQUE;
CREATE INDEX ON Concept (domain) NOTUNIQUE;
```

### `Episode`

```sql
CREATE PROPERTY Episode.uid STRING;
CREATE PROPERTY Episode.text STRING;
CREATE PROPERTY Episode.source_type STRING;
CREATE PROPERTY Episode.language STRING;
CREATE PROPERTY Episode.embedding LIST;
CREATE PROPERTY Episode.created_at DATETIME;

CREATE INDEX ON Episode (uid) UNIQUE;
CREATE INDEX ON Episode (created_at) NOTUNIQUE;
```

### `Claim`

```sql
CREATE PROPERTY Claim.uid STRING;
CREATE PROPERTY Claim.text STRING;
CREATE PROPERTY Claim.normalized_text STRING;
CREATE PROPERTY Claim.confidence DOUBLE;
CREATE PROPERTY Claim.status STRING;
CREATE PROPERTY Claim.embedding LIST;

CREATE INDEX ON Claim (uid) UNIQUE;
```

ArcadeDB usa tipos, propiedades e índices de forma explícita; por ejemplo, para migrar desde Neo4j, su propia guía muestra el patrón `CREATE VERTEX TYPE`, `CREATE EDGE TYPE`, `CREATE PROPERTY`, `CREATE INDEX`. ([ArcadeDB][5])

---

## 9. Naming convention

Esto importa más de lo que parece.

Usaría:

```text
Tipos de vértice: PascalCase
Concept, Episode, Claim, Domain

Tipos de arista: UPPER_SNAKE_CASE
PART_OF, IS_A, SUPPORTED_BY

Propiedades: snake_case
canonical_name, normalized_name, created_at
```

Y todos los nodos importantes con:

```text
uid
created_at
updated_at
source_confidence
```

---

## 10. SQL conceptual para inserción

Crear concepto:

```sql
CREATE VERTEX Concept
SET uid = :uid,
    canonical_name = :canonical_name,
    normalized_name = :normalized_name,
    description = :description,
    domain = :domain,
    created_at = sysdate(),
    updated_at = sysdate();
```

Crear episodio:

```sql
CREATE VERTEX Episode
SET uid = :uid,
    text = :text,
    source_type = :source_type,
    language = :language,
    created_at = sysdate();
```

Conectar concepto con episodio:

```sql
CREATE EDGE MENTIONED_IN
FROM (SELECT FROM Concept WHERE uid = :concept_uid)
TO (SELECT FROM Episode WHERE uid = :episode_uid)
SET confidence = :confidence;
```

---

## 11. La regla más importante para el MCP

El MCP debe ser **opinionado**, no solo un wrapper de ArcadeDB.

Malo:

```text
query_arcadedb(sql)
```

Bueno:

```text
add_knowledge_fragment(text)
find_concept(name)
get_concept_context(name)
upsert_relation(from, relation, to)
```

Porque si le das SQL libre al agente, eventualmente hará queries inconsistentes, duplicará nodos o romperá el diseño.

Puedes dejar `query_readonly` para debugging, pero no como herramienta principal.

---

## 12. Pipeline recomendado para `add_knowledge_fragment`

```text
1. Recibir texto.
2. Crear Episode inmediatamente.
3. LLM extrae:
   - conceptos
   - claims
   - relaciones
   - dominio probable
   - aliases
4. Crear embedding del fragmento.
5. Buscar candidatos:
   - por normalized_name
   - por alias
   - por vector
   - por dominio
6. Resolver entidades:
   - exact match
   - alias match
   - semantic match
   - create new
7. Crear/actualizar Concept.
8. Crear Claim.
9. Crear edges:
   - MENTIONED_IN
   - SUPPORTED_BY
   - PART_OF
   - RELATED_TO
10. Devolver resumen de lo insertado.
```

Salida ideal:

```json
{
  "episode_id": "ep_001",
  "created_concepts": ["Extinción"],
  "updated_concepts": ["Condicionamiento clásico"],
  "created_claims": 2,
  "relations": [
    ["Extinción", "PART_OF", "Condicionamiento clásico"]
  ],
  "needs_review": []
}
```

---

## 13. Cómo evitar duplicados

Usa varias capas:

```text
1. normalized_name UNIQUE
2. Alias → ALIAS_OF → Concept
3. búsqueda vectorial top-k
4. revisión LLM solo entre candidatos
5. si hay duda, marcar needs_review
```

Ejemplo:

```text
"condicionamiento pavloviano"
  ALIAS_OF → "Condicionamiento clásico"
```

No lo guardes como dos conceptos separados.

---

## 14. Stack mínimo

```text
ArcadeDB Docker
MCP Server en Node.js o Python
LLM extractor
Embeddings
Cliente: Manus / Claude Desktop / agente compatible MCP
```

ArcadeDB tiene imagen Docker oficial y soporta grafo, documentos, búsqueda y vector embedding según su descripción pública.

---

## 16. Mi diseño para MVP

```text
ArcadeDB
├── Vertex
│   ├── Episode
│   ├── Concept
│   ├── Claim
│   ├── Domain
│   ├── Topic
│   └── Alias
│
├── Edges
│   ├── PART_OF
│   ├── IS_A
│   ├── RELATED_TO
│   ├── CONTRASTS_WITH
│   ├── PREREQUISITE_FOR
│   ├── SUPPORTED_BY
│   ├── MENTIONED_IN
│   └── ALIAS_OF
│
└── MCP Tools
    ├── add_knowledge_fragment
    ├── search_candidates
    ├── upsert_concept
    ├── link_concepts
    └── get_neighborhood
```

Con eso ya puedes empezar bien.

La frase de diseño sería:

> **ArcadeDB guarda el conocimiento como grafo/documento/vector; MCP impone las reglas de inserción; el LLM solo propone estructura y resuelve ambigüedad local.**
