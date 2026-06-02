from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from app.arcadedb_client import ArcadeDBClient
from app.config import Settings
from app.schema_bootstrap import ensure_database_and_schema
from app.schemas import (
    ALLOWED_RELATIONS,
    CandidateHit,
    ClaimRecord,
    ConceptRecord,
    EpisodeRecord,
    IngestionSummary,
    JobRecord,
    NeighborhoodNode,
    NeighborhoodRelation,
    NeighborhoodResponse,
    UpsertConceptRequest,
    concept_ref_to_normalized,
)
from app.utils import cosine_similarity, make_prefixed_id, normalize_text, utcnow_iso
from app.utils import fit_embedding_dimensions


class KnowledgeStore(Protocol):
    async def bootstrap_schema(self) -> None: ...
    async def check_ready(self) -> bool: ...
    async def create_episode(self, *, uid: str, text: str, source_type: str, tags: list[str], language: str) -> EpisodeRecord: ...
    async def create_job(self, *, uid: str, episode_id: str, status: str) -> JobRecord: ...
    async def get_job(self, job_id: str) -> JobRecord | None: ...
    async def update_job(self, job_id: str, *, status: str, result: IngestionSummary | None = None, error: str | None = None) -> JobRecord: ...
    async def get_episode(self, episode_id: str) -> EpisodeRecord | None: ...
    async def update_episode(self, episode_id: str, *, status: str, error_message: str | None = None, embedding: list[float] | None = None) -> EpisodeRecord: ...
    async def upsert_concept(self, payload: UpsertConceptRequest, *, embedding: list[float], source_confidence: float) -> tuple[ConceptRecord, bool]: ...
    async def get_concept(self, ref: str) -> ConceptRecord | None: ...
    async def search_candidates(self, *, query: str, domain_hint: str | None, query_embedding: list[float] | None, limit: int) -> list[CandidateHit]: ...
    async def create_claim(self, *, text: str, confidence: float, status: str, embedding: list[float]) -> ClaimRecord: ...
    async def create_relation(self, *, from_ref: str, relation: str, to_ref: str, evidence_episode_id: str | None, confidence: float | None = None) -> bool: ...
    async def link_concept_to_episode(self, concept_uid: str, episode_id: str, confidence: float | None = None) -> None: ...
    async def link_claim_to_episode(self, claim_uid: str, episode_id: str, confidence: float | None = None) -> None: ...
    async def link_claim_to_concept(self, claim_uid: str, concept_uid: str, confidence: float | None = None) -> None: ...
    async def get_neighborhood(self, concept_ref: str, depth: int) -> NeighborhoodResponse | None: ...


class ArcadeKnowledgeStore:
    def __init__(self, settings: Settings, client: ArcadeDBClient) -> None:
        self.settings = settings
        self.client = client
        self._embedding_dimensions_cache: int | None = None

    async def bootstrap_schema(self) -> None:
        await ensure_database_and_schema(self.client, self.settings)

    async def check_ready(self) -> bool:
        return await self.client.ready() and await self.client.database_exists()

    async def create_episode(self, *, uid: str, text: str, source_type: str, tags: list[str], language: str) -> EpisodeRecord:
        created_at = utcnow_iso()
        await self.client.command(
            (
                "CREATE VERTEX Episode SET uid = :uid, text = :text, source_type = :source_type, tags = :tags, "
                "language = :language, status = :status, created_at = :created_at"
            ),
            {
                "uid": uid,
                "text": text,
                "source_type": source_type,
                "tags": tags,
                "language": language,
                "status": "queued",
                "created_at": created_at,
            },
        )
        return EpisodeRecord(
            uid=uid,
            text=text,
            source_type=source_type,
            tags=tags,
            language=language,
            status="queued",
            created_at=created_at,
            error_message=None,
        )

    async def create_job(self, *, uid: str, episode_id: str, status: str) -> JobRecord:
        now = utcnow_iso()
        await self.client.command(
            (
                "INSERT INTO IngestionJob SET uid = :uid, episode_id = :episode_id, status = :status, "
                "result = :result, error = :error, created_at = :created_at, updated_at = :updated_at"
            ),
            {
                "uid": uid,
                "episode_id": episode_id,
                "status": status,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        return JobRecord(uid=uid, episode_id=episode_id, status=status, created_at=now, updated_at=now)

    async def get_job(self, job_id: str) -> JobRecord | None:
        rows = await self.client.query(
            (
                "SELECT uid, episode_id, status, result, error, created_at, updated_at "
                "FROM IngestionJob WHERE uid = :uid LIMIT 1"
            ),
            {"uid": job_id},
        )
        if not rows:
            return None
        return self._job_from_row(rows[0])

    async def update_job(
        self,
        job_id: str,
        *,
        status: str,
        result: IngestionSummary | None = None,
        error: str | None = None,
    ) -> JobRecord:
        updated_at = utcnow_iso()
        await self.client.command(
            (
                "UPDATE IngestionJob SET status = :status, result = :result, error = :error, "
                "updated_at = :updated_at WHERE uid = :uid"
            ),
            {
                "uid": job_id,
                "status": status,
                "result": result.model_dump_json() if result else None,
                "error": error,
                "updated_at": updated_at,
            },
        )
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"job not found after update: {job_id}")
        return job

    async def get_episode(self, episode_id: str) -> EpisodeRecord | None:
        rows = await self.client.query(
            (
                "SELECT uid, text, source_type, tags, language, status, error_message, created_at "
                "FROM Episode WHERE uid = :uid LIMIT 1"
            ),
            {"uid": episode_id},
        )
        if not rows:
            return None
        rows[0]["tags"] = rows[0].get("tags") or []
        return EpisodeRecord.model_validate(rows[0])

    async def update_episode(
        self,
        episode_id: str,
        *,
        status: str,
        error_message: str | None = None,
        embedding: list[float] | None = None,
    ) -> EpisodeRecord:
        command = "UPDATE Episode SET status = :status, error_message = :error_message"
        params: dict[str, Any] = {
            "uid": episode_id,
            "status": status,
            "error_message": error_message,
        }
        if embedding is not None:
            command += ", embedding = :embedding"
            params["embedding"] = await self._normalize_embedding(embedding)
        command += " WHERE uid = :uid"
        await self.client.command(command, params)
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"episode not found after update: {episode_id}")
        return episode

    async def upsert_concept(
        self,
        payload: UpsertConceptRequest,
        *,
        embedding: list[float],
        source_confidence: float,
    ) -> tuple[ConceptRecord, bool]:
        embedding = await self._normalize_embedding(embedding)
        normalized_name = normalize_text(payload.canonical_name)
        existing = await self.get_concept(payload.canonical_name)
        now = utcnow_iso()
        if existing:
            aliases = _merge_alias_lists(existing.aliases, payload.aliases)
            await self.client.command(
                (
                    "UPDATE Concept SET canonical_name = :canonical_name, description = :description, "
                    "domain = :domain, aliases = :aliases, embedding = :embedding, "
                    "source_confidence = :source_confidence, updated_at = :updated_at "
                    "WHERE uid = :uid"
                ),
                {
                    "uid": existing.uid,
                    "canonical_name": payload.canonical_name,
                    "description": payload.description or existing.description,
                    "domain": payload.domain,
                    "aliases": aliases,
                    "embedding": embedding,
                    "source_confidence": source_confidence,
                    "updated_at": now,
                },
            )
            await self._ensure_domain(payload.domain)
            await self._ensure_aliases(existing.uid, aliases)
            concept = await self.get_concept(existing.uid)
            if not concept:
                raise ValueError("concept missing after update")
            return concept, False

        concept_uid = make_prefixed_id("cn")
        await self.client.command(
            (
                "CREATE VERTEX Concept SET uid = :uid, canonical_name = :canonical_name, "
                "normalized_name = :normalized_name, description = :description, domain = :domain, "
                "aliases = :aliases, embedding = :embedding, source_confidence = :source_confidence, "
                "created_at = :created_at, updated_at = :updated_at"
            ),
            {
                "uid": concept_uid,
                "canonical_name": payload.canonical_name,
                "normalized_name": normalized_name,
                "description": payload.description,
                "domain": payload.domain,
                "aliases": payload.aliases,
                "embedding": embedding,
                "source_confidence": source_confidence,
                "created_at": now,
                "updated_at": now,
            },
        )
        await self._ensure_domain(payload.domain)
        await self._ensure_aliases(concept_uid, payload.aliases)
        concept = await self.get_concept(concept_uid)
        if not concept:
            raise ValueError("concept missing after create")
        return concept, True

    async def get_concept(self, ref: str) -> ConceptRecord | None:
        normalized = concept_ref_to_normalized(ref)
        rows = await self.client.query(
            (
                "SELECT uid, canonical_name, normalized_name, domain, description, aliases, embedding, "
                "created_at, updated_at FROM Concept "
                "WHERE uid = :ref OR normalized_name = :normalized OR canonical_name = :ref LIMIT 1"
            ),
            {"ref": ref, "normalized": normalized},
        )
        if rows:
            return ConceptRecord.model_validate(rows[0])

        alias_scan_rows = await self.client.query(
            (
                "SELECT uid, canonical_name, normalized_name, domain, description, aliases, embedding, "
                "created_at, updated_at FROM Concept"
            ),
        )
        for row in alias_scan_rows:
            aliases = row.get("aliases") or []
            if any(normalize_text(alias) == normalized for alias in aliases):
                row["aliases"] = aliases
                return ConceptRecord.model_validate(row)
        return None

    async def search_candidates(
        self,
        *,
        query: str,
        domain_hint: str | None,
        query_embedding: list[float] | None,
        limit: int,
    ) -> list[CandidateHit]:
        normalized = normalize_text(query)
        candidates: dict[str, CandidateHit] = {}
        if query_embedding:
            query_embedding = await self._normalize_embedding(query_embedding)

        exact_rows = await self.client.query(
            (
                "SELECT uid, canonical_name, domain, description, aliases, embedding FROM Concept "
                "WHERE normalized_name = :normalized LIMIT :limit"
            ),
            {"normalized": normalized, "limit": limit},
        )
        for row in exact_rows:
            if domain_hint and row.get("domain") != domain_hint:
                continue
            candidates[row["uid"]] = CandidateHit(
                uid=row["uid"],
                canonical_name=row["canonical_name"],
                domain=row["domain"],
                description=row.get("description") or "",
                score=1.0,
                reason="normalized_name",
            )

        alias_scan_rows = await self.client.query(
            (
                "SELECT uid, canonical_name, domain, description, aliases, embedding FROM Concept"
            ),
        )
        for row in alias_scan_rows:
            aliases = row.get("aliases") or []
            matched_alias = next(
                (alias for alias in aliases if normalize_text(alias) == normalized),
                None,
            )
            if not matched_alias:
                continue
            concept_uid = row["uid"]
            if domain_hint and row.get("domain") != domain_hint:
                continue
            candidates[concept_uid] = CandidateHit(
                uid=concept_uid,
                canonical_name=row["canonical_name"],
                domain=row["domain"],
                description=row.get("description") or "",
                score=0.98,
                reason="alias",
                matched_alias=matched_alias,
            )

        try:
            fulltext_rows = await self.client.query(
                (
                    "SELECT uid, canonical_name, domain, description, aliases, embedding FROM Concept "
                    "WHERE SEARCH_INDEX('Concept[canonical_name]', :query)"
                ),
                {"query": query},
            )
        except httpx.HTTPStatusError:
            fulltext_rows = []
        for row in fulltext_rows[: limit * 2]:
            if domain_hint and row.get("domain") != domain_hint:
                continue
            score = 0.82
            existing = candidates.get(row["uid"])
            if existing and existing.score >= score:
                continue
            candidates[row["uid"]] = CandidateHit(
                uid=row["uid"],
                canonical_name=row["canonical_name"],
                domain=row["domain"],
                description=row.get("description") or "",
                score=score,
                reason="full_text",
            )

        if query_embedding:
            try:
                vector_rows = await self.client.query(
                    (
                        "SELECT uid, canonical_name, domain, description, embedding, distance FROM "
                        "(SELECT expand(vectorNeighbors('Concept[embedding]', :query_vector, :limit)))"
                    ),
                    {"query_vector": query_embedding, "limit": limit * self.settings.candidate_limit_multiplier},
                )
            except httpx.HTTPStatusError:
                vector_rows = []
            for row in vector_rows:
                if domain_hint and row.get("domain") != domain_hint:
                    continue
                score = max(0.0, min(1.0, 1.0 - float(row.get("distance", 1.0))))
                existing = candidates.get(row["uid"])
                if existing and existing.score >= score:
                    continue
                candidates[row["uid"]] = CandidateHit(
                    uid=row["uid"],
                    canonical_name=row["canonical_name"],
                    domain=row.get("domain") or "",
                    description=row.get("description") or "",
                    score=score,
                    reason="vector",
                )

        results = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        return results[:limit]

    async def create_claim(self, *, text: str, confidence: float, status: str, embedding: list[float]) -> ClaimRecord:
        embedding = await self._normalize_embedding(embedding)
        normalized_text = normalize_text(text)
        existing = await self.client.query(
            (
                "SELECT uid, text, normalized_text, confidence, status, created_at, embedding "
                "FROM Claim WHERE normalized_text = :normalized_text LIMIT 1"
            ),
            {"normalized_text": normalized_text},
        )
        if existing:
            return ClaimRecord.model_validate(existing[0])

        uid = make_prefixed_id("cl")
        created_at = utcnow_iso()
        await self.client.command(
            (
                "CREATE VERTEX Claim SET uid = :uid, text = :text, normalized_text = :normalized_text, "
                "confidence = :confidence, status = :status, embedding = :embedding, created_at = :created_at"
            ),
            {
                "uid": uid,
                "text": text,
                "normalized_text": normalized_text,
                "confidence": confidence,
                "status": status,
                "embedding": embedding,
                "created_at": created_at,
            },
        )
        return ClaimRecord(
            uid=uid,
            text=text,
            normalized_text=normalized_text,
            confidence=confidence,
            status=status,
            created_at=created_at,
            embedding=embedding,
        )

    async def create_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None,
        confidence: float | None = None,
    ) -> bool:
        from_concept = await self.get_concept(from_ref)
        to_concept = await self.get_concept(to_ref)
        if not from_concept or not to_concept:
            return False
        if from_concept.uid == to_concept.uid:
            return False
        return await self._create_edge_if_missing(
            edge_type=relation,
            from_type="Concept",
            from_uid=from_concept.uid,
            to_type="Concept",
            to_uid=to_concept.uid,
            attributes={
                "confidence": confidence,
                "evidence_episode_id": evidence_episode_id,
                "created_at": utcnow_iso(),
            },
        )

    async def link_concept_to_episode(self, concept_uid: str, episode_id: str, confidence: float | None = None) -> None:
        await self._create_edge_if_missing(
            edge_type="MENTIONED_IN",
            from_type="Concept",
            from_uid=concept_uid,
            to_type="Episode",
            to_uid=episode_id,
            attributes={"confidence": confidence, "created_at": utcnow_iso()},
        )

    async def link_claim_to_episode(self, claim_uid: str, episode_id: str, confidence: float | None = None) -> None:
        await self._create_edge_if_missing(
            edge_type="SUPPORTED_BY",
            from_type="Claim",
            from_uid=claim_uid,
            to_type="Episode",
            to_uid=episode_id,
            attributes={"confidence": confidence, "created_at": utcnow_iso()},
        )

    async def link_claim_to_concept(self, claim_uid: str, concept_uid: str, confidence: float | None = None) -> None:
        await self._create_edge_if_missing(
            edge_type="EXPLAINS",
            from_type="Claim",
            from_uid=claim_uid,
            to_type="Concept",
            to_uid=concept_uid,
            attributes={"confidence": confidence, "created_at": utcnow_iso()},
        )

    async def get_neighborhood(self, concept_ref: str, depth: int) -> NeighborhoodResponse | None:
        concept = await self.get_concept(concept_ref)
        if not concept:
            return None

        nodes: dict[str, NeighborhoodNode] = {}
        relations: dict[tuple[str, str, str, str | None], NeighborhoodRelation] = {}
        frontier = [concept]
        max_hops = 2 if depth == 2 else 1

        for hop in range(max_hops):
            next_frontier: list[ConceptRecord] = []
            seen_next: set[str] = set()
            for current in frontier:
                edge_rows = await self._fetch_neighborhood_edge_rows(current.uid)
                for row in edge_rows:
                    node = self._build_neighborhood_node(row, current.uid)
                    relation = self._build_neighborhood_relation(row)
                    if not node or not relation or node.uid == concept.uid:
                        continue

                    nodes[node.uid] = node
                    relation_key = (
                        relation.from_uid,
                        relation.relation,
                        relation.to_uid,
                        relation.evidence_episode_id,
                    )
                    relations[relation_key] = relation

                    if hop == 0 and node.type == "Concept" and node.uid not in seen_next:
                        related_concept = await self.get_concept(node.uid)
                        if related_concept:
                            next_frontier.append(related_concept)
                            seen_next.add(node.uid)
            frontier = next_frontier
            if not frontier:
                break

        claim_rows = await self._safe_query(
            (
                "MATCH {type: Concept, as: c, where: (uid = :uid)}"
                ".in('EXPLAINS'){as: claim}.out('SUPPORTED_BY'){as: episode} "
                "RETURN claim.uid as claim_uid, claim.text as claim_text, claim.confidence as claim_confidence, "
                "episode.uid as episode_uid, episode.text as episode_text, episode.status as episode_status"
            ),
            {"uid": concept.uid},
        )

        claims: list[dict[str, Any]] = []
        episodes: dict[str, dict[str, Any]] = {}
        for row in claim_rows:
            if row.get("claim_uid"):
                claims.append(
                    {
                        "uid": row["claim_uid"],
                        "text": row.get("claim_text", ""),
                        "confidence": row.get("claim_confidence"),
                    }
                )
            if row.get("episode_uid"):
                episodes[row["episode_uid"]] = {
                    "uid": row["episode_uid"],
                    "text": row.get("episode_text", ""),
                    "status": row.get("episode_status", ""),
                }

        return NeighborhoodResponse(
            concept=NeighborhoodNode(
                uid=concept.uid,
                type="Concept",
                name=concept.canonical_name,
                domain=concept.domain,
                description=concept.description,
            ),
            nodes=list(nodes.values())[: 50 if depth == 2 else 20],
            relations=list(relations.values())[: 50 if depth == 2 else 20],
            claims=claims,
            episodes=list(episodes.values()),
        )

    async def _fetch_neighborhood_edge_rows(self, concept_uid: str) -> list[dict[str, Any]]:
        outgoing = await self._safe_query(
            (
                "MATCH {type: Concept, as: c, where: (uid = :uid)}"
                ".outE(){as: e}.inV(){as: n} "
                "RETURN c.uid as from_uid, c.canonical_name as from_name, "
                "n.uid as to_uid, coalesce(n.canonical_name, n.text, n.name, n.value, n.uid) as to_name, "
                "n.canonical_name as to_concept_name, n.text as to_text, coalesce(n.domain, '') as to_domain, "
                "coalesce(n.description, '') as to_description, type(e) as relation, "
                "e.confidence as confidence, e.evidence_episode_id as evidence_episode_id"
            ),
            {"uid": concept_uid},
        )
        incoming = await self._safe_query(
            (
                "MATCH {type: Concept, as: c, where: (uid = :uid)}"
                ".inE(){as: e}.outV(){as: n} "
                "RETURN n.uid as from_uid, coalesce(n.canonical_name, n.text, n.name, n.value, n.uid) as from_name, "
                "n.canonical_name as from_concept_name, n.text as from_text, coalesce(n.domain, '') as from_domain, "
                "coalesce(n.description, '') as from_description, c.uid as to_uid, c.canonical_name as to_name, "
                "type(e) as relation, e.confidence as confidence, e.evidence_episode_id as evidence_episode_id"
            ),
            {"uid": concept_uid},
        )
        return [*outgoing, *incoming]

    def _build_neighborhood_node(self, row: dict[str, Any], current_uid: str) -> NeighborhoodNode | None:
        if row.get("from_uid") == current_uid:
            prefix = "to"
        elif row.get("to_uid") == current_uid:
            prefix = "from"
        else:
            return None

        node_uid = row.get(f"{prefix}_uid")
        if not node_uid:
            return None

        concept_name = row.get(f"{prefix}_concept_name")
        return NeighborhoodNode(
            uid=node_uid,
            type="Concept" if concept_name else "neighbor",
            name=row.get(f"{prefix}_name") or node_uid,
            domain=(row.get(f"{prefix}_domain") or None) if concept_name else None,
            description=(row.get(f"{prefix}_description") or None) if concept_name else None,
        )

    def _build_neighborhood_relation(self, row: dict[str, Any]) -> NeighborhoodRelation | None:
        from_uid = row.get("from_uid")
        to_uid = row.get("to_uid")
        if not from_uid or not to_uid:
            return None

        return NeighborhoodRelation(
            from_uid=from_uid,
            from_name=row.get("from_name") or from_uid,
            relation=row.get("relation") or "RELATED_TO",
            to_uid=to_uid,
            to_name=row.get("to_name") or to_uid,
            confidence=row.get("confidence"),
            evidence_episode_id=row.get("evidence_episode_id"),
        )

    async def _safe_query(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        *,
        language: str = "sql",
    ) -> list[dict[str, Any]]:
        try:
            return await self.client.query(command, params, language=language)
        except httpx.HTTPStatusError:
            return []

    async def _ensure_domain(self, name: str) -> None:
        normalized_name = normalize_text(name)
        rows = await self.client.query(
            "SELECT uid FROM Domain WHERE normalized_name = :normalized_name LIMIT 1",
            {"normalized_name": normalized_name},
        )
        if rows:
            return
        await self.client.command(
            (
                "CREATE VERTEX Domain SET uid = :uid, name = :name, normalized_name = :normalized_name, "
                "created_at = :created_at"
            ),
            {
                "uid": make_prefixed_id("dm"),
                "name": name,
                "normalized_name": normalized_name,
                "created_at": utcnow_iso(),
            },
        )

    async def _ensure_aliases(self, concept_uid: str, aliases: list[str]) -> None:
        for alias in aliases:
            normalized_value = normalize_text(alias)
            try:
                existing = await self.client.query(
                    "SELECT uid FROM Alias WHERE normalized_value = :normalized_value LIMIT 1",
                    {"normalized_value": normalized_value},
                )
                if existing:
                    alias_uid = existing[0]["uid"]
                else:
                    alias_uid = make_prefixed_id("al")
                    await self.client.command(
                        (
                            "CREATE VERTEX Alias SET uid = :uid, value = :value, normalized_value = :normalized_value, "
                            "created_at = :created_at"
                        ),
                        {
                            "uid": alias_uid,
                            "value": alias,
                            "normalized_value": normalized_value,
                            "created_at": utcnow_iso(),
                        },
                    )
                await self._create_edge_if_missing(
                    edge_type="ALIAS_OF",
                    from_type="Alias",
                    from_uid=alias_uid,
                    to_type="Concept",
                    to_uid=concept_uid,
                    attributes={"created_at": utcnow_iso()},
                )
            except httpx.HTTPStatusError:
                continue

    async def _edge_exists(
        self,
        *,
        edge_type: str,
        from_type: str,
        from_uid: str,
        to_type: str,
        to_uid: str,
    ) -> bool:
        rows = await self.client.query(
            (
                f"MATCH {{type: {from_type}, as: source, where: (uid = :from_uid)}}"
                f".out('{edge_type}'){{type: {to_type}, as: target, where: (uid = :to_uid)}} "
                "RETURN target.uid as target_uid LIMIT 1"
            ),
            {"from_uid": from_uid, "to_uid": to_uid},
            language="sql",
        )
        return bool(rows)

    async def _create_edge_if_missing(
        self,
        *,
        edge_type: str,
        from_type: str,
        from_uid: str,
        to_type: str,
        to_uid: str,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        if edge_type not in ALLOWED_RELATIONS:
            raise ValueError(f"unsupported relation: {edge_type}")
        if await self._edge_exists(
            edge_type=edge_type,
            from_type=from_type,
            from_uid=from_uid,
            to_type=to_type,
            to_uid=to_uid,
        ):
            return False

        params: dict[str, Any] = {"from_uid": from_uid, "to_uid": to_uid}
        set_parts: list[str] = []
        for key, value in (attributes or {}).items():
            params[key] = value
            set_parts.append(f"{key} = :{key}")

        command = (
            f"CREATE EDGE {edge_type} "
            f"FROM (SELECT FROM {from_type} WHERE uid = :from_uid) "
            f"TO (SELECT FROM {to_type} WHERE uid = :to_uid)"
        )
        if set_parts:
            command += " SET " + ", ".join(set_parts)
        await self.client.command(command, params)
        return True

    async def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        return fit_embedding_dimensions(embedding, await self._get_store_embedding_dimensions())

    async def _get_store_embedding_dimensions(self) -> int:
        if self._embedding_dimensions_cache:
            return self._embedding_dimensions_cache

        for record_type in ("Concept", "Episode", "Claim"):
            try:
                rows = await self.client.query(
                    f"SELECT embedding FROM {record_type} WHERE embedding IS NOT NULL LIMIT 1"
                )
            except httpx.HTTPStatusError:
                rows = []
            if not rows:
                continue
            embedding = rows[0].get("embedding")
            if isinstance(embedding, list) and embedding:
                self._embedding_dimensions_cache = len(embedding)
                return self._embedding_dimensions_cache

        self._embedding_dimensions_cache = self.settings.embedding_dimensions
        return self._embedding_dimensions_cache

    def _job_from_row(self, row: dict[str, Any]) -> JobRecord:
        result = row.get("result")
        summary = None
        if result:
            summary = IngestionSummary.model_validate(json.loads(result))
        return JobRecord(
            uid=row["uid"],
            episode_id=row["episode_id"],
            status=row["status"],
            result=summary,
            error=row.get("error"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class InMemoryKnowledgeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.episodes: dict[str, EpisodeRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.concepts: dict[str, ConceptRecord] = {}
        self.aliases: dict[str, str] = {}
        self.claims: dict[str, ClaimRecord] = {}
        self.relations: set[tuple[str, str, str, str | None]] = set()
        self.concept_mentions: set[tuple[str, str]] = set()
        self.claim_support: set[tuple[str, str]] = set()
        self.claim_explains: set[tuple[str, str]] = set()

    async def bootstrap_schema(self) -> None:
        return None

    async def check_ready(self) -> bool:
        return True

    async def create_episode(self, *, uid: str, text: str, source_type: str, tags: list[str], language: str) -> EpisodeRecord:
        episode = EpisodeRecord(
            uid=uid,
            text=text,
            source_type=source_type,
            tags=tags,
            language=language,
            status="queued",
            created_at=utcnow_iso(),
            error_message=None,
        )
        self.episodes[uid] = episode
        return episode

    async def create_job(self, *, uid: str, episode_id: str, status: str) -> JobRecord:
        now = utcnow_iso()
        job = JobRecord(uid=uid, episode_id=episode_id, status=status, created_at=now, updated_at=now)
        self.jobs[uid] = job
        return job

    async def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        *,
        status: str,
        result: IngestionSummary | None = None,
        error: str | None = None,
    ) -> JobRecord:
        job = self.jobs[job_id]
        updated = job.model_copy(update={"status": status, "result": result, "error": error, "updated_at": utcnow_iso()})
        self.jobs[job_id] = updated
        return updated

    async def get_episode(self, episode_id: str) -> EpisodeRecord | None:
        return self.episodes.get(episode_id)

    async def update_episode(
        self,
        episode_id: str,
        *,
        status: str,
        error_message: str | None = None,
        embedding: list[float] | None = None,
    ) -> EpisodeRecord:
        episode = self.episodes[episode_id]
        updated = episode.model_copy(update={"status": status, "error_message": error_message})
        self.episodes[episode_id] = updated
        return updated

    async def upsert_concept(
        self,
        payload: UpsertConceptRequest,
        *,
        embedding: list[float],
        source_confidence: float,
    ) -> tuple[ConceptRecord, bool]:
        existing = await self.get_concept(payload.canonical_name)
        now = utcnow_iso()
        if existing:
            updated = existing.model_copy(
                update={
                    "canonical_name": payload.canonical_name,
                    "domain": payload.domain,
                    "description": payload.description or existing.description,
                    "aliases": _merge_alias_lists(existing.aliases, payload.aliases),
                    "embedding": embedding,
                    "updated_at": now,
                }
            )
            self.concepts[updated.uid] = updated
            for alias in updated.aliases:
                self.aliases[normalize_text(alias)] = updated.uid
            return updated, False

        uid = make_prefixed_id("cn")
        concept = ConceptRecord(
            uid=uid,
            canonical_name=payload.canonical_name,
            normalized_name=normalize_text(payload.canonical_name),
            domain=payload.domain,
            description=payload.description,
            aliases=payload.aliases,
            embedding=embedding,
            created_at=now,
            updated_at=now,
        )
        self.concepts[uid] = concept
        for alias in payload.aliases:
            self.aliases[normalize_text(alias)] = uid
        return concept, True

    async def get_concept(self, ref: str) -> ConceptRecord | None:
        normalized = concept_ref_to_normalized(ref)
        for concept in self.concepts.values():
            if concept.uid == ref or concept.normalized_name == normalized or concept.canonical_name == ref:
                return concept
        alias_uid = self.aliases.get(normalized)
        if alias_uid:
            return self.concepts.get(alias_uid)
        return None

    async def search_candidates(
        self,
        *,
        query: str,
        domain_hint: str | None,
        query_embedding: list[float] | None,
        limit: int,
    ) -> list[CandidateHit]:
        normalized = normalize_text(query)
        results: list[CandidateHit] = []
        for concept in self.concepts.values():
            if domain_hint and concept.domain != domain_hint:
                continue
            if concept.normalized_name == normalized:
                score = 1.0
                reason = "normalized_name"
            elif normalized in [normalize_text(alias) for alias in concept.aliases]:
                score = 0.98
                reason = "alias"
            elif normalized in concept.normalized_name:
                score = 0.85
                reason = "full_text"
            else:
                score = cosine_similarity(query_embedding, concept.embedding) if query_embedding else 0.0
                reason = "vector"
            if score <= 0:
                continue
            results.append(
                CandidateHit(
                    uid=concept.uid,
                    canonical_name=concept.canonical_name,
                    domain=concept.domain,
                    description=concept.description,
                    score=score,
                    reason=reason,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    async def create_claim(self, *, text: str, confidence: float, status: str, embedding: list[float]) -> ClaimRecord:
        normalized_text = normalize_text(text)
        for claim in self.claims.values():
            if claim.normalized_text == normalized_text:
                return claim
        claim = ClaimRecord(
            uid=make_prefixed_id("cl"),
            text=text,
            normalized_text=normalized_text,
            confidence=confidence,
            status=status,
            created_at=utcnow_iso(),
            embedding=embedding,
        )
        self.claims[claim.uid] = claim
        return claim

    async def create_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None,
        confidence: float | None = None,
    ) -> bool:
        from_concept = await self.get_concept(from_ref)
        to_concept = await self.get_concept(to_ref)
        if not from_concept or not to_concept:
            return False
        if from_concept.uid == to_concept.uid:
            return False
        edge = (from_concept.uid, relation, to_concept.uid, evidence_episode_id)
        created = edge not in self.relations
        self.relations.add(edge)
        return created

    async def link_concept_to_episode(self, concept_uid: str, episode_id: str, confidence: float | None = None) -> None:
        self.concept_mentions.add((concept_uid, episode_id))

    async def link_claim_to_episode(self, claim_uid: str, episode_id: str, confidence: float | None = None) -> None:
        self.claim_support.add((claim_uid, episode_id))

    async def link_claim_to_concept(self, claim_uid: str, concept_uid: str, confidence: float | None = None) -> None:
        self.claim_explains.add((claim_uid, concept_uid))

    async def get_neighborhood(self, concept_ref: str, depth: int) -> NeighborhoodResponse | None:
        concept = await self.get_concept(concept_ref)
        if not concept:
            return None
        nodes: dict[str, NeighborhoodNode] = {}
        relations: list[NeighborhoodRelation] = []
        related_uids = {concept.uid}
        for from_uid, relation, to_uid, evidence in self.relations:
            if from_uid == concept.uid or to_uid == concept.uid:
                other_uid = to_uid if from_uid == concept.uid else from_uid
                other = self.concepts.get(other_uid)
                if other:
                    nodes[other.uid] = NeighborhoodNode(
                        uid=other.uid,
                        type="Concept",
                        name=other.canonical_name,
                        domain=other.domain,
                        description=other.description,
                    )
                    relations.append(
                        NeighborhoodRelation(
                            from_uid=from_uid,
                            from_name=self.concepts[from_uid].canonical_name,
                            relation=relation,
                            to_uid=to_uid,
                            to_name=self.concepts[to_uid].canonical_name,
                            evidence_episode_id=evidence,
                        )
                    )
                    related_uids.add(other.uid)

        if depth == 2:
            for from_uid, relation, to_uid, evidence in self.relations:
                if from_uid in related_uids or to_uid in related_uids:
                    for uid in {from_uid, to_uid}:
                        if uid == concept.uid or uid not in self.concepts:
                            continue
                        item = self.concepts[uid]
                        nodes[uid] = NeighborhoodNode(
                            uid=uid,
                            type="Concept",
                            name=item.canonical_name,
                            domain=item.domain,
                            description=item.description,
                        )
                    relations.append(
                        NeighborhoodRelation(
                            from_uid=from_uid,
                            from_name=self.concepts[from_uid].canonical_name,
                            relation=relation,
                            to_uid=to_uid,
                            to_name=self.concepts[to_uid].canonical_name,
                            evidence_episode_id=evidence,
                        )
                    )

        claims: list[dict[str, Any]] = []
        episodes: dict[str, dict[str, Any]] = {}
        for claim_uid, concept_uid in self.claim_explains:
            if concept_uid != concept.uid:
                continue
            claim = self.claims[claim_uid]
            claims.append({"uid": claim.uid, "text": claim.text, "confidence": claim.confidence})
            for supported_claim_uid, episode_id in self.claim_support:
                if supported_claim_uid != claim_uid:
                    continue
                episode = self.episodes[episode_id]
                episodes[episode_id] = {"uid": episode.uid, "text": episode.text, "status": episode.status}

        return NeighborhoodResponse(
            concept=NeighborhoodNode(
                uid=concept.uid,
                type="Concept",
                name=concept.canonical_name,
                domain=concept.domain,
                description=concept.description,
            ),
            nodes=list(nodes.values()),
            relations=relations,
            claims=claims,
            episodes=list(episodes.values()),
        )


def _merge_alias_lists(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for alias in [*existing, *incoming]:
        key = normalize_text(alias)
        if not key or key in seen:
            continue
        merged.append(alias)
        seen.add(key)
    return merged
