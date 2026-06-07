from __future__ import annotations

import json
from collections.abc import Iterable
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
    GetPedagogicalContextResponse,
    IngestionSummary,
    JobRecord,
    NeighborhoodNode,
    NeighborhoodRelation,
    NeighborhoodResponse,
    PEDAGOGICAL_RELATIONS,
    PedagogicalConceptState,
    PedagogicalDomainState,
    PedagogicalEvaluationEvent,
    PedagogicalRecentStats,
    PedagogicalRecalculationTrace,
    TutorContextBundle,
    TutorContextClaim,
    TutorContextConcept,
    TutorContextEvidence,
    TutorContextRelation,
    TutorContextSourceFragment,
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
    async def get_tutor_context_for_episode(self, episode_id: str, depth: int = 1) -> TutorContextBundle | None: ...
    async def get_tutor_context_for_concept(self, concept_ref: str, depth: int = 1) -> TutorContextBundle | None: ...
    async def get_pedagogical_context(self, *, user_id: str) -> GetPedagogicalContextResponse: ...
    async def upsert_pedagogical_concept_state(self, state: PedagogicalConceptState) -> None: ...
    async def upsert_pedagogical_domain_state(self, state: PedagogicalDomainState) -> None: ...
    async def append_pedagogical_evaluation_event(self, event: PedagogicalEvaluationEvent) -> None: ...
    async def get_pedagogical_related_concepts(
        self,
        *,
        concept_uid: str,
        max_depth: int,
        allowed_relations: set[str],
    ) -> list[dict[str, str | int]]: ...


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

    async def get_tutor_context_for_episode(self, episode_id: str, depth: int = 1) -> TutorContextBundle | None:
        episode = await self.get_episode(episode_id)
        if not episode:
            return None

        concept_rows = await self._safe_query(
            (
                "MATCH {type: Episode, as: ep, where: (uid = :uid)}"
                ".in('MENTIONED_IN'){type: Concept, as: concept} "
                "RETURN concept.uid as uid, concept.canonical_name as canonical_name, "
                "concept.domain as domain, coalesce(concept.description, '') as description, "
                "coalesce(concept.aliases, []) as aliases"
            ),
            {"uid": episode_id},
        )
        claim_rows = await self._safe_query(
            (
                "MATCH {type: Episode, as: ep, where: (uid = :uid)}"
                ".in('SUPPORTED_BY'){type: Claim, as: claim} "
                "RETURN claim.uid as uid, claim.text as text, claim.confidence as confidence"
            ),
            {"uid": episode_id},
        )

        concepts: dict[str, TutorContextConcept] = {}
        for row in concept_rows:
            concept = self._tutor_concept_from_row(row)
            concepts[concept.uid] = concept

        claims: dict[str, TutorContextClaim] = {}
        for row in claim_rows:
            claim_uid = str(row.get("uid") or "")
            if not claim_uid:
                continue
            claims[claim_uid] = TutorContextClaim(
                uid=claim_uid,
                text=str(row.get("text") or ""),
                confidence=self._as_optional_float(row.get("confidence")),
                evidence_episode_ids=[episode_id],
            )

        explained_rows = await self._safe_query(
            (
                "MATCH {type: Episode, as: ep, where: (uid = :uid)}"
                ".in('SUPPORTED_BY'){type: Claim, as: claim}.out('EXPLAINS'){type: Concept, as: concept} "
                "RETURN claim.uid as claim_uid, concept.uid as uid, concept.canonical_name as canonical_name, "
                "concept.domain as domain, coalesce(concept.description, '') as description, "
                "coalesce(concept.aliases, []) as aliases"
            ),
            {"uid": episode_id},
        )
        for row in explained_rows:
            concept = self._tutor_concept_from_row(row)
            concepts.setdefault(concept.uid, concept)

        relations = await self._build_tutor_relations(concepts.keys(), allowed_episode_ids={episode_id})
        for relation in relations.values():
            for uid in (relation.from_uid, relation.to_uid):
                if uid in concepts:
                    continue
                related = await self.get_concept(uid)
                if related:
                    concepts[uid] = TutorContextConcept(
                        uid=related.uid,
                        canonical_name=related.canonical_name,
                        domain=related.domain,
                        description=related.description,
                        aliases=list(related.aliases),
                    )
        source_fragments = [self._source_fragment_from_episode(episode)]
        evidence = self._build_tutor_evidence(
            concepts=concepts.values(),
            claims=claims.values(),
            relations=relations.values(),
            source_fragments=source_fragments,
        )

        return TutorContextBundle(
            concepts=list(concepts.values()),
            claims=list(claims.values()),
            relations=list(relations.values()),
            source_fragments=source_fragments,
            evidence=evidence,
        )

    async def get_tutor_context_for_concept(self, concept_ref: str, depth: int = 1) -> TutorContextBundle | None:
        concept = await self.get_concept(concept_ref)
        if not concept:
            return None

        concepts: dict[str, TutorContextConcept] = {
            concept.uid: TutorContextConcept(
                uid=concept.uid,
                canonical_name=concept.canonical_name,
                domain=concept.domain,
                description=concept.description,
                aliases=list(concept.aliases),
            )
        }
        claim_rows = await self._safe_query(
            (
                "MATCH {type: Concept, as: concept, where: (uid = :uid)}"
                ".in('EXPLAINS'){type: Claim, as: claim}.out('SUPPORTED_BY'){type: Episode, as: episode} "
                "RETURN claim.uid as claim_uid, claim.text as claim_text, claim.confidence as claim_confidence, "
                "episode.uid as episode_uid, episode.text as episode_text, episode.status as episode_status, "
                "episode.source_type as episode_source_type, coalesce(episode.tags, []) as episode_tags, "
                "episode.language as episode_language"
            ),
            {"uid": concept.uid},
        )
        mention_rows = await self._safe_query(
            (
                "MATCH {type: Concept, as: concept, where: (uid = :uid)}"
                ".out('MENTIONED_IN'){type: Episode, as: episode} "
                "RETURN episode.uid as episode_uid, episode.text as episode_text, episode.status as episode_status, "
                "episode.source_type as episode_source_type, coalesce(episode.tags, []) as episode_tags, "
                "episode.language as episode_language"
            ),
            {"uid": concept.uid},
        )

        claims: dict[str, TutorContextClaim] = {}
        source_fragments: dict[str, TutorContextSourceFragment] = {}
        for row in claim_rows:
            claim_uid = str(row.get("claim_uid") or "")
            episode_uid = str(row.get("episode_uid") or "")
            if claim_uid:
                claim = claims.get(claim_uid)
                if claim is None:
                    claims[claim_uid] = TutorContextClaim(
                        uid=claim_uid,
                        text=str(row.get("claim_text") or ""),
                        confidence=self._as_optional_float(row.get("claim_confidence")),
                        evidence_episode_ids=[episode_uid] if episode_uid else [],
                    )
                elif episode_uid and episode_uid not in claim.evidence_episode_ids:
                    claim.evidence_episode_ids.append(episode_uid)
            fragment = self._source_fragment_from_row(row)
            if fragment:
                source_fragments[fragment.episode_id] = fragment

        for row in mention_rows:
            fragment = self._source_fragment_from_row(row)
            if fragment:
                source_fragments[fragment.episode_id] = fragment

        relations = await self._build_tutor_relations({concept.uid}, allowed_episode_ids=set())
        for relation in relations.values():
            for episode_id in relation.evidence_episode_ids:
                if episode_id in source_fragments:
                    continue
                episode = await self.get_episode(episode_id)
                if episode:
                    source_fragments[episode_id] = self._source_fragment_from_episode(episode)
        for relation in relations.values():
            for uid, name in ((relation.from_uid, relation.from_name), (relation.to_uid, relation.to_name)):
                if uid in concepts:
                    continue
                related = await self.get_concept(uid)
                if related:
                    concepts[uid] = TutorContextConcept(
                        uid=related.uid,
                        canonical_name=related.canonical_name,
                        domain=related.domain,
                        description=related.description,
                        aliases=list(related.aliases),
                    )

        evidence = self._build_tutor_evidence(
            concepts=[concepts[concept.uid]],
            claims=claims.values(),
            relations=relations.values(),
            source_fragments=source_fragments.values(),
        )
        return TutorContextBundle(
            concepts=list(concepts.values()),
            claims=list(claims.values()),
            relations=list(relations.values()),
            source_fragments=list(source_fragments.values()),
            evidence=evidence,
        )

    async def get_pedagogical_context(self, *, user_id: str) -> GetPedagogicalContextResponse:
        concept_rows = await self._safe_query(
            (
                "SELECT user_id, concept_uid, concept_name, domain, mastery_score_0_to_100, mastery_label, "
                "recent_history_json, recent_stats_json, weaknesses_json, detected_gaps_json, "
                "suggested_questions_json, effective_depth_used, last_evaluated_at, updated_at, "
                "recalculation_traces_json FROM UserConceptMastery WHERE user_id = :user_id"
            ),
            {"user_id": user_id},
        )
        domain_rows = await self._safe_query(
            (
                "SELECT user_id, domain, mastery_score_0_to_100, mastery_label, concept_count, "
                "weak_concept_uids_json, recent_stats_json, updated_at, recalculation_traces_json "
                "FROM UserDomainMastery WHERE user_id = :user_id"
            ),
            {"user_id": user_id},
        )
        event_rows = await self._safe_query(
            (
                "SELECT user_id, concept_uid, concept_name, domain, score_0_to_100, recorded_at, source "
                "FROM UserEvaluationEvent WHERE user_id = :user_id ORDER BY recorded_at DESC LIMIT 20"
            ),
            {"user_id": user_id},
        )

        concepts = [
            PedagogicalConceptState(
                user_id=row["user_id"],
                concept_uid=row["concept_uid"],
                concept_name=row.get("concept_name") or "",
                domain=row["domain"],
                mastery_score_0_to_100=float(row.get("mastery_score_0_to_100") or 0.0),
                mastery_label=row.get("mastery_label") or "muy bajo",
                recent_history=self._parse_model_list(row.get("recent_history_json"), PedagogicalEvaluationEvent),
                recent_stats=self._parse_model(row.get("recent_stats_json"), PedagogicalRecentStats),
                weaknesses=self._parse_string_list(row.get("weaknesses_json")),
                detected_gaps=self._parse_string_list(row.get("detected_gaps_json")),
                suggested_questions=self._parse_string_list(row.get("suggested_questions_json")),
                effective_depth_used=int(row.get("effective_depth_used") or 3),
                last_evaluated_at=row.get("last_evaluated_at"),
                updated_at=row.get("updated_at") or utcnow_iso(),
                recalculation_traces=self._parse_model_list(row.get("recalculation_traces_json"), PedagogicalRecalculationTrace),
            )
            for row in concept_rows
        ]
        domains = [
            PedagogicalDomainState(
                user_id=row["user_id"],
                domain=row["domain"],
                mastery_score_0_to_100=float(row.get("mastery_score_0_to_100") or 0.0),
                mastery_label=row.get("mastery_label") or "muy bajo",
                concept_count=int(row.get("concept_count") or 0),
                weak_concept_uids=self._parse_string_list(row.get("weak_concept_uids_json")),
                recent_stats=self._parse_model(row.get("recent_stats_json"), PedagogicalRecentStats),
                updated_at=row.get("updated_at") or utcnow_iso(),
                recalculation_traces=self._parse_model_list(row.get("recalculation_traces_json"), PedagogicalRecalculationTrace),
            )
            for row in domain_rows
        ]
        events = [PedagogicalEvaluationEvent.model_validate(row) for row in event_rows]
        status = "ok" if concepts and domains else ("sparse" if concepts or domains else "not_found")
        warnings = [] if status != "not_found" else ["empty_user_context"]
        return GetPedagogicalContextResponse(
            user_id=user_id,
            status=status,
            concepts=concepts,
            domains=domains,
            recent_evaluations=events,
            warnings=warnings,
        )

    async def upsert_pedagogical_concept_state(self, state: PedagogicalConceptState) -> None:
        payload = {
            "user_id": state.user_id,
            "concept_uid": state.concept_uid,
            "concept_name": state.concept_name,
            "domain": state.domain,
            "mastery_score_0_to_100": state.mastery_score_0_to_100,
            "mastery_label": state.mastery_label,
            "recent_history_json": json.dumps([item.model_dump() for item in state.recent_history]),
            "recent_stats_json": state.recent_stats.model_dump_json(),
            "weaknesses_json": json.dumps(state.weaknesses),
            "detected_gaps_json": json.dumps(state.detected_gaps),
            "suggested_questions_json": json.dumps(state.suggested_questions),
            "effective_depth_used": state.effective_depth_used,
            "last_evaluated_at": state.last_evaluated_at,
            "updated_at": state.updated_at,
            "recalculation_traces_json": json.dumps([item.model_dump() for item in state.recalculation_traces]),
        }
        rows = await self._safe_query(
            "SELECT user_id FROM UserConceptMastery WHERE user_id = :user_id AND concept_uid = :concept_uid LIMIT 1",
            {"user_id": state.user_id, "concept_uid": state.concept_uid},
        )
        if rows:
            await self.client.command(
                (
                    "UPDATE UserConceptMastery SET concept_name = :concept_name, domain = :domain, "
                    "mastery_score_0_to_100 = :mastery_score_0_to_100, mastery_label = :mastery_label, "
                    "recent_history_json = :recent_history_json, recent_stats_json = :recent_stats_json, "
                    "weaknesses_json = :weaknesses_json, detected_gaps_json = :detected_gaps_json, "
                    "suggested_questions_json = :suggested_questions_json, effective_depth_used = :effective_depth_used, "
                    "last_evaluated_at = :last_evaluated_at, updated_at = :updated_at, "
                    "recalculation_traces_json = :recalculation_traces_json "
                    "WHERE user_id = :user_id AND concept_uid = :concept_uid"
                ),
                payload,
            )
            return
        await self.client.command(
            (
                "INSERT INTO UserConceptMastery SET user_id = :user_id, concept_uid = :concept_uid, "
                "concept_name = :concept_name, domain = :domain, mastery_score_0_to_100 = :mastery_score_0_to_100, "
                "mastery_label = :mastery_label, recent_history_json = :recent_history_json, "
                "recent_stats_json = :recent_stats_json, weaknesses_json = :weaknesses_json, "
                "detected_gaps_json = :detected_gaps_json, suggested_questions_json = :suggested_questions_json, "
                "effective_depth_used = :effective_depth_used, last_evaluated_at = :last_evaluated_at, "
                "updated_at = :updated_at, recalculation_traces_json = :recalculation_traces_json"
            ),
            payload,
        )

    async def upsert_pedagogical_domain_state(self, state: PedagogicalDomainState) -> None:
        payload = {
            "user_id": state.user_id,
            "domain": state.domain,
            "mastery_score_0_to_100": state.mastery_score_0_to_100,
            "mastery_label": state.mastery_label,
            "concept_count": state.concept_count,
            "weak_concept_uids_json": json.dumps(state.weak_concept_uids),
            "recent_stats_json": state.recent_stats.model_dump_json(),
            "updated_at": state.updated_at,
            "recalculation_traces_json": json.dumps([item.model_dump() for item in state.recalculation_traces]),
        }
        rows = await self._safe_query(
            "SELECT user_id FROM UserDomainMastery WHERE user_id = :user_id AND domain = :domain LIMIT 1",
            {"user_id": state.user_id, "domain": state.domain},
        )
        if rows:
            await self.client.command(
                (
                    "UPDATE UserDomainMastery SET mastery_score_0_to_100 = :mastery_score_0_to_100, "
                    "mastery_label = :mastery_label, concept_count = :concept_count, "
                    "weak_concept_uids_json = :weak_concept_uids_json, recent_stats_json = :recent_stats_json, "
                    "updated_at = :updated_at, recalculation_traces_json = :recalculation_traces_json "
                    "WHERE user_id = :user_id AND domain = :domain"
                ),
                payload,
            )
            return
        await self.client.command(
            (
                "INSERT INTO UserDomainMastery SET user_id = :user_id, domain = :domain, "
                "mastery_score_0_to_100 = :mastery_score_0_to_100, mastery_label = :mastery_label, "
                "concept_count = :concept_count, weak_concept_uids_json = :weak_concept_uids_json, "
                "recent_stats_json = :recent_stats_json, updated_at = :updated_at, "
                "recalculation_traces_json = :recalculation_traces_json"
            ),
            payload,
        )

    async def append_pedagogical_evaluation_event(self, event: PedagogicalEvaluationEvent) -> None:
        await self.client.command(
            (
                "INSERT INTO UserEvaluationEvent SET uid = :uid, user_id = :user_id, concept_uid = :concept_uid, "
                "concept_name = :concept_name, domain = :domain, score_0_to_100 = :score_0_to_100, "
                "recorded_at = :recorded_at, source = :source"
            ),
            {
                "uid": make_prefixed_id("uev"),
                "user_id": event.user_id,
                "concept_uid": event.concept_uid,
                "concept_name": event.concept_name,
                "domain": event.domain,
                "score_0_to_100": event.score_0_to_100,
                "recorded_at": event.recorded_at,
                "source": event.source,
            },
        )

    async def get_pedagogical_related_concepts(
        self,
        *,
        concept_uid: str,
        max_depth: int,
        allowed_relations: set[str],
    ) -> list[dict[str, str | int]]:
        seen = {concept_uid}
        frontier = [(concept_uid, 0)]
        results: list[dict[str, str | int]] = []
        while frontier:
            current_uid, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for row in await self._fetch_neighborhood_edge_rows(current_uid):
                relation = str(row.get("relation") or "")
                if relation not in allowed_relations:
                    continue
                related_uid = str(
                    row.get("to_uid") if row.get("from_uid") == current_uid else row.get("from_uid") or ""
                )
                if not related_uid or related_uid == current_uid:
                    continue
                result = {"concept_uid": related_uid, "relation": relation, "depth": depth + 1}
                results.append(result)
                if related_uid not in seen:
                    seen.add(related_uid)
                    frontier.append((related_uid, depth + 1))
        return results

    async def _build_tutor_relations(
        self,
        concept_uids: Iterable[str],
        *,
        allowed_episode_ids: set[str],
    ) -> dict[str, TutorContextRelation]:
        concept_uid_set = {str(uid) for uid in concept_uids}
        relations: dict[str, TutorContextRelation] = {}
        for concept_uid in concept_uid_set:
            for row in await self._fetch_neighborhood_edge_rows(str(concept_uid)):
                relation = self._build_neighborhood_relation(row)
                if relation is None:
                    continue
                if relation.from_uid not in concept_uid_set and relation.to_uid not in concept_uid_set:
                    continue
                if relation.evidence_episode_id and allowed_episode_ids and relation.evidence_episode_id not in allowed_episode_ids:
                    continue
                relation_id = _tutor_relation_uid(
                    relation.from_uid,
                    relation.relation,
                    relation.to_uid,
                    relation.evidence_episode_id,
                )
                item = relations.get(relation_id)
                if item is None:
                    relations[relation_id] = TutorContextRelation(
                        uid=relation_id,
                        from_uid=relation.from_uid,
                        from_name=relation.from_name,
                        relation=relation.relation,
                        to_uid=relation.to_uid,
                        to_name=relation.to_name,
                        confidence=self._as_optional_float(relation.confidence),
                        evidence_episode_ids=[relation.evidence_episode_id] if relation.evidence_episode_id else [],
                    )
                elif relation.evidence_episode_id and relation.evidence_episode_id not in item.evidence_episode_ids:
                    item.evidence_episode_ids.append(relation.evidence_episode_id)
        return relations

    def _tutor_concept_from_row(self, row: dict[str, Any]) -> TutorContextConcept:
        aliases = row.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        return TutorContextConcept(
            uid=str(row.get("uid") or ""),
            canonical_name=str(row.get("canonical_name") or ""),
            domain=str(row.get("domain") or ""),
            description=str(row.get("description") or ""),
            aliases=[str(alias) for alias in aliases],
        )

    def _source_fragment_from_episode(self, episode: EpisodeRecord) -> TutorContextSourceFragment:
        return TutorContextSourceFragment(
            episode_id=episode.uid,
            text=episode.text,
            status=episode.status,
            source_type=episode.source_type,
            tags=list(episode.tags),
            language=episode.language,
        )

    def _source_fragment_from_row(self, row: dict[str, Any]) -> TutorContextSourceFragment | None:
        episode_id = str(row.get("episode_uid") or "")
        if not episode_id:
            return None
        tags = row.get("episode_tags") or []
        if not isinstance(tags, list):
            tags = []
        return TutorContextSourceFragment(
            episode_id=episode_id,
            text=str(row.get("episode_text") or ""),
            status=str(row.get("episode_status") or ""),
            source_type=str(row.get("episode_source_type") or "manual_input"),
            tags=[str(tag) for tag in tags],
            language=str(row.get("episode_language") or "es"),
        )

    def _build_tutor_evidence(
        self,
        *,
        concepts: Any,
        claims: Any,
        relations: Any,
        source_fragments: Any,
    ) -> list[TutorContextEvidence]:
        fragment_ids = {fragment.episode_id for fragment in source_fragments}
        evidence: dict[tuple[str, str, str], TutorContextEvidence] = {}
        for concept in concepts:
            for episode_id in fragment_ids:
                key = ("concept", concept.uid, episode_id)
                evidence[key] = TutorContextEvidence(subject_type="concept", subject_uid=concept.uid, episode_id=episode_id)
        for claim in claims:
            for episode_id in claim.evidence_episode_ids:
                key = ("claim", claim.uid, episode_id)
                evidence[key] = TutorContextEvidence(subject_type="claim", subject_uid=claim.uid, episode_id=episode_id)
        for relation in relations:
            for episode_id in relation.evidence_episode_ids:
                key = ("relation", relation.uid, episode_id)
                evidence[key] = TutorContextEvidence(subject_type="relation", subject_uid=relation.uid, episode_id=episode_id)
        return list(evidence.values())

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

    @staticmethod
    def _as_optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_string_list(raw: object) -> list[str]:
        if not raw:
            return []
        try:
            data = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data]

    @staticmethod
    def _parse_model(raw: object, model: Any) -> Any:
        if not raw:
            return model(
                recent_average=0.0,
                trend="insufficient_data",
                deviation=0.0,
                last_evaluated_at=None,
            )
        try:
            return model.model_validate_json(str(raw))
        except Exception:
            return model(
                recent_average=0.0,
                trend="insufficient_data",
                deviation=0.0,
                last_evaluated_at=None,
            )

    @staticmethod
    def _parse_model_list(raw: object, model: Any) -> list[Any]:
        if not raw:
            return []
        try:
            data = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        items: list[Any] = []
        for entry in data:
            try:
                items.append(model.model_validate(entry))
            except Exception:
                continue
        return items

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
        self.user_concept_mastery: dict[tuple[str, str], PedagogicalConceptState] = {}
        self.user_domain_mastery: dict[tuple[str, str], PedagogicalDomainState] = {}
        self.user_evaluation_events: dict[str, list[PedagogicalEvaluationEvent]] = {}

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

    async def get_tutor_context_for_episode(self, episode_id: str, depth: int = 1) -> TutorContextBundle | None:
        episode = self.episodes.get(episode_id)
        if episode is None:
            return None

        concepts: dict[str, TutorContextConcept] = {}
        for concept_uid, mentioned_episode_id in self.concept_mentions:
            if mentioned_episode_id != episode_id:
                continue
            concept = self.concepts.get(concept_uid)
            if concept:
                concepts[concept.uid] = _tutor_context_concept(concept)

        claims: dict[str, TutorContextClaim] = {}
        for claim_uid, supported_episode_id in self.claim_support:
            if supported_episode_id != episode_id:
                continue
            claim = self.claims.get(claim_uid)
            if claim is None:
                continue
            claims[claim.uid] = TutorContextClaim(
                uid=claim.uid,
                text=claim.text,
                confidence=claim.confidence,
                evidence_episode_ids=[episode_id],
            )
            for explained_claim_uid, concept_uid in self.claim_explains:
                if explained_claim_uid != claim_uid:
                    continue
                concept = self.concepts.get(concept_uid)
                if concept:
                    concepts[concept.uid] = _tutor_context_concept(concept)

        relations = _in_memory_tutor_relations(self, set(concepts), allowed_episode_ids={episode_id})
        for relation in relations.values():
            for related_uid in (relation.from_uid, relation.to_uid):
                if related_uid in concepts:
                    continue
                related = self.concepts.get(related_uid)
                if related:
                    concepts[related.uid] = _tutor_context_concept(related)
        source_fragments = [_source_fragment_from_episode_record(episode)]
        evidence = _build_tutor_evidence(
            concepts=concepts.values(),
            claims=claims.values(),
            relations=relations.values(),
            source_fragments=source_fragments,
        )
        return TutorContextBundle(
            concepts=list(concepts.values()),
            claims=list(claims.values()),
            relations=list(relations.values()),
            source_fragments=source_fragments,
            evidence=evidence,
        )

    async def get_tutor_context_for_concept(self, concept_ref: str, depth: int = 1) -> TutorContextBundle | None:
        concept = await self.get_concept(concept_ref)
        if concept is None:
            return None

        concepts: dict[str, TutorContextConcept] = {concept.uid: _tutor_context_concept(concept)}
        claims: dict[str, TutorContextClaim] = {}
        source_fragments: dict[str, TutorContextSourceFragment] = {}

        for claim_uid, concept_uid in self.claim_explains:
            if concept_uid != concept.uid:
                continue
            claim = self.claims.get(claim_uid)
            if claim is None:
                continue
            evidence_episode_ids = sorted(
                episode_id for supported_claim_uid, episode_id in self.claim_support if supported_claim_uid == claim_uid
            )
            claims[claim.uid] = TutorContextClaim(
                uid=claim.uid,
                text=claim.text,
                confidence=claim.confidence,
                evidence_episode_ids=evidence_episode_ids,
            )
            for episode_id in evidence_episode_ids:
                episode = self.episodes.get(episode_id)
                if episode:
                    source_fragments[episode_id] = _source_fragment_from_episode_record(episode)

        for concept_uid, episode_id in self.concept_mentions:
            if concept_uid != concept.uid:
                continue
            episode = self.episodes.get(episode_id)
            if episode:
                source_fragments[episode_id] = _source_fragment_from_episode_record(episode)

        relations = _in_memory_tutor_relations(self, {concept.uid}, allowed_episode_ids=set())
        for relation in relations.values():
            for episode_id in relation.evidence_episode_ids:
                if episode_id in source_fragments:
                    continue
                episode = self.episodes.get(episode_id)
                if episode:
                    source_fragments[episode_id] = _source_fragment_from_episode_record(episode)
        for relation in relations.values():
            for related_uid in (relation.from_uid, relation.to_uid):
                if related_uid in concepts:
                    continue
                related = self.concepts.get(related_uid)
                if related:
                    concepts[related.uid] = _tutor_context_concept(related)

        evidence = _build_tutor_evidence(
            concepts=[concepts[concept.uid]],
            claims=claims.values(),
            relations=relations.values(),
            source_fragments=source_fragments.values(),
        )
        return TutorContextBundle(
            concepts=list(concepts.values()),
            claims=list(claims.values()),
            relations=list(relations.values()),
            source_fragments=list(source_fragments.values()),
            evidence=evidence,
        )

    async def get_pedagogical_context(self, *, user_id: str) -> GetPedagogicalContextResponse:
        concepts = [
            state
            for (stored_user_id, _), state in self.user_concept_mastery.items()
            if stored_user_id == user_id
        ]
        domains = [
            state
            for (stored_user_id, _), state in self.user_domain_mastery.items()
            if stored_user_id == user_id
        ]
        events = list(reversed(self.user_evaluation_events.get(user_id, [])))[0:20]
        status = "ok" if concepts and domains else ("sparse" if concepts or domains else "not_found")
        warnings = [] if status != "not_found" else ["empty_user_context"]
        return GetPedagogicalContextResponse(
            user_id=user_id,
            status=status,
            concepts=sorted(concepts, key=lambda item: item.updated_at, reverse=True),
            domains=sorted(domains, key=lambda item: item.updated_at, reverse=True),
            recent_evaluations=events,
            warnings=warnings,
        )

    async def upsert_pedagogical_concept_state(self, state: PedagogicalConceptState) -> None:
        self.user_concept_mastery[(state.user_id, state.concept_uid)] = state

    async def upsert_pedagogical_domain_state(self, state: PedagogicalDomainState) -> None:
        self.user_domain_mastery[(state.user_id, state.domain)] = state

    async def append_pedagogical_evaluation_event(self, event: PedagogicalEvaluationEvent) -> None:
        items = self.user_evaluation_events.setdefault(event.user_id, [])
        items.append(event)
        items.sort(key=lambda item: item.recorded_at)

    async def get_pedagogical_related_concepts(
        self,
        *,
        concept_uid: str,
        max_depth: int,
        allowed_relations: set[str],
    ) -> list[dict[str, str | int]]:
        results: list[dict[str, str | int]] = []
        seen = {concept_uid}
        frontier = [(concept_uid, 0)]
        while frontier:
            current_uid, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for from_uid, relation, to_uid, _ in self.relations:
                if relation not in allowed_relations:
                    continue
                if from_uid == current_uid:
                    related_uid = to_uid
                elif to_uid == current_uid:
                    related_uid = from_uid
                else:
                    continue
                results.append({"concept_uid": related_uid, "relation": relation, "depth": depth + 1})
                if related_uid not in seen:
                    seen.add(related_uid)
                    frontier.append((related_uid, depth + 1))
        return results


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


def _tutor_context_concept(concept: ConceptRecord) -> TutorContextConcept:
    return TutorContextConcept(
        uid=concept.uid,
        canonical_name=concept.canonical_name,
        domain=concept.domain,
        description=concept.description,
        aliases=list(concept.aliases),
    )


def _source_fragment_from_episode_record(episode: EpisodeRecord) -> TutorContextSourceFragment:
    return TutorContextSourceFragment(
        episode_id=episode.uid,
        text=episode.text,
        status=episode.status,
        source_type=episode.source_type,
        tags=list(episode.tags),
        language=episode.language,
    )


def _in_memory_tutor_relations(
    store: InMemoryKnowledgeStore,
    concept_uids: set[str],
    *,
    allowed_episode_ids: set[str],
) -> dict[str, TutorContextRelation]:
    relations: dict[str, TutorContextRelation] = {}
    for from_uid, relation, to_uid, evidence_episode_id in store.relations:
        if from_uid not in concept_uids and to_uid not in concept_uids:
            continue
        if evidence_episode_id and allowed_episode_ids and evidence_episode_id not in allowed_episode_ids:
            continue
        relation_id = _tutor_relation_uid(from_uid, relation, to_uid, evidence_episode_id)
        item = relations.get(relation_id)
        if item is None:
            relations[relation_id] = TutorContextRelation(
                uid=relation_id,
                from_uid=from_uid,
                from_name=store.concepts[from_uid].canonical_name,
                relation=relation,
                to_uid=to_uid,
                to_name=store.concepts[to_uid].canonical_name,
                confidence=None,
                evidence_episode_ids=[evidence_episode_id] if evidence_episode_id else [],
            )
        elif evidence_episode_id and evidence_episode_id not in item.evidence_episode_ids:
            item.evidence_episode_ids.append(evidence_episode_id)
    return relations


def _build_tutor_evidence(
    *,
    concepts: Any,
    claims: Any,
    relations: Any,
    source_fragments: Any,
) -> list[TutorContextEvidence]:
    fragment_ids = {fragment.episode_id for fragment in source_fragments}
    evidence: dict[tuple[str, str, str], TutorContextEvidence] = {}
    for concept in concepts:
        for episode_id in fragment_ids:
            evidence[("concept", concept.uid, episode_id)] = TutorContextEvidence(
                subject_type="concept",
                subject_uid=concept.uid,
                episode_id=episode_id,
            )
    for claim in claims:
        for episode_id in claim.evidence_episode_ids:
            evidence[("claim", claim.uid, episode_id)] = TutorContextEvidence(
                subject_type="claim",
                subject_uid=claim.uid,
                episode_id=episode_id,
            )
    for relation in relations:
        for episode_id in relation.evidence_episode_ids:
            evidence[("relation", relation.uid, episode_id)] = TutorContextEvidence(
                subject_type="relation",
                subject_uid=relation.uid,
                episode_id=episode_id,
            )
    return list(evidence.values())


def _tutor_relation_uid(
    from_uid: str,
    relation: str,
    to_uid: str,
    evidence_episode_id: str | None,
) -> str:
    return "|".join([from_uid, relation, to_uid, evidence_episode_id or ""])
