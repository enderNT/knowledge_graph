from __future__ import annotations

import asyncio

from app.ai_provider import AIProvider
from app.config import Settings
from app.schemas import (
    AddKnowledgeFragmentRequest,
    AddKnowledgeFragmentAccepted,
    CandidateHit,
    ConceptResolution,
    IngestionSummary,
    UpsertConceptRequest,
)
from app.store import KnowledgeStore
from app.utils import make_prefixed_id


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: KnowledgeStore,
        ai_provider: AIProvider,
        queue: asyncio.Queue[str],
    ) -> None:
        self.settings = settings
        self.store = store
        self.ai_provider = ai_provider
        self.queue = queue

    async def submit_fragment(self, request: AddKnowledgeFragmentRequest) -> AddKnowledgeFragmentAccepted:
        episode_id = make_prefixed_id("ep")
        job_id = make_prefixed_id("job")
        await self.store.create_episode(
            uid=episode_id,
            text=request.text,
            source_type=request.source_type,
            tags=request.tags,
            language=request.language,
        )
        await self.store.create_job(uid=job_id, episode_id=episode_id, status="queued")
        await self.queue.put(job_id)
        return AddKnowledgeFragmentAccepted(episode_id=episode_id, job_id=job_id, status="queued")

    async def process_job(self, job_id: str) -> None:
        job = await self.store.get_job(job_id)
        if not job:
            return
        episode = await self.store.get_episode(job.episode_id)
        if not episode:
            await self.store.update_job(job_id, status="failed", error="episode not found")
            return

        await self.store.update_job(job_id, status="processing")
        try:
            episode_embedding = await self.ai_provider.embed(episode.text)
            await self.store.update_episode(episode.uid, status="processing", embedding=episode_embedding)
            extraction = await self.ai_provider.extract(episode.text, episode.language, episode.tags)
            summary = IngestionSummary(episode_id=episode.uid, domain=extraction.domain)
            resolved_concepts: dict[str, ConceptResolution] = {}

            for extracted_concept in extraction.concepts:
                concept_embedding = await self.ai_provider.embed(
                    f"{extracted_concept.canonical_name}\n{extracted_concept.description}"
                )
                candidates = await self.store.search_candidates(
                    query=extracted_concept.canonical_name,
                    domain_hint=extraction.domain,
                    query_embedding=concept_embedding,
                    limit=10,
                )
                resolution = await self._resolve_concept(
                    extracted_concept.canonical_name,
                    extraction.domain,
                    extracted_concept.description,
                    extracted_concept.aliases,
                    extracted_concept.confidence,
                    concept_embedding,
                    candidates,
                )
                resolved_concepts[extracted_concept.canonical_name] = resolution
                if resolution.strategy == "ambiguous":
                    summary.needs_review.append(resolution.needs_review_reason or extracted_concept.canonical_name)
                    continue
                if not resolution.concept:
                    continue
                await self.store.link_concept_to_episode(
                    resolution.concept.uid,
                    episode.uid,
                    extracted_concept.confidence,
                )
                if resolution.strategy == "created":
                    summary.created_concepts.append(resolution.concept.canonical_name)
                else:
                    summary.updated_concepts.append(resolution.concept.canonical_name)

            for extracted_claim in extraction.claims:
                claim_embedding = await self.ai_provider.embed(extracted_claim.text)
                claim = await self.store.create_claim(
                    text=extracted_claim.text,
                    confidence=extracted_claim.confidence,
                    status="active",
                    embedding=claim_embedding,
                )
                summary.created_claims += 1
                await self.store.link_claim_to_episode(claim.uid, episode.uid, extracted_claim.confidence)
                for concept_name in extracted_claim.explains:
                    concept_resolution = resolved_concepts.get(concept_name)
                    if concept_resolution and concept_resolution.concept:
                        await self.store.link_claim_to_concept(
                            claim.uid,
                            concept_resolution.concept.uid,
                            extracted_claim.confidence,
                        )

            for relation in extraction.relations:
                from_resolution = resolved_concepts.get(relation.from_name)
                to_resolution = resolved_concepts.get(relation.to_name)
                if not from_resolution or not from_resolution.concept:
                    summary.needs_review.append(f"missing source concept for relation: {relation.from_name}")
                    continue
                if not to_resolution or not to_resolution.concept:
                    summary.needs_review.append(f"missing target concept for relation: {relation.to_name}")
                    continue
                created = await self.store.create_relation(
                    from_ref=from_resolution.concept.uid,
                    relation=relation.relation,
                    to_ref=to_resolution.concept.uid,
                    evidence_episode_id=episode.uid,
                    confidence=relation.confidence,
                )
                if created:
                    summary.relations.append(
                        [
                            from_resolution.concept.canonical_name,
                            relation.relation,
                            to_resolution.concept.canonical_name,
                        ]
                    )

            await self.store.update_episode(episode.uid, status="processed")
            await self.store.update_job(job_id, status="completed", result=summary)
        except Exception as exc:
            error_message = str(exc).strip() or exc.__class__.__name__
            await self.store.update_episode(episode.uid, status="failed", error_message=error_message)
            await self.store.update_job(job_id, status="failed", error=error_message)
            raise

    async def _resolve_concept(
        self,
        canonical_name: str,
        domain: str,
        description: str,
        aliases: list[str],
        confidence: float,
        embedding: list[float],
        candidates: list[CandidateHit],
    ) -> ConceptResolution:
        if candidates:
            top = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            if top.reason in {"normalized_name", "alias"}:
                concept, _ = await self.store.upsert_concept(
                    UpsertConceptRequest(
                        canonical_name=top.canonical_name,
                        aliases=aliases,
                        domain=domain,
                        description=description,
                    ),
                    embedding=embedding,
                    source_confidence=confidence,
                )
                return ConceptResolution(strategy="matched", concept=concept)

            if top.score >= self.settings.resolution_match_threshold:
                if second and abs(top.score - second.score) <= self.settings.resolution_gap_threshold:
                    return ConceptResolution(
                        strategy="ambiguous",
                        needs_review_reason=(
                            f"ambiguous concept match for '{canonical_name}' between "
                            f"{top.canonical_name} and {second.canonical_name}"
                        ),
                    )
                concept, _ = await self.store.upsert_concept(
                    UpsertConceptRequest(
                        canonical_name=top.canonical_name,
                        aliases=aliases,
                        domain=domain,
                        description=description,
                    ),
                    embedding=embedding,
                    source_confidence=confidence,
                )
                return ConceptResolution(strategy="updated", concept=concept)

        concept, created = await self.store.upsert_concept(
            UpsertConceptRequest(
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            ),
            embedding=embedding,
            source_confidence=confidence,
        )
        return ConceptResolution(strategy="created" if created else "updated", concept=concept)
