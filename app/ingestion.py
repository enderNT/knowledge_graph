from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter

from app.ai_provider import AIProvider
from app.config import Settings
from app.schemas import (
    AddKnowledgeFragmentRequest,
    AddKnowledgeFragmentAccepted,
    CandidateHit,
    ConceptResolution,
    ExtractionResult,
    IngestionSummary,
    UpsertConceptRequest,
)
from app.store import ConceptConflictError, KnowledgeStore
from app.trace import bind
from app.utils import make_prefixed_id, normalize_text

logger = logging.getLogger(__name__)


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
            temporal=request.temporal,
            expires_at=request.expires_at,
        )
        await self.store.create_job(uid=job_id, episode_id=episode_id, status="queued")
        await self.queue.put(job_id)
        logger.info(
            "fragment queued",
            extra={"job_id": job_id, "episode_id": episode_id, "tags": request.tags, "language": request.language},
        )
        return AddKnowledgeFragmentAccepted(episode_id=episode_id, job_id=job_id, status="queued")

    async def process_job(self, job_id: str) -> None:
        job = await self.store.get_job(job_id)
        if not job:
            logger.warning("job not found", extra={"job_id": job_id})
            return
        episode = await self.store.get_episode(job.episode_id)
        if not episode:
            await self.store.update_job(job_id, status="failed", error="episode not found")
            logger.error("episode not found for job", extra={"job_id": job_id, "episode_id": job.episode_id})
            return

        bind(run_id=job_id, step="start")
        logger.info("job started", extra={"episode_id": episode.uid, "language": episode.language, "tags": episode.tags})
        t0 = time.monotonic()
        await self.store.update_job(job_id, status="processing")
        try:
            bind(step="embed_episode")
            episode_embedding = await self.ai_provider.embed(episode.text)
            await self.store.update_episode(episode.uid, status="processing", embedding=episode_embedding)
            logger.debug("episode embedded", extra={"episode_id": episode.uid})

            bind(step="extract")
            extraction = await self.ai_provider.extract(episode.text, episode.language, episode.tags)
            logger.info(
                "extraction complete",
                extra={
                    "domain": extraction.domain,
                    "concepts_count": len(extraction.concepts),
                    "claims_count": len(extraction.claims),
                    "relations_count": len(extraction.relations),
                },
            )

            bind(step="vet_extraction")
            vetting = await self.ai_provider.vet_extraction(
                extraction=extraction,
                text=episode.text,
                language=episode.language,
            )
            extraction = ExtractionResult(
                domain=extraction.domain,
                topics=extraction.topics,
                concepts=vetting.concepts,
                claims=vetting.claims,
                relations=vetting.relations,
            )
            logger.info(
                "extraction vetted",
                extra={
                    "concepts_kept": len(extraction.concepts),
                    "claims_kept": len(extraction.claims),
                    "relations_kept": len(extraction.relations),
                },
            )

            summary = IngestionSummary(episode_id=episode.uid, domain=extraction.domain)
            resolved_concepts: dict[str, ConceptResolution] = {}
            claim_support_counts = Counter(
                concept_name
                for extracted_claim in extraction.claims
                for concept_name in extracted_claim.explains
            )
            linked_claim_counts = Counter()

            bind(step="resolve_concepts")
            for extracted_concept in extraction.concepts:
                concept_embedding = await self.ai_provider.embed(
                    "\n".join(
                        part
                        for part in [
                            extracted_concept.canonical_name,
                            extracted_concept.description,
                            *extracted_concept.evidence_quotes,
                        ]
                        if part
                    )
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
                    extracted_concept.evidence_quotes,
                    claim_support_counts.get(extracted_concept.canonical_name, 0),
                    candidates,
                )
                resolved_concepts[extracted_concept.canonical_name] = resolution
                logger.debug(
                    "concept resolved",
                    extra={"concept": extracted_concept.canonical_name, "strategy": resolution.strategy},
                )
                if resolution.strategy in {"ambiguous", "rejected"}:
                    logger.warning(
                        "concept needs review",
                        extra={"concept": extracted_concept.canonical_name, "strategy": resolution.strategy, "reason": resolution.needs_review_reason},
                    )
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

            logger.info(
                "concepts resolved",
                extra={
                    "n_created": len(summary.created_concepts),
                    "n_updated": len(summary.updated_concepts),
                    "needs_review": len(summary.needs_review),
                },
            )

            bind(step="create_claims")
            for extracted_claim in extraction.claims:
                claim_embedding = await self.ai_provider.embed(extracted_claim.text)
                claim = await self.store.create_claim(
                    text=extracted_claim.text,
                    confidence=extracted_claim.confidence,
                    status="active",
                    embedding=claim_embedding,
                    supporting_quote=extracted_claim.supporting_quote,
                )
                summary.created_claims += 1
                logger.debug("claim created", extra={"claim_uid": claim.uid, "explains": extracted_claim.explains})
                await self.store.link_claim_to_episode(claim.uid, episode.uid, extracted_claim.confidence)
                for concept_name in extracted_claim.explains:
                    concept_resolution = resolved_concepts.get(concept_name)
                    if concept_resolution and concept_resolution.concept:
                        await self.store.link_claim_to_concept(
                            claim.uid,
                            concept_resolution.concept.uid,
                            extracted_claim.confidence,
                        )
                        linked_claim_counts[concept_name] += 1
                        bind(step="vet_evidence")
                        await self._persist_pedagogical_evidence(
                            episode_id=episode.uid,
                            claim_uid=claim.uid,
                            claim_text=claim.text,
                            supporting_quote=claim.supporting_quote,
                            concept_uid=concept_resolution.concept.uid,
                            concept_name=concept_resolution.concept.canonical_name,
                            language=episode.language,
                        )

            bind(step="fill_evidence_claims")
            for extracted_concept in extraction.concepts:
                concept_resolution = resolved_concepts.get(extracted_concept.canonical_name)
                if not concept_resolution or not concept_resolution.concept:
                    continue
                if linked_claim_counts[extracted_concept.canonical_name] > 0:
                    continue
                quote = self._best_evidence_quote(extracted_concept.evidence_quotes)
                if quote is None:
                    continue
                quote_embedding = await self.ai_provider.embed(quote)
                claim = await self.store.create_claim(
                    text=quote,
                    confidence=max(0.6, min(0.95, extracted_concept.confidence)),
                    status="active",
                    embedding=quote_embedding,
                    supporting_quote=quote,
                )
                summary.created_claims += 1
                logger.debug(
                    "evidence claim created",
                    extra={"claim_uid": claim.uid, "concept": extracted_concept.canonical_name},
                )
                await self.store.link_claim_to_episode(claim.uid, episode.uid, extracted_concept.confidence)
                await self.store.link_claim_to_concept(
                    claim.uid,
                    concept_resolution.concept.uid,
                    extracted_concept.confidence,
                )
                linked_claim_counts[extracted_concept.canonical_name] += 1
                bind(step="vet_evidence")
                await self._persist_pedagogical_evidence(
                    episode_id=episode.uid,
                    claim_uid=claim.uid,
                    claim_text=claim.text,
                    supporting_quote=quote,
                    concept_uid=concept_resolution.concept.uid,
                    concept_name=concept_resolution.concept.canonical_name,
                    language=episode.language,
                )

            bind(step="create_relations")
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

            bind(step="finalize")
            await self.store.update_episode(episode.uid, status="processed")
            await self.store.update_job(job_id, status="completed", result=summary)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "job completed",
                extra={
                    "duration_ms": elapsed_ms,
                    "created_concepts": len(summary.created_concepts),
                    "updated_concepts": len(summary.updated_concepts),
                    "created_claims": summary.created_claims,
                    "relations": len(summary.relations),
                    "needs_review": len(summary.needs_review),
                    "domain": summary.domain,
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            error_message = str(exc).strip() or exc.__class__.__name__
            bind(step="failed")
            logger.error(
                "job failed",
                extra={"duration_ms": elapsed_ms, "error": error_message},
                exc_info=True,
            )
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
        evidence_quotes: list[str],
        claim_support_count: int,
        candidates: list[CandidateHit],
    ) -> ConceptResolution:
        if candidates:
            top = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            if top.reason in {"normalized_name", "alias"}:
                concept = await self._upsert_or_reuse_existing_identity(
                    canonical_name=top.canonical_name,
                    aliases=aliases,
                    domain=domain,
                    description=description,
                    confidence=confidence,
                    embedding=embedding,
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
                concept = await self._upsert_or_reuse_existing_identity(
                    canonical_name=top.canonical_name,
                    aliases=aliases,
                    domain=domain,
                    description=description,
                    confidence=confidence,
                    embedding=embedding,
                )
                return ConceptResolution(strategy="updated", concept=concept)

        rejection_reason = self._reject_new_concept_reason(
            canonical_name=canonical_name,
            evidence_quotes=evidence_quotes,
            claim_support_count=claim_support_count,
        )
        if rejection_reason:
            return ConceptResolution(
                strategy="rejected",
                needs_review_reason=rejection_reason,
            )

        concept, created = await self._upsert_with_exact_identity_fallback(
            canonical_name=canonical_name,
            aliases=aliases,
            domain=domain,
            description=description,
            confidence=confidence,
            embedding=embedding,
        )
        return ConceptResolution(strategy="created" if created else "updated", concept=concept)

    def _reject_new_concept_reason(
        self,
        *,
        canonical_name: str,
        evidence_quotes: list[str],
        claim_support_count: int,
    ) -> str | None:
        normalized = normalize_text(canonical_name)
        if not normalized:
            return f"rejected weak concept '{canonical_name}': empty normalized name"
        if claim_support_count <= 0 and not self._best_evidence_quote(evidence_quotes):
            return f"rejected weak concept '{canonical_name}': missing claim support or traceable evidence"
        return None

    async def _persist_pedagogical_evidence(
        self,
        *,
        episode_id: str,
        claim_uid: str,
        claim_text: str,
        supporting_quote: str | None,
        concept_uid: str,
        concept_name: str,
        language: str,
    ) -> None:
        effective_quote = supporting_quote or claim_text
        if not effective_quote:
            return
        decision = await self.ai_provider.vet_pedagogical_evidence(
            concept_name=concept_name,
            claim_text=claim_text,
            supporting_quote=effective_quote,
            language=language,
        )
        logger.debug(
            "evidence vetted",
            extra={"concept": concept_name, "status": decision.status, "kind": decision.kind},
        )
        if decision.status != "approved":
            return
        await self.store.create_pedagogical_evidence(
            concept_uid=concept_uid,
            concept_name=concept_name,
            episode_id=episode_id,
            source_claim_uid=claim_uid,
            statement=decision.statement,
            supporting_quote=decision.supporting_quote,
            kind=decision.kind,
            status=decision.status,
            review_notes_json=json.dumps(decision.review_notes, ensure_ascii=True) if decision.review_notes else None,
        )

    @staticmethod
    def _best_evidence_quote(evidence_quotes: list[str]) -> str | None:
        for quote in evidence_quotes:
            normalized = normalize_text(quote)
            if len(normalized) >= 12 and len(normalized.split()) >= 2:
                return quote.strip()
        return None

    async def _upsert_or_reuse_existing_identity(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        domain: str,
        description: str,
        confidence: float,
        embedding: list[float],
    ):
        concept, _ = await self._upsert_with_exact_identity_fallback(
            canonical_name=canonical_name,
            aliases=aliases,
            domain=domain,
            description=description,
            confidence=confidence,
            embedding=embedding,
        )
        return concept

    async def _upsert_with_exact_identity_fallback(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
        domain: str,
        description: str,
        confidence: float,
        embedding: list[float],
    ):
        payload = UpsertConceptRequest(
            canonical_name=canonical_name,
            aliases=aliases,
            domain=domain,
            description=description,
        )
        try:
            return await self.store.upsert_concept(
                payload,
                embedding=embedding,
                source_confidence=confidence,
            )
        except ConceptConflictError:
            existing = await self._find_existing_identity_match(canonical_name=canonical_name, aliases=aliases)
            if existing is None:
                raise
            reused_payload = UpsertConceptRequest(
                uid=existing.uid,
                canonical_name=existing.canonical_name,
                aliases=aliases,
                domain=domain,
                description=description or existing.description,
            )
            concept, _ = await self.store.upsert_concept(
                reused_payload,
                embedding=embedding,
                source_confidence=confidence,
            )
            return concept, False

    async def _find_existing_identity_match(
        self,
        *,
        canonical_name: str,
        aliases: list[str],
    ):
        references = [canonical_name, *aliases]
        for ref in references:
            existing = await self.store.get_concept(ref)
            if existing is not None:
                return existing
        return None
