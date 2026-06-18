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
    PedagogicalEvidenceDecision,
    UpsertConceptRequest,
)
from app.store import ConceptConflictError, KnowledgeStore
from app.trace import bind
from app.trace_copy import trace_summary, trace_title
from app.trace_models import CanonicalTrace
from app.trace_recorder import TraceRecorder
from app.utils import make_prefixed_id, normalize_text

logger = logging.getLogger(__name__)


def _extraction_counts(extraction: ExtractionResult) -> dict[str, int]:
    return {
        "concepts": len(extraction.concepts),
        "claims": len(extraction.claims),
        "relations": len(extraction.relations),
    }


def _trace_status_for_counts(counts: dict[str, int]):
    return "succeeded" if sum(counts.values()) > 0 else "empty"


def _trace_vetting_status(raw: ExtractionResult, vetted: ExtractionResult):
    raw_total = sum(_extraction_counts(raw).values())
    vetted_total = sum(_extraction_counts(vetted).values())
    if vetted_total == 0:
        return "empty"
    if vetted_total < raw_total:
        return "partial"
    return "succeeded"


def _trace_collection_status(*, total: int, needs_review: int = 0, created: int = 0):
    if needs_review and created:
        return "partial"
    if needs_review:
        return "needs_review"
    if total == 0 or created == 0:
        return "empty"
    return "succeeded"


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
        logger.info("job started", extra={"episode_id": episode.uid, "language": episode.language, "tags": episode.tags, "input_shape": {"episode_id": episode.uid, "language": episode.language, "tags_count": len(episode.tags), "text_len": len(episode.text)}})
        t0 = time.monotonic()
        trace = TraceRecorder(
            execution_type="ingestion_job",
            execution_id=job_id,
            episode_id=episode.uid,
        )
        trace.record_step(
            type="fragment_received",
            status="succeeded",
            title=trace_title("fragment_received", "succeeded"),
            summary=trace_summary("fragment_received"),
            input={
                "episode_id": episode.uid,
                "language": episode.language,
                "tags": episode.tags,
                "text_len": len(episode.text),
            },
            output={"job_id": job_id, "episode_id": episode.uid},
        )
        await self.store.update_job(job_id, status="processing")
        try:
            bind(step="embed_episode")
            episode_embedding = await self.ai_provider.embed(episode.text)
            await self.store.update_episode(episode.uid, status="processing", embedding=episode_embedding)
            logger.debug("episode embedded", extra={"episode_id": episode.uid})

            bind(step="extract")
            extraction = await self.ai_provider.extract(episode.text, episode.language, episode.tags)
            raw_extraction = extraction
            extraction_boundary = self.ai_provider.consume_last_llm_boundary_payload()
            logger.info(
                "extraction complete",
                extra={
                    "domain": extraction.domain,
                    "concepts_count": len(extraction.concepts),
                    "claims_count": len(extraction.claims),
                    "relations_count": len(extraction.relations),
                    "output_shape": {"domain": extraction.domain, "concepts": len(extraction.concepts), "claims": len(extraction.claims), "relations": len(extraction.relations)},
                },
            )
            trace.record_step(
                type="knowledge_extracted",
                status=_trace_status_for_counts(_extraction_counts(extraction)),
                title=trace_title("knowledge_extracted", _trace_status_for_counts(_extraction_counts(extraction))),
                summary=trace_summary("knowledge_extracted"),
                input={"language": episode.language, "tags": episode.tags, "text_len": len(episode.text)},
                output={"domain": extraction.domain, **_extraction_counts(extraction)},
                boundary_payload=extraction_boundary,
            )

            bind(step="vet_extraction")
            vetting = await self.ai_provider.vet_extraction(
                extraction=extraction,
                text=episode.text,
                language=episode.language,
            )
            vetting_boundary = self.ai_provider.consume_last_llm_boundary_payload()
            extraction = ExtractionResult(
                domain=extraction.domain,
                topics=extraction.topics,
                concepts=vetting.concepts,
                claims=vetting.claims,
                relations=vetting.relations,
            )
            vetted_status = _trace_vetting_status(raw_extraction, extraction)
            logger.info(
                "extraction vetted",
                extra={
                    "concepts_kept": len(extraction.concepts),
                    "claims_kept": len(extraction.claims),
                    "relations_kept": len(extraction.relations),
                    "output_shape": {"concepts_kept": len(extraction.concepts), "claims_kept": len(extraction.claims), "relations_kept": len(extraction.relations)},
                },
            )
            trace.record_step(
                type="extraction_vetted",
                status=vetted_status,
                title=trace_title("extraction_vetted", vetted_status),
                summary=trace_summary("extraction_vetted"),
                input=_extraction_counts(raw_extraction),
                output={f"{key}_kept": value for key, value in _extraction_counts(extraction).items()},
                detail={"decisions": [decision.model_dump() for decision in vetting.decisions]},
                boundary_payload=vetting_boundary,
            )

            summary = IngestionSummary(episode_id=episode.uid, domain=extraction.domain)
            resolved_concepts: dict[str, ConceptResolution] = {}
            concept_decisions: list[dict[str, object]] = []
            claim_decisions: list[dict[str, object]] = []
            evidence_decisions: list[dict[str, object]] = []
            relation_decisions: list[dict[str, object]] = []
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
                concept_decisions.append(
                    {
                        "concept": extracted_concept.canonical_name,
                        "strategy": resolution.strategy,
                        "status": "needs_review" if resolution.strategy in {"ambiguous", "rejected"} else "succeeded",
                        "concept_uid": resolution.concept.uid if resolution.concept else "",
                        "reason": resolution.needs_review_reason or "",
                    }
                )
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
                    "output_shape": {"created": len(summary.created_concepts), "updated": len(summary.updated_concepts), "needs_review": len(summary.needs_review)},
                },
            )
            concepts_status = _trace_collection_status(
                total=len(extraction.concepts),
                created=len(summary.created_concepts) + len(summary.updated_concepts),
                needs_review=sum(1 for item in concept_decisions if item["status"] == "needs_review"),
            )
            concepts_event = trace.record_step(
                type="concepts_resolved",
                status=concepts_status,
                title=trace_title("concepts_resolved", concepts_status),
                summary=trace_summary("concepts_resolved"),
                input={"concepts": len(extraction.concepts), "domain": extraction.domain},
                output={
                    "created": len(summary.created_concepts),
                    "updated": len(summary.updated_concepts),
                    "needs_review": sum(1 for item in concept_decisions if item["status"] == "needs_review"),
                },
            )
            for decision in concept_decisions:
                trace.record_decision(
                    parent_event_id=concepts_event.event_id,
                    type="concepts_resolved",
                    status=decision["status"],
                    title=trace_title("concepts_resolved", decision["status"], subject=str(decision["concept"])),
                    input={"concept": decision["concept"]},
                    output={"concept_uid": decision["concept_uid"]},
                    detail={"strategy": decision["strategy"], "reason": decision["reason"]},
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
                claim_decisions.append(
                    {
                        "claim_uid": claim.uid,
                        "claim_text": claim.text,
                        "status": "succeeded",
                        "explains": extracted_claim.explains,
                    }
                )
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
                        evidence_decision = await self._persist_pedagogical_evidence(
                            episode_id=episode.uid,
                            claim_uid=claim.uid,
                            claim_text=claim.text,
                            supporting_quote=claim.supporting_quote,
                            concept_uid=concept_resolution.concept.uid,
                            concept_name=concept_resolution.concept.canonical_name,
                            language=episode.language,
                        )
                        if evidence_decision is not None:
                            evidence_decisions.append(
                                {
                                    "concept": concept_resolution.concept.canonical_name,
                                    "claim_uid": claim.uid,
                                    "status": "succeeded" if evidence_decision.status == "approved" else "needs_review",
                                    "decision_status": evidence_decision.status,
                                    "kind": evidence_decision.kind,
                                    "review_notes": evidence_decision.review_notes,
                                }
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
                claim_decisions.append(
                    {
                        "claim_uid": claim.uid,
                        "claim_text": claim.text,
                        "status": "succeeded",
                        "explains": [extracted_concept.canonical_name],
                    }
                )
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
                evidence_decision = await self._persist_pedagogical_evidence(
                    episode_id=episode.uid,
                    claim_uid=claim.uid,
                    claim_text=claim.text,
                    supporting_quote=quote,
                    concept_uid=concept_resolution.concept.uid,
                    concept_name=concept_resolution.concept.canonical_name,
                    language=episode.language,
                )
                if evidence_decision is not None:
                    evidence_decisions.append(
                        {
                            "concept": concept_resolution.concept.canonical_name,
                            "claim_uid": claim.uid,
                            "status": "succeeded" if evidence_decision.status == "approved" else "needs_review",
                            "decision_status": evidence_decision.status,
                            "kind": evidence_decision.kind,
                            "review_notes": evidence_decision.review_notes,
                        }
                    )

            claims_status = _trace_collection_status(total=len(extraction.claims), created=len(claim_decisions))
            claims_event = trace.record_step(
                type="claims_created",
                status=claims_status,
                title=trace_title("claims_created", claims_status),
                summary=trace_summary("claims_created"),
                input={"claims": len(extraction.claims)},
                output={"created": len(claim_decisions)},
            )
            for decision in claim_decisions:
                trace.record_decision(
                    parent_event_id=claims_event.event_id,
                    type="claims_created",
                    status=decision["status"],
                    title=trace_title("claims_created", decision["status"], subject=str(decision["claim_uid"])),
                    input={"claim_text": decision["claim_text"], "explains": decision["explains"]},
                    output={"claim_uid": decision["claim_uid"]},
                )

            approved_evidence = sum(1 for item in evidence_decisions if item["status"] == "succeeded")
            evidence_status = "succeeded" if approved_evidence == len(evidence_decisions) and evidence_decisions else "partial"
            if not evidence_decisions:
                evidence_status = "empty"
            elif approved_evidence == 0:
                evidence_status = "needs_review"
            evidence_event = trace.record_step(
                type="pedagogical_evidence_vetted",
                status=evidence_status,
                title=trace_title("pedagogical_evidence_vetted", evidence_status),
                summary=trace_summary("pedagogical_evidence_vetted"),
                input={"claims": len(claim_decisions)},
                output={"approved": approved_evidence, "needs_review": len(evidence_decisions) - approved_evidence},
            )
            for decision in evidence_decisions:
                trace.record_decision(
                    parent_event_id=evidence_event.event_id,
                    type="pedagogical_evidence_vetted",
                    status=decision["status"],
                    title=trace_title(
                        "pedagogical_evidence_vetted",
                        decision["status"],
                        subject=str(decision["concept"]),
                    ),
                    input={"claim_uid": decision["claim_uid"], "concept": decision["concept"]},
                    output={"decision_status": decision["decision_status"], "kind": decision["kind"]},
                    detail={"review_notes": decision["review_notes"]},
                )

            bind(step="create_relations")
            for relation in extraction.relations:
                from_resolution = resolved_concepts.get(relation.from_name)
                to_resolution = resolved_concepts.get(relation.to_name)
                if not from_resolution or not from_resolution.concept:
                    reason = f"missing source concept for relation: {relation.from_name}"
                    summary.needs_review.append(reason)
                    relation_decisions.append(
                        {
                            "relation": relation.relation,
                            "from": relation.from_name,
                            "to": relation.to_name,
                            "status": "needs_review",
                            "reason": reason,
                        }
                    )
                    continue
                if not to_resolution or not to_resolution.concept:
                    reason = f"missing target concept for relation: {relation.to_name}"
                    summary.needs_review.append(reason)
                    relation_decisions.append(
                        {
                            "relation": relation.relation,
                            "from": relation.from_name,
                            "to": relation.to_name,
                            "status": "needs_review",
                            "reason": reason,
                        }
                    )
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
                relation_decisions.append(
                    {
                        "relation": relation.relation,
                        "from": from_resolution.concept.canonical_name,
                        "to": to_resolution.concept.canonical_name,
                        "status": "succeeded" if created else "skipped",
                        "reason": "" if created else "relation already existed",
                    }
                )

            relation_needs_review = sum(1 for item in relation_decisions if item["status"] == "needs_review")
            relations_status = _trace_collection_status(
                total=len(extraction.relations),
                created=len(summary.relations),
                needs_review=relation_needs_review,
            )
            relations_event = trace.record_step(
                type="relations_created",
                status=relations_status,
                title=trace_title("relations_created", relations_status),
                summary=trace_summary("relations_created"),
                input={"relations": len(extraction.relations)},
                output={"created": len(summary.relations), "needs_review": relation_needs_review},
            )
            for decision in relation_decisions:
                subject = f"{decision['from']} {decision['relation']} {decision['to']}"
                trace.record_decision(
                    parent_event_id=relations_event.event_id,
                    type="relations_created",
                    status=decision["status"],
                    title=trace_title("relations_created", decision["status"], subject=subject),
                    input={"from": decision["from"], "relation": decision["relation"], "to": decision["to"]},
                    output={"created": decision["status"] == "succeeded"},
                    detail={"reason": decision["reason"]},
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
                    "output_shape": {"created_concepts": len(summary.created_concepts), "updated_concepts": len(summary.updated_concepts), "created_claims": summary.created_claims, "relations": len(summary.relations), "needs_review": len(summary.needs_review)},
                },
            )
            final_status = "needs_review" if summary.needs_review else "succeeded"
            trace.record_step(
                type="ingestion_finalized",
                status=final_status,
                title=trace_title("ingestion_finalized", final_status),
                summary=trace_summary("ingestion_finalized"),
                output={
                    "created_concepts": len(summary.created_concepts),
                    "updated_concepts": len(summary.updated_concepts),
                    "created_claims": summary.created_claims,
                    "relations": len(summary.relations),
                    "needs_review": len(summary.needs_review),
                },
            )
            await self._persist_trace_safely(
                trace.close(
                    status=final_status,
                    domain=summary.domain,
                    semantic_counts={
                        "created_concepts": len(summary.created_concepts),
                        "updated_concepts": len(summary.updated_concepts),
                        "created_claims": summary.created_claims,
                        "relations": len(summary.relations),
                        "needs_review": len(summary.needs_review),
                    },
                )
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            error_message = str(exc).strip() or exc.__class__.__name__
            bind(step="failed")
            logger.error(
                "job failed",
                extra={"duration_ms": elapsed_ms, "error": error_message, "error_type": exc.__class__.__name__, "error_message": error_message[:200]},
                exc_info=True,
            )
            trace.record_step(
                type="ingestion_failed",
                status="failed",
                title=trace_title("ingestion_failed", "failed"),
                summary=trace_summary("ingestion_failed"),
                output={"error_type": exc.__class__.__name__, "error_message": error_message[:500]},
            )
            await self.store.update_episode(episode.uid, status="failed", error_message=error_message)
            await self.store.update_job(job_id, status="failed", error=error_message)
            await self._persist_trace_safely(trace.close(status="failed", semantic_counts={"errors": 1}))
            raise

    async def _persist_trace_safely(self, trace: CanonicalTrace) -> None:
        try:
            await self.store.persist_canonical_trace(trace)
        except Exception as exc:
            logger.error(
                "canonical trace persistence failed",
                extra={
                    "trace_id": trace.summary.trace_id,
                    "execution_id": trace.summary.execution_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc)[:300],
                },
            )

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
    ) -> PedagogicalEvidenceDecision | None:
        effective_quote = supporting_quote or claim_text
        if not effective_quote:
            return None
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
            return decision
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
        return decision

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
