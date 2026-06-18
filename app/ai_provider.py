from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import Settings
from app.schemas import (
    ExtractionResult,
    ExtractionVettingResult,
    ExtractedConcept,
    ExtractedClaim,
    ExtractedRelation,
    ExtractionItemDecision,
    GeneratedAdaptiveBlock,
    LearnerResponseGradingDecision,
    PedagogicalEvidenceDecision,
)
from app.trace_models import TraceBoundaryPayload
from app.utils import dedupe_preserve_order, fit_embedding_dimensions, normalize_text, stable_embedding

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
    async def vet_extraction(
        self,
        *,
        extraction: ExtractionResult,
        text: str,
        language: str,
    ) -> ExtractionVettingResult:
        raise NotImplementedError

    @abstractmethod
    async def vet_pedagogical_evidence(
        self,
        *,
        concept_name: str,
        claim_text: str,
        supporting_quote: str,
        language: str,
    ) -> PedagogicalEvidenceDecision:
        raise NotImplementedError

    @abstractmethod
    async def grade_learner_response(
        self,
        *,
        learner_response: str,
        expected_statement: str,
        pedagogical_evidence: list[str],
        concept_name: str,
        question_type: str,
        language: str,
    ) -> LearnerResponseGradingDecision:
        raise NotImplementedError

    @abstractmethod
    async def generate_adaptive_block(
        self,
        *,
        evidence_units: list[dict[str, str]],
        target_dimension: str,
        difficulty: str,
        question_types: list[str],
        served_evidence_uids: list[str],
        item_count: int,
        concept_name: str,
        language: str,
    ) -> GeneratedAdaptiveBlock:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def consume_last_llm_boundary_payload(self) -> TraceBoundaryPayload | None:
        return None


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StubAIProvider(AIProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed(self, text: str) -> list[float]:
        return stable_embedding(text, self.settings.embedding_dimensions)

    async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
        raise NotImplementedError(
            "extraction requires a real LLM provider; set AI_PROVIDER=anthropic or AI_PROVIDER=openai_compatible"
        )

    async def vet_extraction(
        self,
        *,
        extraction: ExtractionResult,
        text: str,
        language: str,
    ) -> ExtractionVettingResult:
        raise NotImplementedError(
            "vet_extraction requires a real LLM provider; set AI_PROVIDER=anthropic or AI_PROVIDER=openai_compatible"
        )

    async def vet_pedagogical_evidence(
        self,
        *,
        concept_name: str,
        claim_text: str,
        supporting_quote: str,
        language: str,
    ) -> PedagogicalEvidenceDecision:
        _ = (concept_name, language)
        statement = " ".join(claim_text.split()).strip(" .")
        quote = " ".join(supporting_quote.split()).strip()
        if not statement or not quote:
            return PedagogicalEvidenceDecision(
                statement=statement or claim_text,
                supporting_quote=quote or supporting_quote,
                status="rejected",
                review_notes=["empty_statement_or_supporting_quote"],
            )
        return PedagogicalEvidenceDecision(
            statement=statement,
            supporting_quote=quote,
            kind="claim",
            status="approved",
            review_notes=[],
        )

    async def grade_learner_response(
        self,
        *,
        learner_response: str,
        expected_statement: str,
        pedagogical_evidence: list[str],
        concept_name: str,
        question_type: str,
        language: str,
    ) -> LearnerResponseGradingDecision:
        from app.pedagogical_assessment import PedagogicalAssessmentService
        from app.schemas import AdaptiveBlockAnswerKey
        assessment = PedagogicalAssessmentService()
        rubric = assessment._rubric_for_statement(expected_statement)
        answer_key = AdaptiveBlockAnswerKey(
            item_id="stub",
            grading_mode="rubric_structured",
            expected=[expected_statement],
            grading_rubric=rubric,
        )
        coverage, precision, score = assessment.grade_open_response(learner_response, answer_key)
        matched, missing = assessment._score_rubric(learner_response, rubric)
        return LearnerResponseGradingDecision(
            coverage=coverage,
            precision=precision,
            score_0_to_1=score,
            matched_points=matched,
            missing_points=missing,
            reasoning="stub: token overlap against rubric",
            used_llm=False,
        )

    async def generate_adaptive_block(
        self,
        *,
        evidence_units: list[dict[str, str]],
        target_dimension: str,
        difficulty: str,
        question_types: list[str],
        served_evidence_uids: list[str],
        item_count: int,
        concept_name: str,
        language: str,
    ) -> GeneratedAdaptiveBlock:
        return GeneratedAdaptiveBlock(items=[])

    # Language-agnostic helpers used by StructuredLLMProvider via fallback_provider

    def _find_matching_concept(self, raw_value: str, concepts: list[str]) -> str | None:
        target = normalize_text(raw_value)
        for concept in concepts:
            if normalize_text(concept) == target:
                return concept
        # Tolerant substring match
        for concept in concepts:
            cn = normalize_text(concept)
            if target and cn and (target in cn or cn in target):
                return concept
        return None

    def _match_concepts_in_text(self, fragment: str, concepts: list[str]) -> list[str]:
        normalized_fragment = normalize_text(fragment)
        matches: list[tuple[int, str]] = []
        for concept in concepts:
            normalized_concept = normalize_text(concept)
            position = normalized_fragment.find(normalized_concept)
            if position >= 0:
                matches.append((position, concept))
        matches.sort(key=lambda item: item[0])
        return dedupe_preserve_order([concept for _, concept in matches])


class StubEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed(self, text: str) -> list[float]:
        return stable_embedding(text, self.settings.embedding_dimensions)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when embeddings use openai_compatible")

        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.openai_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=httpx.Timeout(settings.openai_timeout_seconds, connect=20.0),
        )

    async def embed(self, text: str) -> list[float]:
        t0 = time.monotonic()
        try:
            response = await self.client.post(
                "embeddings",
                json={
                    "model": self.settings.openai_embeddings_model,
                    "input": text,
                    "dimensions": self.settings.embedding_dimensions,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "embed http error",
                extra={"status": exc.response.status_code, "duration_ms": int((time.monotonic() - t0) * 1000), "input_shape": {"text_len": len(text)}, "error_type": exc.__class__.__name__},
            )
            raise
        except httpx.TimeoutException:
            logger.error("embed timeout", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "input_shape": {"text_len": len(text)}, "error_type": "TimeoutException"})
            raise
        payload = response.json()
        embedding = payload["data"][0]["embedding"]
        return fit_embedding_dimensions(embedding, self.settings.embedding_dimensions)

    async def close(self) -> None:
        await self.client.aclose()


class StructuredLLMProvider(AIProvider):
    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.fallback_provider = StubAIProvider(settings)
        self._embedding_provider = embedding_provider
        self._last_llm_boundary_payload: TraceBoundaryPayload | None = None

    async def embed(self, text: str) -> list[float]:
        return await self._embedding_provider.embed(text)

    def consume_last_llm_boundary_payload(self) -> TraceBoundaryPayload | None:
        payload = self._last_llm_boundary_payload
        self._last_llm_boundary_payload = None
        return payload

    async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
        system_prompt = (
            "You extract structured knowledge from user notes. "
            "Return only JSON with keys: domain, topics, concepts, claims, relations. "
            "Use only entities and facts explicitly present in the text. "
            "First identify the central topic and the teachable abstractions in the fragment; only after that use examples as supporting context. "
            "A concept must be a teachable, reusable abstraction that is useful as a standalone graph node. "
            "Prefer canonical noun phrases already present in the fragment. "
            "Do not invent concepts, do not output section headings, date labels, generic tags, or arbitrary adjacent word pairs. "
            "You are responsible for semantic validation: only keep concepts that are meaningful, canonical, and useful as standalone graph nodes. "
            "Do not emit partial fragments, isolated adjectives, or incomplete headings as concepts. "
            "Do not treat concrete examples, sample cases, illustrative instances, variable names, field names, identifiers, literals, accidental names, one-off values, table labels, or overly specific references as concepts. "
            "If a fragment contains an example of a more general idea, extract the general concept only when it is explicitly stated in the text and keep the example only as contextual evidence in claims or evidence_quotes. "
            "If a phrase could be read both as an example and as an idea, prefer the general idea and not the instance. "
            "Treat any example, instance, identifier, accidental name, concrete value, or domain-specific token as context rather than a concept unless the text explicitly states that the specific item is itself the object of study. "
            "Use tags only to infer domain/topics, not as concepts unless the text explicitly mentions them. "
            "Treat editorial, instructional, navigational, formatting, or meta content as non-concepts unless the topic really is that content. "
            "Examples of content that should usually be excluded as concepts or claims: introductions to a manual, directions to the reader, labels like period/date/section, "
            "formatting scaffolds, document navigation, references to prompts, system behavior, retrieval, evidence, fragments, grounding, or internal workflow. "
            "When a phrase is too weak, generic, tabular, or incomplete to stand alone as a useful node, discard it instead of inventing or stretching it. "
            "Concept items must include canonical_name, aliases, description, evidence_quotes, confidence. "
            "Each concept must include 1 to 3 short evidence_quotes copied verbatim from the text. "
            "Claim items must include text, confidence, explains, supporting_quote. "
            "Each claim supporting_quote must be copied verbatim from the text. "
            "Relation items must include from_name, relation, to_name, confidence. "
            "Allowed relation values: PART_OF, IS_A, RELATED_TO, EXPLAINS, CONTRASTS_WITH, "
            "PREREQUISITE_FOR, SUPPORTED_BY, MENTIONED_IN, ALIAS_OF."
        )
        user_prompt = {
            "language": language,
            "tags": tags,
            "text": text,
        }
        t0 = time.monotonic()
        try:
            content = await self._generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=None,
            )
            result = ExtractionResult.model_validate(content)
            logger.debug("llm extract done", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "input_shape": {"language": language, "text_len": len(text), "tags": tags}, "output_shape": {"concepts": len(result.concepts), "claims": len(result.claims), "relations": len(result.relations)}})
            return self._sanitize_llm_extraction(result, text, tags)
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error("llm extract failed", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message, "error_type": exc.__class__.__name__, "error_message": message[:200]})
            raise ValueError(f"llm extraction failed: {message}") from exc

    async def vet_extraction(
        self,
        *,
        extraction: ExtractionResult,
        text: str,
        language: str,
    ) -> ExtractionVettingResult:
        system_prompt = (
            "You are the extraction quality judge for a knowledge graph ingestion pipeline. "
            "You receive a raw extraction (concepts, claims, relations) and the source text it came from. "
            "Return only JSON with keys: concept_decisions, claim_decisions, relation_decisions. "
            "Each list contains objects with: name (identifier), status ('keep', 'drop', or 'repair'), "
            "review_notes (list of short reason strings), repaired_text (string or null — only for 'repair'). "
            "For concepts: 'name' is the canonical_name. "
            "  keep: the concept is teachable, specific, and useful as a standalone graph node. "
            "  drop: the concept is too generic, meta, navigational, a heading, a date label, a non-teachable token, or not explicitly present in the text as an idea. "
            "  repair: the concept is valid but the canonical_name needs improvement; provide repaired_text with the better name. "
            "For claims: 'name' is the first 80 chars of the claim text. "
            "  keep: the claim is a teachable declarative statement grounded in the text. "
            "  drop: the claim is meta-content, navigational, procedural, or not useful to teach. "
            "  repair: the claim is valid but awkwardly worded; provide repaired_text with the corrected statement. "
            "For relations: 'name' is 'from_name → relation → to_name'. "
            "  keep: the relation is explicitly supported by the text. "
            "  drop: the relation is hallucinated, unsupported, or redundant. "
            "Use only the text as the ground truth. Do not invent facts. "
            "Be lenient with concepts that are unambiguously teachable, even if phrased imperfectly."
        )
        concept_list = [
            {"canonical_name": c.canonical_name, "description": c.description, "evidence_quotes": c.evidence_quotes}
            for c in extraction.concepts
        ]
        claim_list = [
            {"text": c.text, "supporting_quote": c.supporting_quote}
            for c in extraction.claims
        ]
        relation_list = [
            {"from_name": r.from_name, "relation": r.relation, "to_name": r.to_name}
            for r in extraction.relations
        ]
        user_prompt = {
            "language": language,
            "text": text,
            "extraction": {
                "concepts": concept_list,
                "claims": claim_list,
                "relations": relation_list,
            },
        }
        t0 = time.monotonic()
        try:
            content = await self._generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )
            return self._apply_vetting_decisions(extraction, content)
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "llm vet_extraction failed",
                extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message, "error_type": exc.__class__.__name__, "error_message": message[:200]},
            )
            raise ValueError(f"llm extraction vetting failed: {message}") from exc
        finally:
            logger.debug(
                "llm vet_extraction done",
                extra={"duration_ms": int((time.monotonic() - t0) * 1000)},
            )

    def _apply_vetting_decisions(
        self,
        extraction: ExtractionResult,
        content: dict[str, Any],
    ) -> ExtractionVettingResult:
        all_decisions: list[ExtractionItemDecision] = []

        # Index decisions by name
        concept_decisions: dict[str, dict[str, Any]] = {
            d["name"]: d for d in content.get("concept_decisions", []) if isinstance(d, dict) and "name" in d
        }
        claim_decisions: dict[str, dict[str, Any]] = {
            d["name"]: d for d in content.get("claim_decisions", []) if isinstance(d, dict) and "name" in d
        }
        relation_decisions: dict[str, dict[str, Any]] = {
            d["name"]: d for d in content.get("relation_decisions", []) if isinstance(d, dict) and "name" in d
        }

        kept_concepts: list[ExtractedConcept] = []
        for concept in extraction.concepts:
            dec = concept_decisions.get(concept.canonical_name, {})
            status = dec.get("status", "keep")
            notes = dec.get("review_notes", [])
            repaired = dec.get("repaired_text")
            all_decisions.append(ExtractionItemDecision(
                item_type="concept",
                name=concept.canonical_name,
                status=status,
                review_notes=notes if isinstance(notes, list) else [notes] if notes else [],
                repaired_text=repaired,
            ))
            if status == "drop":
                logger.debug("concept dropped by vet", extra={"name": concept.canonical_name, "notes": notes})
                continue
            if status == "repair" and repaired:
                concept = concept.model_copy(update={"canonical_name": repaired.strip()})
            kept_concepts.append(concept)

        kept_concept_names = [c.canonical_name for c in kept_concepts]

        kept_claims: list[ExtractedClaim] = []
        for claim in extraction.claims:
            dec_key = claim.text[:80]
            dec = claim_decisions.get(dec_key, {})
            status = dec.get("status", "keep")
            notes = dec.get("review_notes", [])
            repaired = dec.get("repaired_text")
            all_decisions.append(ExtractionItemDecision(
                item_type="claim",
                name=dec_key,
                status=status,
                review_notes=notes if isinstance(notes, list) else [notes] if notes else [],
                repaired_text=repaired,
            ))
            if status == "drop":
                logger.debug("claim dropped by vet", extra={"text": dec_key, "notes": notes})
                continue
            if status == "repair" and repaired:
                claim = claim.model_copy(update={"text": repaired.strip()})
            kept_claims.append(claim)

        kept_relations: list[ExtractedRelation] = []
        for relation in extraction.relations:
            rel_key = f"{relation.from_name} → {relation.relation} → {relation.to_name}"
            dec = relation_decisions.get(rel_key, {})
            status = dec.get("status", "keep")
            notes = dec.get("review_notes", [])
            all_decisions.append(ExtractionItemDecision(
                item_type="relation",
                name=rel_key,
                status=status,
                review_notes=notes if isinstance(notes, list) else [notes] if notes else [],
            ))
            if status == "drop":
                logger.debug("relation dropped by vet", extra={"rel": rel_key, "notes": notes})
                continue
            # Only keep relations whose concepts survived vetting
            if relation.from_name in kept_concept_names and relation.to_name in kept_concept_names:
                kept_relations.append(relation)

        logger.info(
            "vet_extraction applied",
            extra={
                "concepts_kept": len(kept_concepts),
                "concepts_dropped": len(extraction.concepts) - len(kept_concepts),
                "claims_kept": len(kept_claims),
                "claims_dropped": len(extraction.claims) - len(kept_claims),
                "relations_kept": len(kept_relations),
            },
        )
        return ExtractionVettingResult(
            concepts=kept_concepts,
            claims=kept_claims,
            relations=kept_relations,
            decisions=all_decisions,
        )

    async def vet_pedagogical_evidence(
        self,
        *,
        concept_name: str,
        claim_text: str,
        supporting_quote: str,
        language: str,
    ) -> PedagogicalEvidenceDecision:
        system_prompt = (
            "You are the pedagogical evidence distiller and judge for a grounded learning system. "
            "Return only JSON with keys: statement, supporting_quote, kind, status, review_notes. "
            "Your job is to convert a claim plus its source quote into one student-safe pedagogical evidence unit, or reject it. "
            "Use only the provided claim and quote. Do not invent facts. "
            "If the claim is valid but awkward, rewrite it into a clean, self-contained declarative statement about the concept. "
            "Reject content that is editorial, navigational, procedural, formatting-only, document-scaffolding, or internal meta-language unless that is literally the topic. "
            "Examples to usually reject: introductions to a guide/manual, notes to the reader, section labels, reminders about evidence/retrieval/fragments/grounding/prompts/system/process. "
            "Reject weak or non-teachable units such as incomplete headings, isolated labels, raw formatting artifacts, or statements that are not useful to teach the concept. "
            "Approve only if the result is atomic, teachable, grounded in the quote, and appropriate to show to a learner. "
            "If you reject, keep statement close to the best candidate and explain why in review_notes. "
            "Preserve supporting_quote verbatim except for whitespace normalization."
        )
        user_prompt = {
            "language": language,
            "concept_name": concept_name,
            "claim_text": claim_text,
            "supporting_quote": supporting_quote,
        }
        t0 = time.monotonic()
        try:
            content = await self._generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )
            decision = PedagogicalEvidenceDecision.model_validate(content)
            statement = " ".join(decision.statement.split()).strip(" .")
            quote = " ".join(decision.supporting_quote.split()).strip()
            if not statement or not quote:
                return PedagogicalEvidenceDecision(
                    statement=statement or claim_text.strip(),
                    supporting_quote=quote or supporting_quote.strip(),
                    kind=decision.kind,
                    status="rejected",
                    review_notes=[*decision.review_notes, "empty_statement_or_supporting_quote"],
                )
            logger.debug("llm vet done", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "status": decision.status})
            return decision.model_copy(update={"statement": statement, "supporting_quote": quote})
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error("llm vet failed", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message})
            raise ValueError(f"llm pedagogical evidence vetting failed: {message}") from exc

    async def grade_learner_response(
        self,
        *,
        learner_response: str,
        expected_statement: str,
        pedagogical_evidence: list[str],
        concept_name: str,
        question_type: str,
        language: str,
    ) -> LearnerResponseGradingDecision:
        system_prompt = (
            "You are a pedagogical grader for an adaptive learning system. "
            "Evaluate the learner's response against the expected statement and provided evidence. "
            "Return only JSON with keys: coverage, precision, score_0_to_1, matched_points, missing_points, reasoning. "
            "coverage: fraction of expected evidence points addressed by the learner (0.0-1.0). "
            "precision: fraction of the learner's answer that is factually accurate per the evidence (0.0-1.0). "
            "score_0_to_1: overall correctness score (0.0-1.0). "
            "matched_points: list of evidence points the learner adequately covered. "
            "missing_points: list of evidence points the learner missed or got wrong. "
            "reasoning: one-sentence explanation of your judgment. "
            "Use ONLY the provided evidence. Do not invent or assume facts not present in the evidence. "
            "Accept paraphrases, synonyms, and conceptually equivalent formulations as correct. "
            "Grade for understanding, not verbatim recall."
        )
        user_prompt = {
            "language": language,
            "concept_name": concept_name,
            "question_type": question_type,
            "expected_statement": expected_statement,
            "pedagogical_evidence": pedagogical_evidence,
            "learner_response": learner_response,
        }
        t0 = time.monotonic()
        try:
            content = await self._generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )
            decision = LearnerResponseGradingDecision.model_validate({**content, "used_llm": True})
            logger.debug("llm grade done", extra={"duration_ms": int((time.monotonic() - t0) * 1000)})
            return decision
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error("llm grade failed, using stub fallback", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message})
            return await self.fallback_provider.grade_learner_response(
                learner_response=learner_response,
                expected_statement=expected_statement,
                pedagogical_evidence=pedagogical_evidence,
                concept_name=concept_name,
                question_type=question_type,
                language=language,
            )

    async def generate_adaptive_block(
        self,
        *,
        evidence_units: list[dict[str, str]],
        target_dimension: str,
        difficulty: str,
        question_types: list[str],
        served_evidence_uids: list[str],
        item_count: int,
        concept_name: str,
        language: str,
    ) -> GeneratedAdaptiveBlock:
        dimension_focus = {
            "recognition": "recognition of definitions and key facts",
            "recall": "free recall and retrieval of specific information",
            "explanation": "explanation of concepts in the learner's own words",
            "application": "application of knowledge to new scenarios",
        }.get(target_dimension, target_dimension)
        system_prompt = (
            f"You are an expert pedagogical item generator. "
            f"Generate exactly {item_count} assessment items for the concept '{concept_name}', "
            f"targeting {dimension_focus} at {difficulty} difficulty level. "
            "Return only JSON with key 'items' — a list of objects, each containing: "
            "prompt (the full question text), "
            "choices (list of strings for MCQ, empty list for open/cloze/true_false), "
            "expected (list of expected correct answer strings), "
            "correct_choice_indexes (0-based integer indexes of correct choices; empty for non-MCQ), "
            "boolean_answer (true or false for true_false items, null otherwise), "
            "evidence_unit_id (the 'uid' of the evidence unit you used), "
            "rationale (one sentence explaining why this is correct per the evidence). "
            "Rules: "
            "Use ONLY the provided evidence units — do not invent facts. "
            "Do NOT reuse evidence units listed in served_evidence_uids (already shown to the learner). "
            "Vary question language and angle across items — do not repeat the same phrasing. "
            "For MCQ: generate 3-4 choices; distractors must be plausible but clearly incorrect, "
            "same language/length as the correct option, not copies of other evidence, not absurd. "
            "The correct answer must be grounded directly in the evidence. "
            "For open/cloze: expected answers must be derivable from the evidence. "
            "Write in the same language as the evidence and concept."
        )
        user_prompt = {
            "language": language,
            "concept_name": concept_name,
            "target_dimension": target_dimension,
            "difficulty": difficulty,
            "question_types": question_types,
            "item_count": item_count,
            "evidence_units": evidence_units,
            "served_evidence_uids": served_evidence_uids,
        }
        t0 = time.monotonic()
        try:
            content = await self._generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )
            result = GeneratedAdaptiveBlock.model_validate(content)
            logger.debug(
                "llm generate_block done",
                extra={"duration_ms": int((time.monotonic() - t0) * 1000), "items_count": len(result.items), "input_shape": {"concept": concept_name, "dimension": target_dimension, "difficulty": difficulty, "question_types": question_types, "item_count": item_count}, "output_shape": {"items": len(result.items)}},
            )
            return result
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "llm generate_block failed, using stub fallback",
                extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message, "error_type": exc.__class__.__name__, "error_message": message[:200]},
            )
            return await self.fallback_provider.generate_adaptive_block(
                evidence_units=evidence_units,
                target_dimension=target_dimension,
                difficulty=difficulty,
                question_types=question_types,
                served_evidence_uids=served_evidence_uids,
                item_count=item_count,
                concept_name=concept_name,
                language=language,
            )

    @abstractmethod
    async def _generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: dict[str, Any],
        temperature: float | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _sanitize_llm_extraction(self, result: ExtractionResult, text: str, tags: list[str]) -> ExtractionResult:
        domain = tags[0].strip().title() if tags else (result.domain.strip() or "General")
        topics = dedupe_preserve_order([domain, *[t.strip() for t in result.topics if t.strip()]])
        refined_concepts = self._sanitize_concepts(result.concepts, text)
        concept_names = [item["canonical_name"] for item in refined_concepts]
        refined_claims = self._sanitize_claims(result.claims, concept_names, text)
        refined_relations = self._sanitize_relations(result.relations, concept_names)
        return ExtractionResult.model_validate(
            {
                "domain": domain,
                "topics": topics or [domain],
                "concepts": refined_concepts,
                "claims": refined_claims[:8],
                "relations": refined_relations[:12],
            }
        )

    def _sanitize_concepts(self, concepts: list[Any], text: str) -> list[dict[str, Any]]:
        concepts_by_name: dict[str, dict[str, Any]] = {}
        for concept in concepts:
            raw_name = concept.canonical_name if hasattr(concept, "canonical_name") else concept.get("canonical_name", "")
            name = re.sub(r"\s+", " ", raw_name).strip()
            if not name:
                continue
            evidence_source = concept.evidence_quotes if hasattr(concept, "evidence_quotes") else concept.get("evidence_quotes", [])
            evidence_quotes = self._sanitize_quotes(evidence_source, text)
            key = normalize_text(name)
            aliases = [
                alias.strip()
                for alias in (concept.aliases if hasattr(concept, "aliases") else concept.get("aliases", []))
                if alias.strip() and normalize_text(alias) != key
            ]
            description = (concept.description if hasattr(concept, "description") else concept.get("description", "")).strip()
            confidence = float(concept.confidence if hasattr(concept, "confidence") else concept.get("confidence", 0.75))
            if key in concepts_by_name:
                existing = concepts_by_name[key]
                existing["aliases"] = dedupe_preserve_order([*existing["aliases"], *aliases])
                existing["evidence_quotes"] = dedupe_preserve_order([*existing["evidence_quotes"], *evidence_quotes])[:3]
                existing["confidence"] = max(existing["confidence"], confidence)
                if not existing["description"] and description:
                    existing["description"] = description
                continue
            concepts_by_name[key] = {
                "canonical_name": name,
                "aliases": dedupe_preserve_order(aliases),
                "description": description,
                "evidence_quotes": evidence_quotes[:3],
                "confidence": confidence,
            }
        return list(concepts_by_name.values())[:20]

    def _sanitize_claims(self, claims: list[Any], concept_names: list[str], text: str) -> list[dict[str, Any]]:
        refined_claims: list[dict[str, Any]] = []
        seen: set[str] = set()
        for claim in claims:
            text_value = (claim.text if hasattr(claim, "text") else claim.get("text", "")).strip()
            if not text_value:
                continue
            normalized = normalize_text(text_value)
            if normalized in seen:
                continue
            seen.add(normalized)
            explains_source = claim.explains if hasattr(claim, "explains") else claim.get("explains", [])
            explains = [
                matched
                for raw in explains_source
                if (matched := self.fallback_provider._find_matching_concept(raw, concept_names))
            ]
            if not explains:
                explains = self.fallback_provider._match_concepts_in_text(text_value, concept_names)[:3]
            supporting_quote = claim.supporting_quote if hasattr(claim, "supporting_quote") else claim.get("supporting_quote")
            supporting_quote = self._sanitize_quote(supporting_quote, text)
            confidence = float(claim.confidence if hasattr(claim, "confidence") else claim.get("confidence", 0.75))
            refined_claims.append(
                {
                    "text": text_value,
                    "confidence": confidence,
                    "explains": dedupe_preserve_order(explains),
                    "supporting_quote": supporting_quote,
                }
            )
        return refined_claims

    def _sanitize_relations(self, relations: list[Any], concept_names: list[str]) -> list[dict[str, Any]]:
        refined_relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in relations:
            from_name = self.fallback_provider._find_matching_concept(relation.from_name, concept_names)
            to_name = self.fallback_provider._find_matching_concept(relation.to_name, concept_names)
            if not from_name or not to_name or from_name == to_name:
                continue
            key = (from_name, relation.relation, to_name)
            if key in seen:
                continue
            seen.add(key)
            refined_relations.append(
                {
                    "from_name": from_name,
                    "relation": relation.relation,
                    "to_name": to_name,
                    "confidence": relation.confidence,
                }
            )
        return refined_relations

    def _sanitize_quotes(self, quotes: list[str], text: str) -> list[str]:
        sanitized: list[str] = []
        for quote in quotes:
            cleaned = self._sanitize_quote(quote, text)
            if cleaned:
                sanitized.append(cleaned)
        return dedupe_preserve_order(sanitized)

    def _sanitize_quote(self, quote: str | None, text: str) -> str | None:
        if not quote:
            return None
        cleaned = re.sub(r"\s+", " ", quote.strip().strip("\"'")).strip()
        if not cleaned:
            return None
        normalized_text = normalize_text(text)
        normalized_quote = normalize_text(cleaned)
        if len(normalized_quote) < 6:
            return None
        if normalized_quote not in normalized_text:
            return None
        return cleaned

    async def close(self) -> None:
        await self._embedding_provider.close()


class OpenAICompatibleProvider(StructuredLLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=openai_compatible")

        super().__init__(settings, OpenAICompatibleEmbeddingProvider(settings))
        self.client = httpx.AsyncClient(
            base_url=settings.openai_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=httpx.Timeout(settings.openai_timeout_seconds, connect=20.0),
        )

    async def _generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: dict[str, Any],
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.openai_chat_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature

        t0 = time.monotonic()
        request_text = json.dumps(payload, ensure_ascii=True)
        logger.debug("openai request", extra={"model": self.settings.openai_chat_model})
        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "openai http error",
                extra={"status": exc.response.status_code, "duration_ms": int((time.monotonic() - t0) * 1000)},
            )
            raise
        except httpx.TimeoutException:
            logger.error("openai timeout", extra={"duration_ms": int((time.monotonic() - t0) * 1000)})
            raise
        raw_payload = response.json()
        content = raw_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("structured response must be a JSON object")
        self._last_llm_boundary_payload = TraceBoundaryPayload(
            kind="llm",
            provider="openai_compatible",
            model=self.settings.openai_chat_model,
            request_text=request_text,
            response_text=response.text,
            request_json=payload,
            response_json=raw_payload,
            metadata={"parsed_content": parsed},
        )
        logger.debug("openai response", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "model": self.settings.openai_chat_model})
        return parsed

    async def close(self) -> None:
        await super().close()
        client_close = getattr(self.client, "aclose", None)
        if callable(client_close):
            await client_close()


class AnthropicGatewayProvider(StructuredLLMProvider):
    def __init__(self, settings: Settings) -> None:
        embedding_provider = build_embedding_provider(settings, require_explicit=True)
        super().__init__(settings, embedding_provider)

        if not settings.anthropic_gateway_bearer_token:
            raise ValueError("ANTHROPIC_GATEWAY_BEARER_TOKEN is required when AI_PROVIDER=anthropic")
        if not settings.anthropic_chat_model:
            raise ValueError("ANTHROPIC_CHAT_MODEL is required when AI_PROVIDER=anthropic")

        self.client = httpx.AsyncClient(
            base_url=settings.anthropic_gateway_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.anthropic_gateway_bearer_token}"},
            timeout=httpx.Timeout(settings.anthropic_timeout_seconds, connect=20.0),
        )

    async def _generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: dict[str, Any],
        temperature: float | None,
    ) -> dict[str, Any]:
        payload = self._build_gateway_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        t0 = time.monotonic()
        model = self.settings.anthropic_chat_model or ""
        request_text = json.dumps(payload, ensure_ascii=True)
        logger.debug("anthropic request", extra={"model": model})
        try:
            response = await self.client.post("v1/generate-json", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "anthropic http error",
                extra={"status": exc.response.status_code, "duration_ms": int((time.monotonic() - t0) * 1000)},
            )
            raise
        except httpx.TimeoutException:
            logger.error("anthropic timeout", extra={"duration_ms": int((time.monotonic() - t0) * 1000)})
            raise
        raw_payload = response.json()
        content = raw_payload["content_json"]
        if not isinstance(content, dict):
            raise ValueError("gateway content_json must be a JSON object")
        self._last_llm_boundary_payload = TraceBoundaryPayload(
            kind="llm",
            provider="anthropic",
            model=model,
            request_text=request_text,
            response_text=response.text,
            request_json=payload,
            response_json=raw_payload,
            metadata={"parsed_content": content},
        )
        logger.debug("anthropic response", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "model": model})
        return content

    def _build_gateway_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: dict[str, Any],
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.anthropic_chat_model,
            "system_prompt": system_prompt,
            "user_payload_json": user_prompt,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.settings.anthropic_thinking_type:
            thinking: dict[str, Any] = {"type": self.settings.anthropic_thinking_type}
            if self.settings.anthropic_thinking_budget_tokens is not None:
                thinking["budget_tokens"] = self.settings.anthropic_thinking_budget_tokens
            payload["thinking"] = thinking
        if self.settings.anthropic_effort and self.settings.anthropic_thinking_type != "enabled":
            payload["output_config"] = {"effort": self.settings.anthropic_effort}
        return payload

    async def close(self) -> None:
        await super().close()
        client_close = getattr(self.client, "aclose", None)
        if callable(client_close):
            await client_close()


def build_embedding_provider(settings: Settings, *, require_explicit: bool = False) -> EmbeddingProvider:
    provider_name = settings.resolved_embedding_provider if require_explicit else (
        settings.embedding_provider or settings.ai_provider
    )
    if provider_name == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(settings)
    return StubEmbeddingProvider(settings)


def build_ai_provider(settings: Settings) -> AIProvider:
    if settings.ai_provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if settings.ai_provider == "anthropic":
        return AnthropicGatewayProvider(settings)
    return StubAIProvider(settings)
