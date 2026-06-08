from __future__ import annotations

from collections.abc import Iterable

from app.ai_provider import AIProvider
from app.config import Settings
from app.schemas import (
    CandidateHit,
    LearningContextClaim,
    LearningContextDebug,
    LearningContextEpisode,
    LearningContextPrimaryConcept,
    LearningContextRequest,
    LearningContextResponse,
    NeighborhoodRelation,
    TutorContextRequest,
    TutorContextResolvedReference,
    TutorContextResponse,
)
from app.store import KnowledgeStore
from app.utils import normalize_text


_GENERIC_CONCEPT_TOKENS = {
    "aprendizaje",
    "concepto",
    "contenido",
    "cultura",
    "general",
    "historia",
    "materia",
    "memoria",
    "modelo",
    "periodo",
    "proceso",
    "sistema",
    "teoria",
    "tema",
}
_PLACEHOLDER_DESCRIPTION_PREFIXES = {
    "concept extracted from fragment about",
    "descripcion de",
    "description of",
}
_PLACEHOLDER_DESCRIPTIONS = {
    "descripcion",
    "sin descripcion",
    "sin descripción",
    "n a",
    "na",
    "none",
    "pendiente",
    "por definir",
    "tbd",
}
_REASONS_WITH_MANDATORY_SELECTION = {"normalized_name", "alias"}


class LearningContextBuilder:
    def __init__(
        self,
        *,
        settings: Settings,
        store: KnowledgeStore,
        ai_provider: AIProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._ai_provider = ai_provider

    async def build(self, payload: LearningContextRequest) -> LearningContextResponse:
        query_embedding = await self._ai_provider.embed(payload.query)
        candidates = await self._store.search_candidates(
            query=payload.query,
            domain_hint=payload.domain_hint,
            query_embedding=query_embedding,
            limit=payload.candidate_limit,
        )
        selected, selection_reasons = self._select_primary_candidates(candidates, payload.concept_limit)

        if not selected:
            return LearningContextResponse(
                query=payload.query,
                domain_hint=payload.domain_hint,
                status="no_match",
                primary_concepts=[],
                relations=[],
                claims=[],
                episodes=[],
                warnings=self._warnings_for_no_match(candidates),
                debug=LearningContextDebug(
                    candidate_count=len(candidates),
                    selected_concept_uids=[],
                    selection_reasons=selection_reasons,
                ),
            )

        primary_concepts = [self._build_primary_concept(candidate) for candidate in selected]

        relation_map: dict[tuple[str, str, str, str | None], NeighborhoodRelation] = {}
        claim_map: dict[str, LearningContextClaim] = {}
        episode_map: dict[str, LearningContextEpisode] = {}
        if payload.include_neighborhood:
            for candidate in selected:
                neighborhood = await self._store.get_neighborhood(candidate.uid, payload.depth)
                if not neighborhood:
                    continue
                for relation in neighborhood.relations:
                    key = (
                        relation.from_uid,
                        relation.relation,
                        relation.to_uid,
                        relation.evidence_episode_id,
                    )
                    relation_map.setdefault(key, relation)
                for claim in neighborhood.claims:
                    normalized_claim = normalize_text(str(claim.get("text") or ""))
                    claim_key = str(claim.get("uid") or normalized_claim)
                    if not claim_key or claim_key in claim_map:
                        continue
                    claim_map[claim_key] = LearningContextClaim(
                        uid=str(claim.get("uid") or claim_key),
                        text=str(claim.get("text") or ""),
                        confidence=self._as_optional_float(claim.get("confidence")),
                    )
                for episode in neighborhood.episodes:
                    episode_uid = str(episode.get("uid") or "")
                    if not episode_uid or episode_uid in episode_map:
                        continue
                    episode_map[episode_uid] = LearningContextEpisode(
                        uid=episode_uid,
                        text=self._trim_episode_text(str(episode.get("text") or "")),
                        status=str(episode.get("status") or ""),
                    )

        warnings = self._build_warnings(
            candidates=candidates,
            primary_concepts=primary_concepts,
            relations=relation_map.values(),
            claims=claim_map.values(),
            episodes=episode_map.values(),
            include_neighborhood=payload.include_neighborhood,
        )
        status = "ok" if not warnings else "sparse"
        return LearningContextResponse(
            query=payload.query,
            domain_hint=payload.domain_hint,
            status=status,
            primary_concepts=primary_concepts,
            relations=list(relation_map.values()),
            claims=list(claim_map.values())[: payload.claim_limit],
            episodes=list(episode_map.values())[: payload.episode_limit],
            warnings=warnings,
            debug=LearningContextDebug(
                candidate_count=len(candidates),
                selected_concept_uids=[concept.uid for concept in primary_concepts],
                selection_reasons=selection_reasons,
            ),
        )

    def _select_primary_candidates(
        self,
        candidates: list[CandidateHit],
        concept_limit: int,
    ) -> tuple[list[CandidateHit], list[str]]:
        selected: list[CandidateHit] = []
        selection_reasons: list[str] = []
        for candidate in candidates:
            decision = self._selection_decision(candidate)
            if not decision:
                continue
            selected.append(candidate)
            selection_reasons.append(
                f"{candidate.uid}:{candidate.reason}:{candidate.score:.2f}:{decision}"
            )
            if len(selected) >= concept_limit:
                break
        return selected, selection_reasons

    def _selection_decision(self, candidate: CandidateHit) -> str | None:
        return _selection_decision(self._settings, candidate)

    def _build_primary_concept(self, candidate: CandidateHit) -> LearningContextPrimaryConcept:
        quality_flags: list[str] = []
        if self._is_generic_name(candidate.canonical_name):
            quality_flags.append("generic_name")
        if self._is_weak_description(candidate.description):
            quality_flags.append("weak_description")
        if candidate.score < self._settings.learning_context_low_score_threshold:
            quality_flags.append("low_score")
        return LearningContextPrimaryConcept(
            uid=candidate.uid,
            canonical_name=candidate.canonical_name,
            domain=candidate.domain,
            description=candidate.description,
            retrieval_score=candidate.score,
            retrieval_reason=candidate.reason,
            quality_flags=quality_flags,
        )

    def _build_warnings(
        self,
        *,
        candidates: list[CandidateHit],
        primary_concepts: list[LearningContextPrimaryConcept],
        relations: Iterable[NeighborhoodRelation],
        claims: Iterable[LearningContextClaim],
        episodes: Iterable[LearningContextEpisode],
        include_neighborhood: bool,
    ) -> list[str]:
        warnings: list[str] = []
        if self._has_fragmented_results(candidates, primary_concepts):
            warnings.append("fragmented_results")
        if any("generic_name" in concept.quality_flags for concept in primary_concepts):
            warnings.append("generic_primary_concepts")
        if any("weak_description" in concept.quality_flags for concept in primary_concepts):
            warnings.append("weak_concept_descriptions")
        if any("low_score" in concept.quality_flags for concept in primary_concepts):
            warnings.append("low_relevance_candidates")
        if include_neighborhood:
            if not any(True for _ in relations):
                warnings.append("no_neighbor_relations")
            if not any(True for _ in claims):
                warnings.append("no_supporting_claims")
            if not any(True for _ in episodes):
                warnings.append("no_source_episodes")
        return warnings

    def _warnings_for_no_match(self, candidates: list[CandidateHit]) -> list[str]:
        if not candidates:
            return ["no_candidates_found"]
        return ["no_usable_primary_concepts"]

    def _has_fragmented_results(
        self,
        candidates: list[CandidateHit],
        primary_concepts: list[LearningContextPrimaryConcept],
    ) -> bool:
        primary_domains = {concept.domain for concept in primary_concepts if concept.domain}
        if len(primary_domains) > 1:
            return True
        top_candidates = candidates[:3]
        top_domains = {candidate.domain for candidate in top_candidates if candidate.domain}
        if len(top_domains) <= 1 or len(top_candidates) < 2:
            return False
        score_gap = top_candidates[0].score - top_candidates[-1].score
        return score_gap <= self._settings.learning_context_fragmentation_gap_threshold

    def _is_generic_name(self, value: str) -> bool:
        normalized = normalize_text(value)
        if not normalized or len(normalized) <= 3:
            return True
        tokens = normalized.split()
        if normalized in _GENERIC_CONCEPT_TOKENS:
            return True
        return len(tokens) == 1 and tokens[0] in _GENERIC_CONCEPT_TOKENS

    def _is_weak_description(self, value: str) -> bool:
        normalized = normalize_text(value)
        if not normalized:
            return True
        if len(normalized) < self._settings.learning_context_min_description_chars:
            return True
        if normalized in _PLACEHOLDER_DESCRIPTIONS:
            return True
        return any(normalized.startswith(prefix) for prefix in _PLACEHOLDER_DESCRIPTION_PREFIXES)

    def _trim_episode_text(self, text: str) -> str:
        limit = self._settings.learning_context_episode_excerpt_chars
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _as_optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class TutorContextBuilder:
    def __init__(
        self,
        *,
        settings: Settings,
        store: KnowledgeStore,
        ai_provider: AIProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._ai_provider = ai_provider

    async def build(self, payload: TutorContextRequest) -> TutorContextResponse:
        if payload.episode_id:
            resolved = TutorContextResolvedReference(
                input_type="episode_id",
                input_value=payload.episode_id,
                resolved_episode_id=payload.episode_id,
            )
            bundle = await self._store.get_tutor_context_for_episode(payload.episode_id, payload.depth)
            return self._finalize_response(resolved=resolved, bundle=bundle, include_evidence=payload.include_evidence)

        if payload.job_id:
            job = await self._store.get_job(payload.job_id)
            resolved = TutorContextResolvedReference(
                input_type="job_id",
                input_value=payload.job_id,
                resolved_job_id=payload.job_id,
                resolved_episode_id=job.episode_id if job else None,
            )
            if job is None:
                return self._failure_response(resolved, "job_not_found", ["job_not_found"])
            bundle = await self._store.get_tutor_context_for_episode(job.episode_id, payload.depth)
            return self._finalize_response(resolved=resolved, bundle=bundle, include_evidence=payload.include_evidence)

        assert payload.query is not None
        query_embedding = await self._ai_provider.embed(payload.query)
        candidates = await self._store.search_candidates(
            query=payload.query,
            domain_hint=None,
            query_embedding=query_embedding,
            limit=8,
        )
        usable = [candidate for candidate in candidates if _selection_decision(self._settings, candidate)]
        if not usable:
            return self._failure_response(
                TutorContextResolvedReference(input_type="query", input_value=payload.query),
                "query_not_resolved",
                ["no_usable_primary_concepts"] if candidates else ["no_candidates_found"],
            )
        if self._is_ambiguous_query(usable):
            return self._failure_response(
                TutorContextResolvedReference(input_type="query", input_value=payload.query),
                "ambiguous_query_resolution",
                ["ambiguous_query_resolution"],
            )

        primary = usable[0]
        resolved = TutorContextResolvedReference(
            input_type="query",
            input_value=payload.query,
            resolved_concept_uid=primary.uid,
            resolved_concept_name=primary.canonical_name,
            resolution_reason=primary.reason,
        )
        bundle = await self._store.get_tutor_context_for_concept(primary.uid, payload.depth)
        return self._finalize_response(resolved=resolved, bundle=bundle, include_evidence=payload.include_evidence)

    def _finalize_response(
        self,
        *,
        resolved: TutorContextResolvedReference,
        bundle,
        include_evidence: bool,
    ) -> TutorContextResponse:
        if bundle is None:
            return self._failure_response(resolved, "resolved_reference_not_found", ["resolved_reference_not_found"])
        evidence = bundle.evidence if include_evidence else []
        if not self._has_traceable_evidence(bundle, evidence):
            return self._failure_response(
                resolved,
                "insufficient_traceable_evidence",
                self._sufficiency_warnings(bundle, evidence),
            )
        return TutorContextResponse(
            resolved_reference=resolved,
            status="ok",
            concepts=bundle.concepts,
            claims=bundle.claims,
            relations=bundle.relations,
            source_fragments=bundle.source_fragments,
            evidence=evidence,
            pedagogical_evidence=bundle.pedagogical_evidence,
            warnings=[],
            failure_reason=None,
        )

    def _failure_response(
        self,
        resolved: TutorContextResolvedReference,
        failure_reason: str,
        warnings: list[str],
    ) -> TutorContextResponse:
        return TutorContextResponse(
            resolved_reference=resolved,
            status="failed",
            concepts=[],
            claims=[],
            relations=[],
            source_fragments=[],
            evidence=[],
            pedagogical_evidence=[],
            warnings=warnings,
            failure_reason=failure_reason,
        )

    def _has_traceable_evidence(self, bundle, evidence) -> bool:
        return bool(bundle.source_fragments and evidence and bundle.pedagogical_evidence)

    def _sufficiency_warnings(self, bundle, evidence) -> list[str]:
        warnings: list[str] = []
        if not bundle.source_fragments:
            warnings.append("no_source_fragments")
        if not bundle.pedagogical_evidence:
            warnings.append("no_pedagogical_evidence")
        if not evidence:
            warnings.append("no_evidence_links")
        return warnings or ["insufficient_traceable_evidence"]

    def _is_ambiguous_query(self, usable: list[CandidateHit]) -> bool:
        if len(usable) < 2:
            return False
        first, second = usable[0], usable[1]
        if first.uid == second.uid:
            return False
        return abs(first.score - second.score) <= self._settings.learning_context_fragmentation_gap_threshold


def _selection_decision(settings: Settings, candidate: CandidateHit) -> str | None:
    if candidate.reason in _REASONS_WITH_MANDATORY_SELECTION:
        return "exact_or_alias_match"
    if candidate.reason == "full_text" and candidate.score >= settings.learning_context_full_text_min_score:
        return "full_text_above_threshold"
    if candidate.reason == "vector" and candidate.score >= settings.learning_context_vector_min_score:
        return "vector_above_threshold"
    return None
