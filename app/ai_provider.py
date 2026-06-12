from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import Settings
from app.schemas import ExtractionResult, PedagogicalEvidenceDecision
from app.utils import dedupe_preserve_order, fit_embedding_dimensions, normalize_text, stable_embedding

logger = logging.getLogger(__name__)


_CONNECTOR_TOKENS = {"a", "al", "de", "del", "e", "la", "las", "los", "y"}
_QUALIFIER_TOKENS = {"medio", "temprano", "tardio", "tardío", "clasico", "clásico", "operante"}
_GENERIC_CONCEPT_TOKENS = {
    "aprox",
    "aproximado",
    "aproximados",
    "centros",
    "cultura",
    "fechas",
    "general",
    "habito",
    "historia",
    "importantes",
    "madre",
    "periodo",
    "periodos",
    "principalmente",
    "teoria",
    "teoría",
    "todo",
    "zonas",
}
_GENERIC_HEADINGS = {
    "centros importantes",
    "fechas aprox",
    "periodo",
    "periodos aproximados",
    "zonas que habito",
}
_CLAIM_VERB_MARKERS = {
    "contrasta",
    "desarrolla",
    "desarrollaron",
    "desarrollo",
    "desarrolló",
    "es",
    "explica",
    "explican",
    "fue",
    "fueron",
    "ocurre",
    "ocurrio",
    "ocurrió",
    "se",
    "son",
    "surge",
    "surgio",
    "surgió",
    "tiene",
    "tienen",
    "permite",
    "permiten",
}
_CONCEPT_HEAD_PATTERNS = (
    re.compile(
        r"\b(cultura|condicionamiento|aprendizaje|teor[ií]a|modelo|sistema|imperio|civilizaci[oó]n)\s+([a-záéíóúñ]+)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(precl[aá]sico|cl[aá]sico|poscl[aá]sico)\s+(temprano|medio|tard[ií]o)\b",
        flags=re.IGNORECASE,
    ),
)


class AIProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
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

    async def close(self) -> None:
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
        domain = tags[0].strip().title() if tags else "General"
        concepts = self._extract_concepts(text, tags)
        claims = self._extract_claims(text, concepts)
        relation_candidates = self._extract_relations(text, concepts)
        payload = {
            "domain": domain,
            "topics": [domain],
            "concepts": [
                {
                    "canonical_name": name,
                    "aliases": [],
                    "description": f"Concept extracted from fragment about {domain}.",
                    "confidence": 0.8,
                }
                for name in concepts
            ],
            "claims": [
                {
                    "text": claim,
                    "confidence": 0.75,
                    "explains": self._match_concepts_in_text(claim, concepts)[:3],
                }
                for claim in claims[:3]
            ],
            "relations": relation_candidates,
        }
        return self.refine_extraction(
            ExtractionResult.model_validate(payload),
            text,
            tags,
            augment_from_heuristics=True,
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

    def refine_extraction(
        self,
        result: ExtractionResult,
        text: str,
        tags: list[str],
        *,
        augment_from_heuristics: bool,
    ) -> ExtractionResult:
        domain = tags[0].strip().title() if tags else (result.domain.strip() or "General")
        heuristic_concepts = self._extract_concepts(text, tags)
        concept_candidates: list[dict[str, Any]] = [
            {
                "canonical_name": concept.canonical_name,
                "aliases": list(concept.aliases),
                "description": concept.description,
                "confidence": concept.confidence,
            }
            for concept in result.concepts
        ]
        if augment_from_heuristics or len(concept_candidates) < 2:
            concept_candidates.extend(
                {
                    "canonical_name": name,
                    "aliases": [],
                    "description": f"Concept extracted from fragment about {domain}.",
                    "confidence": 0.72,
                }
                for name in heuristic_concepts
            )

        refined_concepts = self._refine_concepts(concept_candidates, text, domain)
        concept_names = [item["canonical_name"] for item in refined_concepts]

        refined_claims = self._refine_claims(result, text, concept_names)
        if not refined_claims:
            refined_claims = [
                {
                    "text": claim,
                    "confidence": 0.75,
                    "explains": self._match_concepts_in_text(claim, concept_names)[:3],
                }
                for claim in self._extract_claims(text, concept_names)[:3]
            ]

        refined_relations = self._refine_relations(
            result=result,
            text=text,
            concept_names=concept_names,
            augment_from_heuristics=augment_from_heuristics,
        )

        topics = [
            self._titleize_phrase(topic)
            for topic in dedupe_preserve_order([domain, *result.topics])
            if topic.strip()
        ]
        return ExtractionResult.model_validate(
            {
                "domain": domain,
                "topics": topics or [domain],
                "concepts": refined_concepts,
                "claims": refined_claims[:3],
                "relations": refined_relations,
            }
        )

    def _extract_concepts(self, text: str, tags: list[str]) -> list[str]:
        concepts: list[str] = []
        concepts.extend(self._extract_table_concepts(text))
        concepts.extend(self._extract_section_concepts(text))
        concepts.extend(self._extract_pattern_concepts(text))
        concepts.extend(self._extract_capitalized_concepts(text))
        return dedupe_preserve_order([item for item in concepts if self._is_valid_concept(item)])

    def _extract_claims(self, text: str, concepts: list[str]) -> list[str]:
        claims: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue
            cleaned_lines = [
                line
                for line in lines
                if not line.endswith(":") and "\t" not in line and normalize_text(line) not in _GENERIC_HEADINGS
            ]
            if not cleaned_lines:
                continue
            joined = " ".join(cleaned_lines)
            joined = joined.replace("aprox.", "aprox").replace("a. C.", "a C").replace("d. C.", "d C")
            for sentence in re.split(r"(?<=[.!?])\s+", joined):
                candidate = sentence.strip()
                if self._looks_like_claim(candidate):
                    claims.append(candidate)

        if claims:
            return dedupe_preserve_order(claims)
        fallback = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        return fallback[:3]

    def _extract_relations(self, text: str, concepts: list[str]) -> list[dict[str, str | float]]:
        relations: list[dict[str, str | float]] = []
        seen: set[tuple[str, str, str]] = set()

        for claim in self._extract_claims(text, concepts):
            mentioned = self._match_concepts_in_text(claim, concepts)
            if len(mentioned) < 2:
                continue
            relation = self._infer_relation_type(claim)
            if not relation:
                continue
            left, right = mentioned[0], mentioned[1]
            if left == right:
                continue
            key = (left, relation, right)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "from_name": left,
                    "relation": relation,
                    "to_name": right,
                    "confidence": 0.72,
                }
            )

        primary_concept = self._infer_primary_concept(concepts)
        if primary_concept:
            current_section: str | None = None
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_section = normalize_text(line[:-1])
                    continue
                if current_section:
                    relation = None
                    direction = "from_primary"
                    if "centro" in current_section:
                        relation = "PART_OF"
                        direction = "to_primary"
                    elif any(keyword in current_section for keyword in ("zona", "region", "ubic")):
                        relation = "RELATED_TO"
                    if relation:
                        for item in self._extract_list_items(line):
                            matched = self._find_matching_concept(item, concepts)
                            if not matched or matched == primary_concept:
                                continue
                            left, right = (
                                (matched, primary_concept)
                                if direction == "to_primary"
                                else (primary_concept, matched)
                            )
                            key = (left, relation, right)
                            if key in seen:
                                continue
                            seen.add(key)
                            relations.append(
                                {
                                    "from_name": left,
                                    "relation": relation,
                                    "to_name": right,
                                    "confidence": 0.68,
                                }
                            )
                    current_section = None
        return relations

    def _refine_concepts(self, candidates: list[dict[str, Any]], text: str, domain: str) -> list[dict[str, Any]]:
        concepts_by_name: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            raw_name = candidate.get("canonical_name", "")
            cleaned_name = self._clean_concept_name(raw_name)
            if not self._is_valid_concept(cleaned_name):
                continue
            key = normalize_text(cleaned_name)
            aliases = [
                self._titleize_phrase(alias)
                for alias in candidate.get("aliases", [])
                if self._is_valid_concept(alias)
            ]
            description = candidate.get("description") or f"Concept extracted from fragment about {domain}."
            confidence = float(candidate.get("confidence", 0.75))
            if key in concepts_by_name:
                merged_aliases = dedupe_preserve_order([*concepts_by_name[key]["aliases"], *aliases])
                concepts_by_name[key]["aliases"] = merged_aliases
                concepts_by_name[key]["confidence"] = max(concepts_by_name[key]["confidence"], confidence)
                if not concepts_by_name[key]["description"] and description:
                    concepts_by_name[key]["description"] = description
                continue
            concepts_by_name[key] = {
                "canonical_name": cleaned_name,
                "aliases": aliases,
                "description": description,
                "confidence": confidence,
            }
        return list(concepts_by_name.values())[:12]

    def _refine_claims(self, result: ExtractionResult, text: str, concept_names: list[str]) -> list[dict[str, Any]]:
        refined_claims: list[dict[str, Any]] = []
        seen: set[str] = set()
        candidates = list(result.claims)
        if not candidates:
            candidates = [
                {
                    "text": sentence,
                    "confidence": 0.75,
                    "explains": self._match_concepts_in_text(sentence, concept_names)[:3],
                }
                for sentence in self._extract_claims(text, concept_names)
            ]

        for candidate in candidates:
            text_value = candidate.text if hasattr(candidate, "text") else candidate.get("text", "")
            text_value = text_value.strip()
            if not self._looks_like_claim(text_value):
                continue
            normalized = normalize_text(text_value)
            if normalized in seen:
                continue
            seen.add(normalized)
            explains_source = candidate.explains if hasattr(candidate, "explains") else candidate.get("explains", [])
            explains = [
                matched
                for raw in explains_source
                if (matched := self._find_matching_concept(raw, concept_names))
            ]
            if not explains:
                explains = self._match_concepts_in_text(text_value, concept_names)[:3]
            confidence = float(candidate.confidence if hasattr(candidate, "confidence") else candidate.get("confidence", 0.75))
            refined_claims.append(
                {
                    "text": text_value,
                    "confidence": confidence,
                    "explains": dedupe_preserve_order(explains),
                }
            )
        return refined_claims

    def _refine_relations(
        self,
        *,
        result: ExtractionResult,
        text: str,
        concept_names: list[str],
        augment_from_heuristics: bool,
    ) -> list[dict[str, Any]]:
        refined_relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in result.relations:
            from_name = self._find_matching_concept(relation.from_name, concept_names)
            to_name = self._find_matching_concept(relation.to_name, concept_names)
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
        if augment_from_heuristics or not refined_relations:
            for relation in self._extract_relations(text, concept_names):
                key = (relation["from_name"], relation["relation"], relation["to_name"])
                if relation["from_name"] == relation["to_name"] or key in seen:
                    continue
                seen.add(key)
                refined_relations.append(relation)
        return refined_relations

    def _extract_table_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        for raw_line in text.splitlines():
            if "\t" not in raw_line:
                continue
            cells = [cell.strip() for cell in re.split(r"\t+", raw_line) if cell.strip()]
            if not cells:
                continue
            label = self._clean_concept_name(cells[0])
            if self._is_valid_concept(label):
                concepts.append(label)
        return concepts

    def _extract_section_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        current_section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_section = normalize_text(line[:-1])
                continue
            if current_section:
                concepts.extend(self._extract_list_items(line))
                current_section = None
        return concepts

    def _extract_pattern_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        lowered = text.lower()
        for pattern in _CONCEPT_HEAD_PATTERNS:
            for match in pattern.finditer(lowered):
                concepts.append(self._clean_concept_name(match.group(0)))
        return concepts

    def _extract_capitalized_concepts(self, text: str) -> list[str]:
        concepts: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "\t" in line or line.endswith(":"):
                continue
            for chunk in re.split(r"(?<=[.!?])\s+", line):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "," in chunk or re.search(r"\s[ye]\s", chunk):
                    concepts.extend(self._extract_list_items(chunk))
                    continue
                concepts.extend(self._extract_capitalized_concepts_from_chunk(chunk))
        return concepts

    def _extract_capitalized_concepts_from_chunk(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", text)
        concepts: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not self._looks_like_proper_token(token):
                index += 1
                continue

            phrase = [token]
            cursor = index + 1
            while cursor < len(tokens):
                if len([item for item in phrase if normalize_text(item) not in _CONNECTOR_TOKENS]) >= 3:
                    break
                current = tokens[cursor]
                current_normalized = normalize_text(current)
                if (
                    current_normalized in _CONNECTOR_TOKENS
                    and cursor + 1 < len(tokens)
                    and self._looks_like_proper_token(tokens[cursor + 1])
                ):
                    phrase.extend([current.lower(), tokens[cursor + 1]])
                    cursor += 2
                    continue
                if current_normalized in _QUALIFIER_TOKENS:
                    phrase.append(current.lower())
                    cursor += 1
                    continue
                if self._looks_like_proper_token(current):
                    phrase.append(current)
                    cursor += 1
                    continue
                break
            concepts.append(self._clean_concept_name(" ".join(phrase)))
            index = cursor
        return concepts

    def _extract_list_items(self, line: str) -> list[str]:
        items: list[str] = []
        raw_parts = [part.strip(" .;") for part in re.split(r",|\sy\s|\se\s", line) if part.strip(" .;")]
        if len(raw_parts) > 1:
            for part in raw_parts:
                items.extend(self._extract_capitalized_concepts_from_chunk(part) or [self._clean_concept_name(part)])
            return items
        return self._extract_capitalized_concepts_from_chunk(line)

    def _infer_relation_type(self, sentence: str) -> str | None:
        normalized = normalize_text(sentence)
        if "contrasta con" in normalized or " versus " in normalized or " vs " in normalized:
            return "CONTRASTS_WITH"
        if "prerrequisito" in normalized:
            return "PREREQUISITE_FOR"
        if "explica" in normalized:
            return "EXPLAINS"
        if "es parte de" in normalized or "forma parte de" in normalized:
            return "PART_OF"
        if "se desarrollo" in normalized or "durante" in normalized:
            return "RELATED_TO"
        return None

    def _infer_primary_concept(self, concepts: list[str]) -> str | None:
        for concept in concepts:
            normalized = normalize_text(concept)
            if normalized.startswith(("cultura ", "condicionamiento ", "aprendizaje ", "teoria ", "teoria ", "imperio ")):
                return concept
        return concepts[0] if concepts else None

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

    def _find_matching_concept(self, raw_value: str, concepts: list[str]) -> str | None:
        target = normalize_text(raw_value)
        for concept in concepts:
            if normalize_text(concept) == target:
                return concept
        return None

    def _looks_like_claim(self, sentence: str) -> bool:
        normalized = normalize_text(sentence)
        if len(normalized.split()) < 4:
            return False
        if normalized in _GENERIC_HEADINGS or "\t" in sentence:
            return False
        return any(marker in normalized.split() for marker in _CLAIM_VERB_MARKERS) or sentence.endswith((".", "!", "?"))

    def _looks_like_proper_token(self, token: str) -> bool:
        return bool(token) and token[0].isupper()

    def _is_valid_concept(self, raw_value: str) -> bool:
        value = self._clean_concept_name(raw_value)
        normalized = normalize_text(value)
        if not normalized or len(normalized) < 3:
            return False
        if normalized in _GENERIC_HEADINGS:
            return False
        if any(char.isdigit() for char in value):
            return False
        meaningful_tokens = [token for token in normalized.split() if token not in _CONNECTOR_TOKENS]
        if not meaningful_tokens:
            return False
        if any(len(token) == 1 for token in meaningful_tokens):
            return False
        if all(token in _GENERIC_CONCEPT_TOKENS for token in meaningful_tokens):
            return False
        if meaningful_tokens[0] in {"aprox", "principalmente", "todo"}:
            return False
        if (
            len(meaningful_tokens) >= 2
            and meaningful_tokens[0] in {"aprendizaje", "condicionamiento", "cultura", "modelo", "sistema", "teoria"}
            and meaningful_tokens[1] in _GENERIC_CONCEPT_TOKENS
        ):
            return False
        return True

    def _clean_concept_name(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value.replace("\t", " ")).strip(" .,:;")
        cleaned = self._trim_predicate_suffix(cleaned)
        return self._titleize_phrase(cleaned)

    def _titleize_phrase(self, value: str) -> str:
        words = [word for word in re.split(r"\s+", value.strip()) if word]
        titled: list[str] = []
        for index, word in enumerate(words):
            normalized = normalize_text(word)
            if index > 0 and normalized in _CONNECTOR_TOKENS:
                titled.append(word.lower())
            else:
                titled.append(word.capitalize() if word.islower() else word)
        return " ".join(titled)

    def _trim_predicate_suffix(self, value: str) -> str:
        words = [word for word in re.split(r"\s+", value.strip()) if word]
        trimmed: list[str] = []
        for index, word in enumerate(words):
            normalized = normalize_text(word)
            if index > 0 and normalized in _CLAIM_VERB_MARKERS:
                break
            trimmed.append(word)
        return " ".join(trimmed).strip()


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
                extra={"status": exc.response.status_code, "duration_ms": int((time.monotonic() - t0) * 1000)},
            )
            raise
        except httpx.TimeoutException:
            logger.error("embed timeout", extra={"duration_ms": int((time.monotonic() - t0) * 1000)})
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

    async def embed(self, text: str) -> list[float]:
        return await self._embedding_provider.embed(text)

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
            logger.debug("llm extract done", extra={"duration_ms": int((time.monotonic() - t0) * 1000)})
            return self._sanitize_llm_extraction(result, text, tags)
        except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.error("llm extract failed", extra={"duration_ms": int((time.monotonic() - t0) * 1000), "error": message})
            raise ValueError(f"llm extraction failed: {message}") from exc

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
        topics = [
            self.fallback_provider._titleize_phrase(topic)
            for topic in dedupe_preserve_order([domain, *result.topics])
            if topic.strip()
        ]
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
            cleaned_name = self.fallback_provider._clean_concept_name(raw_name)
            if not self.fallback_provider._is_valid_concept(cleaned_name):
                continue
            evidence_source = concept.evidence_quotes if hasattr(concept, "evidence_quotes") else concept.get("evidence_quotes", [])
            evidence_quotes = self._sanitize_quotes(evidence_source, text)
            key = normalize_text(cleaned_name)
            aliases = [
                self.fallback_provider._titleize_phrase(alias)
                for alias in (concept.aliases if hasattr(concept, "aliases") else concept.get("aliases", []))
                if alias and normalize_text(alias) != key
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
                "canonical_name": cleaned_name,
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
            if not explains and not supporting_quote:
                continue
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
