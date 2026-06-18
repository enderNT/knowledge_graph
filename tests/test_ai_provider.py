from __future__ import annotations

import asyncio
import json

import pytest

from app.ai_provider import AnthropicGatewayProvider, OpenAICompatibleProvider, StubAIProvider, build_ai_provider
from app.config import Settings
from app.schemas import ExtractionResult, ExtractedConcept, ExtractedClaim, ExtractedRelation
from app.utils import fit_embedding_dimensions


class _CapturedResponse:
    def __init__(self, content: dict[str, object]) -> None:
        self._content = content
        self.text = json.dumps({"choices": [{"message": {"content": json.dumps(self._content)}}]})

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._content)}}]}


class _CapturedClient:
    def __init__(self, content: dict[str, object]) -> None:
        self._content = content
        self.requests: list[dict[str, object]] = []

    async def post(self, *_args, **kwargs):
        self.requests.append(kwargs)
        return _CapturedResponse(self._content)


class _GatewayCapturedResponse:
    def __init__(self, content: dict[str, object]) -> None:
        self._content = content
        self.text = json.dumps({"content_json": self._content})

    def raise_for_status(self):
        return None

    def json(self):
        return {"content_json": self._content}


class _GatewayCapturedClient:
    def __init__(self, content: dict[str, object]) -> None:
        self._content = content
        self.requests: list[dict[str, object]] = []

    async def post(self, *_args, **kwargs):
        self.requests.append(kwargs)
        return _GatewayCapturedResponse(self._content)


def test_stub_extract_raises(settings):
    provider = StubAIProvider(settings)
    with pytest.raises(NotImplementedError, match="requires a real LLM provider"):
        asyncio.run(provider.extract("Texto de prueba.", "es", ["General"]))


def test_stub_vet_extraction_raises(settings):
    provider = StubAIProvider(settings)
    extraction = ExtractionResult(domain="General")
    with pytest.raises(NotImplementedError, match="requires a real LLM provider"):
        asyncio.run(provider.vet_extraction(extraction=extraction, text="Texto.", language="es"))


def test_openai_provider_exposes_full_llm_boundary_payload():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    provider.client = _CapturedClient({"domain": "General", "topics": ["General"], "concepts": [], "claims": [], "relations": []})  # type: ignore[assignment]

    asyncio.run(provider.extract("Texto completo para el LLM.", "es", ["General"]))
    payload = provider.consume_last_llm_boundary_payload()

    assert payload is not None
    assert payload.kind == "llm"
    assert payload.provider == "openai_compatible"
    assert "Texto completo para el LLM." in (payload.request_text or "")
    assert payload.response_text is not None
    assert "\"choices\"" in payload.response_text
    assert "\\\"domain\\\": \\\"General\\\"" in payload.response_text
    assert payload.response_json is not None
    assert payload.metadata["parsed_content"]["domain"] == "General"


def test_llm_sanitize_concepts_keeps_any_non_empty_name():
    """After removing wordlist gates, _sanitize_concepts must not discard concepts for
    language-specific reasons — only empty names are invalid."""
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    text = "Learning theory: general culture and period of history."
    # These names would have been rejected by old _GENERIC_CONCEPT_TOKENS / _GENERIC_HEADINGS
    llm_concepts = [
        {
            "canonical_name": "Learning Theory",
            "aliases": [],
            "description": "Framework for how learning occurs.",
            "evidence_quotes": ["Learning theory"],
            "confidence": 0.9,
        },
        {
            "canonical_name": "General Culture",
            "aliases": [],
            "description": "Broad cultural context.",
            "evidence_quotes": ["general culture"],
            "confidence": 0.8,
        },
    ]
    result = provider._sanitize_concepts(llm_concepts, text)
    names = [item["canonical_name"] for item in result]
    assert "Learning Theory" in names
    assert "General Culture" in names


def test_llm_sanitize_claims_keeps_claims_without_spanish_verbs():
    """After removing _looks_like_claim gate, claims that lack Spanish verb markers
    but are valid English/Portuguese sentences must be kept."""
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    text = "Classical conditioning associates stimuli and responses through repeated pairing."
    claim_text = "Classical conditioning associates stimuli through pairing."
    claims = [
        {
            "text": claim_text,
            "confidence": 0.9,
            "explains": ["Classical conditioning"],
            "supporting_quote": "Classical conditioning associates stimuli",
        }
    ]
    result = provider._sanitize_claims(claims, ["Classical conditioning"], text)
    assert len(result) == 1
    assert result[0]["text"] == claim_text


def test_llm_sanitize_claims_keeps_multilingual_claims_equally():
    """English, Spanish and Portuguese claims must pass through _sanitize_claims identically."""
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    texts_and_claims = [
        (
            "Operant conditioning shapes behavior through reinforcement.",
            "Operant conditioning shapes behavior through reinforcement.",
            "en",
        ),
        (
            "O condicionamento operante molda o comportamento por reforço.",
            "O condicionamento operante molda o comportamento por reforço.",
            "pt",
        ),
        (
            "El condicionamiento operante moldea la conducta por refuerzo.",
            "El condicionamiento operante moldea la conducta por refuerzo.",
            "es",
        ),
    ]
    for text, claim_text, _lang in texts_and_claims:
        claims = [{"text": claim_text, "confidence": 0.85, "explains": [], "supporting_quote": claim_text[:40]}]
        result = provider._sanitize_claims(claims, [], text)
        assert len(result) == 1, f"claim dropped for language {_lang}: {claim_text}"


def test_llm_vet_extraction_keep_drop_repair():
    """vet_extraction must apply keep/drop/repair decisions from the LLM."""
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)

    extraction = ExtractionResult(
        domain="Psicología",
        topics=["Psicología"],
        concepts=[
            ExtractedConcept(canonical_name="Condicionamiento Clásico", confidence=0.9, evidence_quotes=["clásico"]),
            ExtractedConcept(canonical_name="Periodos Aproximados", confidence=0.5, evidence_quotes=[]),
        ],
        claims=[
            ExtractedClaim(text="El condicionamiento clásico asocia estímulos.", confidence=0.9, explains=["Condicionamiento Clásico"]),
            ExtractedClaim(text="Introducción al manual.", confidence=0.4, explains=[]),
        ],
        relations=[
            ExtractedRelation(from_name="Condicionamiento Clásico", relation="RELATED_TO", to_name="Periodos Aproximados"),
        ],
    )
    text = "El condicionamiento clásico asocia estímulos. Introducción al manual. clásico"

    vet_response = {
        "concept_decisions": [
            {"name": "Condicionamiento Clásico", "status": "keep", "review_notes": []},
            {"name": "Periodos Aproximados", "status": "drop", "review_notes": ["non_teachable_heading"]},
        ],
        "claim_decisions": [
            {"name": "El condicionamiento clásico asocia estímulos.", "status": "keep", "review_notes": []},
            {"name": "Introducción al manual.", "status": "drop", "review_notes": ["meta_content"]},
        ],
        "relation_decisions": [
            {"name": "Condicionamiento Clásico → RELATED_TO → Periodos Aproximados", "status": "drop", "review_notes": ["target_dropped"]},
        ],
    }
    provider.client = _CapturedClient(vet_response)

    result = asyncio.run(
        provider.vet_extraction(extraction=extraction, text=text, language="es")
    )

    assert len(result.concepts) == 1
    assert result.concepts[0].canonical_name == "Condicionamiento Clásico"
    assert len(result.claims) == 1
    assert result.claims[0].text == "El condicionamiento clásico asocia estímulos."
    assert len(result.relations) == 0

    dropped_concept_decisions = [d for d in result.decisions if d.item_type == "concept" and d.status == "drop"]
    assert any(d.name == "Periodos Aproximados" for d in dropped_concept_decisions)


def test_llm_vet_extraction_repair_updates_concept_name():
    """A 'repair' decision must update the concept's canonical_name."""
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)

    extraction = ExtractionResult(
        domain="Test",
        concepts=[
            ExtractedConcept(canonical_name="Condicionamiento clasico", confidence=0.9, evidence_quotes=[]),
        ],
    )
    text = "Condicionamiento clasico."

    vet_response = {
        "concept_decisions": [
            {
                "name": "Condicionamiento clasico",
                "status": "repair",
                "review_notes": ["casing"],
                "repaired_text": "Condicionamiento Clásico",
            }
        ],
        "claim_decisions": [],
        "relation_decisions": [],
    }
    provider.client = _CapturedClient(vet_response)
    result = asyncio.run(provider.vet_extraction(extraction=extraction, text=text, language="es"))

    assert len(result.concepts) == 1
    assert result.concepts[0].canonical_name == "Condicionamiento Clásico"


def test_fit_embedding_dimensions_reduces_to_target_size():
    fitted = fit_embedding_dimensions([float(index) for index in range(12)], 4)

    assert len(fitted) == 4
    assert fitted == [1.0, 4.0, 7.0, 10.0]


def test_openai_extractor_sanitizes_without_heuristic_augmentation():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)

    provider.client = _CapturedClient(
        {
            "domain": "Programacion COBOL",
            "topics": ["Programacion COBOL"],
            "concepts": [
                {
                    "canonical_name": "Formato Decimal Empaquetado",
                    "aliases": ["decimal empaquetado"],
                    "description": "Representa numeros en formato compacto.",
                    "evidence_quotes": ["Packed Decimal Format"],
                    "confidence": 0.92,
                }
            ],
            "claims": [
                {
                    "text": "Packed Decimal Format is a numeric representation in COBOL.",
                    "confidence": 0.91,
                    "explains": ["Formato Decimal Empaquetado"],
                    "supporting_quote": "Packed Decimal Format",
                }
            ],
            "relations": [],
        }
    )

    text = (
        "Numeric Representations in COBOL\n"
        "- Packed Decimal Format\n"
        "- Single Precision Floating Point\n"
    )
    extraction = asyncio.run(provider.extract(text, "en", ["Programacion COBOL"]))

    concept_names = [item.canonical_name for item in extraction.concepts]
    assert "Formato Decimal Empaquetado" in concept_names
    assert extraction.concepts[0].evidence_quotes == ["Packed Decimal Format"]
    assert extraction.claims[0].supporting_quote == "Packed Decimal Format"


def test_openai_extractor_prompt_excludes_examples_and_identifiers():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    provider.client = _CapturedClient(
        {
            "domain": "Programacion",
            "topics": ["Programacion"],
            "concepts": [],
            "claims": [],
            "relations": [],
        }
    )

    asyncio.run(provider.extract("customer_id = 42", "es", ["Programacion"]))

    request = provider.client.requests[0]
    system_prompt = request["json"]["messages"][0]["content"]

    assert "A concept must be a teachable, reusable abstraction" in system_prompt
    assert "Do not treat concrete examples, sample cases, illustrative instances, variable names, field names, identifiers, literals" in system_prompt
    assert "If a fragment contains an example of a more general idea, extract the general concept only when it is explicitly stated in the text" in system_prompt
    assert "If a phrase could be read both as an example and as an idea, prefer the general idea and not the instance" in system_prompt


def test_openai_extractor_supports_technical_examples_as_context_only():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    provider.client = _CapturedClient(
        {
            "domain": "Bases de Datos",
            "topics": ["Bases de Datos"],
            "concepts": [
                {
                    "canonical_name": "Normalizacion",
                    "aliases": [],
                    "description": "Reduce redundancia y mejora la consistencia de datos.",
                    "evidence_quotes": ["La normalización evita duplicidad"],
                    "confidence": 0.95,
                }
            ],
            "claims": [
                {
                    "text": "La normalización evita duplicidad en tablas como users y orders.",
                    "confidence": 0.88,
                    "explains": ["Normalizacion"],
                    "supporting_quote": "La normalización evita duplicidad",
                }
            ],
            "relations": [],
        }
    )

    text = (
        "La normalización evita duplicidad. "
        "En el ejemplo aparecen tablas users y orders, el campo customer_id y el literal 'PENDING'."
    )
    extraction = asyncio.run(provider.extract(text, "es", ["Bases de Datos"]))

    concept_names = [item.canonical_name for item in extraction.concepts]
    assert concept_names == ["Normalizacion"]
    assert extraction.claims[0].supporting_quote == "La normalización evita duplicidad"


def test_openai_extractor_supports_pedagogical_examples_as_context_only():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)
    provider.client = _CapturedClient(
        {
            "domain": "Psicologia",
            "topics": ["Psicologia"],
            "concepts": [
                {
                    "canonical_name": "Condicionamiento Clasico",
                    "aliases": [],
                    "description": "Aprendizaje por asociacion entre estimulos.",
                    "evidence_quotes": ["El condicionamiento clásico asocia estímulos"],
                    "confidence": 0.96,
                }
            ],
            "claims": [
                {
                    "text": "El ejemplo del perro de Pavlov ilustra que el condicionamiento clásico asocia estímulos.",
                    "confidence": 0.9,
                    "explains": ["Condicionamiento Clasico"],
                    "supporting_quote": "El condicionamiento clásico asocia estímulos",
                }
            ],
            "relations": [],
        }
    )

    text = (
        "El condicionamiento clásico asocia estímulos. "
        "El perro de Pavlov es un ejemplo clásico usado para explicar esta idea."
    )
    extraction = asyncio.run(provider.extract(text, "es", ["Psicologia"]))

    concept_names = [item.canonical_name for item in extraction.concepts]
    assert concept_names == ["Condicionamiento Clasico"]
    assert extraction.claims[0].supporting_quote == "El condicionamiento clásico asocia estímulos"


def test_openai_extractor_fails_instead_of_falling_back_to_heuristics():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        OPENAI_API_KEY="sk-test",
        AI_PROVIDER="openai_compatible",
        embedding_dimensions=16,
    )
    provider = OpenAICompatibleProvider(settings)

    class _FailingClient:
        async def post(self, *_args, **_kwargs):
            raise ValueError("boom")

    provider.client = _FailingClient()

    with pytest.raises(ValueError, match="llm extraction failed"):
        asyncio.run(provider.extract("Texto de prueba", "es", ["General"]))


def test_settings_accept_anthropic_provider():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
    )

    assert settings.ai_provider == "anthropic"


def test_settings_reject_anthropic_adaptive_thinking_for_haiku():
    with pytest.raises(ValueError, match="ANTHROPIC_THINKING_TYPE=adaptive is not supported for Claude Haiku 4.5"):
        Settings(
            app_env="test",
            API_KEY="test-api-key",
            ARCADEDB_ROOT_PASSWORD="test-password",
            AI_PROVIDER="anthropic",
            ANTHROPIC_CHAT_MODEL="claude-haiku-4-5",
            ANTHROPIC_THINKING_TYPE="adaptive",
        )


def test_settings_require_budget_for_manual_thinking():
    with pytest.raises(ValueError, match="ANTHROPIC_THINKING_BUDGET_TOKENS is required when ANTHROPIC_THINKING_TYPE=enabled"):
        Settings(
            app_env="test",
            API_KEY="test-api-key",
            ARCADEDB_ROOT_PASSWORD="test-password",
            AI_PROVIDER="anthropic",
            ANTHROPIC_CHAT_MODEL="claude-haiku-4-5",
            ANTHROPIC_THINKING_TYPE="enabled",
        )


def test_build_ai_provider_requires_embedding_provider_for_anthropic():
    settings = Settings(
        _env_file=None,
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
    )

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER is required when AI_PROVIDER=anthropic"):
        build_ai_provider(settings)


def test_build_ai_provider_returns_anthropic_gateway_provider():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
    )

    provider = build_ai_provider(settings)

    assert isinstance(provider, AnthropicGatewayProvider)


def test_anthropic_gateway_extractor_sanitizes_response():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)
    provider.client = _GatewayCapturedClient(
        {
            "domain": "Bases de Datos",
            "topics": ["Bases de Datos"],
            "concepts": [
                {
                    "canonical_name": "Normalizacion",
                    "aliases": [],
                    "description": "Reduce redundancia.",
                    "evidence_quotes": ["La normalización evita duplicidad"],
                    "confidence": 0.95,
                }
            ],
            "claims": [
                {
                    "text": "La normalización evita duplicidad en tablas relacionadas.",
                    "confidence": 0.88,
                    "explains": ["Normalizacion"],
                    "supporting_quote": "La normalización evita duplicidad",
                }
            ],
            "relations": [],
        }
    )

    text = (
        "La normalización evita duplicidad. "
        "En el ejemplo aparecen tablas users y orders, el campo customer_id y el literal 'PENDING'."
    )
    extraction = asyncio.run(provider.extract(text, "es", ["Bases de Datos"]))

    assert [item.canonical_name for item in extraction.concepts] == ["Normalizacion"]
    request = provider.client.requests[0]
    assert request["json"]["model"] == "claude-test"
    assert request["json"]["user_payload_json"]["text"] == text


def test_anthropic_gateway_vetting_validates_response():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)
    provider.client = _GatewayCapturedClient(
        {
            "statement": "La normalizacion reduce duplicidad.",
            "supporting_quote": "La normalización evita duplicidad",
            "kind": "claim",
            "status": "approved",
            "review_notes": [],
        }
    )

    decision = asyncio.run(
        provider.vet_pedagogical_evidence(
            concept_name="Normalizacion",
            claim_text="La normalizacion reduce duplicidad.",
            supporting_quote="La normalización evita duplicidad",
            language="es",
        )
    )

    assert decision.status == "approved"
    request = provider.client.requests[0]
    assert request["json"]["temperature"] == 0.1
    assert request["json"]["user_payload_json"]["concept_name"] == "Normalizacion"


def test_anthropic_gateway_supports_sonnet_adaptive_thinking_with_effort():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-sonnet-4-6",
        ANTHROPIC_THINKING_TYPE="adaptive",
        ANTHROPIC_EFFORT="medium",
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)
    provider.client = _GatewayCapturedClient(
        {
            "domain": "Programacion",
            "topics": ["Programacion"],
            "concepts": [],
            "claims": [],
            "relations": [],
        }
    )

    asyncio.run(provider.extract("customer_id = 42", "es", ["Programacion"]))

    request = provider.client.requests[0]["json"]
    assert request["model"] == "claude-sonnet-4-6"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "medium"}


def test_anthropic_gateway_supports_haiku_extended_thinking_budget():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-haiku-4-5",
        ANTHROPIC_THINKING_TYPE="enabled",
        ANTHROPIC_THINKING_BUDGET_TOKENS=8000,
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)
    provider.client = _GatewayCapturedClient(
        {
            "domain": "Programacion",
            "topics": ["Programacion"],
            "concepts": [],
            "claims": [],
            "relations": [],
        }
    )

    asyncio.run(provider.extract("customer_id = 42", "es", ["Programacion"]))

    request = provider.client.requests[0]["json"]
    assert request["model"] == "claude-haiku-4-5"
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "output_config" not in request


def test_anthropic_gateway_extract_reports_invalid_gateway_payload():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)

    class _InvalidGatewayResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content_json": ["not-an-object"]}

    class _InvalidGatewayClient:
        async def post(self, *_args, **_kwargs):
            return _InvalidGatewayResponse()

    provider.client = _InvalidGatewayClient()

    with pytest.raises(ValueError, match="llm extraction failed: gateway content_json must be a JSON object"):
        asyncio.run(provider.extract("Texto de prueba", "es", ["General"]))


def test_anthropic_gateway_extract_reports_http_errors():
    settings = Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AI_PROVIDER="anthropic",
        EMBEDDING_PROVIDER="stub",
        ANTHROPIC_GATEWAY_BEARER_TOKEN="gateway-secret",
        ANTHROPIC_CHAT_MODEL="claude-test",
        embedding_dimensions=16,
    )
    provider = AnthropicGatewayProvider(settings)

    class _FailingGatewayClient:
        async def post(self, *_args, **_kwargs):
            raise ValueError("gateway boom")

    provider.client = _FailingGatewayClient()

    with pytest.raises(ValueError, match="llm extraction failed: gateway boom"):
        asyncio.run(provider.extract("Texto de prueba", "es", ["General"]))
