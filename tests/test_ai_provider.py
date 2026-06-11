from __future__ import annotations

import asyncio
import json

import pytest

from app.ai_provider import AnthropicGatewayProvider, OpenAICompatibleProvider, StubAIProvider, build_ai_provider
from app.config import Settings
from app.utils import fit_embedding_dimensions


class _CapturedResponse:
    def __init__(self, content: dict[str, object]) -> None:
        self._content = content

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


def test_stub_extractor_prefers_named_entities_and_structured_sections(settings):
    provider = StubAIProvider(settings)
    text = """
    La cultura madre mesoamericana fue la olmeca.

    Periodos aproximados:

    Periodo\tFechas aprox.
    Preclásico temprano\t2500–1200 a. C.
    Preclásico medio\t1200–400 a. C.
    Preclásico tardío\t400 a. C.–200 d. C.

    La cultura olmeca se desarrolló sobre todo durante el Preclásico medio, aprox. 1200–400 a. C.

    Zonas que habitó:
    Principalmente la costa del Golfo de México, en el sur de Veracruz y el occidente de Tabasco.

    Centros importantes:
    San Lorenzo, La Venta y Tres Zapotes.
    """.strip()

    extraction = asyncio.run(provider.extract(text, "es", ["Mesoamérica", "Historia"]))
    concept_names = {item.canonical_name for item in extraction.concepts}
    relation_tuples = {(item.from_name, item.relation, item.to_name) for item in extraction.relations}

    assert {
        "Cultura Olmeca",
        "Preclásico Temprano",
        "Preclásico Medio",
        "Preclásico Tardío",
        "Golfo de México",
        "Veracruz",
        "Tabasco",
        "San Lorenzo",
        "La Venta",
        "Tres Zapotes",
    }.issubset(concept_names)
    assert "Historia" not in concept_names
    assert "Periodos Aproximados" not in concept_names
    assert "Todo Durante" not in concept_names
    assert "Preclásico Temprano a C" not in concept_names
    assert "Preclásico Medio a C" not in concept_names
    assert ("Cultura Olmeca", "RELATED_TO", "Preclásico Medio") in relation_tuples
    assert ("San Lorenzo", "PART_OF", "Cultura Olmeca") in relation_tuples


def test_stub_extractor_keeps_explicit_contrast_relations(settings):
    provider = StubAIProvider(settings)
    text = "El condicionamiento clásico contrasta con el condicionamiento operante."

    extraction = asyncio.run(provider.extract(text, "es", ["Psicología"]))
    concept_names = [item.canonical_name for item in extraction.concepts]

    assert "Condicionamiento Clásico" in concept_names
    assert "Condicionamiento Operante" in concept_names
    assert extraction.relations
    assert extraction.relations[0].relation == "CONTRASTS_WITH"
    assert extraction.relations[0].from_name == "Condicionamiento Clásico"
    assert extraction.relations[0].to_name == "Condicionamiento Operante"


def test_stub_extractor_trims_predicate_suffixes_from_concepts(settings):
    provider = StubAIProvider(settings)
    text = (
        "Hardware y software son componentes fundamentales. "
        "Redes y conectividad permiten comunicar dispositivos."
    )

    extraction = asyncio.run(provider.extract(text, "es", ["Tecnología"]))
    concept_names = {item.canonical_name for item in extraction.concepts}

    assert {"Hardware", "Software", "Redes", "Conectividad"}.issubset(concept_names)
    assert all(" Son " not in f" {name} " for name in concept_names)
    assert all(" Permiten " not in f" {name} " for name in concept_names)


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


def test_build_ai_provider_requires_embedding_provider_for_anthropic():
    settings = Settings(
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
