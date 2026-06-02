from __future__ import annotations

import asyncio

from app.ai_provider import StubAIProvider
from app.utils import fit_embedding_dimensions


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
