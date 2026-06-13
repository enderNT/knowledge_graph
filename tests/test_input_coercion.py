from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AdaptiveSessionConstraints,
    AdaptiveSessionStartRequest,
    ApplyPrereqReliefRequest,
    UpdateSRFromBlockRequest,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hybrid", "hybrid"),
        ("  AUTO  ", "auto"),
        ("recovery", "recovery"),
    ],
)
def test_study_mode_is_coerced(raw: str, expected: str):
    request = AdaptiveSessionStartRequest(user_id="u1", query="memoria", study_mode=raw)
    assert request.study_mode == expected


def test_study_mode_invalid_value_raises_clear_error():
    with pytest.raises(ValidationError) as exc_info:
        AdaptiveSessionStartRequest(user_id="u1", query="memoria", study_mode="mixed")
    # The offending field must be named so the agent can self-correct.
    assert "study_mode" in str(exc_info.value)


def test_question_types_coerce_separators_and_case():
    constraints = AdaptiveSessionConstraints(
        allowed_question_types=["Multiple-Choice-Single", "TRUE_FALSE", "open"]
    )
    assert constraints.allowed_question_types == [
        "multiple_choice_single",
        "true_false",
        "open",
    ]


def test_preferred_max_difficulty_coerced_and_none_passes():
    assert AdaptiveSessionConstraints(preferred_max_difficulty="Advanced").preferred_max_difficulty == "advanced"
    assert AdaptiveSessionConstraints(preferred_max_difficulty=None).preferred_max_difficulty is None


@pytest.mark.parametrize(
    ("coverage_in", "coverage_out"),
    [
        (0.85, 0.85),   # already 0-1, untouched
        (85, 0.85),     # sent on 0-100 scale, rescaled
        (100, 1.0),
        (1, 1.0),       # boundary stays as-is
        (0, 0.0),
    ],
)
def test_coverage_rescaling(coverage_in, coverage_out):
    request = UpdateSRFromBlockRequest(
        user_id="u",
        concept_uid="c",
        dimension="Recall",
        block_verdict="partial_high",
        block_difficulty="Intermediate",
        coverage=coverage_in,
        precision=0.5,
    )
    assert request.coverage == pytest.approx(coverage_out)
    # enums coerced too
    assert request.dimension == "recall"
    assert request.block_difficulty == "intermediate"


def test_sr_enums_invalid_value_names_field():
    with pytest.raises(ValidationError) as exc_info:
        UpdateSRFromBlockRequest(
            user_id="u",
            concept_uid="c",
            dimension="memoria",
            block_verdict="correct",
            block_difficulty="easy",
        )
    text = str(exc_info.value)
    assert "dimension" in text or "block_difficulty" in text


def test_prereq_relief_dimension_coerced():
    request = ApplyPrereqReliefRequest(
        user_id="u",
        source_concept_uid="c",
        source_dimension="Application",
        quality_q=3,
    )
    assert request.source_dimension == "application"
