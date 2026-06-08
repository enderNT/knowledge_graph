from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from app.config import Settings
from app.pedagogical_assessment import PedagogicalAssessmentService
from app.schemas import (
    AgentToolDebug,
    EvaluateAnswerResponse,
    ExplainTopicResponse,
    GenerateQuizResponse,
    TutorContextResponse,
)


class AgentContentGenerator(ABC):
    mode: Literal["structured_llm", "stub"]

    @abstractmethod
    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        tutor_context: TutorContextResponse,
    ) -> ExplainTopicResponse:
        raise NotImplementedError

    @abstractmethod
    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        tutor_context: TutorContextResponse,
    ) -> GenerateQuizResponse:
        raise NotImplementedError

    @abstractmethod
    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None,
        expected_answer: str | None,
        tutor_context: TutorContextResponse,
    ) -> EvaluateAnswerResponse:
        raise NotImplementedError


class SharedPedagogicalAgentContentGenerator(AgentContentGenerator):
    mode: Literal["structured_llm", "stub"] = "stub"

    def __init__(self) -> None:
        self._assessment = PedagogicalAssessmentService()

    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        tutor_context: TutorContextResponse,
    ) -> ExplainTopicResponse:
        return self._assessment.build_explanation(
            query=query,
            domain_hint=domain_hint,
            audience=audience,
            focus=focus,
            include_examples=include_examples,
            tutor_context=tutor_context,
            debug=_build_debug(tutor_context, self.mode),
        )

    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        tutor_context: TutorContextResponse,
    ) -> GenerateQuizResponse:
        return self._assessment.build_quiz(
            query=query,
            domain_hint=domain_hint,
            difficulty=difficulty,
            question_count=question_count,
            question_type=question_type,
            tutor_context=tutor_context,
            debug=_build_debug(tutor_context, self.mode),
        )

    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None,
        expected_answer: str | None,
        tutor_context: TutorContextResponse,
    ) -> EvaluateAnswerResponse:
        return self._assessment.evaluate_answer(
            query=query,
            question=question,
            learner_answer=learner_answer,
            expected_answer=expected_answer,
            tutor_context=tutor_context,
            debug=_build_debug(tutor_context, self.mode),
        )


def build_agent_content_generator(settings: Settings) -> AgentContentGenerator:
    _ = settings
    return SharedPedagogicalAgentContentGenerator()


def _build_debug(
    tutor_context: TutorContextResponse,
    generation_mode: Literal["structured_llm", "stub", "skipped"],
) -> AgentToolDebug:
    return AgentToolDebug(
        retrieval_status="ok" if tutor_context.status == "ok" else "no_match",
        used_neighborhood=False,
        generation_mode=generation_mode,
        source_concept_uids=[item.uid for item in tutor_context.concepts],
        source_claim_uids=[item.uid for item in tutor_context.claims],
    )
