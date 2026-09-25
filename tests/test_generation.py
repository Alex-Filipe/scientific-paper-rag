from dataclasses import dataclass, field

import pytest

from paper_rag.domain.models import Chunk, RetrievedChunk
from paper_rag.infrastructure.generation import (
    ABSTENTION_ANSWER,
    ExtractiveAnswerGenerator,
    OpenAIAnswerGenerator,
    create_answer_generator,
)
from paper_rag.settings import Settings


@dataclass(frozen=True, slots=True)
class FakeResponse:
    output_text: str


@dataclass(slots=True)
class FakeResponsesAPI:
    output_text: str
    calls: list[dict[str, str | int]] = field(default_factory=list)

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        max_output_tokens: int,
    ) -> FakeResponse:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input": input,
                "max_output_tokens": max_output_tokens,
            }
        )
        return FakeResponse(self.output_text)


@dataclass(frozen=True, slots=True)
class FakeOpenAIClient:
    responses: FakeResponsesAPI


def make_context(
    text: str = "RAG combina busca de documentos com geração de linguagem."
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="paper-1::chunk-0",
            document_id="paper-1",
            document_title="A Survey of RAG",
            source="survey.pdf",
            position=0,
            text=text,
        ),
        score=0.91,
    )


def test_openai_generator_synthesizes_answer_from_cited_context() -> None:
    responses = FakeResponsesAPI("RAG combina recuperação e geração [1].")
    generator = OpenAIAnswerGenerator(
        client=FakeOpenAIClient(responses=responses),
        model="gpt-6-astra",
    )

    answer = generator.generate("O que é RAG?", [make_context()])

    assert answer == "RAG combina recuperação e geração [1]."
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-6-astra"
    assert "O que é RAG?" in str(call["input"])
    assert "RAG combina busca de documentos" in str(call["input"])
    assert "instruções" in str(call["instructions"])


def test_openai_generator_abstains_without_calling_provider_for_empty_context() -> None:
    responses = FakeResponsesAPI("Esta resposta nunca será usada.")
    generator = OpenAIAnswerGenerator(
        client=FakeOpenAIClient(responses=responses),
        model="gpt-6-astra",
    )

    assert generator.generate("O que é RAG?", []) == ABSTENTION_ANSWER
    assert responses.calls == []


def test_openai_generator_preserves_model_abstention() -> None:
    generator = OpenAIAnswerGenerator(
        client=FakeOpenAIClient(responses=FakeResponsesAPI(ABSTENTION_ANSWER)),
        model="gpt-6-astra",
    )

    assert generator.generate("O que é RAG?", [make_context()]) == ABSTENTION_ANSWER


@pytest.mark.parametrize("model_answer", ["Resposta sem referência.", "Resposta [2]."])
def test_openai_generator_falls_back_when_citations_are_invalid(model_answer: str) -> None:
    generator = OpenAIAnswerGenerator(
        client=FakeOpenAIClient(responses=FakeResponsesAPI(model_answer)),
        model="gpt-6-astra",
    )

    answer = generator.generate("O que é RAG?", [make_context()])

    assert answer.startswith("Evidências mais relevantes encontradas:")
    assert "[1]" in answer
    assert model_answer not in answer


def test_extractive_backend_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    generator = create_answer_generator("extractive", "gpt-6-astra")

    assert isinstance(generator, ExtractiveAnswerGenerator)


def test_openai_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_answer_generator("openai", "gpt-6-astra")


def test_settings_load_generation_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_GENERATION_BACKEND", "openai")
    monkeypatch.setenv("RAG_OPENAI_MODEL", "example-model")

    settings = Settings.from_environment()

    assert settings.generation_backend == "openai"
    assert settings.openai_model == "example-model"
