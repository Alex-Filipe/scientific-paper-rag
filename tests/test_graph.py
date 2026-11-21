from collections.abc import Callable, Sequence

import pytest
from conftest import ServiceFactory

from paper_rag.generation import ABSTENTION_ANSWER
from paper_rag.models import AnswerStatus
from paper_rag.service import RagService

InstallLLM = Callable[..., object]
PAPER = (
    "RAG combines document retrieval with language generation. "
    "Embeddings support semantic retrieval from a document collection."
)


@pytest.fixture
def service(make_service: ServiceFactory) -> RagService:
    rag = make_service()
    rag.ingest_text("Retrieval-Augmented Generation", PAPER, "paper.md")
    return rag


def nodes(answer_steps: Sequence[object]) -> list[str]:
    return [step.node for step in answer_steps]  # type: ignore[attr-defined]


def test_extractive_flow_lists_evidence(service: RagService) -> None:
    answer = service.ask("How does retrieval use embeddings?")

    assert answer.status is AnswerStatus.EXTRACTIVE
    assert answer.citations[0].title == "Retrieval-Augmented Generation"
    assert "embeddings" in answer.text.lower()
    assert nodes(answer.steps) == ["retrieve", "grade", "generate"]


def test_empty_corpus_abstains(make_service: ServiceFactory) -> None:
    answer = make_service().ask("What is RAG?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.text == ABSTENTION_ANSWER
    assert not answer.citations


def test_llm_answer_with_valid_citation(service: RagService, with_fake_llm: InstallLLM) -> None:
    with_fake_llm(service, ["Sim.", "RAG combina recuperação e geração [1]."])

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.text == "RAG combina recuperação e geração [1]."
    assert answer.citations[0].cited is True
    assert nodes(answer.steps) == ["retrieve", "grade", "generate", "validate"]


@pytest.mark.parametrize("reply", ["Resposta sem referência.", "Resposta [9]."])
def test_invalid_citations_fall_back_to_evidence(
    service: RagService, with_fake_llm: InstallLLM, reply: str
) -> None:
    with_fake_llm(service, ["sim", reply])

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.FALLBACK
    assert answer.text.startswith("Evidências mais relevantes encontradas:")
    assert reply not in answer.text


def test_model_abstention_is_preserved(service: RagService, with_fake_llm: InstallLLM) -> None:
    with_fake_llm(service, ["sim", ABSTENTION_ANSWER])

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.text == ABSTENTION_ANSWER


def test_weak_evidence_triggers_query_rewrite(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["Não", "semantic retrieval embeddings", "sim", "Usa embeddings [1]."])

    answer = service.ask("Como funciona?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.search_query == "semantic retrieval embeddings"
    assert nodes(answer.steps) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "validate",
    ]


def test_rewrites_are_bounded(service: RagService, with_fake_llm: InstallLLM) -> None:
    with_fake_llm(service, ["não", "outra consulta", "não"], max_rewrites=1)

    answer = service.ask("Pergunta sem resposta")

    assert answer.status is AnswerStatus.ABSTAINED
    assert nodes(answer.steps).count("rewrite") == 1
    assert nodes(answer.steps)[-1] == "abstain"


def test_unreadable_grade_keeps_all_chunks(service: RagService, with_fake_llm: InstallLLM) -> None:
    with_fake_llm(service, ["talvez", "RAG combina recuperação e geração [1]."])

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.ANSWERED
    assert "ilegível" in answer.steps[1].detail


def test_grade_judges_each_chunk(service: RagService, with_fake_llm: InstallLLM) -> None:
    service.ingest_text("Chunking", "Chunking splits papers into retrieval units.", "chunk.md")
    retrieved = service.retrieve("retrieval", 4)
    assert len(retrieved) == 2
    with_fake_llm(service, ["não", "sim", "Resposta [1]."])

    answer = service.ask("retrieval")

    assert [citation.chunk_id for citation in answer.citations] == [
        retrieved[1].metadata["chunk_id"]
    ]
    assert answer.steps[1].detail == "1 de 2 trecho(s) úteis."


def test_rewrite_without_words_keeps_original_question(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["não", "0.195", "não"], max_rewrites=1)

    answer = service.ask("Qual é o BLEU?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.search_query == "Qual é o BLEU?"
    assert "inválida" in answer.steps[2].detail


def test_stream_emits_steps_tokens_and_answer(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["sim", "Resposta [1]."])

    events = list(service.stream("O que é RAG?"))

    kinds = [event["event"] for event in events]
    assert kinds[0] == "step"
    assert "token" in kinds
    assert kinds[-1] == "answer"
    streamed = "".join(event["data"]["text"] for event in events if event["event"] == "token")
    assert streamed == "Resposta [1]."
    assert events[-1]["data"]["status"] == "answered"


def test_stream_validates_before_running(service: RagService) -> None:
    with pytest.raises(ValueError, match="empty"):
        service.stream("   ")


def test_separate_grader_handles_grade_and_rewrite(service: RagService) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from paper_rag.generation import GenerationBackend
    from paper_rag.graph import RagNodes, build_graph

    generator = FakeListChatModel(responses=["RAG combina recuperação e geração [1]."])
    grader = FakeListChatModel(responses=["não", "retrieval augmented generation", "sim"])
    service.graph = build_graph(
        RagNodes(
            search=service.search,
            llm=generator,
            backend=GenerationBackend.OLLAMA,
            max_rewrites=1,
            grader=grader,
        )
    )

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.search_query == "retrieval augmented generation"
    assert nodes(answer.steps)[-2:] == ["generate", "validate"]
    assert nodes(answer.steps)[:5] == ["retrieve", "grade", "rewrite", "retrieve", "grade"]


def test_numbers_missing_from_cited_chunk_fall_back(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["sim", "RAG recupera 42 documentos [1]."])

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.FALLBACK
    assert answer.steps[-1].node == "validate"
    assert "Números sem suporte nos trechos citados (42)" in answer.steps[-1].detail


def test_numeric_validation_can_be_disabled(service: RagService) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from paper_rag.generation import GenerationBackend
    from paper_rag.graph import RagNodes, build_graph

    service.graph = build_graph(
        RagNodes(
            search=service.search,
            llm=FakeListChatModel(responses=["RAG recupera 42 documentos [1]."]),
            backend=GenerationBackend.OLLAMA,
            grade_documents=False,
            validate_numbers=False,
        )
    )

    answer = service.ask("O que é RAG?")

    assert answer.status is AnswerStatus.ANSWERED
    assert nodes(answer.steps) == ["retrieve", "grade", "generate", "validate"]
