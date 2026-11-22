from collections.abc import Callable

import httpx
import pytest
from conftest import ServiceFactory, make_pdf
from fastapi.testclient import TestClient

from paper_rag.api import create_app
from paper_rag.models import AnswerStatus, Step
from paper_rag.service import RagService
from paper_rag.ui.client import (
    AnswerEvent,
    ApiClient,
    ApiError,
    ErrorEvent,
    ServerEvent,
    StepEvent,
    TokenEvent,
    parse_sse,
    to_stream_event,
)

InstallLLM = Callable[..., object]


def api(service: RagService) -> ApiClient:
    return ApiClient(TestClient(create_app(service)))


def offline(handler: Callable[[httpx.Request], httpx.Response]) -> ApiClient:
    transport = httpx.MockTransport(handler)
    return ApiClient(httpx.Client(base_url="http://api.test", transport=transport))


def test_parse_sse_follows_the_event_stream_rules() -> None:
    lines = [
        ": comment",
        "event: step",
        'data: {"node": "retrieve",',
        'data: "detail": "ok"}',
        "",
        "",
        'data:{"text": "sem espaço"}',
        "",
        "event: answer",
        'data: {"cut": true}',
    ]

    assert list(parse_sse(lines)) == [
        ServerEvent("step", {"node": "retrieve", "detail": "ok"}),
        ServerEvent("message", {"text": "sem espaço"}),
    ]


def test_parse_sse_rejects_invalid_json() -> None:
    with pytest.raises(ApiError, match="JSON inválido"):
        list(parse_sse(["event: token", "data: {oops", ""]))


def test_stream_events_are_typed() -> None:
    answer = {"answer": "A [1].", "status": "answered", "citations": [], "steps": []}

    assert to_stream_event(ServerEvent("step", {"node": "grade", "detail": "x"})) == StepEvent(
        Step("grade", "x")
    )
    assert to_stream_event(ServerEvent("token", {"text": "A"})) == TokenEvent("A")
    typed = to_stream_event(ServerEvent("answer", answer))
    assert isinstance(typed, AnswerEvent)
    assert typed.answer.status is AnswerStatus.ANSWERED
    assert to_stream_event(ServerEvent("error", {"detail": "Ollama"})) == ErrorEvent("Ollama")
    assert to_stream_event(ServerEvent("ping", {})) is None


def test_client_reads_config_documents_and_uploads(make_service: ServiceFactory) -> None:
    client = api(make_service())

    chunks = client.upload("resnet.pdf", make_pdf([["Residual."], ["4.49% top-5 error."]]))

    assert chunks == 2
    assert client.config().generation_backend == "extractive"
    [document] = client.documents()
    assert (document.title, document.source, document.pages) == ("resnet", "resnet.pdf", 2)


def test_client_streams_steps_tokens_and_the_final_answer(
    make_service: ServiceFactory, with_fake_llm: InstallLLM
) -> None:
    service = make_service()
    service.ingest_text("RAG", "RAG combines retrieval with generation.", "rag")
    with_fake_llm(service, ["RAG combina busca e geração [1]."])

    events = list(api(service).stream("O que é RAG?", top_k=2, grade_documents=False))

    assert [event.step.node for event in events if isinstance(event, StepEvent)] == [
        "retrieve",
        "grade",
        "generate",
        "validate",
    ]
    tokens = "".join(event.text for event in events if isinstance(event, TokenEvent))
    assert tokens == "RAG combina busca e geração [1]."
    final = events[-1]
    assert isinstance(final, AnswerEvent)
    assert final.answer.citations[0].cited is True


def test_client_reports_api_errors(make_service: ServiceFactory) -> None:
    client = api(make_service())

    with pytest.raises(ApiError, match="400: Question cannot be empty"):
        list(client.stream("   "))
    with pytest.raises(ApiError, match="400: Send a .pdf"):
        client.upload("paper.docx", b"x")


def test_client_reports_validation_errors(make_service: ServiceFactory) -> None:
    with pytest.raises(ApiError, match="422"):
        list(api(make_service()).stream("ok", top_k=99))


def test_client_explains_an_unreachable_api() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = offline(refuse)

    with pytest.raises(ApiError, match="Não consegui falar com a API em http://api.test"):
        client.config()
    with pytest.raises(ApiError, match="ConnectError"):
        list(client.stream("O que é RAG?"))


def test_client_yields_error_events_from_the_stream() -> None:
    body = 'event: step\ndata: {"node": "retrieve", "detail": "ok"}\n\n'
    body += 'event: error\ndata: {"detail": "Ollama is not running"}\n\n'
    client = offline(
        lambda request: httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )
    )

    events = list(client.stream("O que é RAG?"))

    assert events == [StepEvent(Step("retrieve", "ok")), ErrorEvent("Ollama is not running")]
