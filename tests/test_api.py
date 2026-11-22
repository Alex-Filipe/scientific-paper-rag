import json
from collections.abc import Callable

from conftest import ServiceFactory, make_pdf
from fastapi.testclient import TestClient

from paper_rag.api import create_app


def make_client(make_service: ServiceFactory) -> TestClient:
    return TestClient(create_app(make_service()))


def test_ingest_and_ask_return_structured_answer(make_service: ServiceFactory) -> None:
    client = make_client(make_service)

    created = client.post(
        "/documents",
        json={"title": "RAG", "text": "RAG combines retrieval with generation.", "source": "rag"},
    )
    response = client.post("/questions", json={"question": "What does RAG combine?"})

    assert created.status_code == 201
    assert created.json() == {"chunks_created": 1}
    body = response.json()
    assert body["status"] == "extractive"
    assert body["citations"][0]["source"] == "rag"
    assert {"index", "page", "excerpt", "rank", "cited", "text"} <= set(body["citations"][0])
    assert body["citations"][0]["text"] == "RAG combines retrieval with generation."
    assert [step["node"] for step in body["steps"]] == ["retrieve", "grade", "generate"]


def test_upload_accepts_text_files(make_service: ServiceFactory) -> None:
    client = make_client(make_service)

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.md", b"# Notes\n\nChunking keeps context.", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json() == {"filename": "notes.md", "chunks_created": 1}


def test_upload_pdf_twice_replaces_it_and_cites_pages(make_service: ServiceFactory) -> None:
    service = make_service()
    client = TestClient(create_app(service))
    pdf = make_pdf([["Deep residual networks."], ["The 152-layer ResNet reaches 4.49% error."]])

    for _ in range(2):
        response = client.post(
            "/documents/upload", files={"file": ("resnet.pdf", pdf, "application/pdf")}
        )
        assert response.status_code == 201
        assert response.json() == {"filename": "resnet.pdf", "chunks_created": 2}
    body = client.post("/questions", json={"question": "152-layer error", "top_k": 1}).json()

    assert service.search.index.count() == 2
    citation = body["citations"][0]
    assert (citation["source"], citation["title"], citation["page"]) == ("resnet.pdf", "resnet", 2)


def test_upload_rejects_unsupported_files(make_service: ServiceFactory) -> None:
    response = make_client(make_service).post(
        "/documents/upload", files={"file": ("paper.docx", b"x", "application/octet-stream")}
    )

    assert response.status_code == 400


def test_stream_sends_server_sent_events(make_service: ServiceFactory) -> None:
    client = make_client(make_service)
    client.post("/documents", json={"title": "RAG", "text": "RAG uses retrieval.", "source": "r"})

    with client.stream("POST", "/questions/stream", json={"question": "retrieval?"}) as response:
        body = "".join(response.iter_text())

    assert response.headers["content-type"].startswith("text/event-stream")
    events = [block for block in body.split("\n\n") if block]
    assert events[0].startswith("event: step")
    final = events[-1].splitlines()
    assert final[0] == "event: answer"
    assert json.loads(final[1].removeprefix("data: "))["status"] == "extractive"


def test_documents_lists_each_indexed_document(make_service: ServiceFactory) -> None:
    client = make_client(make_service)
    pdf = make_pdf([["Deep residual networks."], ["The 152-layer ResNet reaches 4.49% error."]])
    client.post("/documents/upload", files={"file": ("resnet.pdf", pdf, "application/pdf")})
    client.post("/documents", json={"title": "Attention", "text": "Transformers.", "source": "a"})

    response = client.get("/documents")

    assert response.status_code == 200
    documents = response.json()
    assert [(doc["title"], doc["source"], doc["chunks"], doc["pages"]) for doc in documents] == [
        ("Attention", "a", 1, None),
        ("resnet", "resnet.pdf", 2, 2),
    ]
    assert all(len(doc["document_id"]) == 16 for doc in documents)


def test_config_describes_the_service_settings(make_service: ServiceFactory) -> None:
    extractive = make_client(make_service).get("/config").json()
    ollama = TestClient(
        create_app(make_service(generation_backend="ollama", grader_model="qwen2.5:7b-instruct"))
    ).get("/config")

    assert extractive["generation_backend"] == "extractive"
    assert (extractive["model"], extractive["grader_model"]) == (None, None)
    assert (extractive["top_k"], extractive["grade_documents"]) == (4, False)
    assert extractive["validate_numbers"] is True
    assert ollama.json()["model"] == "qwen2.5:1.5b-instruct"
    assert ollama.json()["grader_model"] == "qwen2.5:7b-instruct"


def test_question_options_reach_the_graph(
    make_service: ServiceFactory, with_fake_llm: Callable[..., object]
) -> None:
    service = make_service()
    service.ingest_text("RAG", "RAG combines retrieval with generation.", "rag")
    with_fake_llm(service, ["RAG recupera 42 documentos [1]."])
    client = TestClient(create_app(service))

    body = client.post(
        "/questions",
        json={"question": "O que é RAG?", "grade_documents": False, "validate_numbers": False},
    ).json()

    assert body["status"] == "answered"
    assert body["steps"][1]["detail"].startswith("Avaliação pulada")


def test_invalid_question_is_rejected(make_service: ServiceFactory) -> None:
    client = make_client(make_service)

    assert client.post("/questions", json={"question": "   "}).status_code == 400
    assert client.post("/questions/stream", json={"question": "   "}).status_code == 400
