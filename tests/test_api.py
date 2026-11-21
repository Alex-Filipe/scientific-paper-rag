import json

from conftest import ServiceFactory
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
    assert {"index", "page", "excerpt", "rank", "cited"} <= set(body["citations"][0])
    assert [step["node"] for step in body["steps"]] == ["retrieve", "grade", "generate"]


def test_upload_accepts_text_files(make_service: ServiceFactory) -> None:
    client = make_client(make_service)

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.md", b"# Notes\n\nChunking keeps context.", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json() == {"filename": "notes.md", "chunks_created": 1}


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


def test_invalid_question_is_rejected(make_service: ServiceFactory) -> None:
    client = make_client(make_service)

    assert client.post("/questions", json={"question": "   "}).status_code == 400
    assert client.post("/questions/stream", json={"question": "   "}).status_code == 400
