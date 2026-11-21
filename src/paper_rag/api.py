import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paper_rag.documents import SUPPORTED_SUFFIXES
from paper_rag.service import RagService, build_service


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = "api"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=20)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service: RagService | None = None) -> FastAPI:
    rag = service or build_service()
    application = FastAPI(title="Scientific Paper RAG", version="0.2.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/documents", status_code=201)
    def ingest_document(request: DocumentRequest) -> dict[str, int]:
        try:
            count = rag.ingest_text(title=request.title, text=request.text, source=request.source)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"chunks_created": count}

    @application.post("/documents/upload", status_code=201)
    def upload_document(file: UploadFile) -> dict[str, object]:
        filename = Path(file.filename or "").name
        if Path(filename).suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail="Send a .pdf, .txt or .md file")
        with TemporaryDirectory(prefix="paper-rag-upload-") as directory:
            path = Path(directory) / filename
            path.write_bytes(file.file.read())
            try:
                # The temporary path changes on every upload; the file name is the stable source,
                # so uploading the same file again replaces it instead of duplicating it.
                count = rag.ingest_file(path, title=path.stem, source=filename)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        return {"filename": filename, "chunks_created": count}

    @application.post("/questions")
    def ask_question(request: QuestionRequest) -> dict[str, object]:
        try:
            answer = rag.ask(request.question, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (ConnectionError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return answer.to_dict()

    @application.post("/questions/stream")
    def stream_question(request: QuestionRequest) -> StreamingResponse:
        """Server-Sent Events: `step`, `token`, then `answer` (or `error`)."""
        try:
            events = rag.stream(request.question, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        def encode(stream: Iterator[dict[str, Any]]) -> Iterator[str]:
            try:
                for event in stream:
                    yield _sse(event["event"], event["data"])
            except (ConnectionError, RuntimeError) as error:
                yield _sse("error", {"detail": str(error)})

        return StreamingResponse(encode(events), media_type="text/event-stream")

    return application


app = create_app()
