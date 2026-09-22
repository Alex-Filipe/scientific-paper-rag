from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_rag.bootstrap import Container, build_container


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = "api"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=20)


def create_app(container: Container | None = None) -> FastAPI:
    services = container or build_container()
    application = FastAPI(title="Scientific Paper RAG", version="0.1.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/documents", status_code=201)
    def ingest_document(request: DocumentRequest) -> dict[str, int]:
        try:
            count = services.ingest_document.execute(
                title=request.title,
                text=request.text,
                source=request.source,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"chunks_created": count}

    @application.post("/questions")
    def ask_question(request: QuestionRequest) -> dict[str, object]:
        try:
            answer = services.ask_question.execute(request.question, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "answer": answer.text,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "title": citation.title,
                    "source": citation.source,
                    "excerpt": citation.excerpt,
                    "score": citation.score,
                }
                for citation in answer.citations
            ],
        }

    return application


app = create_app()
