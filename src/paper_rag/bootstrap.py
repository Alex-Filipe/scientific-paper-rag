from dataclasses import dataclass

from paper_rag.application.ask import AskQuestion
from paper_rag.application.chunking import WordChunker
from paper_rag.application.ingest import IngestDocument
from paper_rag.infrastructure.embeddings import HashingEmbedder
from paper_rag.infrastructure.generation import ExtractiveAnswerGenerator
from paper_rag.infrastructure.sqlite_repository import SQLiteChunkRepository
from paper_rag.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    ingest_document: IngestDocument
    ask_question: AskQuestion


def build_container(settings: Settings | None = None) -> Container:
    resolved = settings or Settings.from_environment()
    embedder = HashingEmbedder()
    repository = SQLiteChunkRepository(resolved.database_path)

    return Container(
        ingest_document=IngestDocument(
            chunker=WordChunker(resolved.chunk_size, resolved.chunk_overlap),
            embedder=embedder,
            repository=repository,
        ),
        ask_question=AskQuestion(
            embedder=embedder,
            repository=repository,
            generator=ExtractiveAnswerGenerator(),
            default_top_k=resolved.top_k,
        ),
    )
