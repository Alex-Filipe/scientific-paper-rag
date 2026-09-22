from dataclasses import dataclass

from paper_rag.application.ask import AskQuestion
from paper_rag.application.chunking import WordChunker
from paper_rag.application.ingest import IngestDocument
from paper_rag.application.retrieve import RetrieveChunks
from paper_rag.infrastructure.embeddings import HashingEmbedder
from paper_rag.infrastructure.generation import ExtractiveAnswerGenerator
from paper_rag.infrastructure.sqlite_repository import SQLiteChunkRepository
from paper_rag.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    ingest_document: IngestDocument
    retrieve_chunks: RetrieveChunks
    ask_question: AskQuestion


def build_container(settings: Settings | None = None) -> Container:
    resolved = settings or Settings.from_environment()
    embedder = HashingEmbedder()
    repository = SQLiteChunkRepository(resolved.database_path)
    retriever = RetrieveChunks(embedder=embedder, repository=repository)

    return Container(
        ingest_document=IngestDocument(
            chunker=WordChunker(resolved.chunk_size, resolved.chunk_overlap),
            embedder=embedder,
            repository=repository,
        ),
        retrieve_chunks=retriever,
        ask_question=AskQuestion(
            retriever=retriever,
            generator=ExtractiveAnswerGenerator(),
            default_top_k=resolved.top_k,
        ),
    )
