from dataclasses import dataclass

from paper_rag.adapters.embeddings import create_embedder
from paper_rag.adapters.generation import create_answer_generator
from paper_rag.adapters.storage import SQLiteChunkRepository
from paper_rag.core.ask import AskQuestion
from paper_rag.core.ingest import IngestDocument, WordChunker
from paper_rag.core.retrieval import RetrievalMode, RetrieveChunks
from paper_rag.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    ingest_document: IngestDocument
    retrieve_chunks: RetrieveChunks
    ask_question: AskQuestion


def build_container(settings: Settings | None = None) -> Container:
    resolved = settings or Settings.from_environment()
    configured_embedder = create_embedder(
        backend=resolved.embedding_backend,
        model_name=resolved.embedding_model,
        cache_dir=resolved.model_cache_dir,
    )
    embedder = configured_embedder.embedder
    repository = SQLiteChunkRepository(
        resolved.database_path,
        embedding_space=configured_embedder.embedding_space,
    )
    retriever = RetrieveChunks(
        embedder=embedder,
        repository=repository,
        mode=RetrievalMode(resolved.retrieval_mode),
    )

    return Container(
        ingest_document=IngestDocument(
            chunker=WordChunker(resolved.chunk_size, resolved.chunk_overlap),
            embedder=embedder,
            repository=repository,
        ),
        retrieve_chunks=retriever,
        ask_question=AskQuestion(
            retriever=retriever,
            generator=create_answer_generator(resolved.generation_backend, resolved.openai_model),
            default_top_k=resolved.top_k,
        ),
    )
