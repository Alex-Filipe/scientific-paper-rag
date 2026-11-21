import hashlib
from collections.abc import Sequence
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def collection_name_for(embedding_space: str) -> str:
    """Each embedding space gets its own collection, so incompatible vectors never mix."""
    return "chunks-" + hashlib.sha1(embedding_space.encode()).hexdigest()[:12]


class ChunkIndex:
    """Persistent Chroma collection holding the chunks of one embedding space."""

    def __init__(self, directory: Path, embeddings: Embeddings, embedding_space: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.embedding_space = embedding_space
        self.store = Chroma(
            collection_name=collection_name_for(embedding_space),
            embedding_function=embeddings,
            persist_directory=str(directory),
            client_settings=ChromaSettings(anonymized_telemetry=False),
            collection_metadata={"embedding_space": embedding_space},
            collection_configuration={"hnsw": {"space": "cosine"}},
        )

    def replace_document(self, document_id: str, chunks: Sequence[Document]) -> None:
        """Drop every chunk of a document before adding its new version (no orphans)."""
        self.store.delete(where={"document_id": document_id})
        if chunks:
            self.store.add_documents(
                list(chunks), ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks]
            )

    def all_chunks(self) -> list[Document]:
        stored = self.store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=dict(metadata or {}), id=chunk_id)
            for chunk_id, text, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"], strict=True
            )
        ]

    def count(self) -> int:
        return len(self.store.get(include=[])["ids"])
