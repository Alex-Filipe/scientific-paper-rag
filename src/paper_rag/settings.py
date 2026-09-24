import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = Path(".data/rag.db")
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k: int = 4
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: Path = Path(".data/models")
    retrieval_mode: str = "hybrid"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("RAG_DB_PATH", ".data/rag.db")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "220")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "40")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "hashing"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            model_cache_dir=Path(os.getenv("RAG_MODEL_CACHE_DIR", ".data/models")),
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "hybrid"),
        )
