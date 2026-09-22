import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = Path(".data/rag.db")
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k: int = 4

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("RAG_DB_PATH", ".data/rag.db")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "220")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "40")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
        )
