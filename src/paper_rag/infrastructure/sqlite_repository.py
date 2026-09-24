import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from paper_rag.domain.models import Chunk, RetrievedChunk


class SQLiteChunkRepository:
    """Persistent local baseline. Similarity is computed in memory for small corpora."""

    def __init__(self, database_path: Path, embedding_space: str = "hashing-v1") -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._embedding_space = embedding_space
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_space TEXT NOT NULL DEFAULT 'hashing-v1'
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()}
            if "embedding_space" not in columns:
                connection.execute(
                    "ALTER TABLE chunks ADD COLUMN embedding_space "
                    "TEXT NOT NULL DEFAULT 'hashing-v1'"
                )
                connection.execute(
                    "UPDATE chunks SET id = id || '::hashing-v1' WHERE id NOT LIKE '%::hashing-v1'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_space ON chunks (embedding_space)"
            )
            connection.commit()

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")

        rows = [
            (
                f"{chunk.id}::{self._embedding_space}",
                chunk.document_id,
                chunk.document_title,
                chunk.source,
                chunk.position,
                chunk.text,
                json.dumps(list(embedding)),
                self._embedding_space,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with closing(self._connect()) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (id, document_id, document_title, source, position, text, embedding,
                     embedding_space)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, document_title, source, position, text, embedding
                FROM chunks
                WHERE embedding_space = ?
                """,
                (self._embedding_space,),
            ).fetchall()

        results = []
        for row in rows:
            stored_embedding = json.loads(row[6])
            chunk = Chunk(
                id=row[0].removesuffix(f"::{self._embedding_space}"),
                document_id=row[1],
                document_title=row[2],
                source=row[3],
                position=row[4],
                text=row[5],
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=self._cosine_similarity(query_embedding, stored_embedding),
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions must match")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        denominator = left_norm * right_norm
        return numerator / denominator if denominator else 0.0
