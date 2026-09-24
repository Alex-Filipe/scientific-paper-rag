import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from paper_rag.domain.models import Chunk, RetrievedChunk
from paper_rag.infrastructure.embeddings import TOKEN_PATTERN

LEXICAL_STOP_WORDS = frozenset(
    {
        "a",
        "as",
        "ao",
        "aos",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "para",
        "por",
        "que",
        "qual",
        "quais",
        "se",
        "um",
        "uma",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "how",
        "what",
        "which",
        "when",
        "where",
        "why",
    }
)


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
            fts_exists = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
                ).fetchone()
                is not None
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_title,
                    source,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
            if not fts_exists:
                stored_chunks = connection.execute(
                    "SELECT id, document_title, source, text FROM chunks"
                ).fetchall()
                indexed_chunks = {
                    stored_id.rsplit("::", maxsplit=1)[0]: (
                        stored_id.rsplit("::", maxsplit=1)[0],
                        title,
                        source,
                        text,
                    )
                    for stored_id, title, source, text in stored_chunks
                }
                connection.executemany(
                    """
                    INSERT INTO chunks_fts (chunk_id, document_title, source, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    list(indexed_chunks.values()),
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
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(chunk.id,) for chunk in chunks],
            )
            connection.executemany(
                """
                INSERT INTO chunks_fts (chunk_id, document_title, source, text)
                VALUES (?, ?, ?, ?)
                """,
                [(chunk.id, chunk.document_title, chunk.source, chunk.text) for chunk in chunks],
            )
            connection.commit()

    def search_lexical(self, query: str, top_k: int) -> list[Chunk]:
        if top_k <= 0:
            return []

        terms = dict.fromkeys(
            token
            for token in TOKEN_PATTERN.findall(query.casefold())
            if token not in LEXICAL_STOP_WORDS
        )
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, c.document_title, c.source, c.position, c.text
                FROM chunks_fts
                JOIN chunks AS c
                  ON c.id = chunks_fts.chunk_id || '::' || ?
                WHERE chunks_fts MATCH ? AND c.embedding_space = ?
                ORDER BY bm25(chunks_fts, 0.0, 2.0, 1.0, 1.0), c.position
                LIMIT ?
                """,
                (self._embedding_space, match_query, self._embedding_space, top_k),
            ).fetchall()

        return [
            Chunk(
                id=row[0].removesuffix(f"::{self._embedding_space}"),
                document_id=row[1],
                document_title=row[2],
                source=row[3],
                position=row[4],
                text=row[5],
            )
            for row in rows
        ]

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
