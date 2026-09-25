import sqlite3
from contextlib import closing
from pathlib import Path

from paper_rag.adapters.storage import SQLiteChunkRepository
from paper_rag.core.contracts import Chunk
from paper_rag.core.retrieval import ReciprocalRankFusion


def make_chunk(chunk_id: str, text: str, position: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        document_title="Technical notes",
        source=f"{chunk_id}.txt",
        position=position,
        text=text,
    )


def test_sqlite_lexical_search_matches_diacritic_variants(tmp_path: Path) -> None:
    repository = SQLiteChunkRepository(tmp_path / "lexical.db")
    repository.add(
        [
            make_chunk("target", "A responsabilidade sistêmica define o limite."),
            make_chunk("other", "O procedimento descreve a manutenção preventiva.", 1),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = repository.search_lexical("responsabilidade sistemica", top_k=2)

    assert [chunk.id for chunk in results] == ["target"]


def test_sqlite_lexical_search_ignores_common_stop_words(tmp_path: Path) -> None:
    repository = SQLiteChunkRepository(tmp_path / "stop-words.db")
    repository.add([make_chunk("only-english-article", "A reranker receives candidates.")], [[1.0]])

    results = repository.search_lexical("a o de que and the", top_k=3)

    assert results == []


def test_sqlite_lexical_index_updates_when_a_chunk_is_reingested(tmp_path: Path) -> None:
    repository = SQLiteChunkRepository(tmp_path / "update.db")
    repository.add([make_chunk("same", "The solar array is offline.")], [[1.0]])
    repository.add([make_chunk("same", "The wind turbine is offline.")], [[1.0]])

    assert repository.search_lexical("solar array", top_k=3) == []
    assert [chunk.id for chunk in repository.search_lexical("wind turbine", top_k=3)] == ["same"]


def test_existing_database_is_backfilled_into_the_lexical_index(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_title TEXT NOT NULL,
                source TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunks
                (id, document_id, document_title, source, position, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-chunk",
                "legacy-document",
                "Legacy procedure",
                "legacy.txt",
                0,
                "The cobalt catalyst reduces reaction time.",
                "[1.0]",
            ),
        )
        connection.commit()

    repository = SQLiteChunkRepository(database_path)

    results = repository.search_lexical("cobalt catalyst", top_k=1)

    assert [chunk.id for chunk in results] == ["legacy-chunk"]


def test_reciprocal_rank_fusion_promotes_chunks_shared_by_both_rankers() -> None:
    first = make_chunk("first", "First passage")
    shared = make_chunk("shared", "Shared passage", 1)
    last = make_chunk("last", "Last passage", 2)

    results = ReciprocalRankFusion().fuse([[first, shared], [shared, last]], top_k=3)

    assert [result.chunk.id for result in results] == ["shared", "first", "last"]
    assert results[0].score > results[1].score
