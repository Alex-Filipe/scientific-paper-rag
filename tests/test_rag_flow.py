from pathlib import Path

from paper_rag.bootstrap import build_container
from paper_rag.settings import Settings


def test_ingest_and_retrieve_relevant_evidence(tmp_path: Path) -> None:
    container = build_container(
        Settings(
            database_path=tmp_path / "test.db",
            chunk_size=8,
            chunk_overlap=2,
            top_k=2,
        )
    )
    container.ingest_document.execute(
        title="Retrieval-Augmented Generation",
        text=(
            "RAG combines document retrieval with language generation. "
            "Embeddings support semantic retrieval from a document collection."
        ),
        source="paper.md",
    )

    answer = container.ask_question.execute("How does retrieval use embeddings?")

    assert answer.citations
    assert answer.citations[0].title == "Retrieval-Augmented Generation"
    assert "embeddings" in answer.text.lower()


def test_question_without_documents_has_no_citations(tmp_path: Path) -> None:
    container = build_container(Settings(database_path=tmp_path / "empty.db"))

    answer = container.ask_question.execute("What is RAG?")

    assert not answer.citations
    assert "não encontrei" in answer.text.lower()
