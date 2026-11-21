from pathlib import Path

import pytest
from langchain_core.documents import Document

from paper_rag.documents import load_file, make_document_id, make_splitter, split_into_chunks


def test_splitter_measures_size_and_overlap_in_words() -> None:
    chunks = make_splitter(chunk_size=4, chunk_overlap=2).split_text(
        "one two three four five six seven"
    )

    assert chunks[0] == "one two three four"
    assert chunks[1] == "three four five six"


@pytest.mark.parametrize(("size", "overlap"), [(4, 4), (4, -1), (0, 0)])
def test_splitter_rejects_invalid_sizes(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="Chunk"):
        make_splitter(size, overlap)


def test_chunks_keep_page_numbers_and_stable_ids() -> None:
    pages = [
        Document(page_content="alpha beta gamma", metadata={"page": 0, "producer": "x"}),
        Document(page_content="delta epsilon", metadata={"page": 1}),
    ]

    chunks = split_into_chunks(
        pages, title=" Paper ", source="paper.pdf", splitter=make_splitter(220, 40)
    )

    document_id = make_document_id("paper.pdf", "Paper")
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"{document_id}:0",
        f"{document_id}:1",
    ]
    assert chunks[0].metadata["document_title"] == "Paper"
    assert "producer" not in chunks[0].metadata


def test_chunks_from_plain_text_have_no_page() -> None:
    chunks = split_into_chunks(
        [Document(page_content="texto simples")],
        title="Nota",
        source="nota.md",
        splitter=make_splitter(220, 40),
    )

    assert "page" not in chunks[0].metadata


@pytest.mark.parametrize(("title", "text"), [("", "texto"), ("Título", "   ")])
def test_chunks_require_title_and_text(title: str, text: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        split_into_chunks(
            [Document(page_content=text)],
            title=title,
            source="x",
            splitter=make_splitter(220, 40),
        )


def test_load_file_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Título\n\nConteúdo com acentuação.", encoding="utf-8")

    documents = load_file(path)

    assert "acentuação" in documents[0].page_content


def test_load_file_rejects_unsupported_types(tmp_path: Path) -> None:
    path = tmp_path / "paper.docx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported"):
        load_file(path)
