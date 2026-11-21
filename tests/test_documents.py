from pathlib import Path

import pytest
from conftest import ServiceFactory, make_pdf
from langchain_core.documents import Document

from paper_rag.documents import (
    clean_pdf_text,
    load_file,
    make_document_id,
    make_splitter,
    split_into_chunks,
)


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


def test_load_file_reads_pdf_page_by_page(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    pages_lines = [["Residual learn-", "ing eases training."], ["Top-5 error: 4.49%."]]
    path.write_bytes(make_pdf(pages_lines))

    pages = load_file(path)
    chunks = split_into_chunks(
        pages, title="ResNet", source=str(path), splitter=make_splitter(220, 40)
    )

    assert [page.metadata["page"] for page in pages] == [0, 1]
    assert pages[0].page_content == "Residual learning eases training."
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert "4.49%" in chunks[1].page_content


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("more diﬃcult classiﬁcation", "more difficult classification"),
        ("learn-\ning residual func-  \n  tions", "learning residual functions"),
        ("ResNet-\n34 and VGG-\nnets", "ResNet-\n34 and VGG-\nnets"),
        ("self-attention", "self-attention"),
    ],
)
def test_clean_pdf_text_fixes_ligatures_and_hyphenation(raw: str, expected: str) -> None:
    assert clean_pdf_text(raw) == expected


def test_load_file_rejects_unsupported_types(tmp_path: Path) -> None:
    path = tmp_path / "paper.docx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported"):
        load_file(path)


def test_reingesting_a_file_by_another_path_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_service: ServiceFactory
) -> None:
    path = tmp_path / "paper.md"
    path.write_text("Residual learning eases the training of deep networks.", encoding="utf-8")
    service = make_service()
    monkeypatch.chdir(tmp_path)

    service.ingest_file(Path("paper.md"))
    service.ingest_file(path)

    assert service.search.index.count() == 1
    assert service.retrieve("residual", 1)[0].metadata["source"] == str(path.resolve())
