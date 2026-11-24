"""Docling parser: page and section logic runs on fake Docling items, so no models are needed.

The real conversion is opt-in (`PAPER_RAG_DOCLING_TESTS=1`, with the `docling` extra installed):
it loads the layout and table models, which the CI does not install.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import ServiceFactory, make_pdf
from langchain_core.documents import Document

from paper_rag import docling_parser
from paper_rag.docling_parser import (
    Block,
    Span,
    blocks_from_docling,
    clean_docling_text,
    pages_from_blocks,
    split_by_page,
    table_text,
)
from paper_rag.documents import (
    PaperLoader,
    chunk_sections,
    drop_sections,
    is_reference_section,
    make_splitter,
    split_into_chunks,
)
from paper_rag.settings import Settings


def item(
    label: str,
    text: str = "",
    *,
    pages: tuple[tuple[int, int, int], ...] | None = None,
    orig: str = "",
    marker: str = "",
    grid: list[list[str]] | None = None,
) -> SimpleNamespace:
    spans = pages if pages is not None else ((1, 0, len(text)),)
    return SimpleNamespace(
        label=SimpleNamespace(value=label),
        text=text,
        orig=orig or text,
        marker=marker,
        prov=[SimpleNamespace(page_no=page, charspan=(start, end)) for page, start, end in spans],
        data=SimpleNamespace(grid=[[SimpleNamespace(text=cell) for cell in row] for row in grid])
        if grid is not None
        else None,
    )


class FakeDoclingDocument:
    def __init__(self, *items: SimpleNamespace) -> None:
        self.items = items

    def iterate_items(self, with_groups: bool = True) -> Iterator[tuple[Any, int]]:
        assert with_groups is False
        for entry in self.items:
            yield entry, 1


def test_table_rows_become_lines_without_repeated_spanned_cells() -> None:
    rows = [
        ["method", "method", "error (%)"],
        ["Maxout [9]", "Maxout [9]", "9.38"],
        ["ResNet", "110", " 6.43 "],
        ["", "", ""],
    ]

    assert table_text(rows) == "method error (%)\nMaxout [9] 9.38\nResNet 110 6.43"


def test_docling_text_gets_ligatures_spaces_and_number_hyphens_back() -> None:
    assert clean_docling_text("Our 152layer  net ﬁts\n3×3 ﬁlters") == (
        "Our 152-layer net fits 3×3 filters"
    )
    assert clean_docling_text("50k images, 1st place, 1e4 iterations") == (
        "50k images, 1st place, 1e4 iterations"
    )


def test_an_item_merged_across_pages_is_cut_at_the_nearest_space() -> None:
    text = "In addition, highway networks have not shown gains."
    # Docling's offsets ignore the line-break hyphen it removed ("high-way"), so the page change
    # can land inside a word.
    spans = [Span(page=2, start=0, end=18), Span(page=3, start=18, end=53)]

    assert split_by_page(text, spans) == [
        (2, "In addition, highway"),
        (3, "networks have not shown gains."),
    ]


def test_spans_on_the_same_page_stay_together() -> None:
    text = "Left column text continues in the right column."
    spans = [Span(page=4, start=0, end=17), Span(page=4, start=18, end=47)]

    assert split_by_page(text, spans) == [(4, text)]
    assert split_by_page("", []) == []


def test_reference_sections_are_recognized_in_several_forms() -> None:
    assert is_reference_section("References")
    assert is_reference_section("7. Bibliography")
    assert is_reference_section("Referências")
    assert not is_reference_section("4.1. ImageNet Classification")
    assert not is_reference_section("")


def test_blocks_skip_pictures_and_furniture_and_keep_tables_markers_and_formulas() -> None:
    document = FakeDoclingDocument(
        item("page_header", "CVPR 2016"),
        item("section_header", "4.1. ImageNet Classification", pages=((5, 0, 28),)),
        item("picture", "0 10 20 30 40 50"),
        item("caption", "Table 2. Top-1 error.", pages=((5, 0, 21),)),
        item(
            "table",
            pages=((5, 0, 0),),
            grid=[["", "plain", "ResNet"], ["34 layers", "28.54", "25.03"]],
        ),
        item("formula", "", orig="y = F(x, {Wi}) + x. (1)", pages=((5, 0, 23),)),
        item(
            "text",
            "Second, compared to its plain counterpart, the 34layer ResNet wins.",
            pages=((5, 0, 41), (6, 41, 67)),
        ),
        item("section_header", "References", pages=((9, 0, 10),)),
        item(
            "list_item",
            "S. Ioffe and C. Szegedy. Batch normalization.",
            marker="[16]",
            pages=((9, 0, 45),),
        ),
        item("list_item", "first point", marker="-", pages=((9, 0, 11),)),
        item("page_footer", "774"),
        item("text", "no provenance", pages=()),
    )

    blocks = list(blocks_from_docling(document))

    assert blocks == [
        Block(5, "4.1. ImageNet Classification", heading=True),
        Block(5, "Table 2. Top-1 error."),
        Block(5, "plain ResNet\n34 layers 28.54 25.03"),
        Block(5, "y = F(x, {Wi}) + x. (1)"),
        Block(5, "Second, compared to its plain counterpart,"),
        Block(6, "the 34-layer ResNet wins."),
        Block(9, "References", heading=True),
        Block(9, "[16] S. Ioffe and C. Szegedy. Batch normalization."),
        Block(9, "first point"),
    ]


def test_pages_hold_their_blocks_and_where_each_section_starts() -> None:
    blocks = [
        Block(1, "Deep Residual Learning", heading=True),
        Block(1, "Kaiming He"),
        Block(1, "Abstract", heading=True),
        Block(1, "Deeper networks are harder to train."),
        Block(2, "We address the degradation problem."),
        Block(2, "  "),
        Block(2, "2. Related Work", heading=True),
        Block(2, "VLAD encodes residual vectors."),
    ]

    pages = pages_from_blocks(blocks, "resnet.pdf")

    assert [page.metadata["page"] for page in pages] == [0, 1]
    assert pages[0].page_content == (
        "Deep Residual Learning\n\nKaiming He\n\nAbstract\n\nDeeper networks are harder to train."
    )
    assert pages[0].metadata["sections"] == ((0, "Deep Residual Learning"), (36, "Abstract"))
    assert pages[0].page_content[36:].startswith("Abstract")
    # The section of the previous page carries over to the top of the next one.
    assert pages[1].metadata["sections"] == ((0, "Abstract"), (37, "2. Related Work"))
    assert pages[1].metadata["source"] == "resnet.pdf"


def test_text_before_the_first_heading_has_no_section() -> None:
    pages = pages_from_blocks([Block(1, "Preprint"), Block(1, "Title", heading=True)], "x.pdf")

    assert pages[0].metadata["sections"] == ((0, ""), (10, "Title"))


def test_each_chunk_gets_the_section_covering_most_of_it() -> None:
    text = "aaaa bbbb cccc dddd eeee"
    sections = [(0, "Intro"), (10, "Method")]

    labels = chunk_sections(text, ["aaaa bbbb cccc", "cccc dddd eeee", "missing"], sections)

    assert labels == ["Intro", "Method", "Method"]
    assert chunk_sections(text, ["aaaa"], []) == [""]


def test_chunks_carry_the_section_and_stay_on_their_page() -> None:
    pages = [
        Document(
            page_content="Abstract\n\nShort abstract text.\n\n1. Introduction\n\n" + "word " * 30,
            metadata={"page": 0, "sections": ((0, "Abstract"), (32, "1. Introduction"))},
        ),
        Document(page_content="Plain page without sections.", metadata={"page": 1}),
    ]

    chunks = split_into_chunks(
        pages, title="Paper", source="paper.pdf", splitter=make_splitter(20, 5)
    )

    assert [(chunk.metadata["page"], chunk.metadata.get("section")) for chunk in chunks] == [
        (1, "Abstract"),
        (1, "1. Introduction"),
        (1, "1. Introduction"),
        (2, None),
    ]
    assert "sections" not in chunks[0].metadata


def test_paper_loader_dispatches_pdfs_to_the_chosen_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf([["Residual learning."]]))
    calls: list[tuple[Path, str]] = []

    def fake_docling(pdf: Path, source: str) -> list[Document]:
        calls.append((pdf, source))
        return [Document(page_content="from docling", metadata={"source": source, "page": 0})]

    monkeypatch.setattr(docling_parser, "load_pdf_with_docling", fake_docling)

    assert PaperLoader(path, "docling").load()[0].page_content == "from docling"
    assert PaperLoader(path).load()[0].page_content == "Residual learning."
    assert calls == [(path, str(path))]
    with pytest.raises(ValueError, match="pdfminer"):
        PaperLoader(path, "pdfminer")


def test_service_uses_the_configured_parser_and_cites_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_service: ServiceFactory
) -> None:
    path = tmp_path / "resnet.pdf"
    path.write_bytes(make_pdf([["unused"]]))

    def fake_docling(pdf: Path, source: str) -> list[Document]:
        blocks = [
            Block(6, "4.1. ImageNet Classification", heading=True),
            Block(6, "ResNet-152 19.38 4.49"),
        ]
        return pages_from_blocks(blocks, source)

    monkeypatch.setattr(docling_parser, "load_pdf_with_docling", fake_docling)
    service = make_service(pdf_parser="docling")

    assert service.ingest_file(path) == 1
    answer = service.ask("ResNet-152 top-5 error")
    assert answer.citations[0].page == 6
    assert answer.citations[0].section == "4.1. ImageNet Classification"
    assert service.config().pdf_parser == "docling"


def test_unknown_parser_fails_when_the_service_starts(make_service: ServiceFactory) -> None:
    with pytest.raises(ValueError, match="pdfminer"):
        make_service(pdf_parser="pdfminer")


def test_parser_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PDF_PARSER", "docling")

    assert Settings.from_environment().pdf_parser == "docling"


@pytest.mark.skipif(
    os.getenv("PAPER_RAG_DOCLING_TESTS") != "1", reason="set PAPER_RAG_DOCLING_TESTS=1 to run"
)
def test_docling_converts_a_real_pdf_page_by_page(tmp_path: Path) -> None:  # pragma: no cover
    pytest.importorskip("docling")
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf([["Residual learning eases training."], ["Top-5 error: 4.49%."]]))

    pages = PaperLoader(path, "docling").load()

    assert [page.metadata["page"] for page in pages] == [0, 1]
    assert "Residual learning eases training." in pages[0].page_content
    assert "4.49%" in pages[1].page_content


def test_dropping_a_section_keeps_the_other_offsets_right() -> None:
    page = Document(
        page_content=(
            "5. Conclusion\n\nIt works.\n\nReferences\n\n[1] A. Author.\n\nA. Appendix\n\nMore."
        ),
        metadata={
            "page": 8,
            "sections": ((0, "5. Conclusion"), (26, "References"), (54, "A. Appendix")),
        },
    )

    kept = drop_sections(page, is_reference_section)

    assert kept.page_content == "5. Conclusion\n\nIt works.\n\nA. Appendix\n\nMore."
    assert kept.metadata["sections"] == ((0, "5. Conclusion"), (26, "A. Appendix"))
    assert kept.page_content[26:].startswith("A. Appendix")
    assert kept.metadata["page"] == 8
    assert drop_sections(page, lambda heading: False) is page


def test_bibliography_is_left_out_but_the_page_count_stays() -> None:
    blocks = [
        Block(8, "5. Conclusion", heading=True),
        Block(8, "Residual nets work."),
        Block(9, "References", heading=True),
        Block(9, "[1] Y. Bengio. Learning long-term dependencies."),
    ]
    pages = pages_from_blocks(blocks, "resnet.pdf")
    splitter = make_splitter(220, 40)

    kept = split_into_chunks(
        pages, title="ResNet", source="x", splitter=splitter, skip_references=True
    )
    full = split_into_chunks(pages, title="ResNet", source="x", splitter=splitter)

    assert [chunk.page_content for chunk in kept] == ["5. Conclusion\n\nResidual nets work."]
    assert [chunk.metadata["page_count"] for chunk in kept] == [9]
    assert [chunk.metadata["section"] for chunk in full] == ["5. Conclusion", "References"]
    with pytest.raises(ValueError, match="bibliography"):
        split_into_chunks(
            pages[1:], title="ResNet", source="x", splitter=splitter, skip_references=True
        )


def test_service_leaves_the_bibliography_out_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_service: ServiceFactory
) -> None:
    path = tmp_path / "resnet.pdf"
    path.write_bytes(make_pdf([["unused"]]))

    def fake_docling(pdf: Path, source: str) -> list[Document]:
        blocks = [
            Block(1, "Residual learning eases training."),
            Block(2, "References", heading=True),
            Block(2, "[16] S. Ioffe and C. Szegedy. Batch normalization."),
        ]
        return pages_from_blocks(blocks, source)

    monkeypatch.setattr(docling_parser, "load_pdf_with_docling", fake_docling)
    default = make_service(pdf_parser="docling")
    keeping = make_service(
        pdf_parser="docling", index_references=True, chroma_dir=tmp_path / "keep"
    )

    assert default.ingest_file(path) == 1
    assert keeping.ingest_file(path) == 2
    assert default.documents()[0].pages == 2
    assert (default.config().index_references, keeping.config().index_references) == (
        False,
        True,
    )


def test_bibliography_setting_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_INDEX_REFERENCES", "true")

    assert Settings.from_environment().index_references is True
