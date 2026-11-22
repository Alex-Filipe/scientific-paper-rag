"""PDF parsing with Docling (optional extra `docling`): layout, reading order, sections, tables.

Docling runs a layout model and TableFormer over each page, so the text loses the artifacts of
`pypdf` (`T able`, glyph names, axis numbers of the figures) and gains section headings. The
conversion is split in two steps so the page logic can be tested without the models:
`blocks_from_docling` reads a `DoclingDocument`, and `pages_from_blocks` builds one LangChain
document per page and section, so chunks never cross a page and carry their section.
"""

import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

HEADING_LABELS = frozenset({"title", "section_header"})
SKIPPED_LABELS = frozenset(
    {"picture", "chart", "page_header", "page_footer", "document_index", "empty_value"}
)
REFERENCE_SECTIONS = ("references", "bibliography", "referências", "referencias")
SNAP_WINDOW = 20
# Docling drops the hyphen of every word broken across lines, so "34-\nlayer" becomes
# "34layer"; a number glued to a 4+ letter word is put back as "34-layer".
_NUMBER_GLUED_TO_WORD = re.compile(r"(?<=\d)(?=[a-z]{4,})")


@dataclass(frozen=True, slots=True)
class Block:
    """One piece of text in reading order: a paragraph, heading, table, caption or list item."""

    page: int
    """1-based PDF page."""
    text: str
    heading: bool = False


@dataclass(frozen=True, slots=True)
class Span:
    """Where part of an item lies: Docling's provenance (page and character span)."""

    page: int
    start: int
    end: int


def is_reference_section(section: str) -> bool:
    """True for the bibliography heading ("References", "7. Bibliography", "Referências")."""
    words = re.sub(r"^[\d.\s]+", "", section).strip().casefold()
    return words.startswith(REFERENCE_SECTIONS)


def clean_docling_text(text: str) -> str:
    """NFKC (ligatures), single spaces and the hyphen Docling drops after a number."""
    collapsed = " ".join(unicodedata.normalize("NFKC", text).split())
    return _NUMBER_GLUED_TO_WORD.sub("-", collapsed)


def table_text(rows: Iterable[Sequence[str]]) -> str:
    """One line per row, cells separated by spaces (`ResNet-152 21.43 5.71`).

    Docling repeats a spanned cell in every column it covers; repeats are dropped, so
    `Maxout [9] | Maxout [9] | 9.38` becomes `Maxout [9] 9.38`.
    """
    lines = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            value = clean_docling_text(cell)
            if value and (not cells or cells[-1] != value):
                cells.append(value)
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)


def split_by_page(text: str, spans: Sequence[Span]) -> list[tuple[int, str]]:
    """Cut an item that Docling merged across pages at the page change.

    The character spans are offsets in the text before Docling removed line-break hyphens, so
    they can be a few characters off; the cut moves to the nearest space to keep words whole.
    """
    if not spans:
        return []
    pieces: list[tuple[int, str]] = []
    start, page = 0, spans[0].page
    for previous, current in zip(spans, spans[1:], strict=False):
        if current.page == previous.page:
            continue
        cut = max(start, _snap_to_space(text, current.start))
        pieces.append((page, text[start:cut]))
        start, page = cut, current.page
    pieces.append((page, text[start:]))
    return [(number, piece.strip()) for number, piece in pieces if piece.strip()]


def _snap_to_space(text: str, position: int) -> int:
    position = min(max(position, 0), len(text))
    for distance in range(SNAP_WINDOW + 1):
        for candidate in (position + distance, position - distance):
            if 0 <= candidate < len(text) and text[candidate].isspace():
                return candidate
    return position


@dataclass(slots=True)
class _Page:
    parts: list[str]
    sections: list[tuple[int, str]]
    length: int = 0


def pages_from_blocks(blocks: Iterable[Block], source: str) -> list[Document]:
    """One document per page, paragraphs separated by blank lines, in reading order.

    `page` is 0-based like the `pypdf` loader. `sections` lists `(offset, heading)` pairs: where
    each section starts in the page text, beginning with the section carried over from the
    previous page. The splitter later labels each chunk with the section covering most of it.
    A document per section instead would cut chunks at every heading; on the ResNet paper those
    fragments lowered the semantic Recall@10 from 21 to 17 of 21 questions.
    """
    pages: dict[int, _Page] = {}
    section = ""
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if block.heading:
            section = text
        page = pages.setdefault(block.page, _Page(parts=[], sections=[]))
        offset = page.length + 2 if page.parts else 0
        if not page.sections or page.sections[-1][1] != section:
            page.sections.append((offset, section))
        page.parts.append(text)
        page.length = offset + len(text)
    return [
        Document(
            page_content="\n\n".join(page.parts),
            metadata={
                "source": source,
                "page": number - 1,
                "sections": tuple(page.sections),
            },
        )
        for number, page in sorted(pages.items())
    ]


def blocks_from_docling(document: Any) -> Iterator[Block]:
    """Read a `DoclingDocument` body in reading order.

    Pictures (and the axis labels inside them), page headers and footers are skipped; captions
    and footnotes are kept. Formulas without recognized LaTeX fall back to the PDF text, and
    list items keep their marker, so the bibliography still reads `[16] S. Ioffe...`.
    """
    for item, _level in document.iterate_items(with_groups=False):
        label = str(getattr(item.label, "value", item.label))
        provenance = list(getattr(item, "prov", None) or [])
        if label in SKIPPED_LABELS or not provenance:
            continue
        if label == "table":
            rows = [[str(cell.text) for cell in row] for row in item.data.grid]
            yield Block(page=int(provenance[0].page_no), text=table_text(rows))
            continue
        text = str(getattr(item, "text", "") or getattr(item, "orig", "") or "")
        spans = [
            Span(page=int(prov.page_no), start=int(prov.charspan[0]), end=int(prov.charspan[1]))
            for prov in provenance
        ]
        marker = str(getattr(item, "marker", "") or "")
        for index, (page, piece) in enumerate(split_by_page(text, spans)):
            cleaned = clean_docling_text(piece)
            if label == "list_item" and index == 0 and any(char.isalnum() for char in marker):
                cleaned = f"{marker} {cleaned}"
            yield Block(page=page, text=cleaned, heading=label in HEADING_LABELS)


@cache
def _converter() -> Any:
    """Build the converter once per process: loading the layout and table models takes seconds."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "The docling parser needs the optional extra: pip install -e '.[docling]'"
        ) from error

    # Born-digital papers carry their text; OCR would only add time and its own errors.
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def load_pdf_with_docling(path: Path, source: str) -> list[Document]:  # pragma: no cover
    """Convert a PDF with Docling (downloads the models on first use) into page documents."""
    result = _converter().convert(path)
    return pages_from_blocks(blocks_from_docling(result.document), source)
