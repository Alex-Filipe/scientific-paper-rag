import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt", ".md"})
_LOWERCASE = "a-zà-öø-ÿ"
HYPHENATED_LINE_BREAK = re.compile(rf"(?<=[{_LOWERCASE}])-[ \t]*\n[ \t]*(?=[{_LOWERCASE}])")
REFERENCE_SECTIONS = ("references", "bibliography", "referências", "referencias")


class PdfParser(StrEnum):
    PYPDF = "pypdf"
    """Text layer of each page; fast and light, but keeps PDF typesetting artifacts."""
    DOCLING = "docling"
    """Layout and table models (optional extra `docling`); adds the `section` of each block."""


class PaperLoader(BaseLoader):
    """Load PDFs page by page (0-based `page` metadata) and text/Markdown as one document."""

    def __init__(self, path: Path, parser: PdfParser | str = PdfParser.PYPDF) -> None:
        self.path = path
        self.parser = PdfParser(parser)

    def lazy_load(self) -> Iterator[Document]:
        suffix = self.path.suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {suffix or 'no extension'}")
        source = str(self.path)
        if suffix != ".pdf":
            yield Document(
                page_content=self.path.read_text(encoding="utf-8"), metadata={"source": source}
            )
            return

        if self.parser is PdfParser.DOCLING:
            from paper_rag.docling_parser import load_pdf_with_docling

            yield from load_pdf_with_docling(self.path, source)
            return

        from pypdf import PdfReader

        for page_number, page in enumerate(PdfReader(self.path).pages):
            yield Document(
                page_content=clean_pdf_text(page.extract_text() or ""),
                metadata={"source": source, "page": page_number},
            )


def clean_pdf_text(text: str) -> str:
    """Undo PDF typesetting: ligatures (ﬁ -> fi) and words hyphenated across lines.

    Only lowercase-lowercase breaks are joined, so "learn-\\ning" becomes "learning"; a compound
    split at the line end ("high-\\nlevel") also loses its hyphen, an accepted trade-off.
    """
    return HYPHENATED_LINE_BREAK.sub("", unicodedata.normalize("NFKC", text))


def count_words(text: str) -> int:
    return len(text.split())


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Split by paragraphs, then lines, then words; sizes are measured in words."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be between zero and size - 1")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_words,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )


def make_document_id(source: str, title: str) -> str:
    return hashlib.sha256(f"{source}:{title}".encode()).hexdigest()[:16]


def load_file(path: Path, parser: PdfParser | str = PdfParser.PYPDF) -> list[Document]:
    """Load a file into LangChain documents; PDFs yield at least one document per page."""
    return PaperLoader(path, parser).load()


def is_reference_section(section: str) -> bool:
    """True for the bibliography heading ("References", "7. Bibliography", "Referências")."""
    words = re.sub(r"^[\d.\s]+", "", section).strip().casefold()
    return words.startswith(REFERENCE_SECTIONS)


def drop_sections(page: Document, drop: Callable[[str], bool]) -> Document:
    """Remove from a page the text of the sections `drop` selects, keeping the offsets right.

    Only pages that list their `sections` (docling parser) can lose text; others are returned
    as they are.
    """
    sections: Sequence[tuple[int, str]] = page.metadata.get("sections") or ()
    if not any(drop(heading) for _, heading in sections):
        return page
    text = page.page_content
    ends = [start for start, _ in sections[1:]] + [len(text)]
    parts: list[str] = []
    kept: list[tuple[int, str]] = []
    length = 0
    for (start, heading), end in zip(sections, ends, strict=True):
        if drop(heading):
            continue
        kept.append((length, heading))
        parts.append(text[start:end])
        length += end - start
    return Document(
        page_content="".join(parts).rstrip(), metadata=page.metadata | {"sections": tuple(kept)}
    )


def chunk_sections(
    text: str, chunks: Sequence[str], sections: Sequence[tuple[int, str]]
) -> list[str]:
    """Label each chunk with the section that covers most of it.

    `sections` holds `(offset, heading)` pairs in ascending offset order, as the docling parser
    gives them per page. Chunks are located in order (they overlap, so each search starts just
    after the previous chunk's start); a chunk that cannot be found keeps the previous position.
    """
    if not sections:
        return [""] * len(chunks)
    ends = [start for start, _ in sections[1:]] + [len(text)]
    ranges = [(start, end, heading) for (start, heading), end in zip(sections, ends, strict=True)]
    labels: list[str] = []
    cursor = 0
    for chunk in chunks:
        found = text.find(chunk, cursor)
        start = found if found >= 0 else cursor
        end = start + len(chunk)
        cursor = start + 1
        labels.append(max(ranges, key=lambda item: min(end, item[1]) - max(start, item[0]))[2])
    return labels


def split_into_chunks(
    pages: Sequence[Document],
    *,
    title: str,
    source: str,
    splitter: RecursiveCharacterTextSplitter,
    skip_references: bool = False,
) -> list[Document]:
    """Split pages into chunks carrying only the metadata the RAG needs.

    Each page is split on its own, so a chunk never crosses a page. Pages that list their
    `sections` (docling parser) give each chunk the section covering most of it and, with
    `skip_references`, leave the bibliography out. `page_count` keeps the number of pages of
    the document even when its last pages hold only references.
    """
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Document title cannot be empty")
    if not any(page.page_content.strip() for page in pages):
        raise ValueError("Document text cannot be empty")

    document_id = make_document_id(source, clean_title)
    numbers = [number for page in pages if isinstance(number := page.metadata.get("page"), int)]
    chunks: list[Document] = []
    for original in pages:
        page = drop_sections(original, is_reference_section) if skip_references else original
        page_number = page.metadata.get("page")
        texts = [text for text in splitter.split_text(page.page_content) if text.strip()]
        sections = chunk_sections(page.page_content, texts, page.metadata.get("sections") or ())
        for text, section in zip(texts, sections, strict=True):
            position = len(chunks)
            chunk_id = f"{document_id}:{position}"
            metadata: dict[str, str | int] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_title": clean_title,
                "source": source,
                "position": position,
            }
            if isinstance(page_number, int):
                metadata["page"] = page_number + 1
                metadata["page_count"] = max(numbers) + 1
            if section:
                metadata["section"] = section
            chunks.append(Document(page_content=text, metadata=metadata, id=chunk_id))
    if not chunks:
        raise ValueError("Document has no text outside the bibliography")
    return chunks
