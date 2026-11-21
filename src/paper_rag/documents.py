import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt", ".md"})


class PaperLoader(BaseLoader):
    """Load PDFs page by page (0-based `page` metadata) and text/Markdown as one document."""

    def __init__(self, path: Path) -> None:
        self.path = path

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

        from pypdf import PdfReader

        for page_number, page in enumerate(PdfReader(self.path).pages):
            yield Document(
                page_content=page.extract_text() or "",
                metadata={"source": source, "page": page_number},
            )


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


def load_file(path: Path) -> list[Document]:
    """Load a file into LangChain documents; PDFs yield one document per page."""
    return PaperLoader(path).load()


def split_into_chunks(
    pages: Sequence[Document],
    *,
    title: str,
    source: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split pages into chunks carrying only the metadata the RAG needs."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Document title cannot be empty")
    if not any(page.page_content.strip() for page in pages):
        raise ValueError("Document text cannot be empty")

    document_id = make_document_id(source, clean_title)
    chunks: list[Document] = []
    for page in pages:
        page_number = page.metadata.get("page")
        for text in splitter.split_text(page.page_content):
            if not text.strip():
                continue
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
            chunks.append(Document(page_content=text, metadata=metadata, id=chunk_id))
    return chunks
