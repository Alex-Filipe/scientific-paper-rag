from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from paper_rag.generation import GenerationBackend
from paper_rag.graph import RagNodes, build_graph
from paper_rag.service import RagService, build_service
from paper_rag.settings import Settings

ServiceFactory = Callable[..., RagService]


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a minimal text PDF (Helvetica, one line per entry) without extra dependencies."""
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, lines in zip(page_ids, pages, strict=True):
        shown = " T* ".join(f"({line}) Tj" for line in lines)
        content = f"BT /F1 12 Tf 14 TL 72 720 Td {shown} ET".encode()
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_id + 1)
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(pdf)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(chroma_dir=tmp_path / "chroma")


@pytest.fixture
def make_service(settings: Settings) -> ServiceFactory:
    def factory(**overrides: object) -> RagService:
        return build_service(replace(settings, **overrides))  # type: ignore[arg-type]

    return factory


@pytest.fixture
def with_fake_llm() -> Callable[[RagService, Sequence[str]], FakeListChatModel]:
    """Swap the service graph for one driven by scripted LLM replies."""

    def install(
        service: RagService, replies: Sequence[str], max_rewrites: int = 1
    ) -> FakeListChatModel:
        llm = FakeListChatModel(responses=list(replies))
        service.graph = build_graph(
            RagNodes(
                search=service.search,
                llm=llm,
                backend=GenerationBackend.OLLAMA,
                max_rewrites=max_rewrites,
            )
        )
        return llm

    return install  # type: ignore[return-value]
