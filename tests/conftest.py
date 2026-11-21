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
