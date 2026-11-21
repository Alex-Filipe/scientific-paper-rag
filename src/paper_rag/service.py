from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph

from paper_rag.documents import load_file, make_splitter, split_into_chunks
from paper_rag.embeddings import create_embeddings
from paper_rag.generation import GenerationBackend, create_chat_model
from paper_rag.graph import RagNodes, RagState, build_graph
from paper_rag.models import Answer, AnswerStatus, Citation, Step
from paper_rag.retrieval import HybridSearch, RetrievalMode
from paper_rag.settings import Settings
from paper_rag.vectorstore import ChunkIndex

EXCERPT_CHARS = 280


@dataclass(frozen=True, slots=True)
class Trace:
    """One graph run with everything the evaluation needs to look inside it."""

    answer: Answer
    contexts: tuple[Document, ...]
    """Final contexts; citation numbers in the answer refer to these."""
    updates: tuple[tuple[str, dict[str, Any]], ...]
    """State update of each node, in execution order (e.g. contexts before and after grade)."""


@dataclass(slots=True)
class RagService:
    """Entry point shared by the CLI, the HTTP API and the evaluation runner."""

    search: HybridSearch
    splitter: RecursiveCharacterTextSplitter
    graph: CompiledStateGraph[RagState, None, RagState, RagState]
    default_top_k: int = 4

    def ingest_text(self, title: str, text: str, source: str) -> int:
        return self._ingest([Document(page_content=text)], title=title, source=source)

    def ingest_file(self, path: Path, title: str | None = None, source: str | None = None) -> int:
        """Ingest a file; `source` (default: absolute path) and title identify the document,
        so ingesting the same source again replaces it instead of duplicating it."""
        return self._ingest(
            load_file(path), title=title or path.stem, source=source or str(path.resolve())
        )

    def retrieve(self, question: str, top_k: int) -> list[Document]:
        return self.search.search(question, top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        return _to_answer(self.graph.invoke(self._initial_state(question, top_k)))

    def trace(
        self,
        question: str,
        top_k: int | None = None,
        *,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> Trace:
        """Run the graph like `ask`, keeping each node's update; callbacks see every LLM call."""
        config: RunnableConfig = {"callbacks": list(callbacks)}
        updates: list[tuple[str, dict[str, Any]]] = []
        final_state: dict[str, Any] = {}
        for mode, chunk in self.graph.stream(
            self._initial_state(question, top_k),
            config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                node_updates = cast(dict[str, dict[str, Any] | None], chunk)
                updates.extend((node, dict(update or {})) for node, update in node_updates.items())
            else:
                final_state = cast(dict[str, Any], chunk)
        return Trace(
            answer=_to_answer(final_state),
            contexts=tuple(final_state.get("contexts", [])),
            updates=tuple(updates),
        )

    def stream(self, question: str, top_k: int | None = None) -> Iterator[dict[str, Any]]:
        """Validate eagerly, then yield `step`, `token` and a final `answer` event."""
        return self._stream(self._initial_state(question, top_k))

    def _stream(self, initial_state: RagState) -> Iterator[dict[str, Any]]:
        final_state: dict[str, Any] = {}
        for mode, chunk in self.graph.stream(
            initial_state,
            stream_mode=["updates", "messages", "values"],
        ):
            if mode == "updates":
                updates = cast(dict[str, dict[str, Any] | None], chunk)
                for update in updates.values():
                    for step in (update or {}).get("steps", []):
                        yield {"event": "step", "data": asdict(step)}
            elif mode == "messages":
                message, metadata = cast(tuple[BaseMessage, dict[str, Any]], chunk)
                if metadata.get("langgraph_node") == "generate" and message.text:
                    yield {"event": "token", "data": {"text": message.text}}
            elif mode == "values":
                final_state = cast(dict[str, Any], chunk)
        yield {"event": "answer", "data": _to_answer(final_state).to_dict()}

    def _ingest(self, pages: list[Document], *, title: str, source: str) -> int:
        chunks = split_into_chunks(pages, title=title, source=source, splitter=self.splitter)
        self.search.index.replace_document(str(chunks[0].metadata["document_id"]), chunks)
        return len(chunks)

    def _initial_state(self, question: str, top_k: int | None) -> RagState:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        limit = top_k or self.default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        return {
            "question": question,
            "top_k": limit,
            "search_query": question.strip(),
            "rewrites": 0,
            "steps": [],
        }


def _to_answer(state: dict[str, Any]) -> Answer:
    contexts: list[Document] = state.get("contexts", [])
    cited = set(state.get("cited", []))
    citations = tuple(
        Citation(
            index=index,
            chunk_id=str(context.metadata.get("chunk_id", context.id or "")),
            title=str(context.metadata.get("document_title", "")),
            source=str(context.metadata.get("source", "")),
            page=_optional_int(context.metadata.get("page")),
            excerpt=context.page_content[:EXCERPT_CHARS],
            rank=int(context.metadata.get("rank", index)),
            score=float(context.metadata.get("score", 0.0)),
            cited=index in cited,
        )
        for index, context in enumerate(contexts, start=1)
    )
    steps: list[Step] = state.get("steps", [])
    return Answer(
        text=str(state.get("answer", "")),
        status=AnswerStatus(state.get("status", AnswerStatus.ABSTAINED)),
        citations=citations,
        steps=tuple(steps),
        search_query=str(state.get("search_query", "")),
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def build_service(settings: Settings | None = None) -> RagService:
    resolved = settings or Settings.from_environment()
    configured = create_embeddings(
        backend=resolved.embedding_backend,
        model_name=resolved.embedding_model,
        cache_dir=resolved.model_cache_dir,
    )
    index = ChunkIndex(resolved.chroma_dir, configured.embeddings, configured.embedding_space)
    search = HybridSearch(
        index=index,
        mode=RetrievalMode(resolved.retrieval_mode),
        candidate_k=resolved.candidate_k,
    )
    nodes = RagNodes(
        search=search,
        llm=create_chat_model(resolved),
        backend=GenerationBackend(resolved.generation_backend),
        grade_documents=resolved.grade_documents,
        max_rewrites=resolved.max_rewrites,
    )
    return RagService(
        search=search,
        splitter=make_splitter(resolved.chunk_size, resolved.chunk_overlap),
        graph=build_graph(nodes),
        default_top_k=resolved.top_k,
    )
