"""Corrective RAG flow as a LangGraph state machine.

    retrieve -> grade -> generate -> validate -> END
                  |         \\-> END (extractive answer or model abstention)
                  |  \\-> abstain -> END
                  \\-> rewrite -> retrieve   (at most `max_rewrites` times)

Without an LLM (extractive backend) grading and rewriting are skipped, so the flow is
retrieve -> generate (extractive) or retrieve -> abstain. Grade and rewrite are off by default.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paper_rag.generation import (
    ABSTENTION_ANSWER,
    CITATION_PATTERN,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    GenerationBackend,
    answer_prompt_for,
    cited_indexes,
    extractive_answer,
    format_evidence,
    is_abstention,
    is_usable_query,
    parse_grade,
    recitation_target,
    unsupported_numbers,
)
from paper_rag.models import AnswerStatus, Step
from paper_rag.retrieval import HybridSearch


class RagState(TypedDict, total=False):
    question: str
    top_k: int
    search_query: str
    rewrites: int
    contexts: list[Document]
    draft: str
    """The model's answer before validation."""
    answer: str
    status: AnswerStatus
    cited: list[int]
    steps: Annotated[list[Step], operator.add]
    grade_documents: bool
    """Per-question override of `RagNodes.grade_documents`."""
    validate_numbers: bool
    """Per-question override of `RagNodes.validate_numbers`."""


@dataclass(slots=True)
class RagNodes:
    search: HybridSearch
    llm: BaseChatModel | None
    backend: GenerationBackend
    grade_documents: bool = True
    max_rewrites: int = 1
    grader: BaseChatModel | None = None
    """Model for grade and rewrite; None uses `llm`."""
    validate_numbers: bool = True
    """Reject answers whose numbers are not in the question or the cited chunks."""

    @property
    def judge(self) -> BaseChatModel:
        assert self.llm is not None
        return self.grader or self.llm

    def retrieve(self, state: RagState) -> dict[str, Any]:
        query = state["search_query"]
        contexts = self.search.search(query, state["top_k"])
        return {
            "contexts": contexts,
            "steps": [Step("retrieve", f"Busca por “{query}”: {len(contexts)} trecho(s).")],
        }

    def grade(self, state: RagState) -> dict[str, Any]:
        contexts = state["contexts"]
        if not contexts:
            return {"steps": [Step("grade", "Nenhum trecho para avaliar.")]}
        if self.llm is None or not state.get("grade_documents", self.grade_documents):
            return {"steps": [Step("grade", "Avaliação pulada (sem LLM ou desativada).")]}

        # Sequential calls keep replies aligned with chunks (and local servers serialize anyway).
        chain = GRADE_PROMPT | self.judge
        kept: list[Document] = []
        unreadable = 0
        for context in contexts:
            reply = chain.invoke(
                {"question": state["question"], "chunk": context.page_content[:4_000].strip()}
            )
            verdict = parse_grade(reply.text)
            if verdict is None:
                unreadable += 1
            if verdict is not False:  # an unreadable verdict keeps the chunk
                kept.append(context)
        detail = f"{len(kept)} de {len(contexts)} trecho(s) úteis."
        if unreadable:
            detail += f" {unreadable} resposta(s) ilegível(is) do avaliador; trecho(s) mantido(s)."
        return {"contexts": kept, "steps": [Step("grade", detail)]}

    def route_after_grade(self, state: RagState) -> Literal["generate", "rewrite", "abstain"]:
        if state["contexts"]:
            return "generate"
        if self.llm is not None and state.get("rewrites", 0) < self.max_rewrites:
            return "rewrite"
        return "abstain"

    def rewrite(self, state: RagState) -> dict[str, Any]:
        reply = (REWRITE_PROMPT | self.judge).invoke({"question": state["question"]})
        candidate = reply.text.strip().strip("\"'“”")
        if is_usable_query(candidate):
            new_query = candidate
            detail = f"Evidência fraca; nova consulta: “{new_query}”."
        else:
            new_query = state["question"].strip()
            detail = f"Consulta reescrita inválida (“{candidate}”); mantive a pergunta original."
        return {
            "search_query": new_query,
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [Step("rewrite", detail)],
        }

    def generate(self, state: RagState) -> dict[str, Any]:
        contexts = state["contexts"]
        if self.llm is None:
            return {
                "answer": extractive_answer(contexts),
                "status": AnswerStatus.EXTRACTIVE,
                "cited": [],
                "steps": [Step("generate", "Modo extrativo: listando as evidências.")],
            }

        reply = (answer_prompt_for(self.backend) | self.llm).invoke(
            {"question": state["question"].strip(), "evidence": format_evidence(contexts)}
        )
        text = reply.text.strip()
        if is_abstention(text):
            return {
                "answer": ABSTENTION_ANSWER,
                "status": AnswerStatus.ABSTAINED,
                "cited": [],
                "steps": [Step("generate", "O modelo se absteve por falta de evidência.")],
            }
        return {"draft": text, "steps": [Step("generate", "Resposta gerada pelo modelo.")]}

    def route_after_generate(self, state: RagState) -> Literal["validate", "end"]:
        return "validate" if "status" not in state else "end"

    def validate(self, state: RagState) -> dict[str, Any]:
        """Keep the draft only if its citations exist and its numbers come from what it cites.

        A draft citing one chunk whose numbers sit in another retrieved chunk's matching line is
        re-cited to that chunk instead of falling back.
        """
        draft, contexts = state["draft"], state["contexts"]
        cited = cited_indexes(draft, len(contexts))
        if cited is None:
            return _fallback(contexts, "Citações ausentes ou inválidas; mostrando evidências.")
        if state.get("validate_numbers", self.validate_numbers):
            sources = [contexts[index - 1].page_content for index in cited]
            missing = unsupported_numbers(draft, state["question"], sources)
            if missing:
                texts = [context.page_content for context in contexts]
                target = (
                    recitation_target(draft, state["question"], texts) if len(cited) == 1 else None
                )
                if target is None:
                    return _fallback(
                        contexts,
                        f"Números sem suporte nos trechos citados ({', '.join(missing)}); "
                        "mostrando evidências.",
                    )
                return {
                    "answer": CITATION_PATTERN.sub(f"[{target}]", draft),
                    "status": AnswerStatus.ANSWERED,
                    "cited": [target],
                    "steps": [
                        Step(
                            "validate",
                            f"Citação corrigida de {cited} para [{target}], o trecho que "
                            f"contém {', '.join(missing)}.",
                        )
                    ],
                }
        return {
            "answer": draft,
            "status": AnswerStatus.ANSWERED,
            "cited": cited,
            "steps": [Step("validate", f"Resposta com citações válidas: {cited}.")],
        }

    def abstain(self, state: RagState) -> dict[str, Any]:
        del state
        return {
            "answer": ABSTENTION_ANSWER,
            "status": AnswerStatus.ABSTAINED,
            "cited": [],
            "steps": [Step("abstain", "Sem evidência suficiente para responder.")],
        }


def _fallback(contexts: list[Document], detail: str) -> dict[str, Any]:
    return {
        "answer": extractive_answer(contexts),
        "status": AnswerStatus.FALLBACK,
        "cited": [],
        "steps": [Step("validate", detail)],
    }


def build_graph(nodes: RagNodes) -> CompiledStateGraph[RagState, None, RagState, RagState]:
    graph = StateGraph(RagState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("generate", nodes.generate)
    graph.add_node("validate", nodes.validate)
    graph.add_node("abstain", nodes.abstain)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", nodes.route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "generate", nodes.route_after_generate, {"validate": "validate", "end": END}
    )
    graph.add_edge("validate", END)
    graph.add_edge("abstain", END)
    return graph.compile()
