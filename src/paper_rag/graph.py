"""Corrective RAG flow as a LangGraph state machine.

    retrieve -> grade -> generate -> END
                  |  \\-> abstain -> END
                  \\-> rewrite -> retrieve   (at most `max_rewrites` times)

Without an LLM (extractive backend) grading and rewriting are skipped, so the flow is
retrieve -> generate (extractive) or retrieve -> abstain.
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
    GRADE_PROMPT,
    REWRITE_PROMPT,
    GenerationBackend,
    answer_prompt_for,
    cited_indexes,
    extractive_answer,
    format_evidence,
    is_abstention,
    parse_grades,
)
from paper_rag.models import AnswerStatus, Step
from paper_rag.retrieval import HybridSearch


class RagState(TypedDict, total=False):
    question: str
    top_k: int
    search_query: str
    rewrites: int
    contexts: list[Document]
    answer: str
    status: AnswerStatus
    cited: list[int]
    steps: Annotated[list[Step], operator.add]


@dataclass(slots=True)
class RagNodes:
    search: HybridSearch
    llm: BaseChatModel | None
    backend: GenerationBackend
    grade_documents: bool = True
    max_rewrites: int = 1

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
        if self.llm is None or not self.grade_documents:
            return {"steps": [Step("grade", "Avaliação pulada (sem LLM ou desativada).")]}

        reply = (GRADE_PROMPT | self.llm).invoke(
            {"question": state["question"], "evidence": format_evidence(contexts)}
        )
        useful = parse_grades(reply.text, len(contexts))
        if useful is None:
            return {"steps": [Step("grade", "Resposta do avaliador ilegível; mantive todos.")]}
        kept = [contexts[index - 1] for index in useful]
        return {
            "contexts": kept,
            "steps": [Step("grade", f"{len(kept)} de {len(contexts)} trecho(s) úteis.")],
        }

    def route_after_grade(self, state: RagState) -> Literal["generate", "rewrite", "abstain"]:
        if state["contexts"]:
            return "generate"
        if self.llm is not None and state.get("rewrites", 0) < self.max_rewrites:
            return "rewrite"
        return "abstain"

    def rewrite(self, state: RagState) -> dict[str, Any]:
        assert self.llm is not None
        reply = (REWRITE_PROMPT | self.llm).invoke({"question": state["question"]})
        new_query = reply.text.strip().strip("\"'“”") or state["question"]
        return {
            "search_query": new_query,
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [Step("rewrite", f"Evidência fraca; nova consulta: “{new_query}”.")],
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
        cited = cited_indexes(text, len(contexts))
        if cited is None:
            return {
                "answer": extractive_answer(contexts),
                "status": AnswerStatus.FALLBACK,
                "cited": [],
                "steps": [
                    Step("validate", "Citações ausentes ou inválidas; mostrando evidências.")
                ],
            }
        return {
            "answer": text,
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


def build_graph(nodes: RagNodes) -> CompiledStateGraph[RagState, None, RagState, RagState]:
    graph = StateGraph(RagState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("generate", nodes.generate)
    graph.add_node("abstain", nodes.abstain)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", nodes.route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()
