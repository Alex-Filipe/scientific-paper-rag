"""Streamlit chat screen: `paper-rag ui` (or `streamlit run src/paper_rag/ui/app.py`).

Talks only to the HTTP API (`paper-rag serve`), whose address comes from `PAPER_RAG_API_URL`.
"""

import os
import time
from dataclasses import dataclass

import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from paper_rag.models import AnswerStatus, Step
from paper_rag.ui import view
from paper_rag.ui.client import DEFAULT_API_URL, ApiClient, ApiError

API_URL_ENV = "PAPER_RAG_API_URL"
MAX_TOP_K = 10


@dataclass(frozen=True, slots=True)
class QuestionOptions:
    top_k: int
    grade_documents: bool
    validate_numbers: bool


@st.cache_resource
def shared_http_client(base_url: str) -> httpx.Client:
    return ApiClient.connect(base_url).http


def api_client() -> ApiClient:
    """A new wrapper on every run around a long-lived `httpx.Client`.

    When a project file changes, Streamlit unloads every `paper_rag` module, so an `ApiClient`
    kept from an earlier run would yield events of the old classes and the `isinstance` checks
    of the new ones would drop them all. `httpx` is not reloaded. Tests inject `http_client`.
    """
    http = st.session_state.get("http_client") or shared_http_client(
        os.getenv(API_URL_ENV, DEFAULT_API_URL)
    )
    return ApiClient(http)


def sidebar(client: ApiClient) -> QuestionOptions | None:
    with st.sidebar:
        st.header("Paper RAG")
        try:
            config = client.config()
        except ApiError as error:
            st.error(str(error), icon=":material/cloud_off:")
            st.caption("Suba a API em outro terminal com `paper-rag serve` e recarregue a página.")
            return None
        st.caption(f"API em {client.base_url}")

        st.subheader("Artigos")
        upload_form(client)
        try:
            documents = client.documents()
        except ApiError as error:
            st.error(str(error))
            documents = []
        if documents:
            for document in documents:
                st.markdown(view.document_label(document), help=document.source)
        else:
            st.caption("Nenhum artigo indexado ainda.")

        st.subheader("Modelo")
        for line in view.model_lines(config):
            st.caption(line)

        st.subheader("Pergunta")
        top_k = st.slider("Trechos recuperados (top-k)", 1, MAX_TOP_K, min(config.top_k, MAX_TOP_K))
        has_llm = config.model is not None
        no_llm = "Sem LLM na API (modo extrativo)." if not has_llm else None
        grade = st.toggle(
            "Avaliar trechos",
            value=config.grade_documents and has_llm,
            disabled=not has_llm,
            help=no_llm or "Nó grade: o LLM julga cada trecho antes da geração.",
        )
        if grade and (warning := view.grade_warning(config)):
            st.caption(warning)
        validate = st.toggle(
            "Validar números",
            value=config.validate_numbers,
            disabled=not has_llm,
            help=no_llm
            or "Nó validate: descarta a resposta se um número não estiver nos trechos citados.",
        )
        if st.button("Limpar conversa", icon=":material/delete_sweep:"):
            st.session_state.turns = []
    return QuestionOptions(top_k=top_k, grade_documents=grade, validate_numbers=validate)


def upload_form(client: ApiClient) -> None:
    with st.form("upload", clear_on_submit=True, border=False):
        file = st.file_uploader("Enviar artigo", type=["pdf", "txt", "md"])
        submitted = st.form_submit_button("Indexar", icon=":material/upload_file:")
    if not submitted or file is None:
        return
    with st.spinner(f"Indexando {file.name}…"):
        try:
            chunks = client.upload(file.name, file.getvalue())
        except ApiError as error:
            st.error(str(error))
            return
    st.toast(f"{file.name}: {chunks} trechos indexados.", icon=":material/check:")


def render_result(turn: view.Turn, slot: DeltaGenerator) -> None:
    """Final text (replacing the streamed tokens), status badge, notices and sources."""
    if turn.error:
        slot.error(turn.error, icon=":material/error:")
        return
    answer = turn.answer
    if answer is None:
        slot.warning("A API encerrou o stream sem enviar a resposta final.")
        return
    slot.markdown(view.answer_markdown(answer.text))
    badge = view.STATUS_BADGES[answer.status]
    st.badge(badge.label, icon=badge.icon, color=badge.color, help=badge.help)
    if fix := turn.citation_fix:
        st.info(fix.detail, icon=":material/build:")
    if draft := turn.discarded_draft:
        with st.expander("Resposta do modelo descartada pela validação", icon=":material/undo:"):
            if note := turn.validation_note:
                st.caption(note)
            st.markdown(view.answer_markdown(draft))
    if not answer.citations:
        return
    st.caption(view.sources_heading(answer))
    expand_cited = answer.status == AnswerStatus.ANSWERED
    for citation in answer.citations:
        with st.expander(
            view.source_label(citation),
            expanded=citation.cited and expand_cited,
            icon=view.source_icon(citation),
        ):
            st.markdown(view.excerpt_markdown(citation.excerpt))
            st.caption(
                f"{citation.source} · posição {citation.rank} na busca · "
                f"RRF {view.format_score(citation.score)}"
            )


def render_history(turns: list[view.Turn]) -> None:
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(view.escape_markdown(turn.question))
        with st.chat_message("assistant"):
            with st.status(view.summary_label(turn), state="error" if turn.error else "complete"):
                for step in turn.steps:
                    st.markdown(view.step_markdown(step))
            render_result(turn, st.empty())


def ask(client: ApiClient, question: str, options: QuestionOptions) -> view.Turn:
    turn = view.Turn(question=question)
    with st.chat_message("user"):
        st.markdown(view.escape_markdown(question))
    with st.chat_message("assistant"):
        status = st.status("Consultando o grafo…", expanded=True)
        slot = st.empty()

        def on_step(step: Step) -> None:
            status.markdown(view.step_markdown(step))
            # Without `expanded`, the update clears the field and the browser collapses the box.
            status.update(label=view.running_label(step), expanded=True)

        started = time.perf_counter()
        try:
            events = client.stream(
                question,
                top_k=options.top_k,
                grade_documents=options.grade_documents,
                validate_numbers=options.validate_numbers,
            )
            slot.write_stream(view.follow(events, turn, on_step))
        except ApiError as error:
            turn.error = str(error)
        turn.seconds = time.perf_counter() - started
        status.update(
            label=view.summary_label(turn),
            state="error" if turn.error else "complete",
            expanded=False,
        )
        render_result(turn, slot)
    return turn


def main() -> None:
    st.set_page_config(page_title="Paper RAG", page_icon=":material/menu_book:")
    client = api_client()
    options = sidebar(client)
    turns: list[view.Turn] = st.session_state.setdefault("turns", [])

    st.title("Perguntas sobre os artigos")
    if not turns:
        st.caption(
            "As respostas citam os trechos como [1], [2]…; os passos do grafo aparecem "
            "enquanto a resposta é gerada."
        )
    render_history(turns)
    question = st.chat_input("Pergunte sobre os artigos indexados", disabled=options is None)
    if question and options is not None:
        turns.append(ask(client, str(question), options))


main()
