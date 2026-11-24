"""Presentation logic of the chat screen, free of Streamlit so it can be unit tested."""

import re
from collections.abc import Callable, Iterable, Iterator, Set
from dataclasses import dataclass, field, fields
from typing import Literal

from paper_rag.models import Answer, AnswerStatus, Citation, DocumentInfo, ServiceConfig, Step
from paper_rag.numbers import NUMBER, claimed_readings, number_readings
from paper_rag.ui.client import AnswerEvent, ErrorEvent, StepEvent, StreamEvent, TokenEvent

MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~$])")
LIST_MARKER = re.compile(r"^(\s*)(\d+)([.)])(?=\s)|^(\s*)([-+])(?=\s)")
BadgeColor = Literal["green", "blue", "orange", "gray"]


@dataclass(frozen=True, slots=True)
class Badge:
    label: str
    color: BadgeColor
    icon: str
    help: str


STATUS_BADGES = {
    AnswerStatus.ANSWERED: Badge(
        "respondida",
        "green",
        ":material/check_circle:",
        "Resposta do modelo com citações e números conferidos pelo nó validate.",
    ),
    AnswerStatus.EXTRACTIVE: Badge(
        "extrativa",
        "blue",
        ":material/format_list_bulleted:",
        "Sem LLM configurado: a resposta lista os trechos recuperados.",
    ),
    AnswerStatus.FALLBACK: Badge(
        "fallback",
        "orange",
        ":material/warning:",
        "A validação descartou a resposta do modelo; os trechos aparecem no lugar dela.",
    ),
    AnswerStatus.ABSTAINED: Badge(
        "abstenção",
        "gray",
        ":material/block:",
        "Não há evidência suficiente nos artigos indexados.",
    ),
}

UNVERIFIED_BADGE = Badge(
    "não verificada",
    "orange",
    ":material/help:",
    "O modelo não citou os trechos, então a validação não conferiu a resposta; confira nas fontes.",
)
UNVERIFIED_WARNING = (
    "Resposta não verificada: o modelo não citou os trechos, então a validação não conferiu nem "
    "as fontes nem os números. Confira nos trechos recuperados abaixo."
)
MISSING_CITATIONS = "Citações ausentes"
"""Start of the `validate` detail when the draft has no valid `[n]` (see `graph.validate`)."""

STEP_TITLES = {
    "retrieve": "Busca",
    "grade": "Avaliação dos trechos",
    "rewrite": "Reescrita da pergunta",
    "generate": "Geração",
    "validate": "Validação",
    "abstain": "Abstenção",
}


@dataclass(slots=True)
class Turn:
    """One question and everything its stream produced, kept to redraw the chat history."""

    question: str
    steps: list[Step] = field(default_factory=list)
    draft: str = ""
    """Tokens of the `generate` node, streamed before validation."""
    answer: Answer | None = None
    error: str | None = None
    seconds: float | None = None

    def apply(self, event: StreamEvent) -> str:
        """Record one event; returns the token text to show (empty for other events)."""
        if isinstance(event, StepEvent):
            self.steps.append(event.step)
        elif isinstance(event, TokenEvent):
            self.draft += event.text
            return event.text
        elif isinstance(event, AnswerEvent):
            self.answer = event.answer
        elif isinstance(event, ErrorEvent):
            self.error = event.detail
        return ""

    @property
    def citation_fix(self) -> Step | None:
        """The `validate` step that re-cited the answer to another chunk, if any."""
        return next((step for step in self.steps if is_citation_fix(step)), None)

    @property
    def discarded_draft(self) -> str | None:
        """The model's draft when validation replaced it with the evidence (`fallback`)."""
        if self.answer is None or self.answer.status != AnswerStatus.FALLBACK:
            return None
        return self.draft.strip() or None

    @property
    def validation_note(self) -> str | None:
        """Why the draft fell back, from the `validate` step."""
        if self.discarded_draft is None:
            return None
        return next((step.detail for step in self.steps if step.node == "validate"), None)

    @property
    def unverified_draft(self) -> str | None:
        """The draft of a fallback caused only by missing citations, shown flagged as unverified.

        Such a draft is often a faithful paraphrase that forgot its `[n]`, and the evidence shown
        instead is no better. A draft rejected for unsupported numbers stays hidden: it is wrong.
        """
        note = self.validation_note
        return self.discarded_draft if note and note.startswith(MISSING_CITATIONS) else None


def revive(turn: object) -> Turn:
    """A turn kept in the session, rebuilt with the current `Turn` class if it is an old one.

    Streamlit reloads `paper_rag.ui` when a file changes; turns stored before that are instances
    of the previous class and would miss any property added since.
    """
    if isinstance(turn, Turn):
        return turn
    return Turn(**{item.name: getattr(turn, item.name) for item in fields(Turn)})


def follow(
    events: Iterable[StreamEvent], turn: Turn, on_step: Callable[[Step], None]
) -> Iterator[str]:
    """Feed `events` into `turn`, call `on_step` for each step and yield the token text.

    This is the generator `st.write_stream` consumes: steps go to the status box as they
    arrive, tokens to the answer area; the final `answer` event replaces the tokens afterwards.
    """
    for event in events:
        text = turn.apply(event)
        if isinstance(event, StepEvent):
            on_step(event.step)
        if text:
            yield text


def is_citation_fix(step: Step) -> bool:
    return step.node == "validate" and step.detail.startswith("Citação corrigida")


def step_title(node: str) -> str:
    return STEP_TITLES.get(node, node)


def step_markdown(step: Step) -> str:
    icon = ":material/build:" if is_citation_fix(step) else ":material/check:"
    return f"{icon} **{step_title(step.node)}:** {escape_markdown(step.detail)}"


def summary_label(turn: Turn) -> str:
    """Label of the closed status box, e.g. `4 passos em 3,2 s · citação corrigida`."""
    if turn.error:
        return "Erro ao consultar a API"
    label = f"{len(turn.steps)} passo{'s' if len(turn.steps) != 1 else ''}"
    if turn.seconds is not None:
        label += f" em {format_seconds(turn.seconds)}"
    if turn.citation_fix is not None:
        label += " · citação corrigida"
    elif turn.unverified_draft is not None:
        label += " · resposta não verificada"
    elif turn.discarded_draft is not None:
        label += " · resposta descartada"
    return label


def running_label(step: Step) -> str:
    return f"{step_title(step.node)} concluída…"


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + " s"


def format_score(score: float) -> str:
    return f"{score:.4f}".replace(".", ",")


def source_label(citation: Citation) -> str:
    page = f", p. {citation.page}" if citation.page is not None else ""
    section = f" · {escape_markdown(citation.section)}" if citation.section else ""
    cited = " · citada" if citation.cited else ""
    return f"[{citation.index}] {escape_markdown(citation.title)}{page}{section}{cited}"


def source_icon(citation: Citation) -> str:
    return ":material/bookmark:" if citation.cited else ":material/description:"


def sources_heading(answer: Answer) -> str:
    # Equality, not identity: turns kept in the session survive Streamlit's module reloads.
    return "Fontes" if answer.status == AnswerStatus.ANSWERED else "Trechos recuperados"


def document_label(document: DocumentInfo) -> str:
    pages = f"{document.pages} p. · " if document.pages is not None else ""
    return f"**{escape_markdown(document.title)}** · {pages}{document.chunks} trechos"


def model_lines(config: ServiceConfig) -> list[str]:
    if config.model is None:
        lines = [f"Geração: **{config.generation_backend}** (sem LLM)"]
    else:
        lines = [f"Geração: **{config.generation_backend}** · `{config.model}`"]
        if config.grader_model:
            lines.append(f"Avaliador (grade): `{config.grader_model}`")
    lines.append(f"Busca: {config.retrieval_mode} · embeddings {config.embedding_backend}")
    references = "" if config.index_references else " · sem bibliografia no índice"
    sections = references if config.pdf_parser != "pypdf" else ""
    lines.append(f"Leitura de PDF: {config.pdf_parser}{sections}")
    return lines


def grade_warning(config: ServiceConfig) -> str | None:
    """Warn when grading would reuse the small generator, which the evaluation advised against."""
    if config.model is None or config.grader_model:
        return None
    return (
        f"O grade vai usar o próprio `{config.model}`. Com o 1,5B ele troca acertos por "
        "abstenções; prefira `RAG_GRADER_MODEL=qwen2.5:7b-instruct` ao subir a API."
    )


def escape_markdown(text: str) -> str:
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def answer_markdown(text: str) -> str:
    """Model text keeps its Markdown, but `$` must not turn into LaTeX."""
    return text.replace("$", r"\$")


def badge(turn: Turn) -> Badge | None:
    if turn.answer is None:
        return None
    return UNVERIFIED_BADGE if turn.unverified_draft else STATUS_BADGES[turn.answer.status]


def highlight_readings(turn: Turn) -> set[float]:
    """Numbers to mark in the sources: the ones an accepted answer adds to the question.

    These are the numbers the `validate` node looked for in the cited chunks, so marking them
    lets the reader check the support without hunting through a table. Unverified drafts get
    none: in practice their numbers were figure and equation references (`Fig. 3`, `Eqn.(1)`).
    """
    if turn.answer is None or turn.answer.status != AnswerStatus.ANSWERED:
        return set()
    return claimed_readings(turn.answer.text, turn.question)


def source_markdown(citation: Citation, highlight: Set[float] = frozenset()) -> str:
    """The whole chunk (older APIs only send the excerpt), with the given numbers marked."""
    return excerpt_markdown(citation.text or citation.excerpt, highlight)


def excerpt_markdown(text: str, highlight: Set[float] = frozenset()) -> str:
    """Chunk text as Markdown with its line breaks (table rows) kept and no accidental lists."""
    return "  \n".join(_escape_line(line, highlight) for line in text.strip().splitlines())


def _escape_line(line: str, highlight: Set[float]) -> str:
    escaped = LIST_MARKER.sub(
        lambda match: (
            f"{match[1]}{match[2]}\\{match[3]}" if match[2] else f"{match[4]}\\{match[5]}"
        ),
        escape_markdown(line),
    )
    if not highlight:
        return escaped
    # Escaping never touches digits, dots or commas, so numbers survive it intact.
    return NUMBER.sub(lambda match: _mark(match[0], highlight), escaped)


def _mark(token: str, highlight: Set[float]) -> str:
    if not number_readings(token) & highlight:
        return token
    number = token.rstrip()  # NUMBER may swallow the space before the next table cell
    return f":orange-background[**{number}**]{token[len(number) :]}"
