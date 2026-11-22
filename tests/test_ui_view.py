from dataclasses import asdict, dataclass

from paper_rag.models import Answer, AnswerStatus, Citation, DocumentInfo, ServiceConfig, Step
from paper_rag.ui import view
from paper_rag.ui.client import AnswerEvent, ErrorEvent, StepEvent, TokenEvent

FIX = Step("validate", "Citação corrigida de [1] para [2], o trecho que contém 4.49.")


def citation(
    index: int = 1, *, cited: bool = True, page: int | None = 7, section: str = ""
) -> Citation:
    return Citation(
        index=index,
        chunk_id=f"doc:{index}",
        title="Deep Residual Learning",
        source="resnet.pdf",
        page=page,
        excerpt="ResNet-152 19.38 4.49",
        rank=index,
        score=0.0325,
        cited=cited,
        section=section,
    )


def answer(status: AnswerStatus, text: str = "Erro de 4,49% [2].") -> Answer:
    return Answer(text, status, (citation(1, cited=False), citation(2)), (), "erro")


def config(**overrides: object) -> ServiceConfig:
    values: dict[str, object] = {
        "generation_backend": "ollama",
        "model": "qwen2.5:1.5b-instruct",
        "grader_model": None,
        "embedding_backend": "hashing",
        "retrieval_mode": "hybrid",
        "top_k": 4,
        "grade_documents": False,
        "max_rewrites": 0,
        "validate_numbers": True,
    }
    return ServiceConfig(**(values | overrides))  # type: ignore[arg-type]


def test_follow_routes_steps_and_yields_tokens() -> None:
    turn = view.Turn("Qual o erro?")
    seen: list[str] = []
    final = answer(AnswerStatus.ANSWERED)
    events = [
        StepEvent(Step("retrieve", "4 trechos")),
        TokenEvent("Erro "),
        TokenEvent("de 4,49% [1]."),
        StepEvent(Step("generate", "ok")),
        StepEvent(FIX),
        AnswerEvent(final),
    ]

    tokens = list(view.follow(events, turn, lambda step: seen.append(step.node)))

    assert tokens == ["Erro ", "de 4,49% [1]."]
    assert seen == ["retrieve", "generate", "validate"]
    assert turn.draft == "Erro de 4,49% [1]."
    assert turn.answer is final
    assert turn.citation_fix == FIX
    assert turn.discarded_draft is None


def test_fallback_keeps_the_discarded_draft_and_its_reason() -> None:
    turn = view.Turn("Qual o erro?")
    reason = Step("validate", "Números sem suporte nos trechos citados (5.71); mostrando…")
    for event in [TokenEvent("Erro de 5,71% [1]."), StepEvent(reason)]:
        turn.apply(event)
    turn.apply(AnswerEvent(answer(AnswerStatus.FALLBACK, "Evidências mais relevantes...")))

    assert turn.discarded_draft == "Erro de 5,71% [1]."
    assert turn.validation_note == reason.detail
    assert turn.citation_fix is None


def test_error_event_is_recorded() -> None:
    turn = view.Turn("Qual o erro?")

    turn.apply(ErrorEvent("Ollama is not running"))

    assert turn.error == "Ollama is not running"
    assert view.summary_label(turn) == "Erro ao consultar a API"


def test_summary_label_counts_steps_and_flags_validation() -> None:
    turn = view.Turn("Qual o erro?", steps=[Step("retrieve", "x"), FIX], seconds=3.24)
    single = view.Turn("Oi", steps=[Step("abstain", "x")])
    dropped = view.Turn("Oi", steps=[Step("validate", "x")], draft="5,71% [1].")
    dropped.answer = answer(AnswerStatus.FALLBACK)

    assert view.summary_label(turn) == "2 passos em 3,2 s · citação corrigida"
    assert view.summary_label(single) == "1 passo"
    assert view.summary_label(dropped) == "1 passo · resposta descartada"


def test_step_markdown_highlights_citation_fixes_and_escapes() -> None:
    assert view.step_markdown(FIX).startswith(":material/build: **Validação:** Citação corrigida")
    assert r"\[1\]" in view.step_markdown(FIX)
    assert view.step_markdown(Step("custom", "$5")) == r":material/check: **custom:** \$5"
    assert view.running_label(Step("retrieve", "x")) == "Busca concluída…"


def test_source_labels_show_page_and_citation() -> None:
    assert view.source_label(citation(2)) == "[2] Deep Residual Learning, p. 7 · citada"
    assert view.source_label(citation(1, cited=False, page=None)) == "[1] Deep Residual Learning"
    assert view.source_label(citation(3, section="4.1. ImageNet *Classification*")) == (
        r"[3] Deep Residual Learning, p. 7 · 4.1. ImageNet \*Classification\* · citada"
    )
    assert view.source_icon(citation(1, cited=False)) == ":material/description:"
    assert view.sources_heading(answer(AnswerStatus.ANSWERED)) == "Fontes"
    assert view.sources_heading(answer(AnswerStatus.FALLBACK)) == "Trechos recuperados"
    assert view.format_score(0.0325) == "0,0325"


def test_document_and_model_descriptions() -> None:
    pdf = DocumentInfo("id", "Deep_Residual", "resnet.pdf", 42, 9)
    text = DocumentInfo("id", "Notas", "notes.md", 1, None)

    assert view.document_label(pdf) == r"**Deep\_Residual** · 9 p. · 42 trechos"
    assert view.document_label(text) == "**Notas** · 1 trechos"
    assert view.model_lines(config(grader_model="qwen2.5:7b-instruct")) == [
        "Geração: **ollama** · `qwen2.5:1.5b-instruct`",
        "Avaliador (grade): `qwen2.5:7b-instruct`",
        "Busca: hybrid · embeddings hashing",
        "Leitura de PDF: pypdf",
    ]
    assert view.model_lines(config(generation_backend="extractive", model=None))[0] == (
        "Geração: **extractive** (sem LLM)"
    )


def test_grade_warning_only_when_the_generator_grades() -> None:
    assert "RAG_GRADER_MODEL" in (view.grade_warning(config()) or "")
    assert view.grade_warning(config(grader_model="qwen2.5:7b-instruct")) is None
    assert view.grade_warning(config(model=None)) is None


def test_markdown_helpers_keep_lines_and_avoid_latex() -> None:
    assert view.excerpt_markdown("Table 3\nResNet-152 *19.38*\n") == (
        "Table 3  \nResNet-152 \\*19.38\\*"
    )
    assert view.excerpt_markdown("1. Introduction\n- item\n2) x\n12.5 error") == (
        "1\\. Introduction  \n\\- item  \n2\\) x  \n12.5 error"
    )
    assert view.answer_markdown("Custa $5 e $6 [1].") == r"Custa \$5 e \$6 [1]."
    assert view.format_seconds(14.66) == "14,7 s"


def test_sources_show_the_whole_chunk_with_claimed_numbers_marked() -> None:
    table = "ResNet-101 21.75 6.05\nResNet-152 21.43 5.71\n1. Introduction"
    whole = Citation(**(asdict(citation()) | {"text": table}))
    turn = view.Turn("Erro top-5 do ResNet-152?")
    turn.apply(AnswerEvent(answer(AnswerStatus.ANSWERED, "O ResNet-152 chega a 5,71% [2].")))

    highlight = view.highlight_readings(turn)

    assert highlight == {5.71}
    assert view.source_markdown(whole, highlight) == (
        "ResNet-101 21.75 6.05  \n"
        "ResNet-152 21.43 :orange-background[**5.71**]  \n"
        "1\\. Introduction"
    )
    assert view.source_markdown(citation()) == "ResNet-152 19.38 4.49"


def test_marks_keep_table_spacing_and_skip_unchecked_answers() -> None:
    fallback = view.Turn("Erro?", answer=answer(AnswerStatus.FALLBACK, "Evidências: 5.71"))

    assert view.excerpt_markdown("plain 28.54 10.02", {28.54}) == (
        "plain :orange-background[**28.54**] 10.02"
    )
    assert view.highlight_readings(fallback) == set()
    assert view.highlight_readings(view.Turn("Erro?")) == set()


def test_uncited_draft_is_shown_as_unverified() -> None:
    uncited = Step("validate", "Citações ausentes ou inválidas; mostrando evidências.")
    turn = view.Turn("Erro top-5 do ResNet-152?", steps=[uncited], draft="Chega a 5,71%.")
    turn.answer = answer(AnswerStatus.FALLBACK, "Evidências mais relevantes encontradas: ...")
    numbers = view.Turn("Erro?", steps=[Step("validate", "Números sem suporte (5.71)")])
    numbers.draft, numbers.answer = "Chega a 5,71% [1].", turn.answer

    assert turn.unverified_draft == "Chega a 5,71%."
    assert view.badge(turn) == view.UNVERIFIED_BADGE
    assert view.summary_label(turn) == "1 passo · resposta não verificada"
    assert view.highlight_readings(turn) == set()
    assert numbers.unverified_draft is None
    assert view.badge(numbers) == view.STATUS_BADGES[AnswerStatus.FALLBACK]
    assert view.highlight_readings(numbers) == set()
    assert view.badge(view.Turn("Erro?")) is None


def test_revive_rebuilds_turns_of_a_reloaded_class() -> None:
    @dataclass
    class OldTurn:
        question: str
        steps: list[Step]
        draft: str
        answer: Answer | None
        error: str | None
        seconds: float | None

    old = OldTurn("Erro?", [FIX], "", None, None, 1.0)
    current = view.Turn("Erro?")

    revived = view.revive(old)

    assert isinstance(revived, view.Turn)
    assert revived.citation_fix == FIX
    assert view.revive(current) is current
