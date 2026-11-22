"""The Streamlit screen run headless by `AppTest`, talking to the API in-process (no network)."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import ServiceFactory
from fastapi.testclient import TestClient

from paper_rag.api import create_app
from paper_rag.service import RagService

pytest.importorskip("streamlit", reason="needs the `ui` extra")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).parents[1] / "src" / "paper_rag" / "ui" / "app.py"
InstallLLM = Callable[..., object]


def launch(http: httpx.Client) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["http_client"] = http
    return app.run()


def screen(service: RagService) -> AppTest:
    return launch(TestClient(create_app(service)))


def texts(elements: Iterable[Any]) -> str:
    return "\n".join(str(element.value) for element in elements)


@pytest.fixture
def service(make_service: ServiceFactory) -> RagService:
    rag = make_service()
    rag.ingest_text("Deep Residual Learning", "ResNet-152 reaches 4.49% top-5 error.", "resnet")
    return rag


def test_sidebar_lists_documents_and_settings(service: RagService) -> None:
    app = screen(service)

    assert not app.exception
    sidebar = texts(app.sidebar.markdown)
    assert r"**Deep Residual Learning** · 1 trechos" in sidebar
    assert "Geração: **extractive** (sem LLM)" in texts(app.sidebar.caption)
    assert app.sidebar.slider[0].value == 4
    assert [toggle.disabled for toggle in app.sidebar.toggle] == [True, True]


def test_answer_replaces_tokens_and_lists_sources(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["O ResNet-152 chega a 4.49% de erro top-5 [1]."])
    app = screen(service)

    app.chat_input[0].set_value("Qual o erro top-5 do ResNet-152?").run()

    assert not app.exception
    shown = texts(app.markdown)
    assert "O ResNet-152 chega a 4.49% de erro top-5 [1]." in shown
    assert "**Validação:** Resposta com citações válidas: \\[1\\]." in shown
    labels = [expander.label for expander in app.expander]
    assert labels == ["[1] Deep Residual Learning · citada"]
    assert "ResNet-152 reaches :orange-background[**4.49**]% top-5 error." in shown
    assert app.session_state["turns"][0].answer.status == "answered"

    app.chat_input[0].set_value("E o top-1?").run()

    assert len(app.session_state["turns"]) == 2
    assert [message.name for message in app.chat_message] == ["user", "assistant"] * 2
    assert app.status[0].label.startswith("4 passos em ")


def test_fallback_shows_the_discarded_draft(service: RagService, with_fake_llm: InstallLLM) -> None:
    with_fake_llm(service, ["O erro é 5.71% [1]."])
    app = screen(service)

    app.chat_input[0].set_value("Qual o erro top-5?").run()

    labels = [expander.label for expander in app.expander]
    assert labels[0] == "Resposta do modelo descartada pela validação"
    assert "Números sem suporte nos trechos citados (5.71)" in texts(app.caption)
    assert "Evidências mais relevantes encontradas" in texts(app.markdown)


def test_uncited_draft_replaces_the_evidence_with_a_warning(
    service: RagService, with_fake_llm: InstallLLM
) -> None:
    with_fake_llm(service, ["O ResNet-152 chega a 4.49% de erro top-5."])
    app = screen(service)

    app.chat_input[0].set_value("Qual o erro top-5?").run()

    assert not app.exception
    assert "Resposta não verificada" in texts(app.warning)
    shown = texts(app.markdown)
    assert "O ResNet-152 chega a 4.49% de erro top-5." in shown
    assert "Evidências mais relevantes encontradas" not in shown
    assert ":orange-background" not in shown
    labels = [expander.label for expander in app.expander]
    assert "Resposta do modelo descartada pela validação" not in labels
    assert app.status[0].label.endswith("resposta não verificada")


def test_unreachable_api_disables_the_chat() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app = launch(httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(refuse)))

    assert not app.exception
    assert "Não consegui falar com a API" in texts(app.sidebar.error)
    assert app.chat_input[0].disabled
