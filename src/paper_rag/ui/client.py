"""HTTP client for the API, including the Server-Sent Events of `POST /questions/stream`.

The stream is a POST, so the browser `EventSource` (GET only) cannot read it; this module parses
the `text/event-stream` lines from an `httpx` response instead. It has no Streamlit dependency.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from paper_rag.models import Answer, AnswerStatus, Citation, DocumentInfo, ServiceConfig, Step

DEFAULT_API_URL = "http://127.0.0.1:8000"
# Generation with a local 7B grader takes ~15 s per question; reads wait well beyond that.
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=300.0)


class ApiError(RuntimeError):
    """The API is unreachable or answered with an error."""


@dataclass(frozen=True, slots=True)
class ServerEvent:
    """One raw Server-Sent Event: its name and JSON payload."""

    event: str
    data: Any


@dataclass(frozen=True, slots=True)
class StepEvent:
    step: Step


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    answer: Answer


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    detail: str


StreamEvent = StepEvent | TokenEvent | AnswerEvent | ErrorEvent


def parse_sse(lines: Iterable[str]) -> Iterator[ServerEvent]:
    """Parse `text/event-stream` lines (without line terminators) into events.

    Follows the WHATWG rules the API relies on: `event:` names the event (default `message`),
    `data:` lines are joined with newlines, `:` starts a comment, a blank line dispatches, and an
    event cut off before its blank line is dropped.
    """
    name, data_lines = "message", list[str]()
    for line in lines:
        if not line:
            if data_lines:
                yield ServerEvent(name, _json("\n".join(data_lines)))
            name, data_lines = "message", list[str]()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            name = value
        elif field == "data":
            data_lines.append(value)


def to_stream_event(event: ServerEvent) -> StreamEvent | None:
    """Type the API events; unknown event names are ignored (None)."""
    data = event.data if isinstance(event.data, dict) else {}
    if event.event == "step":
        return StepEvent(Step(node=str(data.get("node", "")), detail=str(data.get("detail", ""))))
    if event.event == "token":
        return TokenEvent(str(data.get("text", "")))
    if event.event == "answer":
        return AnswerEvent(answer_from_dict(data))
    if event.event == "error":
        return ErrorEvent(str(data.get("detail", "Erro desconhecido na API.")))
    return None


def answer_from_dict(data: dict[str, Any]) -> Answer:
    """Rebuild an `Answer` from the JSON of `POST /questions` or of the `answer` event."""
    return Answer(
        text=str(data.get("answer", "")),
        status=AnswerStatus(data.get("status", AnswerStatus.ABSTAINED)),
        citations=tuple(Citation(**citation) for citation in data.get("citations", [])),
        steps=tuple(Step(**step) for step in data.get("steps", [])),
        search_query=str(data.get("search_query", "")),
    )


class ApiClient:
    """Thin typed wrapper over the API; any `httpx.Client` works (tests pass a `TestClient`)."""

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    @classmethod
    def connect(cls, base_url: str = DEFAULT_API_URL) -> "ApiClient":
        return cls(httpx.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT))

    @property
    def base_url(self) -> str:
        return str(self.http.base_url).rstrip("/")

    def config(self) -> ServiceConfig:
        return ServiceConfig(**self._request("GET", "/config"))

    def documents(self) -> list[DocumentInfo]:
        return [DocumentInfo(**document) for document in self._request("GET", "/documents")]

    def upload(self, filename: str, content: bytes) -> int:
        """Ingest a file; returns the number of chunks created."""
        body = self._request("POST", "/documents/upload", files={"file": (filename, content)})
        return int(body["chunks_created"])

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        grade_documents: bool | None = None,
        validate_numbers: bool | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield the typed events of `POST /questions/stream` as they arrive."""
        payload = _question(
            question,
            {
                "top_k": top_k,
                "grade_documents": grade_documents,
                "validate_numbers": validate_numbers,
            },
        )
        try:
            with self.http.stream("POST", "/questions/stream", json=payload) as response:
                if response.is_error:
                    response.read()
                    raise ApiError(_error_detail(response))
                for raw in parse_sse(response.iter_lines()):
                    event = to_stream_event(raw)
                    if event is not None:
                        yield event
        except httpx.HTTPError as error:
            raise ApiError(self._unreachable(error)) from error

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApiError(self._unreachable(error)) from error
        if response.is_error:
            raise ApiError(_error_detail(response))
        return response.json()

    def _unreachable(self, error: httpx.HTTPError) -> str:
        if isinstance(error, httpx.TimeoutException):
            return f"A API em {self.base_url} demorou demais para responder."
        return f"Não consegui falar com a API em {self.base_url} ({type(error).__name__})."


def _question(question: str, options: dict[str, Any]) -> dict[str, Any]:
    given = {key: value for key, value in options.items() if value is not None}
    return {"question": question} | given


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ApiError(f"Evento SSE com JSON inválido: {text[:80]!r}") from error


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return f"API respondeu {response.status_code}: {detail or response.text[:200]}"
