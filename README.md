# Scientific Paper RAG

RAG backend para consultar artigos científicos com respostas e citações rastreáveis. Funciona
localmente, sem Docker, serviços externos ou chave de API.

```text
documento -> parsing -> chunks -> embeddings -> SQLite
pergunta  -> embeddings -> retrieval -> resposta + citações
```

## Como executar

Requer Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

```bash
paper-rag ingest data/artigo.pdf
paper-rag ask "Qual é a principal contribuição do artigo?"
paper-rag serve
```

API interativa: `http://127.0.0.1:8000/docs`.

## Qualidade

```bash
ruff check .
mypy src
pytest
```

O CI executa essas verificações em cada push e pull request. A evolução planejada é substituir os
adaptadores locais por embeddings semânticos, busca híbrida, reranking e avaliação. As decisões de
projeto estão em [`docs/architecture.md`](docs/architecture.md).
