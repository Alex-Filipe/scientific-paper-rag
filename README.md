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
paper-rag evaluate evaluation/retrieval_baseline.json
paper-rag serve
```

API interativa: `http://127.0.0.1:8000/docs`.

Para usar embeddings semânticos locais, instale a opção `semantic` (o modelo é baixado na primeira
execução) e selecione o mesmo backend ao ingerir e consultar:

```bash
python -m pip install -e ".[semantic]"
RAG_EMBEDDING_BACKEND=semantic paper-rag ingest data/artigo.pdf
RAG_EMBEDDING_BACKEND=semantic paper-rag ask "Qual é a contribuição?"
paper-rag evaluate evaluation/retrieval_paraphrases.json --embedding both
```

Hashing permanece como backend padrão e serve de referência. `--embedding both` mostra as métricas
de cada método e a diferença entre eles. O modelo fica em `.data/models`; os vetores de cada método
ficam isolados no SQLite.

## Qualidade

```bash
ruff check .
mypy src
pytest
```

O CI verifica o baseline lexical e bloqueia regressões com perguntas parafraseadas no backend
semântico. As decisões estão em
[`docs/architecture.md`](docs/architecture.md).
