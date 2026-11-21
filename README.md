# Scientific Paper RAG

RAG backend para consultar artigos científicos com respostas e citações rastreáveis (documento e
página), construído com **LangChain** e orquestrado por um grafo **LangGraph** corretivo. O modo
extrativo roda localmente sem Docker, serviço externo ou chave de API; a síntese com LLM é opcional.

```text
ingestão: documento -> loader -> text splitter -> embeddings -> Chroma
pergunta: retrieve (BM25 + vetores, RRF) -> grade -> [rewrite -> retrieve] -> generate | abstain
```

| Etapa | Tecnologia |
|---|---|
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`, tamanho em palavras) |
| Vetores | Chroma local via `langchain-chroma` |
| Busca | Híbrida: BM25+ e vetores fundidos por Reciprocal Rank Fusion |
| Geração | `ChatOllama` (local) ou `ChatOpenAI` (Responses API), com citações validadas |
| Orquestração | LangGraph: avalia evidência, reescreve a pergunta e se abstém quando necessário |
| Entrada | CLI (Typer) e API FastAPI com streaming SSE |

Detalhes e decisões em [`docs/architecture.md`](docs/architecture.md).

## Como executar

Requer Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

```bash
paper-rag ingest data/artigo.pdf
paper-rag ask "Qual é a principal contribuição do artigo?" --show-steps
paper-rag evaluate evaluation/retrieval_baseline.json
paper-rag serve
```

API interativa: `http://127.0.0.1:8000/docs`.

| Endpoint | Função |
|---|---|
| `POST /documents` | Ingere texto (`title`, `text`, `source`) |
| `POST /documents/upload` | Ingere um arquivo `.pdf`, `.txt` ou `.md`; reenviar o mesmo nome substitui a versão anterior |
| `POST /questions` | Resposta estruturada: texto, status, citações com página e passos do grafo |
| `POST /questions/stream` | Mesmo fluxo via Server-Sent Events: `step`, `token` e `answer` (o `answer` final prevalece sobre os tokens) |

## Embeddings semânticos

Instale a opção `semantic` (o modelo é baixado na primeira execução para `.data/models`) e use o
mesmo backend ao ingerir e consultar:

```bash
python -m pip install -e ".[semantic]"
RAG_EMBEDDING_BACKEND=semantic paper-rag ingest data/artigo.pdf
RAG_EMBEDDING_BACKEND=semantic paper-rag ask "Qual é a contribuição?"
paper-rag evaluate evaluation/retrieval_paraphrases.json --embedding semantic --retrieval-mode both
```

Hashing é o backend padrão e serve de referência. `--embedding both` compara os embedders e
`--retrieval-mode both` compara busca vetorial e híbrida. Cada backend tem sua própria coleção no
Chroma (`.data/chroma`).

## Síntese com LLM (opcional)

O padrão `extractive` lista as evidências sem chamar um serviço externo. Para usar um modelo local,
instale o [Ollama](https://ollama.com), inicie o serviço e baixe um modelo pequeno:

```bash
ollama pull qwen2.5:1.5b-instruct
RAG_GENERATION_BACKEND=ollama RAG_OLLAMA_MODEL=qwen2.5:1.5b-instruct \
  paper-rag ask "Qual é a principal contribuição do artigo?" --show-steps
```

Se o serviço não iniciar pelo aplicativo, rode `ollama serve` em outro terminal.

Para usar a API da OpenAI, instale a opção `openai` e configure a chave no ambiente; o uso da API
pode gerar custos:

```bash
python -m pip install -e ".[openai]"
export OPENAI_API_KEY="sua-chave"
RAG_GENERATION_BACKEND=openai RAG_OPENAI_MODEL=gpt-6-astra \
  paper-rag ask "Qual é a principal contribuição do artigo?"
```

As respostas devem citar os trechos como `[1]`, `[2]` etc.; se o modelo não fornecer citações
válidas, o sistema mostra as evidências em vez de uma síntese sem referências.

Por padrão o fluxo é linear (busca → geração, uma chamada ao LLM). Os nós corretivos são
opcionais: `RAG_GRADE_DOCUMENTS=true` avalia cada trecho recuperado e `RAG_MAX_REWRITES=1`
reescreve a pergunta quando nenhum trecho ajuda. `RAG_GRADER_MODEL` permite usar nesses nós um
modelo maior que o gerador. Com o `qwen2.5:1.5b-instruct`, a avaliação mostrou que o avaliador não
distingue o trecho certo dos demais e troca respostas certas por abstenções. Com o
`qwen2.5:7b-instruct` só no `grade`, as respostas erradas ou inventadas caíram de 10 para 2 em 26
perguntas, ao custo de ~15 s por pergunta; veja a
[avaliação da geração](docs/architecture.md#avaliação-da-geração).

```bash
ollama pull qwen2.5:7b-instruct
RAG_GENERATION_BACKEND=ollama RAG_GRADE_DOCUMENTS=true RAG_GRADER_MODEL=qwen2.5:7b-instruct \
  paper-rag ask "Qual é o erro top-5 do ResNet-152?" --show-steps
```

## Qualidade

```bash
ruff check .
mypy src
pytest
```

O CI bloqueia regressões tanto na busca vetorial quanto na híbrida, usando perguntas diretas e
parafraseadas. A versão anterior, implementada sem frameworks, está na tag `v0.1.0-manual`.

A avaliação da **geração** chama o LLM, por isso fica fora do CI. Ela compara o fluxo linear com
o grafo corretivo em 26 perguntas sobre o artigo do ResNet (diretas, vagas, de tabela, em
português e sem resposta). O PDF não é versionado: baixe
[o artigo](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
e informe a pasta:

```bash
RAG_GENERATION_BACKEND=ollama paper-rag evaluate-generation \
  --papers-dir ~/Downloads --embedding both --runs 3
paper-rag evaluate-generation --report .data/evaluation/generation-resnet-<data>.jsonl
```

O relatório traz acertos, parciais, erradas com citação válida, abstenções corretas e falsas,
`fallback`, falsos negativos do `grade`, chamadas ao LLM, tokens e latência por pergunta, além de
uma comparação pareada com o fluxo linear. Os vereditos são lexicais (fatos esperados na
resposta); confira as respostas no JSONL gerado.
