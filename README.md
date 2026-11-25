# Scientific Paper RAG

RAG para consultar artigos científicos com respostas e citações rastreáveis (documento e
página), construído com **LangChain** e orquestrado por um grafo **LangGraph** corretivo, com uma
tela de chat em **Streamlit** sobre a API. O modo extrativo roda localmente sem Docker, serviço
externo ou chave de API; a síntese com LLM é opcional.

```text
ingestão: documento -> loader (pypdf ou Docling) -> text splitter -> embeddings -> Chroma
pergunta: retrieve (BM25 + vetores, RRF) -> [rerank] -> grade -> [rewrite -> retrieve] -> generate | abstain
```

| Etapa | Tecnologia |
|---|---|
| Leitura de PDF | `pypdf` (padrão) ou **Docling** (opcional: layout, seções e tabelas) |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`, tamanho em palavras) |
| Vetores | Chroma local via `langchain-chroma` |
| Busca | Híbrida: BM25+ e vetores fundidos por Reciprocal Rank Fusion |
| Reordenação | Cross-encoder multilíngue pré-treinado (FastEmbed, ONNX), opcional |
| Geração | `ChatOllama` (local) ou `ChatOpenAI` (Responses API), com citações validadas |
| Orquestração | LangGraph: avalia evidência, reescreve a pergunta e se abstém quando necessário |
| Entrada | CLI (Typer), API FastAPI com streaming SSE e tela Streamlit |

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
| `GET /documents` | Lista os documentos indexados: título, origem, nº de trechos e de páginas |
| `GET /config` | Configuração do serviço (só leitura): backend e modelo, avaliador, embeddings, leitura de PDF, reranker, `top_k` e chaves padrão |
| `POST /documents` | Ingere texto (`title`, `text`, `source`) |
| `POST /documents/upload` | Ingere um arquivo `.pdf`, `.txt` ou `.md`; reenviar o mesmo nome substitui a versão anterior |
| `POST /questions` | Resposta estruturada: texto, status, citações (página, seção, trecho inteiro em `text`, início em `excerpt` e, com o reranker, `search_rank` e `rerank_score`) e passos do grafo |
| `POST /questions/stream` | Mesmo fluxo via Server-Sent Events: `step`, `token` e `answer` (o `answer` final prevalece sobre os tokens) |

As duas rotas de pergunta aceitam, além de `question` e `top_k`, as chaves opcionais
`grade_documents` e `validate_numbers`, que valem só para aquela pergunta e substituem
`RAG_GRADE_DOCUMENTS` e `RAG_VALIDATE_NUMBERS`.

## Tela (Streamlit)

Chat sobre a API: barra lateral com os artigos indexados, upload, modelo (só leitura), top-k e as
chaves "avaliar trechos" e "validar números"; os passos do grafo aparecem enquanto a resposta é
gerada, e cada resposta traz o selo do status (`respondida`, `fallback`, `não verificada`,
`abstenção`, `extrativa`) e as fontes com página: o trecho inteiro, com as citadas abertas e os
números da resposta grifados. Quando o modelo esquece as citações, a resposta aparece marcada como
não verificada em vez de ser trocada pelas evidências. A tela só conversa com a API, então suba as
duas em terminais separados:

```bash
python -m pip install -e ".[ui]"
RAG_GENERATION_BACKEND=ollama paper-rag serve
```

```bash
paper-rag ui
```

A tela abre em `http://localhost:8501`. `--api-url` aponta para outra API e `--headless` não abre o
navegador. As chaves da tela valem por pergunta; o modelo e o avaliador (`RAG_GRADER_MODEL`) são
escolhidos ao subir a API. Detalhes em
[`docs/architecture.md`](docs/architecture.md#tela-streamlit).

## Leitura de PDF com Docling (opcional)

O padrão lê a camada de texto de cada página com `pypdf`: é leve e rápido, mas deixa marcas da
diagramação (`T able`, `W e`, nomes de glifos, números dos eixos das figuras). O
[Docling](https://github.com/docling-project/docling) usa modelos de layout e de tabela e devolve
o texto limpo, com a seção de cada trecho:

```bash
python -m pip install -e ".[docling]"
RAG_PDF_PARSER=docling paper-rag ingest data/artigo.pdf
RAG_PDF_PARSER=docling RAG_GENERATION_BACKEND=ollama paper-rag serve
```

- As citações e a tela mostram a seção (`p. 6 · 4.1. ImageNet Classification`), e a página
  continua exata: o trecho nunca cruza páginas.
- As tabelas viram uma linha por linha (`ResNet-152 19.38 4.49`), o formato que o `validate` usa.
- A bibliografia fica fora do índice e não ocupa vagas do top-k; `RAG_INDEX_REFERENCES=true` a
  mantém. Com `pypdf` não há seções, então ela continua indexada.
- Custo: o extra instala o PyTorch (cerca de 2 GB no ambiente), a primeira conversão baixa cerca
  de 500 MB de modelos do Hugging Face e um artigo de 9 páginas leva ~10 s, contra 0,2 s do
  `pypdf`. Não usa OCR: serve para PDFs com texto, como os de artigos.

No artigo do ResNet, a busca ficou empatada com o `pypdf`. Na geração com o
`qwen2.5:1.5b-instruct`, das 23 respostas (de 52) que mudaram, 15 melhoraram e 7 pioraram, e o
prompt ficou 17% menor. A comparação completa está em
[`docs/architecture.md`](docs/architecture.md#leitura-de-pdf-docling).

## Reranker (opcional)

A busca híbrida traz 20 candidatos e um cross-encoder pré-treinado lê cada par (pergunta, trecho)
para escolher os `top_k` que vão para o modelo. Ele roda em ONNX na CPU pelo FastEmbed, sem
PyTorch, e o modelo padrão, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (471 MB, Apache 2.0), é
multilíngue, então funciona com perguntas em português sobre artigos em inglês:

```bash
python -m pip install -e ".[rerank]"
RAG_RERANK=true paper-rag ask "Qual é a taxa de aprendizado inicial no ImageNet?" --show-steps
RAG_RERANK=true RAG_GENERATION_BACKEND=ollama paper-rag serve
```

- O reranker vira o passo `rerank` do grafo, que na tela aparece como "Reordenação" e diz de que
  posição da busca veio cada trecho. `RAG_RERANK_MODEL` troca o modelo.
- O modelo baixa na primeira pergunta para `.data/models`; cada pergunta leva cerca de 1 s a mais
  num MacBook M4.

No artigo do ResNet (26 perguntas, sem LLM), o trecho que responde à pergunta passou a aparecer no
top-4 em 78 de 84 casos, contra 69 só com a busca (dois parsers × dois embeddings), e em primeiro
lugar em 54, contra 36. Um modelo maior (`jina-reranker-v2`) acertou 2 a mais, mas leva o dobro do
tempo e proíbe uso comercial. A comparação está em
[`docs/architecture.md`](docs/architecture.md#reranker-cross-encoder).

Na geração com o `qwen2.5:1.5b-instruct`, a resposta passou a citar o trecho certo em 57 de 84
casos, contra 45. Das 79 respostas (de 104) que mudaram, 22 melhoraram e 15 pioraram; boa parte
das perdas é o modelo pequeno lendo a coluna errada da tabela mesmo com o trecho certo.

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

As respostas devem citar os trechos como `[1]`, `[2]` etc. O nó `validate` confere as citações e
os números. Se uma citação não existir, ou se um número da resposta não estiver na pergunta nem
nos trechos citados, o sistema mostra as evidências em vez da síntese. A exceção é quando o
número está na linha certa de outro trecho recuperado: aí a citação é corrigida para esse trecho
(`RAG_VALIDATE_NUMBERS=false` desliga a checagem de números).

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
