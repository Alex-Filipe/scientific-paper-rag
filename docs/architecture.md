# Arquitetura

## Visão geral

Monólito modular em Python. O pipeline usa os pacotes mantidos do ecossistema LangChain
(`langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-ollama`,
`langchain-openai`) e orquestra a pergunta com um grafo **LangGraph**.

```text
ingestão:  arquivo -> PaperLoader -> RecursiveCharacterTextSplitter -> embeddings -> Chroma
pergunta:  LangGraph (retrieve -> grade -> [rewrite -> retrieve] -> generate | abstain)
```

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `documents.py` | `PaperLoader` (PDF por página com `pypdf`, TXT/MD) e chunking por palavras |
| `embeddings.py` | `HashingEmbeddings` (baseline sem download) e `FastEmbedEmbeddings` (semântico, CPU) |
| `vectorstore.py` | `ChunkIndex`: coleção Chroma persistente por espaço de embeddings |
| `retrieval.py` | `BM25Retriever`, `RRFRetriever` e `HybridSearch` (BM25 + vetores via RRF) |
| `generation.py` | Prompts, fábrica de modelos de chat, validação de citações e modo extrativo |
| `graph.py` | Grafo LangGraph corretivo |
| `service.py` | `RagService`: ingestão, busca, `ask` e `stream`, usado por CLI, API e avaliação |
| `models.py` | Resposta estruturada: texto, status, citações (com página) e passos do grafo |
| `api.py` / `cli.py` | Portas de entrada; a API expõe streaming por Server-Sent Events |
| `evaluation/` | Dataset versionado, Recall@K, MRR e gates do CI |

## Grafo LangGraph

```text
START -> retrieve -> grade --(há trechos úteis)--> generate -> END
                       |--(nenhum útil e ainda pode reescrever)--> rewrite -> retrieve
                       \--(nenhum útil)--> abstain -> END
```

- **retrieve:** busca híbrida com `RAG_CANDIDATE_K` candidatos por ranker e corte final em `top_k`.
- **grade:** o LLM indica quais trechos ajudam (`1, 3` ou `nenhum`). Resposta ilegível mantém
  todos os trechos. Desligado sem LLM ou com `RAG_GRADE_DOCUMENTS=false`.
- **rewrite:** o LLM reescreve a pergunta como consulta técnica. Limitado por `RAG_MAX_REWRITES`.
- **generate:** sem LLM, lista as evidências (`extractive`). Com LLM, exige citações `[n]`
  válidas; se faltarem ou apontarem para trechos inexistentes, mostra as evidências (`fallback`).
- **abstain:** responde que não há evidência suficiente.

Cada nó registra um `Step`. Em `POST /questions/stream` os passos chegam como eventos `step`, os
tokens da geração como `token` e a resposta final como `answer`.

## Decisões

1. **Chroma local** guarda vetores e textos sem servidor nem Docker. Cada espaço de embeddings tem
   sua própria coleção, evitando comparar vetores incompatíveis; trocar de backend exige reingerir.
2. **Reingestão sem órfãos:** antes de adicionar um documento, todos os chunks dele são removidos.
3. **Busca híbrida por padrão:** BM25+ (sem acentos e sem stop words, só chunks com termos em
   comum) e busca vetorial fundidos por Reciprocal Rank Fusion. Cada ranker contribui
   `RAG_CANDIDATE_K` candidatos antes do corte, para que a fusão tenha o que reordenar.
4. **Sem `langchain-community` e sem `langchain-classic`:** o `langchain-community` foi
   descontinuado em 2026 e o `langchain-classic` é legado. Loader, BM25, RRF e FastEmbed são
   implementações pequenas sobre as interfaces de `langchain-core` (`BaseLoader`,
   `BaseRetriever`, `Embeddings`), como o próprio LangChain recomenda para componentes simples.
5. **BM25 em memória:** o índice lexical é reconstruído a partir do Chroma em cada busca. É
   suficiente para dezenas de artigos; para corpus grande, trocar por um índice persistente.
6. **Geração segura:** os prompts tratam as evidências como dado não confiável, exigem citações e
   definem uma frase exata de abstenção. O modo extrativo continua sendo o padrão.
7. **Avaliação no CI:** Recall@K e MRR bloqueiam regressões na busca vetorial e híbrida, com
   perguntas diretas e parafraseadas.

## Comparação com a versão manual (`v0.1.0-manual`)

Mesmos datasets, `top_k=3`:

| Dataset | Configuração | Manual (Recall@3 / MRR) | LangChain (Recall@3 / MRR) |
|---|---|---|---|
| baseline | hashing/vector | 1,000 / 0,900 | 1,000 / 0,900 |
| baseline | hashing/hybrid | 1,000 / 1,000 | 1,000 / 1,000 |
| baseline | semantic/vector e hybrid | 1,000 / 1,000 | 1,000 / 1,000 |
| paráfrases | hashing/vector e hybrid | 0,800 / 0,467 | 0,800 / 0,567 |
| paráfrases | semantic/vector e hybrid | 1,000 / 1,000 | 1,000 / 1,000 |

Os datasets são pequenos (5 documentos curtos), então servem para detectar regressões, não para
medir qualidade em artigos reais.

## Próximas melhorias planejadas

1. Dataset de avaliação com artigos reais e avaliação da geração (fidelidade, suporte das
   citações, abstenção).
2. Parser que preserve a estrutura de artigos (duas colunas, seções), por exemplo Docling.
3. Reranker entre a busca e o grafo.
4. Tela consumindo a API (resposta estruturada e SSE já disponíveis).
5. Observabilidade de latência, erros e tokens por consulta.
