# Arquitetura

## Por que um monólito modular?

O sistema ainda não possui escala ou equipes independentes que justifiquem microsserviços. Um único
processo reduz o custo operacional, enquanto módulos e interfaces preservam limites claros.

## Camadas

- `core`: modelos, contratos e fluxo principal — ingestão, busca e perguntas.
- `adapters`: integrações concretas de parsing, embeddings, armazenamento e geração.
- `evaluation`: dataset versionado, métricas e execução de avaliações do retrieval.
- `api` e `cli`: portas de entrada para os mesmos casos de uso.
- `bootstrap`: composição explícita das dependências.

Para aprender o fluxo, comece por `core/ingest.py` (documento até banco) e depois siga
`core/ask.py` → `core/retrieval.py` → `adapters/generation.py` (pergunta até resposta).

## Decisões iniciais

1. SQLite armazena chunks, vetores e o índice lexical FTS5, evitando um servidor local. A busca
   vetorial percorre o corpus e serve como baseline para corpus pequeno.
2. Hashing permanece como baseline sem dependências extras. FastEmbed é opcional e executa em CPU
   via ONNX Runtime com o modelo multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Cada espaço de embeddings é identificado no SQLite para evitar comparar vetores incompatíveis.
   Artigos precisam ser ingeridos novamente quando se muda o backend.
4. A recuperação padrão é híbrida: combina ranking BM25 do FTS5 com ranking vetorial via Reciprocal
   Rank Fusion. `RAG_RETRIEVAL_MODE=vector` mantém disponível o modo vetorial isolado.
5. O gerador extrativo é o padrão local. Um gerador OpenAI opcional usa a Responses API para sintetizar
   respostas apenas com os trechos recuperados, exige citações válidas e abstém quando não há evidência.
   A chave permanece em `OPENAI_API_KEY`; nunca é salva nas configurações nem no banco.
6. O core conhece contratos, não bibliotecas de RAG. Os adapters implementam esses contratos,
   permitindo trocar tecnologias sem reescrever o fluxo principal.
7. Docker fica fora do caminho de desenvolvimento. Poderá ser adicionado como opção de entrega.
8. Recall@K e MRR formam a baseline automática que protege o retrieval contra regressões.

## Próximas melhorias planejadas

Estas melhorias ainda não estão implementadas; a ordem pode mudar conforme os testes e o uso do
projeto:

1. Adicionar um gerador local via Ollama, começando com um modelo Qwen pequeno e quantizado. Escolher
   o tamanho após verificar a memória disponível na máquina; não requer Docker.
2. Criar avaliação de geração com respostas de referência, medindo fidelidade às evidências,
   relevância, suporte das citações e comportamento de abstinência.
3. Avaliar um reranker para reordenar os trechos recuperados antes de enviá-los ao gerador.
4. Testar robustez contra prompt injection em documentos e, se houver múltiplos usuários, filtrar
   documentos por permissões antes da recuperação.
5. Adicionar observabilidade de latência, erros, tokens e custo por consulta.
