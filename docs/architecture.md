# Arquitetura

## Por que um monólito modular?

O sistema ainda não possui escala ou equipes independentes que justifiquem microsserviços. Um único
processo reduz o custo operacional, enquanto módulos e interfaces preservam limites claros.

## Camadas

- `domain`: entidades e contratos que não dependem de FastAPI, SQLite ou provedores de LLM.
- `application`: casos de uso de ingestão e consulta.
- `infrastructure`: implementações substituíveis de parsing, embeddings, armazenamento e geração.
- `evaluation`: dataset versionado, métricas e execução de avaliações do retrieval.
- `api` e `cli`: portas de entrada para os mesmos casos de uso.
- `bootstrap`: composição explícita das dependências.

## Decisões iniciais

1. SQLite armazena chunks, vetores e o índice lexical FTS5, evitando um servidor local. A busca
   vetorial percorre o corpus e serve como baseline para corpus pequeno.
2. Hashing permanece como baseline sem dependências extras. FastEmbed é opcional e executa em CPU
   via ONNX Runtime com o modelo multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Cada espaço de embeddings é identificado no SQLite para evitar comparar vetores incompatíveis.
   Artigos precisam ser ingeridos novamente quando se muda o backend.
4. A recuperação padrão é híbrida: combina ranking BM25 do FTS5 com ranking vetorial via Reciprocal
   Rank Fusion. `RAG_RETRIEVAL_MODE=vector` mantém disponível o modo vetorial isolado.
5. O gerador inicial é extrativo: apresenta evidências encontradas e não tenta inventar uma síntese.
6. O domínio conhece protocolos, não bibliotecas de RAG. Isso permite comparar implementações sem
   reescrever os casos de uso.
7. Docker fica fora do caminho de desenvolvimento. Poderá ser adicionado como opção de entrega.
8. Recall@K e MRR formam a baseline automática que protege o retrieval contra regressões.

## Próximas decisões técnicas

- reranking;
- provedor de LLM e resposta fundamentada;
- observabilidade, custo e latência.
