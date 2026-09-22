# Arquitetura

## Por que um monólito modular?

O sistema ainda não possui escala ou equipes independentes que justifiquem microsserviços. Um único
processo reduz o custo operacional, enquanto módulos e interfaces preservam limites claros.

## Camadas

- `domain`: entidades e contratos que não dependem de FastAPI, SQLite ou provedores de LLM.
- `application`: casos de uso de ingestão e consulta.
- `infrastructure`: implementações substituíveis de parsing, embeddings, armazenamento e geração.
- `api` e `cli`: portas de entrada para os mesmos casos de uso.
- `bootstrap`: composição explícita das dependências.

## Decisões iniciais

1. SQLite armazena chunks e vetores, evitando um servidor local. A busca é linear e serve somente
   como baseline para corpus pequeno.
2. O embedder local usa hashing de tokens. Ele torna o pipeline reproduzível, mas não compreende
   semântica.
3. O gerador inicial é extrativo: apresenta evidências encontradas e não tenta inventar uma síntese.
4. O domínio conhece protocolos, não bibliotecas de RAG. Isso permite comparar implementações sem
   reescrever os casos de uso.
5. Docker fica fora do caminho de desenvolvimento. Poderá ser adicionado como opção de entrega.

## Próximas decisões técnicas

- modelo de embeddings e estratégia de chunking;
- busca vetorial versus híbrida;
- reranking;
- provedor de LLM e resposta fundamentada;
- dataset e métricas de avaliação;
- observabilidade, custo e latência.
