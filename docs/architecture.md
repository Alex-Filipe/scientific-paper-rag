# Arquitetura

## Visão geral

Monólito modular em Python. O pipeline usa os pacotes mantidos do ecossistema LangChain
(`langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-ollama`,
`langchain-openai`) e orquestra a pergunta com um grafo **LangGraph**.

```text
ingestão:  arquivo -> PaperLoader (pypdf | Docling) -> RecursiveCharacterTextSplitter -> embeddings -> Chroma
pergunta:  LangGraph (retrieve -> [rerank] -> grade -> [rewrite -> retrieve] -> generate -> validate | abstain)
```

Por padrão `grade` e `rewrite` ficam desligados e o fluxo é linear (`retrieve -> generate`); a
[avaliação da geração](#avaliação-da-geração) explica por quê. O `rerank` (cross-encoder) é
opcional e só entra no grafo quando ligado ([Reranker](#reranker-cross-encoder)).

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `documents.py` | `PaperLoader` (PDF por página com `pypdf` ou Docling, TXT/MD), limpeza de ligaduras/hifenização e chunking por palavras, com a seção de cada trecho e a bibliografia fora |
| `docling_parser.py` | Leitura com Docling (extra `docling`): itens em ordem de leitura → uma página por documento, com os inícios de seção, tabelas linha a linha e itens cortados na troca de página |
| `embeddings.py` | `HashingEmbeddings` (baseline sem download) e `FastEmbedEmbeddings` (semântico, CPU) |
| `vectorstore.py` | `ChunkIndex`: coleção Chroma persistente por espaço de embeddings |
| `retrieval.py` | `BM25Retriever`, `RRFRetriever` e `HybridSearch` (BM25 + vetores via RRF) |
| `rerank.py` | `CrossEncoderReranker`: reordena os candidatos da busca com um cross-encoder do FastEmbed (ONNX, extra `rerank`) |
| `generation.py` | Prompts, fábrica de modelos de chat, validação de citações e modo extrativo |
| `numbers.py` | Leitura de números (`4,49`/`4.49`, `50k`) usada pelo `validate` e pela tela, sem LangChain |
| `graph.py` | Grafo LangGraph corretivo |
| `service.py` | `RagService`: ingestão, busca, `ask` e `stream`, usado por CLI, API e avaliação |
| `models.py` | Resposta estruturada: texto, status, citações (com página) e passos do grafo |
| `api.py` / `cli.py` | Portas de entrada; a API expõe streaming por Server-Sent Events |
| `ui/` | Tela Streamlit: `client.py` (cliente `httpx` e parser do SSE), `view.py` (lógica de apresentação sem Streamlit) e `app.py` (a tela) |
| `evaluation/` | Busca: datasets, Recall@K, MRR e gates do CI. Geração: dataset com artigo real, vereditos, auditoria do `grade` e relatório (só CLI) |

## Grafo LangGraph

```text
START -> retrieve -> [rerank] -> grade --(há trechos úteis)--> generate --(resposta do LLM)--> validate -> END
                                   |                                \--(extrativo ou abstenção)--> END
                                   |--(nenhum útil e ainda pode reescrever)--> rewrite -> retrieve
                                   \--(nenhum útil)--> abstain -> END
```

- **retrieve:** busca híbrida com `RAG_CANDIDATE_K` candidatos por ranker e corte final em `top_k`.
  Com o reranker ligado, devolve os `RAG_CANDIDATE_K` primeiros da fusão, sem cortar.
- **rerank:** só existe com `RAG_RERANK=true`. Um cross-encoder dá uma nota a cada par (pergunta
  original, trecho) e mantém os `top_k` de nota mais alta. O passo diz de que posições da busca
  vieram os trechos que ficaram; cada citação guarda `search_rank` e `rerank_score`.
- **grade:** o LLM julga cada trecho separadamente (`sim`/`não`, uma chamada por trecho). Resposta
  ilegível mantém o trecho. Opcional: `RAG_GRADE_DOCUMENTS=true` (padrão `false`; sem LLM é sempre
  pulado), ou `grade_documents` numa pergunta da API.
- **rewrite:** o LLM reescreve a pergunta como consulta técnica. Limitado por `RAG_MAX_REWRITES`
  (padrão `0`). `grade` e `rewrite` podem usar outro modelo com `RAG_GRADER_MODEL`.
  Se a consulta reescrita não tiver palavras (por exemplo, só um número), a pergunta original é
  mantida.
- **generate:** sem LLM, lista as evidências (`extractive`). Com LLM, gera um rascunho; se o
  modelo se abstiver, termina aí.
- **validate:** aceita o rascunho só se as citações `[n]` existirem e se todo número da resposta
  aparecer na pergunta ou nos trechos citados (`RAG_VALIDATE_NUMBERS`, padrão `true`). A
  comparação aceita formatos diferentes (`4,49`/`4.49`, `50.000`/`50k`, `1,7 milhão`/`1.7M`) e
  ignora as referências do artigo (`[10, 41]`). Se a resposta cita um único trecho e os números
  estão em outro trecho recuperado, numa linha que também tem uma palavra da pergunta (como em
  `34 layers 28.54 25.03`), a citação é corrigida para esse trecho. Nos demais casos, mostra as
  evidências (`fallback`). `validate_numbers` numa pergunta da API substitui `RAG_VALIDATE_NUMBERS`.
- **abstain:** responde que não há evidência suficiente.

Cada nó registra um `Step`. Em `POST /questions/stream` os passos chegam como eventos `step`, os
tokens da geração como `token` e a resposta final como `answer`. Os tokens saem antes da validação
das citações: o cliente deve substituir o texto parcial pelo do evento `answer`, que pode ser
`fallback` (evidências) ou `abstained`.

## Decisões

1. **Chroma local** guarda vetores e textos sem servidor nem Docker. Cada espaço de embeddings tem
   sua própria coleção, evitando comparar vetores incompatíveis; trocar de backend exige reingerir.
2. **Reingestão sem órfãos nem duplicatas:** antes de adicionar um documento, todos os chunks dele
   são removidos. O documento é identificado por título + origem: na CLI, o caminho absoluto; no
   upload da API, o nome do arquivo (reenviar o mesmo nome substitui a versão anterior). O mesmo
   artigo ingerido pela CLI e pela API vira dois documentos.
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
8. **Avaliação trecho a trecho:** com o `qwen2.5:1.5b`, pedir a lista dos trechos úteis gerava
   números inexistentes (`3, 5, 6, 7, 8` para 4 trechos), e artigos têm suas próprias referências
   entre colchetes (`[40]`). Um `sim`/`não` por trecho é o formato que o modelo pequeno segue.
9. **Fluxo linear por padrão:** com o `qwen2.5:1.5b-instruct`, o `grade` manteve 27% dos
   trechos-gabarito e 27% dos demais, ou seja, não filtrou relevância; só se absteve mais. O
   `rewrite` não trouxe ganho e somou 2,5 a 3,4 chamadas por pergunta. Com o
   `qwen2.5:7b-instruct` só no `grade` (`RAG_GRADER_MODEL`), os erros caem de 10 para 2 em 26
   perguntas, mas cada pergunta leva ~15 s; fica como opção, não como padrão.
10. **Avaliação da geração fora do CI:** depende do Ollama e leva cerca de 4 s por pergunta. Os
    testes cobrem o avaliador com um LLM falso.
11. **Validação como nó próprio:** a validação saiu de dentro do `generate` e virou o nó
    `validate`, com uma checagem a mais: números sem suporte nos trechos citados, e a correção da
    citação quando o número está na linha certa de outro trecho. São regras lexicais, sem LLM.
12. **Docling opcional, `pypdf` como padrão:** o Docling limpa o texto e dá as seções, mas
    instala o PyTorch, baixa modelos e leva ~10 s por artigo; na busca empatou com o `pypdf`. O CI
    não instala o extra: a lógica de páginas e seções é testada com itens falsos do Docling, e a
    conversão real é um teste opcional (`PAPER_RAG_DOCLING_TESTS=1`).
13. **Bibliografia fora do índice (com seções):** as referências não respondem perguntas sobre o
    conteúdo e tiravam vagas do top-k. Só o Docling sabe onde ela começa; `RAG_INDEX_REFERENCES`
    a mantém.
14. **Reranker opcional, como nó próprio:** o cross-encoder pré-treinado melhora a busca em todas
    as combinações de parser e embedding, mas baixa um modelo de 471 MB e soma cerca de 1 s por
    pergunta. Fica desligado por padrão, como o embedding semântico e o Docling, e vira o nó
    `rerank` quando ligado, para aparecer como passo na tela. Um LLM como reranker e um modelo
    treinado por nós foram descartados (ver [Reranker](#reranker-cross-encoder)).

## Teste com artigo real

ResNet (He et al., CVPR 2016, 9 páginas em duas colunas), `qwen2.5:1.5b-instruct` no Ollama,
`top_k=4`, 7 perguntas (diretas, vaga, sem resposta, numérica e duas em português):

- **Extração (`pypdf`):** a ordem das colunas sai correta e as tabelas viram linhas legíveis
  (`ResNet-152 19.38 4.49`). Ligaduras (`ﬁ`) e hifenização de fim de linha agora são limpas. Ainda
  ficam espaços espúrios (`W e`, `T able`, `CIF AR-10`), números dos eixos das figuras e expoentes
  perdidos (`1.8×109`); o parser opcional com Docling resolve os dois primeiros
  ([Leitura de PDF](#leitura-de-pdf-docling)).
- **Páginas citadas:** corretas (conferidas com `pdftotext`), porque os chunks nunca cruzam páginas.
- **Latência:** 2,5 a 5,1 s por pergunta na CLI (média 4,0 s, MacBook M4), incluindo até 9
  chamadas ao LLM quando há reescrita.
- **Grafo:** o avaliador de 1,5B quase não discrimina. Com o prompt "ajuda a responder", descarta
  trechos bons (falsas abstenções, sobretudo com pergunta em português e artigo em inglês). Com
  um prompt tolerante, aprova tudo e o fluxo vira linear. A reescrita costuma produzir uma
  resposta inventada em vez de uma consulta. A [avaliação da geração](#avaliação-da-geração)
  mediu isso com dataset.
- **Geração:** a validação garante citações existentes, não respostas fiéis. O modelo errou
  perguntas diretas ("vanishing gradients" em vez de degradação) citando trechos válidos.

## Avaliação da geração

Pergunta: o grafo corretivo dá respostas melhores que o fluxo linear com o mesmo modelo?

**Método.** `paper-rag evaluate-generation` com `evaluation/generation_resnet.json`: 26 perguntas
sobre o ResNet (6 diretas, 4 vagas, 5 de tabela, 6 em português, 5 sem resposta no artigo). Cada
pergunta traz a resposta de referência, os fatos esperados (com variantes como `4,49`/`4.49`) e
citações literais com página. Um trecho é **gabarito** quando contém uma dessas citações. O
avaliador registra, por pergunta, o veredito (correta, parcial, errada, abstenção, `fallback`),
se a resposta citou um trecho-gabarito, o que o `grade` manteve ou descartou, as chamadas ao LLM
por nó, os tokens e a latência. Configurações: linear (`RAG_GRADE_DOCUMENTS=false`,
`RAG_MAX_REWRITES=0`), só grade e grade + rewrite; `top_k=4`; `qwen2.5:1.5b-instruct`; MacBook M4.

**Cache do Ollama.** O Ollama 0.34 guarda em cache vários prompts: um prompt de 1.600 tokens
repetido cai de 2 s para 0,05 s. Como as configurações compartilham prompts, a latência dependia
da ordem das rodadas. O avaliador descarrega e recarrega o modelo antes de cada pergunta (cerca de
0,7 s, fora do tempo medido). Com o cache limpo, 3 rodadas deram respostas idênticas; com o cache
ativo, 2 de 156 respostas mudaram. A variação entre rodadas vista no teste anterior vinha do cache,
não da amostragem.

**Resultados** (vereditos revisados à mão; o juiz lexical concordou em 48 de 64 respostas
distintas, sendo generoso nas vagas e rígido em traduções). As tabelas de configurações foram
medidas antes do nó `validate`; o efeito dele está no fim da seção.

| | linear | grade | grade + rewrite |
|---|---|---|---|
| **hashing** | | | |
| Corretas / parciais (de 21) | 8 / 6 | 6 / 2 | 6 / 2 |
| Erradas com citação válida | 6 | 4 | 5 |
| Abstenções falsas | 0 | 9 | 8 |
| `fallback` | 1 | 0 | 0 |
| Sem resposta: abstém / inventa (de 5) | 1 / 4 | 4 / 1 | 4 / 1 |
| Chamadas ao LLM por pergunta | 1,0 | 4,5 | 7,0 |
| Latência média | 3,3 s | 4,1 s | 5,2 s |
| **semantic** | | | |
| Corretas / parciais (de 21) | 9 / 4 | 4 / 2 | 5 / 2 |
| Erradas com citação válida | 7 | 2 | 3 |
| Abstenções falsas | 0 | 13 | 11 |
| `fallback` | 1 | 0 | 0 |
| Sem resposta: abstém / inventa (de 5) | 1 / 4 | 4 / 1 | 4 / 1 |
| Chamadas ao LLM por pergunta | 1,0 | 4,3 | 7,7 |
| Latência média | 4,1 s | 4,4 s | 5,8 s |

- **O `grade` não filtra relevância com o 1,5B:** manteve 27% dos trechos-gabarito e 27% dos
  demais trechos das perguntas respondíveis (26% e 26% com semantic). Nas perguntas sem resposta
  manteve 5 a 10%, por isso se abstém mais. O efeito é trocar respostas erradas por abstenções
  falsas, e zerou as perguntas de tabela.
- **O `rewrite` não trouxe ganho:** os mesmos acertos do grade (um a mais com semantic, um erro a
  mais com hashing) por 2,5 a 3,4 chamadas a mais por pergunta.
- **O gargalo é o gerador:** o trecho-gabarito está no top-4 em 17 de 21 perguntas nos dois
  backends, mas o modelo acerta 8 ou 9. Ele pega a linha errada da tabela, cita referências do
  artigo como se fossem fatos e responde mal em português (1 de 6).
- **Respostas longas:** o pico de 13,5 s não veio do `grade`. O gerador divagou até o limite de
  512 tokens numa pergunta vaga.

**Avaliador maior.** Num teste só do avaliador (os 4 trechos da primeira busca de cada pergunta,
hashing), o prompt não salvou o 1,5B: com o prompt atual ele mantém 27% dos trechos-gabarito e
27% dos demais; com um prompt tolerante ("palavras-chave ou significado relacionado"), 95% e 85%,
o que equivale ao fluxo linear. O `qwen2.5:7b-instruct` separa de fato: mantém 64% dos
trechos-gabarito, 16% dos demais e nenhum trecho das perguntas sem resposta, a 2,7 s por trecho.
O prompt tolerante não melhorou o 7B. A rodada completa com o 7B só no `grade`
(`RAG_GRADER_MODEL`, gerador 1,5B, hashing, uma rodada):

| | linear | grade (1,5B) | grade (7B) |
|---|---|---|---|
| Corretas / parciais (de 21) | 8 / 6 | 6 / 2 | 9 / 3 |
| Erradas com citação válida | 6 | 4 | 2 |
| Abstenções falsas | 0 | 9 | 6 |
| `fallback` | 1 | 0 | 1 |
| Sem resposta: abstém / inventa (de 5) | 1 / 4 | 4 / 1 | 5 / 0 |
| Corretas + abstenções corretas (de 26) | 9 | 10 | 14 |
| Latência média | 3,3 s | 4,1 s | 14,7 s |

Com o 7B como avaliador, as respostas erradas ou inventadas caem de 10 para 2 e todas as perguntas
sem resposta viram abstenção. Em troca, ele ainda descarta 36% dos trechos-gabarito, sobretudo de
tabela e das perguntas em português (6 abstenções falsas), e a latência fica 4,5 vezes maior.
Um acerto a mais ou a menos não é significativo com 21 perguntas; a queda dos erros é.

Decisão: fluxo linear por padrão, porque roda com o modelo pequeno e responde em ~3 s. Quem
prefere não receber respostas inventadas pode usar `RAG_GRADE_DOCUMENTS=true` com
`RAG_GRADER_MODEL=qwen2.5:7b-instruct` (~15 s por pergunta e um modelo de 4,7 GB). O `rewrite`
fica desligado nos dois casos. O prompt do avaliador foi escolhido com as mesmas perguntas do
teste, então os números do 7B são um pouco otimistas.

**Validação de números.** Com o fluxo linear (cache limpo, uma rodada, mesmos rascunhos do
gerador), corretas / erradas + inventadas, em 21 respondíveis e 5 sem resposta:

| | sem validação | só checagem | checagem + correção da citação |
|---|---|---|---|
| hashing | 8 / 10 | 6 / 8 | 8 / 9 |
| semantic | 9 / 11 | 6 / 7 | 8 / 9 |

Só a checagem trocou 11 de 52 respostas por evidências: 6 erradas ou inventadas e 5 corretas. Em
4 dessas corretas o número estava certo, mas citado no trecho errado (o 28,54% citado como [1]
quando estava em [2]); a quinta tinha um detalhe inventado. A correção da citação recupera essas
4, agora citando o trecho certo, e readmite 3 erradas cujo número aparece numa linha com uma
palavra da pergunta (o 73,8% da tabela do VOC, o 3,57% e um "Fig. 7"). Uma regra mais simples,
que corrige a citação para qualquer trecho com o número, recuperou 5 e readmitiu 4, e no hashing
voltou exatamente ao resultado sem validação. Somando os dois backends, o padrão (checagem +
correção) fica com 1 correta a menos e 3 erradas a menos que sem validação, e as respostas
corretas passam a citar o trecho certo. A regra foi escolhida olhando estes mesmos 11 casos, então
o ganho é otimista. A checagem não pega números copiados da linha errada do trecho citado (o
5,71% da Tabela 3 no lugar do 4,49%), nem respostas sem números.

## Tela (Streamlit)

`paper-rag ui` roda `streamlit run src/paper_rag/ui/app.py` com o endereço da API em
`PAPER_RAG_API_URL` (padrão `http://127.0.0.1:8000`). A tela não importa o serviço: tudo passa
pela API, como faria um front-end separado.

```text
barra lateral: GET /config, GET /documents, POST /documents/upload, top-k e chaves por pergunta
chat:          POST /questions/stream -> step (st.status) -> token (st.write_stream) -> answer
```

- **SSE por POST:** o `EventSource` do navegador só faz GET, então `client.py` lê o
  `text/event-stream` da resposta `httpx` linha a linha (`parse_sse`, regras do WHATWG: `event:`,
  `data:` em várias linhas, comentários, evento incompleto descartado) e entrega eventos tipados
  (`StepEvent`, `TokenEvent`, `AnswerEvent`, `ErrorEvent`).
- **Tokens antes da validação:** os tokens do `generate` aparecem enquanto chegam, mas o texto
  final vem do evento `answer`, que pode ter a citação corrigida ou ter virado `fallback`. A
  correção de citação aparece em destaque. No `fallback`, a tela separa os dois motivos do
  `validate`:
  - **Sem citação válida:** o rascunho ocupa o lugar da resposta, com o selo `não verificada` e um
    aviso de que nem as fontes nem os números foram conferidos. A lista de evidências que a API
    manda nesse caso repete as fontes. No teste real, o rascunho era uma tradução fiel do trecho
    [3] que só esqueceu o `[3]`, enquanto 3 das 4 evidências eram referências bibliográficas.
  - **Número sem suporte:** a tela mostra as evidências e guarda o rascunho, que tem um número
    inventado, num expander fechado com o motivo.

  A regra do `validate` não mudou; a resposta continua com status `fallback`.
- **Chaves por pergunta:** "avaliar trechos" e "validar números" vão no corpo da pergunta
  (`grade_documents`, `validate_numbers`) e entram no estado do grafo; o modelo e o avaliador
  (`RAG_GRADER_MODEL`) continuam fixos ao subir a API, e a tela só os mostra. Com o grade ligado e
  sem avaliador separado, a tela avisa que o 1,5B troca acertos por abstenções.
- **Fontes:** cada trecho recuperado vira um `st.expander` com título e página, aberto quando foi
  citado, com o **trecho inteiro** (campo `text` da citação; o `excerpt` tem só os 280 primeiros
  caracteres e muitas vezes não chega ao número citado), não o PDF. Numa resposta aceita, os
  números que ela acrescenta à pergunta, os mesmos que o `validate` procurou, aparecem grifados
  em todos os trechos (`numbers.claimed_readings`, com as mesmas leituras `4,49`/`4.49`).
  Abaixo de cada trecho vem a posição na busca e a nota RRF; com o reranker, a posição depois de
  reordenar, a posição na busca e a nota do cross-encoder ("posição 2 após reordenar (4ª na
  busca) · nota 4,26"). Com o reranker, a barra lateral mostra o modelo e o passo "Reordenação"
  entra na caixa dos passos.
- **Compatibilidade:** o cliente ignora campos que não conhece e transforma payloads inesperados
  em `ApiError`, para que uma API mais nova não derrube a tela com um traceback.
- **Recarga de módulos:** quando um arquivo muda, o Streamlit descarrega os módulos da pasta da
  tela (`paper_rag.ui.*`), mas não os outros módulos do pacote. Um `ApiClient` guardado na sessão
  passaria a emitir eventos das classes antigas, e os `isinstance` das novas descartariam tudo (a
  resposta sumia em silêncio). Por isso a sessão guarda só o `httpx.Client` e o `ApiClient` é
  recriado a cada execução. Os turnos da conversa são recriados com a classe `Turn` atual
  (`view.revive`), e o status é comparado por valor, não por identidade. Mudanças fora de `ui/`
  (por exemplo, em `models.py`) só valem depois de reiniciar a tela.
- **Testes sem rede nem LLM:** o cliente roda contra o `TestClient` da própria API (que é um
  `httpx.Client`) e contra `httpx.MockTransport`; a tela roda pelo `AppTest` do Streamlit, com o
  modelo falso dos testes do grafo. O coverage mede `src/paper_rag` por pasta, porque o Streamlit
  executa o script como módulo `app`, fora do pacote.

**Teste real** (2026-11-22, `qwen2.5:1.5b-instruct`, hashing, índice novo): upload do PDF do
ResNet pela tela (42 trechos, 9 páginas, como na CLI); perguntas com o passo a passo aparecendo
durante a execução e os tokens chegando antes do selo; uma pergunta com "avaliar trechos" ligado
(3 de 4 trechos mantidos) e uma sem resposta no artigo (BLEU no WMT14), em que o modelo se absteve
com palavras próprias e citou `[1-3]`, e o `validate` a trocou por `fallback`. Na pergunta sobre o
erro top-1 da plain-34, a resposta citava 28,54% em `[1]` e o `excerpt` mostrava só a Tabela 1 do
começo do trecho; com o trecho inteiro, a linha `34 layers 28.54 25.03` da Tabela 2 aparece com o
número grifado. Na pergunta sobre o top-5 do ResNet-152 com modelo único, o 5,71% grifado em três
linhas mostra que a resposta veio da Tabela 3 (10-crop), não da linha de modelo único.

## Leitura de PDF (Docling)

`RAG_PDF_PARSER=docling` (extra `docling`, Docling 2.151) lê o PDF com o modelo de layout e o
TableFormer, sem OCR. `docling_parser.py` percorre os itens do `DoclingDocument` na ordem de
leitura:

- **Ignorados:** figuras (e o texto de dentro delas, como os números dos eixos), cabeçalhos e
  rodapés de página.
- **Mantidos:** parágrafos, títulos de seção, legendas, notas de rodapé, fórmulas (o texto do PDF,
  já que o reconhecimento de fórmulas fica desligado) e itens de lista com o marcador, para que a
  bibliografia continue `[16] S. Ioffe…`.
- **Tabelas:** uma linha por linha da tabela, com as células separadas por espaço e sem as
  repetições de células mescladas (`Maxout [9] 9.38`). É o formato que o `validate` usa para
  corrigir citações (`34 layers 28.54 25.03`).
- **Página exata:** o Docling junta num item só o parágrafo que continua na outra coluna ou na
  página seguinte. As partes são separadas na troca de página pelos `charspan` da proveniência.
  Esses deslocamentos ignoram o hífen de quebra de linha que o Docling remove, então o corte vai
  para o espaço mais próximo.
- **Hífens:** o Docling tira o hífen de toda palavra quebrada no fim da linha (`34layer`,
  `highlevel`). Um número colado numa palavra de 4 ou mais letras volta a ter hífen (`34-layer`);
  as palavras ficam como estão.
- **Seções:** cada página vira um documento com a lista de onde cada seção começa. O splitter
  corta a página como faz com o `pypdf`, e cada trecho recebe a seção que ocupa a maior parte dele.
  Um documento por seção foi testado e descartado: picava os trechos (alguns com 2 palavras) e o
  gabarito no top-10 com semantic caiu de 21 para 17.
- **Bibliografia:** a seção de referências (`References`, `Bibliography`, `Referências`) sai do
  texto antes do corte, salvo com `RAG_INDEX_REFERENCES=true`. Cada trecho guarda `page_count`,
  então o ResNet continua listado com 9 páginas, embora a 9 só tenha referências.

**Comparação no ResNet** (sem LLM, script avulso fora do repositório, MacBook M4). Os gates de
busca do CI usam textos sintéticos, não PDF, e não mudam.

| Texto | `pypdf` | Docling |
|---|---|---|
| Tempo de leitura | 0,2 s | 10 a 14 s (inclui carregar os modelos) |
| Palavras / trechos | 7.322 / 42 | 6.570 / 34 |
| Espaços espúrios (`T able`, `W e`, `Y . Bengio`, `CIF AR-10`) | 161 | 0 |
| Nomes de glifos (`rel/uni0075`) | 12 | 0 |
| Linhas só de números (eixos das figuras) | 9 | 0 |
| Expoentes perdidos (`1.8×109`) | 6 | 6 |
| Trechos de bibliografia | 6 | 0 |
| Citações do dataset de geração encontradas | 36 de 36 | 36 de 36 |

| Busca (21 perguntas com resposta) | `pypdf` | Docling |
|---|---|---|
| Gabarito no top-4, hashing / semantic | 17 / 17 | 17 / 18 |
| MRR no top-4, hashing / semantic | 0,58 / 0,64 | 0,56 / 0,58 |
| Gabarito no top-10, hashing / semantic | 19 / 21 | 19 / 20 |
| Vagas do top-4 com bibliografia (de 104), hashing / semantic | 7 / 1 | 0 / 0 |

Sem tirar a bibliografia, o Docling ficava com 16 / 18 no top-4. As diferenças são de 1 pergunta
em 21 e mudam de lado conforme o backend, então a busca empatou; o ganho está no texto que o
modelo e a tela recebem e nas seções.

**Geração** (`paper-rag evaluate-generation`, fluxo linear, `qwen2.5:1.5b-instruct`, cache do
Ollama limpo, uma rodada, 26 perguntas; `.data/evaluation/parser-*.jsonl`). Vereditos lexicais,
com revisão manual só das 23 respostas (de 52) que mudaram de um parser para o outro; três delas
foram corrigidas (por exemplo, "gatilhos" é uma resposta certa para "gates").

| | `pypdf` hashing | Docling hashing | `pypdf` semantic | Docling semantic |
|---|---|---|---|---|
| Corretas / parciais (de 21) | 11 / 0 | 13 / 0 | 8 / 1 | 12 / 2 |
| Erradas com citação válida | 8 | 6 | 8 | 7 |
| `fallback` | 2 | 2 | 4 | 0 |
| Sem resposta: abstém ou `fallback` / inventa (de 5) | 1 / 4 | 3 / 2 | 1 / 4 | 1 / 4 |
| Citou um trecho-gabarito | 12 | 12 | 9 | 12 |
| Tokens de entrada por pergunta | 1.724 | 1.441 | 1.790 | 1.458 |
| Latência média | 2,7 s | 3,0 s | 3,4 s | 3,2 s |

Nas 23 respostas que mudaram, 15 melhoraram com o Docling, 7 pioraram e 1 ficou igual. Os
ganhos vieram sobretudo de perguntas diretas e de tabela (o 27,2 de mAP do COCO e os 11,3 bilhões
de FLOPs passaram a sair certos) e de duas invenções nas perguntas sem resposta que viraram
abstenção ou `fallback`. As perdas foram respostas que estavam certas e passaram a pegar a linha
errada (`7.93%` no lugar de `28.54%`), uma resposta "não venceu nenhuma competição" e um BLEU
inventado. O prompt ficou 17% menor, porque o texto não carrega mais eixos, rodapés e
bibliografia. Com o cache limpo, a mesma entrada dá a mesma resposta (etapa 2), então as
diferenças vêm do texto e dos trechos, não de amostragem. Mesmo assim, pequenas mudanças no texto
fazem o modelo de 1,5B trocar respostas nos dois sentidos, e com 26 perguntas de um artigo só o
saldo de +8 é indicativo, não conclusivo. Os números absolutos não são comparáveis com os da
[avaliação da geração](#avaliação-da-geração), que teve todas as respostas revisadas à mão.

**Limitações que ficam:**

- Na página 1 do ResNet, o Docling põe a coluna da direita (Figura 1 e o fim da introdução) antes
  do Abstract. A página continua certa, mas um trecho mistura as duas colunas.
- Os expoentes continuam perdidos (`1.8×109`), e a Tabela 1 (arquitetura, com colchetes) continua
  ilegível com os dois parsers.
- Palavras com hífen quebradas no fim da linha perdem o hífen (`highlevel`).
- Rótulos de figura às vezes grudam em parágrafos vizinhos (`iter. (1e4)`).

## Reranker (cross-encoder)

A busca híbrida ordena os trechos fundindo duas listas (BM25 e vetores) sem nunca ler a pergunta
e o trecho juntos. Um cross-encoder lê os dois numa única passada do modelo e dá uma nota ao par.
Com `RAG_RERANK=true`, o `retrieve` passa os 20 primeiros candidatos da fusão
(`RAG_CANDIDATE_K`) e o nó `rerank` mantém os `top_k` de nota mais alta.

```bash
python -m pip install -e ".[rerank]"
RAG_RERANK=true paper-rag ask "Qual é o erro top-5 do ResNet-152?" --show-steps
```

- **FastEmbed, sem PyTorch:** o `TextCrossEncoder` do FastEmbed roda o modelo em ONNX na CPU; é a
  mesma biblioteca do embedding semântico. O modelo baixa na primeira pergunta para
  `RAG_MODEL_CACHE_DIR` (`.data/models`).
- **Modelo:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache 2.0), um MiniLM multilíngue
  treinado no mMARCO, a versão traduzida do MS MARCO em 14 idiomas, inclusive português. Ele não
  está na lista do FastEmbed e é registrado com `add_custom_model` na primeira carga.
  `RAG_RERANK_MODEL` troca o modelo (por exemplo, `jinaai/jina-reranker-v2-base-multilingual`).
- **Texto da nota:** o cross-encoder recebe a pergunta original, não a consulta reescrita, porque
  a resposta tem de atender à pergunta. Empates mantêm a ordem da busca.
- **Testes sem download:** os testes trocam o modelo por um cross-encoder falso (uma nota por
  palavra da pergunta encontrada no trecho); o CI não instala o extra nem baixa modelo.

**Escolha do modelo** (sem LLM, script avulso fora do repositório, MacBook M4): as 26 perguntas do
dataset de geração sobre o ResNet; um trecho é gabarito quando contém uma das citações esperadas.
Para cada pergunta, os mesmos 20 candidatos da busca híbrida foram reordenados por três modelos:
um pequeno só em inglês, o mmarco e o maior multilíngue que o FastEmbed oferece.

| Gabarito no top-4 (de 21) / MRR@4 | só a busca | MiniLM-L6 (inglês) | mmarco (multilíngue) | jina v2 (multilíngue) |
|---|---|---|---|---|
| `pypdf`, hashing | 17 / 0,58 | 18 / 0,69 | 19 / 0,74 | 20 / 0,80 |
| `pypdf`, semantic | 17 / 0,64 | 20 / 0,75 | 20 / 0,77 | 21 / 0,85 |
| Docling, hashing | 17 / 0,56 | 19 / 0,68 | 20 / 0,77 | 20 / 0,78 |
| Docling, semantic | 18 / 0,58 | 20 / 0,69 | 19 / 0,76 | 19 / 0,77 |
| **Soma no top-4 (de 84)** | 69 | 77 | 78 | 80 |
| Gabarito em 1º lugar (de 84) | 36 | 47 | 54 | 57 |
| Entrou / saiu do top-4 | | 12 / 4 | 13 / 4 | 14 / 3 |
| Tempo extra por pergunta (20 pares) | | 0,5 a 0,65 s | 0,9 a 1,2 s | 2,0 a 2,5 s |
| Download / licença | | 91 MB / Apache 2.0 | 471 MB / Apache 2.0 | 1,1 GB / CC-BY-NC-4.0 |

- **O ganho vem de ligar um reranker, qualquer um:** os três melhoram as quatro combinações. O
  gabarito no top-4 vai de 69 para 77 a 80 (o máximo possível era 83, porque numa pergunta o
  gabarito não está entre os 20 candidatos) e o MRR@4 médio, de 0,59 para 0,70 a 0,80. Entre os
  modelos, a diferença é de 1 a 3 perguntas em 84.
- **As perdas são quase todas em português:** "O que é a opção B de atalhos no artigo?" com
  Docling caiu da posição 2 ou 3 para 11 a 17, e o modelo só em inglês jogou a pergunta sobre a
  taxa de aprendizado da posição 1 para a 14. Nas 6 perguntas em português (24 casos), o
  gabarito no top-4 fica em 18 só com a busca, 17 com o MiniLM, 18 com o mmarco e 20 com o jina:
  o ganho vem das perguntas em inglês, e o mmarco só empata com a busca em português.
- **Por que o mmarco:** fica perto do jina (78 × 80 no top-4, 0,76 × 0,80 de MRR), custa metade
  do tempo e um terço do download, e tem licença Apache. O jina proíbe uso comercial
  (CC-BY-NC-4.0). O MiniLM em inglês é o mais rápido, mas é o que mais erra em português.
  `BAAI/bge-reranker-base`, também do FastEmbed, cobre só inglês e chinês e ficou de fora.
- **Nota mínima para se abster:** testei se a maior nota do reranker separa as perguntas sem
  resposta das demais. Não separa: em nenhum modelo um limiar exclui mais de 1 das 5 sem resposta
  sem excluir também perguntas com resposta. O reranker só reordena; abster continua com o `grade`.
- **Gates do CI com o reranker** (fora do CI, que não baixa modelo): baseline e paráfrases ficam
  em 1,000 / 1,000 (Recall@3 / MRR) com hashing e semantic; nas paráfrases com hashing, a busca
  sem reranker fica em 0,800 / 0,567.

**Geração** (`paper-rag evaluate-generation`, fluxo linear, `qwen2.5:1.5b-instruct`, cache do
Ollama limpo, uma rodada, 26 perguntas; `.data/evaluation/rerank-*.jsonl` comparado com os
`parser-*.jsonl` da etapa do Docling, que têm a mesma configuração sem reranker). Vereditos
lexicais, antes → depois:

| | `pypdf` hashing | `pypdf` semantic | Docling hashing | Docling semantic |
|---|---|---|---|---|
| Corretas / parciais (de 21) | 11 / 0 → 9 / 1 | 8 / 0 → 10 / 1 | 12 / 0 → 14 / 0 | 13 / 1 → 14 / 0 |
| Erradas com citação válida | 8 → 9 | 9 → 9 | 7 → 6 | 7 → 6 |
| `fallback` | 2 → 2 | 4 → 1 | 2 → 1 | 0 → 1 |
| Sem resposta: abstém ou `fallback` / inventa (de 5) | 1 / 4 → 1 / 4 | 1 / 4 → 1 / 4 | 3 / 2 → 1 / 4 | 1 / 4 → 1 / 4 |
| Gabarito no top-4 | 17 → 19 | 17 → 20 | 17 → 20 | 18 → 19 |
| Citou um trecho-gabarito | 12 → 13 | 9 → 14 | 12 → 15 | 12 → 15 |
| Latência média | 2,7 → 3,6 s | 3,4 → 4,3 s | 3,0 → 4,1 s | 3,2 → 4,6 s |

Como a ordem dos trechos no prompt muda, 79 das 104 respostas mudaram de texto. Revisei todas à
mão: **22 melhoraram, 15 pioraram e 42 ficaram equivalentes** (5 / 6, 6 / 3, 4 / 3 e 7 / 3 nas
quatro colunas, na mesma ordem). Os vereditos lexicais erram em alguns casos, nos dois sentidos:
"atalhos com gatilhos" é a resposta certa sobre as highway networks, e "Evidências [1]: Não
encontrei evidências…" é uma abstenção, embora tenha passado pelo `validate` como resposta.

- **Ganhos:** perguntas de tabela (o 27,2 de mAP do COCO nos dois embeddings com `pypdf`, os 11,3
  bilhões de FLOPs, o 6,43% com 1,7 milhão de parâmetros), o tamanho do CIFAR-10 e a taxa de
  aprendizado em português, perguntas sobre degradação que passaram a citar o erro de treino em
  vez de overfitting, e duas invenções a menos nas perguntas sem resposta (um BLEU e uma acurácia
  no MNIST).
- **Perdas:** o modelo agora recebe a Tabela 2 em primeiro lugar e lê a coluna errada da linha
  `34 layers 28.54 25.03` (25,03% em vez de 28,54%, em 3 das 4 colunas); o erro do ensemble virou
  7,40%; duas perguntas sem resposta que davam `fallback` passaram a inventar um número.
- **Repetição do prompt:** em 3 respostas com `pypdf` o modelo devolveu as próprias instruções
  ("Responda no idioma da pergunta…"), e o `validate` as aceitou, porque elas contêm `[1]`. Isso
  não tinha acontecido em nenhuma rodada anterior.
- **Custo:** cerca de 1 s a mais por pergunta (de 0,8 a 1,4 s), ou 30 a 45%.

O reranker melhora o que o gerador recebe: o trecho-gabarito vai para o top-4 em 78 de 84 casos
(eram 69) e a resposta cita um trecho-gabarito em 57 (eram 45). A resposta melhora menos, com
saldo de +7 em 79 respostas que mudaram, porque o gerador de 1,5B continua errando a linha ou a
coluna da tabela mesmo com o trecho certo em `[1]`. Com 26 perguntas de um artigo só, o saldo é
indicativo, não conclusivo.

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

1. Observabilidade de latência, erros e tokens por consulta.
2. Leitura de PDF: ordem da primeira página e expoentes com o Docling (ver
   [limitações](#leitura-de-pdf-docling)).
3. Reranker: medir a versão quantizada do mmarco (119 MB, só ARM) e, com um dataset maior,
   ajustar um reranker para artigos científicos.
4. `validate`: recusar a resposta que repete as instruções do prompt (visto 3 vezes com o
   reranker; hoje passa porque as instruções contêm `[1]`).
