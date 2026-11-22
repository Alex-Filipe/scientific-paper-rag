# Arquitetura

## Visão geral

Monólito modular em Python. O pipeline usa os pacotes mantidos do ecossistema LangChain
(`langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-ollama`,
`langchain-openai`) e orquestra a pergunta com um grafo **LangGraph**.

```text
ingestão:  arquivo -> PaperLoader -> RecursiveCharacterTextSplitter -> embeddings -> Chroma
pergunta:  LangGraph (retrieve -> grade -> [rewrite -> retrieve] -> generate -> validate | abstain)
```

Por padrão `grade` e `rewrite` ficam desligados e o fluxo é linear (`retrieve -> generate`); a
[avaliação da geração](#avaliação-da-geração) explica por quê.

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `documents.py` | `PaperLoader` (PDF por página com `pypdf` e limpeza de ligaduras/hifenização, TXT/MD) e chunking por palavras |
| `embeddings.py` | `HashingEmbeddings` (baseline sem download) e `FastEmbedEmbeddings` (semântico, CPU) |
| `vectorstore.py` | `ChunkIndex`: coleção Chroma persistente por espaço de embeddings |
| `retrieval.py` | `BM25Retriever`, `RRFRetriever` e `HybridSearch` (BM25 + vetores via RRF) |
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
START -> retrieve -> grade --(há trechos úteis)--> generate --(resposta do LLM)--> validate -> END
                       |                                \--(extrativo ou abstenção)--> END
                       |--(nenhum útil e ainda pode reescrever)--> rewrite -> retrieve
                       \--(nenhum útil)--> abstain -> END
```

- **retrieve:** busca híbrida com `RAG_CANDIDATE_K` candidatos por ranker e corte final em `top_k`.
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

## Teste com artigo real

ResNet (He et al., CVPR 2016, 9 páginas em duas colunas), `qwen2.5:1.5b-instruct` no Ollama,
`top_k=4`, 7 perguntas (diretas, vaga, sem resposta, numérica e duas em português):

- **Extração (`pypdf`):** a ordem das colunas sai correta e as tabelas viram linhas legíveis
  (`ResNet-152 19.38 4.49`). Ligaduras (`ﬁ`) e hifenização de fim de linha agora são limpas. Ainda
  ficam espaços espúrios (`W e`, `T able`, `CIF AR-10`), números dos eixos das figuras e expoentes
  perdidos (`1.8×109`); isso fica para o parser da etapa 4.
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
  final vem do evento `answer`, que pode ter a citação corrigida ou ter virado `fallback`. Quando
  vira `fallback`, o rascunho descartado fica num expander com o motivo dado pelo `validate`, e a
  correção de citação aparece em destaque.
- **Chaves por pergunta:** "avaliar trechos" e "validar números" vão no corpo da pergunta
  (`grade_documents`, `validate_numbers`) e entram no estado do grafo; o modelo e o avaliador
  (`RAG_GRADER_MODEL`) continuam fixos ao subir a API, e a tela só os mostra. Com o grade ligado e
  sem avaliador separado, a tela avisa que o 1,5B troca acertos por abstenções.
- **Fontes:** cada trecho recuperado vira um `st.expander` com título e página, aberto quando foi
  citado, com o **trecho inteiro** (campo `text` da citação; o `excerpt` tem só os 280 primeiros
  caracteres e muitas vezes não chega ao número citado), não o PDF. Numa resposta aceita, os
  números que ela acrescenta à pergunta, os mesmos que o `validate` procurou, aparecem grifados
  em todos os trechos (`numbers.claimed_readings`, com as mesmas leituras `4,49`/`4.49`).
- **Compatibilidade:** o cliente ignora campos que não conhece e transforma payloads inesperados
  em `ApiError`, para que uma API mais nova não derrube a tela com um traceback.
- **Recarga de módulos:** quando um arquivo muda, o Streamlit descarrega os módulos da pasta da
  tela (`paper_rag.ui.*`), mas não os outros módulos do pacote. Um `ApiClient` guardado na sessão
  passaria a emitir eventos das classes antigas, e os `isinstance` das novas descartariam tudo (a
  resposta sumia em silêncio). Por isso a sessão guarda só o `httpx.Client` e o `ApiClient` é
  recriado a cada execução; o status é comparado por valor, não por identidade. Mudanças fora de
  `ui/` (por exemplo, em `models.py`) só valem depois de reiniciar a tela.
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

1. Parser que preserve a estrutura de artigos (duas colunas, seções), por exemplo Docling.
2. Reranker entre a busca e o grafo.
3. Observabilidade de latência, erros e tokens por consulta.
