# Tech Challenge — Fase 1: Análise Exploratória de Dados de NPS

Projeto da pós-graduação FIAP (Fase 1) com o objetivo de entender, a partir de dados
operacionais de um e-commerce, quais fatores explicam a satisfação do cliente medida
pelo NPS (Net Promoter Score).

## Objetivo do Projeto

A empresa só descobre se um cliente ficou insatisfeito depois que toda a jornada de
compra já aconteceu — a nota de NPS é coletada no pós-venda. Isso deixa o negócio
reativo diante de problemas de entrega, atendimento e qualidade.

Este notebook (`EDA_TECHALLENGE_FASE1.ipynb`) cobre o **Item 3 do desafio — Análise
Exploratória de Dados —** e busca responder, com evidências estatísticas, quatro
perguntas centrais:

1. Quais fatores têm maior correlação com o NPS?
2. O que mais gera clientes Detratores?
3. Existe um "ponto de ruptura" — um momento específico em que a satisfação despenca?
4. Que perfil de cliente (idade, ticket, tempo de relacionamento, recompra) tende a
   dar notas mais altas ou mais baixas?

## Descrição da Base de Dados

Arquivo: `desafio_nps_fase_1.csv` — uma linha por cliente/pedido. Principais colunas
utilizadas na análise:

| Coluna | Descrição |
|---|---|
| `nps_score` | Nota de satisfação do cliente (0–10), coletada no pós-venda. **Variável-alvo da análise.** |
| `customer_id` / `order_id` | Identificadores de cliente e pedido. |
| `customer_age` | Idade do cliente. |
| `customer_region` | Região do cliente. |
| `customer_tenure_months` | Tempo de relacionamento com a empresa, em meses. |
| `order_value` | Valor do pedido (R$). |
| `items_quantity` | Quantidade de itens no pedido. |
| `discount_value` | Valor do desconto aplicado (R$). |
| `payment_installments` | Número de parcelas do pagamento. |
| `delivery_time_days` | Tempo total de entrega, em dias. |
| `delivery_delay_days` | Dias de atraso na entrega (0 = entregue no prazo). |
| `delivery_attempts` | Número de tentativas de entrega. |
| `freight_value` | Valor do frete (R$). |
| `complaints_count` | Número de reclamações registradas pelo cliente. |
| `customer_service_contacts` | Número de contatos com o suporte. |
| `resolution_time_days` | Tempo de resolução do atendimento, em dias (para quem acionou o suporte). |
| `repeat_purchase_30d` | Se o cliente comprou novamente em até 30 dias (0/1). |
| `csat_internal_score` | Métrica interna de satisfação (CSAT), avaliada à parte do NPS. |

A partir dessas colunas originais, o notebook cria as seguintes variáveis derivadas
na Seção 2.0 (*Tratamento e Engenharia de Features*):

| Variável criada | Tipo | Baseada em | Regra / Faixas |
|---|---|---|---|
| `nps_categoria` | Classificação | `nps_score` | `Promotor` (9–10), `Neutro` (7–8), `Detrator` (0–6) — régua padrão de NPS. |
| `entrega_atrasada` | Flag binária (0/1) | `delivery_delay_days` | `1` se `delivery_delay_days > 0`, senão `0`. |
| `acionou_suporte` | Flag binária (0/1) | `customer_service_contacts` | `1` se `customer_service_contacts > 0`, senão `0`. |
| `faixa_etaria` | Faixa | `customer_age` | `18-29`, `30-44`, `45-59`, `60+` (cortes em 18, 30, 45, 60, 70). |
| `faixa_ticket` | Faixa (quartil) | `order_value` | `Baixo`, `Médio-baixo`, `Médio-alto`, `Alto` (quartis, `q=4`). |
| `faixa_tempo_relacionamento` | Faixa | `customer_tenure_months` | `Até 6 meses`, `6 meses a 1 ano`, `1 a 3 anos`, `3 a 5 anos`, `Mais de 5 anos` (cortes em 1, 7, 13, 37, 61, 125 meses). |
| `perfil_parcelamento` | Faixa | `payment_installments` | `À vista (1x)`, `Curto prazo (2x-3x)`, `Médio prazo (4x-6x)`, `Longo prazo (7x+)` (cortes em 1, 2, 4, 7, 15 parcelas). |
| `faixa_desconto` | Faixa (quartil) | `discount_value` | `Desconto Baixo`, `Desconto Médio-baixo`, `Desconto Médio-alto`, `Desconto Alto` (quartis, `q=4`). |
| `faixa_atraso_entrega` | Faixa | `delivery_delay_days` | `00.Sem Atraso`, `01.Atraso Leve (1-2d)`, `02.Atraso Moderado (3-5d)`, `03.Atraso Crítico (6d+)` (cortes em 0, 1, 3, 6, 12 dias). |
| `faixa_tempo_resolucao` | Faixa condicional | `resolution_time_days` + `acionou_suporte` | Para quem não acionou o suporte: `00.Não acionou`. Para quem acionou: `01.Imediata (D0)`, `02.Rápida (1-2d)`, `03.Padrão (3-6d)`, `04.Longa (7d+)` (cortes em 0, 1, 3, 7, 15 dias). |
| `faixa_reclamacao` | Faixa condicional | `complaints_count` | Sem reclamações: `00.Sem reclamações`. Com reclamações: `01.Fricção Baixa (1-3)`, `02.Fricção Moderada (4-5)`, `04.Fricção Alta (6+)` (cortes em 0, 4, 6, 15 reclamações). |

Todas as variáveis de faixa usam `pd.cut` (cortes fixos definidos por regra de
negócio) ou `pd.qcut` (quartis da distribuição real), com `right=False` — ou seja,
o limite inferior de cada faixa é incluído e o superior é excluído. Essas variáveis
são usadas ao longo de toda a EDA (Seções 3.0 a 5.0).

## Metodologia Utilizada

O notebook está organizado em seções sequenciais:

1. **Imports e configurações (0.0):** bibliotecas (`pandas`, `numpy`, `matplotlib`,
   `seaborn`, `scipy`) e paleta de cores padrão para as categorias de NPS.
2. **Importação e inspeção da base (1.0):** carga do CSV, verificação de tipos e
   nulos (`df.info()`), estatísticas descritivas (`df.describe()`) e distribuição
   de clientes por região.
3. **Tratamento e engenharia de features (2.0):** classificação dos clientes na
   régua padrão de NPS (Promotor 9–10 / Neutro 7–8 / Detrator 0–6), cálculo do NPS
   consolidado da empresa (% Promotores − % Detratores), criação de flags binárias
   e segmentação em faixas de negócio (idade, ticket, tempo de relacionamento,
   parcelamento, desconto, atraso na entrega, tempo de resolução, reclamações).
4. **Análise de correlação com o NPS (3.0):** matriz de correlação de Pearson entre
   todas as variáveis numéricas e o `nps_score`, visualizada em heatmap.
5. **Análise por variáveis categóricas (4.0):** tabelas dinâmicas (crosstabs) cruzando
   o NPS com região, faixa de atraso, faixa etária, ticket, tempo de relacionamento,
   parcelamento, desconto, tentativas de entrega e recompra.
6. **Insights consolidados (5.0):**
   - **5.1** Ranking de correlação com o NPS;
   - **5.2** Taxa de Detratores pelos principais fatores geradores de insatisfação,
     com teste de hipótese não-paramétrico (Mann-Whitney U) comparando clientes com
     e sem atraso na entrega;
   - **5.3** Identificação do ponto de ruptura na experiência (NPS médio por dia de
     atraso e por número de reclamações);
   - **5.4** Perfil demográfico e comportamental de Promotores e Detratores (faixa
     etária, tempo de relacionamento, ticket, recompra, SLA de suporte);
   - **5.5** Scorecard qualitativo consolidando o impacto relativo de cada fator.

**Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.

## Como Reproduzir os Resultados

Este notebook foi desenvolvido para rodar no **Google Colab**, com caminhos de
arquivo absolutos apontando para `/content/`. Para reproduzir os resultados:

### 1. Abrir o notebook no Google Colab
Faça upload de `EDA_TECHALLENGE_FASE1.ipynb` no Colab (ou abra direto do GitHub via
*File → Open notebook → GitHub*).

### 2. Criar as pastas esperadas pelo notebook
No ambiente do Colab, crie as pastas de dados e de saída de gráficos:
```python
import os
os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/graficos', exist_ok=True)
```

### 3. Disponibilizar o dataset
Faça upload do arquivo `desafio_nps_fase_1.csv` para `/content/data/`, de forma que
o caminho final seja:
```
/content/data/desafio_nps_fase_1.csv
```
(Você pode arrastar o arquivo direto na aba "Arquivos" do Colab, ou montar o Google
Drive e copiar o arquivo para essa pasta.)

### 4. Executar todas as células
Rode o notebook de cima para baixo (*Ambiente de execução → Executar tudo*). As
seções dependem umas das outras na ordem em que aparecem (engenharia de features
antes da EDA, por exemplo), então a execução sequencial é importante.

### 5. Gráficos gerados
Os gráficos das Seções 3.0 e 5.0 são salvos automaticamente em `/content/graficos/`
durante a execução (ex.: `heatmap_correlacao.png`, `ranking_correlacao_nps.png`,
`taxa_detratores_por_fator.png`, `ponto_ruptura.png`, `perfil_demografico.png`,
entre outros). Para mantê-los após o fim da sessão do Colab, faça o download da
pasta `/content/graficos/` ou copie-a para o Google Drive antes de encerrar.

### Rodando localmente (alternativa ao Colab)
Se preferir rodar fora do Colab, ajuste os caminhos de `pd.read_csv(...)` e de todos
os `plt.savefig(...)` no notebook, substituindo `/content/data/` e `/content/graficos/`
pelos caminhos relativos do seu ambiente (ex.: `data/` e `reports/`), e instale as
dependências com:
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```
