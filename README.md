
# Rota 03 - Machine Learning

## Contexto e Problema de Negócio

No dinâmico mercado atual, a retenção de talentos tornou-se um fator crítico para o sucesso sustentável das organizações. A alta rotatividade de pessoal (churn) gera custos elevados com demissão, recrutamento e treinamento, além de impactar a estabilidade e o desempenho das equipes.

Este projeto aborda o desafio de gerenciar grandes volumes de dados de Recursos Humanos para antecipar e mitigar a perda de funcionários-chave. O principal problema de negócio é a falta de uma abordagem proativa para identificar os funcionários com maior risco de desligamento, fazendo com que a empresa atue de forma reativa, quando o talento já foi perdido.

## Objetivos do Projeto

O objetivo deste projeto é desenvolver um modelo de **Machine Learning (Aprendizado Supervisionado)** capaz de prever com precisão se um funcionário deixará a organização (`Attrition = 1`).

Os objetivos secundários incluem:
* Identificar as **principais variáveis** (fatores) que mais influenciam a decisão de um funcionário sair.
* Fornecer **insights estratégicos** e **recomendações** baseadas em dados para o departamento de RH.
* Criar uma ferramenta que permita à empresa tomar **medidas preventivas** e eficazes para reter talentos.

## Tecnologias Utilizadas

* **Linguagem Principal:** Python
* **Bibliotecas de Análise:** Pandas, NumPy
* **Bibliotecas de Visualização:** Matplotlib, Seaborn
* **Bibliotecas de ML:** Scikit-learn, XGBoost, TensorFlow
* **Ambiente de Desenvolvimento:** Google Colab
* **Insumos:** Bases de dados (CSV) do departamento de RH

## Dashboards do Projeto

### 1. Visão Geral
* **Descrição:** Resumo dos funcionários, taxa de rotatividade e principais indicadores-chave da empresa.

![Visão Geral](dashboards_screenshots/dash%2001-rota%2003.jpg)

### 2. Perfil do Funcionário
* **Descrição:** Distribuição dos funcionários por departamento, sexo e estado civil.

![Perfil do Funcionário](dashboards_screenshots/dash%2002-rota%2003.jpg)

### 3. Distribuição de Dados
* **Descrição:** Distribuição das principais variáveis numéricas, como idade, salário mensal e anos de experiência.

![Distribuição de Dados](dashboards_screenshots/dash%2003-rota%2003.jpg)

### 4. Modelos de Machine Learning e Conclusões
* **Descrição:** Comparação de algoritmos treinados e insights estratégicos para retenção de talentos.

![Modelos de Machine Learning](dashboards_screenshots/dash%2004-rota%2003.jpg)


## Principais Insights e Conclusões

A análise exploratória e o modelo de Machine Learning (Random Forest) permitiram identificar os fatores críticos que levam à rotatividade de funcionários (churn) e propor ações estratégicas para o RH.

### Resultados Principais

* **Taxa de Rotatividade (Baseline):** A empresa possui uma taxa de churn de **16%**, indicando que 1 em cada 6 funcionários deixou a organização.
* **Modelo Preditivo (Random Forest):** O modelo Random Forest foi o mais eficaz em prever quais funcionários têm maior probabilidade de sair.
* **Perfil de Risco Identificado:** O maior risco de desligamento está concentrado em **funcionários mais jovens, com salários mais baixos e com menor tempo de experiência** na empresa.
* **Fatores-Chave de Desligamento:** O modelo confirmou que as 3 variáveis mais importantes para prever a saída de um funcionário são:
    1.  **Idade**
    2.  **Salário Mensal**
    3.  **Total de Anos de Experiência**
* **Pontos de Atenção:** Foi identificada uma concentração de desligamentos no departamento de **Pesquisa e Desenvolvimento (P&D)** e entre funcionários com estado civil **"Casado"**.

### Recomendações Estratégicas (Ações para o RH)

Com base nesses resultados, as seguintes ações proativas podem ser tomadas para reter talentos:

1.  **Uso do Modelo Preditivo:** Aplicar o modelo Random Forest para **identificar funcionários com alto risco** de saída e incluí-los em programas de retenção preventivos.
2.  **Ações de Retenção Focadas:** Criar programas de incentivo direcionados ao perfil de risco, principalmente para os **mais jovens e com salários mais baixos**.
3.  **Revisão de Carreiras e Salários:** Focar na retenção por meio de **promoções internas** e **revisão de políticas salariais** para garantir a competitividade.
4.  **Desenvolvimento e Engajamento:** Oferecer mais oportunidades de **crescimento e capacitação** para aumentar o engajamento e a satisfação.
5.  **Monitoramento Contínuo:** Acompanhar as métricas de rotatividade por departamento (com foco em P&D) e faixa salarial para antecipar novos problemas.

### Slides de Conclusão e Recomendações

![Conclusões do Projeto](dashboards_screenshots/dash%2005-rota%2003.jpg)
![Recomendações do Projeto](dashboards_screenshots/dash%2006-rota%2003.jpg)


##  Códigos e Cálculos Relevantes

### Código / Função 1

* **O que faz:** carregar e limpar dados

```python
import pandas as pd
df = pd.read_csv('datasets/rh_data.csv')
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)
```

### Código / Função 2

* **O que faz:** dividir dados em treino e teste

```python
from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Código / Função 3

* **O que faz:** treinar modelo Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

## Como Rodar o Projeto

1. Abrir o Google Colab.
2. Importar os arquivos CSV disponíveis na pasta `datasets/`.
3. Limpar e preparar os dados (tratar nulos, duplicados, codificar variáveis categóricas).
4. Dividir os dados em treino e teste (com seed para reprodutibilidade).
5. Criar novas variáveis explicativas (feature engineering).
6. Treinar modelos de Machine Learning (Logistic Regression, Random Forest, XGBoost).
7. Avaliar os modelos usando métricas como precisão, recall e F1-score.
8. Gerar gráficos e dashboards para análise e interpretação dos resultados.

### Observações

* Os dados são fictícios, para demonstração.
---
## Autor
**Leticia Gama de Souza**
[LinkedIn](https://www.linkedin.com/in/leticia-gama-code) | [GitHub](https://github.com/LeticiaGama-dev)

