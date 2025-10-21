
# Rota 03 - Machine Learning

##  Introdução

Projeto de Machine Learning para previsão de rotatividade de funcionários, incluindo análise exploratória, preparação de dados, treinamento de modelos supervisionados e apresentação de resultados.
Objetivo: fornecer insights sobre retenção de talentos e ajudar a empresa a tomar decisões estratégicas para reduzir a rotatividade.

##  Tecnologias Utilizadas

* Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow)
* Google Colab
* CSVs fornecidos pelo departamento de RH

##  Como Rodar o Projeto

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

##  Dashboards do Projeto

As imagens dos dashboards estão na pasta [`dashboards_screenshots`](./dashboards_screenshots).

### Dash 01: Visão Geral
- **Arquivo:** `dash 01-rota 03.jpg`  
- **Descrição:** Resumo dos funcionários, taxa de rotatividade e principais indicadores-chave da empresa.

### Dash 02: Perfil do Funcionário
- **Arquivo:** `dash 02-rota 03.jpg`  
- **Descrição:** Distribuição dos funcionários por departamento, sexo e estado civil.

### Dash 03: Distribuição de Dados
- **Arquivo:** `dash 03-rota 03.jpg`  
- **Descrição:** Distribuição das principais variáveis numéricas, como idade, salário mensal e anos de experiência.

### Dash 04: Modelos de Machine Learning e Conclusões
- **Arquivo:** `dash 04-rota 03.jpg`  
- **Descrição:** Comparação de algoritmos treinados e insights estratégicos para retenção de talentos.


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

 **Estrutura do Repositório**

```
rota-03-machine-learning/
│
├─ datasets/               # Bases de dados (CSV)
├─ dashboards_screenshots/ # Prints de gráficos e dashboards
├─ notebooks/              # Notebooks Python / Colab
├─ src/                    # Scripts do projeto
├─ README.md
├─ .gitignore (opcional)
```

🔗 **Link do Repositório**
[https://github.com/LeticiaGama-dev/rota-03-machine-learning](https://github.com/LeticiaGama-dev/rota-03-machine-learning)

---
Meu Contato:

[LinkedIn]  (www.linkedin.com/in/leticia-gama-code)
