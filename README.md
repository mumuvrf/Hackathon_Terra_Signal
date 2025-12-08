# Hackathon_Terra_Signal

Um projeto de aprendizado de máquina para previsão de churn de clientes em um conjunto de dados de empresa de telecomunicações. Este projeto implementa um pipeline ML completo incluindo processamento de dados, treinamento do modelo, previsões e uma API REST para inferência em tempo real.

## 📋 Visão Geral do Projeto

Este projeto de hackathon visa prever o churn de clientes (a probabilidade de um cliente descontinuar seu serviço) para uma empresa de telecomunicações fictícia chamada Terra Signal. A solução inclui:

- **Pipeline de Processamento de Dados**: Limpar e transformar dados brutos de clientes
- **Treinamento do Modelo**: Regressão Logística com codificação One-Hot
- **Previsões em Lote**: Processar novos dados de clientes e gerar previsões
- **API REST**: Endpoint FastAPI para previsões de churn em tempo real e recomendações

## 👥 Autores

- Miqueias Ayron
- Aimée Ibrahim
- Vinícios Rodrigues

## 📁 Estrutura do Projeto

```
Hackathon_Terra_Signal/
├── data/
│   ├── history.csv              # Dados históricos de clientes com rótulos de churn
│   ├── inference.csv            # Novos dados de clientes sem rótulo para previsão
│   └── prediction.csv           # Previsões geradas para inference.csv
├── models/
│   ├── model.pkl                # Modelo de Regressão Logística treinado
│   └── transformer.pkl          # Transformador One-Hot encoder ajustado
├── notebooks/
│   ├── NOTEBOOK.ipynb           # Notebook inicial com fluxo de trabalho básico
│   ├── CHURN.ipynb              # Análise exploratória de dados e desenvolvimento do modelo
│   └── history_analysis.ipynb   # Análise detalhada dos dados históricos
├── src/
│   ├── app.py                   # Aplicação FastAPI com endpoints de previsão e recomendação
│   ├── train.py                 # Script de treinamento do modelo
│   ├── predict.py               # Script de previsão em lote
│   ├── process.py               # Utilitários de processamento de dados
│   └── __init__.py              # Inicialização do pacote
├── ADD_USERS.ipynb              # Notebook utilitário para colaboração de workspace
├── requirements.txt             # Dependências do Python
├── LICENSE                      # Apache License 2.0
└── README.md                    # Este arquivo
```

## 🔧 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip ou conda

### Instalação

1. Clone ou baixe o projeto:
```bash
cd c:\dev\Hackathon_Terra_Signal
```

2. Crie um ambiente virtual (opcional mas recomendado):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📊 Dados

### history.csv
- **Propósito**: Conjunto de dados de treinamento com churn de clientes rotulado
- **Características**: Demografia de clientes, uso de serviços, informações de faturamento
- **Alvo**: Coluna `Churn` (Sim/Não)
- **Linhas**: Registros históricos de clientes

### inference.csv
- **Propósito**: Novos dados de clientes sem rótulo para previsão
- **Características**: Mesma estrutura de history.csv mas sem o rótulo Churn
- **Saída**: Previsões salvas em `prediction.csv`

### prediction.csv
- **Propósito**: Previsões geradas para inference.csv
- **Conteúdo**: Dados originais de clientes + valores de Churn previstos

## 🚀 Uso

### 1. Processamento de Dados
O módulo `process.py` gerencia limpeza de dados e engenharia de características:

```python
from src.process import process

df = process('../data/history.csv')
```

**Etapas de Processamento:**
- Converter permanência (tenure) e cobranças para formato numérico
- Preencher TotalCharges ausentes usando: `TotalCharges = tenure * MonthlyCharges`
- Criar categorias binned:
  - `tenureCategory`: Agrupa permanência em intervalos (0-11, 12-23, 24-35, 36-47, 48+)
  - `MonthlyChargesCategory`: Agrupa cobranças mensais (0-40, 41-80, 80+)

### 2. Treinamento do Modelo
Treine o modelo de previsão de churn:

```bash
python src/train.py
```

**Detalhes do Modelo:**
- **Algoritmo**: Regressão Logística com balanceamento de classes
- **Pré-processamento**: Codificação One-Hot para variáveis categóricas
- **Saída**: 
  - `models/model.pkl` - Classificador treinado
  - `models/transformer.pkl` - Encoder ajustado

### 3. Previsões em Lote
Gere previsões para novos dados:

```bash
python src/predict.py
```

Isto irá:
- Carregar o modelo treinado e o transformador
- Processar `data/inference.csv`
- Gerar previsões
- Salvar resultados em `data/prediction.csv`

### 4. API REST
Inicie o servidor FastAPI para previsões em tempo real:

```bash
uvicorn src.app:app --reload
```

#### Endpoints da API

**POST /predict** - Prever churn para um único cliente
```json
Solicitação:
{
  "gender": "Male",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1572,
  "Contract": "Month-to-month",
  ...outras características...
}

Resposta:
{
  "Churn": "Yes"
}
```

**POST /recommend** - Obter recomendações para reduzir risco de churn
```json
Solicitação:
{
  "Contract": "Month-to-month",
  ...outras características...
}

Resposta:
{
  "recommendations": [
    "Recomendar um plano de contrato de maior duração (1 ano ou 2 anos)."
  ]
}
```

## 📓 Notebooks

### NOTEBOOK.ipynb
Notebook inicial cobrindo:
- Carregamento e exploração de dados
- Fluxo de trabalho de treinamento do modelo
- Execução de previsão em lote
- Visualização de resultados

### CHURN.ipynb
Análise detalhada incluindo:
- Análise Exploratória de Dados (EDA)
- Distribuições de características e correlações
- Análise de taxa de churn
- Métricas de desempenho do modelo
- Visualizações e insights

### history_analysis.ipynb
Análise histórica de dados aprofundada com:
- Análise de segmento de cliente
- Padrões de uso de serviço
- Tendências de faturamento
- Estatísticas detalhadas

## 📦 Dependências Principais

| Pacote | Versão | Propósito |
|--------|--------|----------|
| pandas | 2.3.3 | Manipulação de dados |
| scikit-learn | (via lightgbm) | Algoritmos ML & pré-processamento |
| lightgbm | 4.6.0 | Gradient boosting |
| fastapi | (incluído) | Framework REST API |
| numpy | 2.3.5 | Computação numérica |
| matplotlib | 3.10.7 | Visualização de dados |
| jupyter | (via ipykernel) | Notebooks interativos |

Veja `requirements.txt` para lista completa.

## 🔄 Fluxo de Trabalho

1. **Explorar Dados**: Use CHURN.ipynb para entender o conjunto de dados
2. **Preparar Dados**: process.py gerencia limpeza e engenharia de características
3. **Treinar Modelo**: Execute `python src/train.py` para construir o modelo
4. **Fazer Previsões**: Execute `python src/predict.py` para previsões em lote
5. **Implantar API**: Execute servidor FastAPI para previsões em tempo real
6. **Obter Insights**: Verifique recomendações da API para estratégias de retenção de clientes

## 📈 Desempenho do Modelo

O modelo de Regressão Logística com características categóricas codificadas em One-Hot fornece:
- Classificação binária (Churn: Sim/Não)
- Treinamento ponderado por classe para lidar com dados desbalanceados
- Importância de características através de coeficientes de regressão
- Recomendações acionáveis para retenção de clientes

## 🛠️ Engenharia de Características

A solução cria características derivadas a partir de dados brutos:

- **tenureCategory**: Permanência de cliente agrupada em 5 intervalos
- **MonthlyChargesCategory**: Cobranças mensais agrupadas em 3 brackets

Essas características categóricas melhoram a interpretabilidade e desempenho do modelo.

## 📄 Licença

Este projeto é licenciado sob a Apache License 2.0. Veja o arquivo `LICENSE` para detalhes.

## 🤝 Contribuindo

Para adicionar outros membros da equipe a este workspace:
1. Use o notebook `ADD_USERS.ipynb`
2. Ou use os recursos de compartilhamento de workspace do VS Code

## 📧 Suporte

Para dúvidas ou problemas, entre em contato com os autores do projeto ou consulte os comentários do notebook individual para detalhes de implementação específicos.
