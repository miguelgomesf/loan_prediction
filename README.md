# loan_prediction
Treinamento de modelos em python para previsão de aprovação de empréstimos com base em dados de clientes.

## Objetivo

Criar um pipeline de machine learning que:

- Trata dados faltantes
- Transforma dados para melhor aproveitamento dos modelos treinados
- Balanceia as classes com SMOTE
- Treina e avalia modelos Decision Tree, Random Forest e XGBoost
- Ajusta o treshold para aumentar precisão dos modelos Random Forest e XGBoost

## Dataset

O dataset foi obtido via kaggle:

> [Loan Status Prediction](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction/data)

*Dataset não incluido no repositório por questão de licença. Pode-se baixá-lo a partir do link acima*

## Como executar

1. Clone este repositório:
```bash
git clone https://github.com/miguelgomesf/loan_prediction.git
```
2. Instale as dependências
```bash
pip install -r requirements.txt
```
3. Execute o notebook
- Acesse o arquivo loan_prediction.ipynb via [Google Colab](https://colab.research.google.com/drive/1awOHU-io4M6RRyZvQBjzfINfpn7p7e7Y?usp=sharing)
- Ou localmente via Jupyter Notebook

## Tecnologias e bibliotecas
- Python 3
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Raciocínio por trás do projeto
