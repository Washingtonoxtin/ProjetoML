# fix_baseline.py
import json

def create_simple_fixed_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "introducao",
                "metadata": {},
                "source": ["# ETAPA 3: MODELO BASELINE - CORRIGIDO"]
            },
            {
                "cell_type": "markdown",
                "id": "importacoes",
                "metadata": {},
                "source": ["## 1. IMPORTAÇÕES"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "importacoes-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd",
                    "import numpy as np",
                    "import matplotlib.pyplot as plt",
                    "from sklearn.model_selection import train_test_split",
                    "from sklearn.linear_model import LinearRegression",
                    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",
                    "import joblib",
                    "import os",
                    "",
                    "print('Bibliotecas importadas!')"
                ]
            },
            {
                "cell_type": "markdown", 
                "id": "carregar-dados",
                "metadata": {},
                "source": ["## 2. CARREGAR DADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "carregar-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_csv('dataset_preprocessado.csv')",
                    "print(f'Dataset: {df.shape}')",
                    "print('Colunas:', list(df.columns))"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "corrigir-dados",
                "metadata": {}, 
                "source": ["## 3. CORRIGIR DADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "corrigir-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# REMOVER COLUNA sale_id",
                    "if 'sale_id' in df.columns:",
                    "    df = df.drop('sale_id', axis=1)",
                    "    print('Coluna sale_id removida')",
                    "",
                    "# CONVERTER TODAS AS COLUNAS OBJECT PARA NUMÉRICAS",
                    "object_cols = df.select_dtypes(include=['object']).columns",
                    "for col in object_cols:",
                    "    if col != 'monthly_sales':",
                    "        # Tentar converter para numérico",
                    "        df[col] = pd.to_numeric(df[col], errors='coerce')",
                    "        # Preencher NaN com 0",
                    "        df[col] = df[col].fillna(0)",
                    "        print(f'Convertida: {col}')",
                    "",
                    "print('Todas as colunas convertidas!')",
                    "print('Tipos finais:', df.dtypes.value_counts())"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "preparar-dados",
                "metadata": {},
                "source": ["## 4. PREPARAR DADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "preparar-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "X = df.drop('monthly_sales', axis=1)",
                    "y = df['monthly_sales']",
                    "",
                    "print(f'X: {X.shape}, y: {y.shape}')",
                    "print('Todas as colunas são numéricas:', X.select_dtypes(exclude=['number']).shape[1] == 0)"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "divisao-dados",
                "metadata": {},
                "source": ["## 5. DIVISÃO DOS DADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "divisao-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
                    "X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)",
                    "",
                    "print(f'Treino: {X_train.shape}')",
                    "print(f'Validação: {X_val.shape}')",
                    "print(f'Teste: {X_test.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "treinar-modelo",
                "metadata": {},
                "source": ["## 6. TREINAR MODELO"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "treinar-modelo-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "modelo = LinearRegression()",
                    "modelo.fit(X_train, y_train)",
                    "",
                    "print('Modelo treinado!')",
                    "",
                    "y_pred_train = modelo.predict(X_train)",
                    "y_pred_val = modelo.predict(X_val)",
                    "",
                    "print('Previsões feitas!')"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "metricas-avaliacao",
                "metadata": {},
                "source": ["## 7. MÉTRICAS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "metricas-avaliacao-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "r2_treino = r2_score(y_train, y_pred_train)",
                    "r2_val = r2_score(y_val, y_pred_val)",
                    "",
                    "print(f'R² Treino: {r2_treino:.4f}')",
                    "print(f'R² Validação: {r2_val:.4f}')",
                    "print(f'Diferença: {abs(r2_treino - r2_val):.4f}')"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "salvar-modelo",
                "metadata": {},
                "source": ["## 8. SALVAR MODELO"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "salvar-modelo-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('models', exist_ok=True)",
                    "joblib.dump(modelo, 'models/modelo_baseline.pkl')",
                    "print('Modelo salvo!')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('03_Modelo_Baseline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("NOTEBOOK CRIADO: 03_Modelo_Baseline.ipynb")

if __name__ == "__main__":
    create_simple_fixed_notebook()