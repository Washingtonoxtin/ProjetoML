# create_correct_notebook.py
import json

def create_correct_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "introducao",
                "metadata": {},
                "source": [
                    "# 🎯 PRÉ-PROCESSAMENTO DE DADOS\n\n",
                    "## OBJETIVOS\n",
                    "- Tratamento de valores faltantes\n",
                    "- Encoding de variáveis categóricas\n", 
                    "- Tratamento de outliers\n",
                    "- Normalização dos dados\n",
                    "- Feature engineering\n",
                    "- Salvamento do dataset processado\n\n",
                    "## 📁 ARQUIVOS GERADOS\n",
                    "- `dataset_preprocessado.csv` - Dataset pré-processado\n",
                    "- `models/scaler.pkl` - Scaler para novas predições"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "importacoes",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- IMPORTAÇÕES NECESSÁRIAS ---\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "import joblib\n",
                    "import os\n\n",
                    "# Configurações de visualização\n",
                    "sns.set(style=\"whitegrid\", palette=\"pastel\")\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\n\n",
                    "print(\"✅ Bibliotecas importadas com sucesso!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "carregar-dados",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- CARREGAR DADOS ---\n",
                    "print(\"📥 Carregando dataset...\")\n",
                    "df = pd.read_csv(\"dataset_explorado.csv\")\n\n",
                    "print(f\"📊 Dimensões iniciais: {df.shape}\")\n",
                    "print(f\"🔍 Valores faltantes iniciais: {df.isna().sum().sum()}\")\n\n",
                    "# Visualizar primeiras linhas\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown", 
                "id": "valores-faltantes",
                "metadata": {},
                "source": [
                    "## 1. 🧹 TRATAMENTO DE VALORES FALTANTES"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "tratamento-faltantes",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- TRATAMENTO DE VALORES FALTANTES ---\n",
                    "print(\"🔍 Tratando valores faltantes...\")\n\n",
                    "# Estratégia direta e agressiva\n",
                    "for coluna in df.columns:\n",
                    "    if df[coluna].isna().sum() > 0:\n",
                    "        if df[coluna].dtype in ['int64', 'float64']:\n",
                    "            # Numéricas: mediana\n",
                    "            valor = df[coluna].median()\n",
                    "            df[coluna] = df[coluna].fillna(valor)\n",
                    "        else:\n",
                    "            # Categóricas: moda ou 'MISSING'\n",
                    "            if len(df[coluna].mode()) > 0:\n",
                    "                valor = df[coluna].mode()[0]\n",
                    "            else:\n",
                    "                valor = 'MISSING'\n",
                    "            df[coluna] = df[coluna].fillna(valor)\n",
                    "        print(f\"✅ {coluna}: {df[coluna].isna().sum()} faltantes restantes\")\n\n",
                    "print(f\"🎯 Valores faltantes totais: {df.isna().sum().sum()}\")"
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
    
    # Salvar notebook correto
    with open('02_Preprocessamento_CORRETO.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ NOTEBOOK CORRETO CRIADO!")
    print("📓 Arquivo: 02_Preprocessamento_CORRETO.ipynb")
    print("🚀 Execute no Jupyter!")

if __name__ == "__main__":
    create_correct_notebook()