# create_notebook.py
import json

def create_preprocessing_notebook():
    """Cria o notebook Jupyter completo do zero"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "introducao",
                "metadata": {},
                "source": [
                    "# 🎯 PRÉ-PROCESSAMENTO DE DADOS\\n\\n",
                    "## OBJETIVOS\\n",
                    "- Tratamento de valores faltantes\\n",
                    "- Encoding de variáveis categóricas\\n", 
                    "- Tratamento de outliers\\n",
                    "- Normalização dos dados\\n",
                    "- Feature engineering\\n",
                    "- Salvamento do dataset processado\\n\\n",
                    "## 📁 ARQUIVOS GERADOS\\n",
                    "- `dataset_preprocessado.csv` - Dataset pré-processado\\n",
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
                    "# --- IMPORTAÇÕES NECESSÁRIAS ---\\n",
                    "import pandas as pd\\n",
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "import seaborn as sns\\n",
                    "from sklearn.impute import SimpleImputer\\n",
                    "from sklearn.preprocessing import StandardScaler\\n",
                    "import joblib\\n",
                    "import os\\n\\n",
                    "# Configurações de visualização\\n",
                    "sns.set(style=\\\"whitegrid\\\", palette=\\\"pastel\\\")\\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\\n\\n",
                    "print(\\\"✅ Bibliotecas importadas com sucesso!\\\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "carregar-dados",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- CARREGAR DADOS ---\\n",
                    "print(\\\"📥 Carregando dataset...\\\")\\n",
                    "df = pd.read_csv(\\\"dataset_explorado.csv\\\")\\n\\n",
                    "print(f\\\"📊 Dimensões iniciais: {df.shape}\\\")\\n",
                    "print(f\\\"🔍 Valores faltantes iniciais: {df.isna().sum().sum()}\\\")\\n\\n",
                    "# Visualizar primeiras linhas\\n",
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
                    "# --- TRATAMENTO DE VALORES FALTANTES ---\\n",
                    "print(\\\"🔍 Tratando valores faltantes...\\\")\\n\\n",
                    "# Estratégia direta e agressiva\\n",
                    "for coluna in df.columns:\\n",
                    "    if df[coluna].isna().sum() > 0:\\n",
                    "        if df[coluna].dtype in ['int64', 'float64']:\\n",
                    "            # Numéricas: mediana\\n",
                    "            valor = df[coluna].median()\\n",
                    "            df[coluna] = df[coluna].fillna(valor)\\n",
                    "        else:\\n",
                    "            # Categóricas: moda ou 'MISSING'\\n",
                    "            if len(df[coluna].mode()) > 0:\\n",
                    "                valor = df[coluna].mode()[0]\\n",
                    "            else:\\n",
                    "                valor = 'MISSING'\\n",
                    "            df[coluna] = df[coluna].fillna(valor)\\n",
                    "        print(f\\\"✅ {coluna}: {df[coluna].isna().sum()} faltantes restantes\\\")\\n\\n",
                    "print(f\\\"🎯 Valores faltantes totais: {df.isna().sum().sum()}\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "encoding-categorico", 
                "metadata": {},
                "source": [
                    "## 2. 🔄 ENCODING DE VARIÁVEIS CATEGÓRICAS"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "aplicar-encoding",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- ENCODING CATEGÓRICO ---\\n",
                    "print(\\\"🔍 Aplicando encoding categórico...\\\")\\n\\n",
                    "# One-Hot Encoding\\n",
                    "categoricas = ['product_category', 'payment_methods']\\n",
                    "for col in categoricas:\\n",
                    "    if col in df.columns:\\n",
                    "        df = pd.get_dummies(df, columns=[col], drop_first=True, prefix=col)\\n\\n",
                    "# Encoding ordinal\\n",
                    "map_ordinal = {'Low': 0, 'Medium': 1, 'High': 2}\\n",
                    "df['competition_level'] = df['competition_level'].map(map_ordinal)\\n",
                    "df['seasonality'] = df['seasonality'].map(map_ordinal)\\n\\n",
                    "# Encoding binário\\n",
                    "df['free_shipping'] = df['free_shipping'].map({'Yes': 1, 'No': 0})\\n\\n",
                    "print(f\\\"✅ Encoding aplicado! Novo formato: {df.shape}\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "tratamento-outliers",
                "metadata": {},
                "source": [
                    "## 3. 📊 TRATAMENTO DE OUTLIERS"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "aplicar-outliers",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- TRATAMENTO DE OUTLIERS ---\\n",
                    "print(\\\"🔍 Tratando outliers...\\\")\\n\\n",
                    "numericas = ['marketing_spend', 'website_traffic', 'avg_price', 'conversion_rate']\\n\\n",
                    "for col in numericas:\\n",
                    "    if col in df.columns:\\n",
                    "        Q1, Q3 = df[col].quantile([0.25, 0.75])\\n",
                    "        IQR = Q3 - Q1\\n",
                    "        limite_inf = Q1 - 1.5 * IQR\\n",
                    "        limite_sup = Q3 + 1.5 * IQR\\n\\n",
                    "        # Aplicar limites\\n",
                    "        df[col] = np.clip(df[col], limite_inf, limite_sup)\\n",
                    "        print(f\\\"✅ {col}: outliers tratados\\\")\\n\\n",
                    "print(\\\"🎯 Outliers tratados com sucesso!\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "normalizacao",
                "metadata": {},
                "source": [
                    "## 4. ⚖️ NORMALIZAÇÃO (SCALING)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "aplicar-normalizacao",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- NORMALIZAÇÃO ---\\n",
                    "print(\\\"🔍 Aplicando normalização...\\\")\\n\\n",
                    "colunas_normalizar = [\\n",
                    "    'marketing_spend', 'website_traffic', 'conversion_rate',\\n",
                    "    'avg_product_rating', 'avg_price', 'customer_reviews',\\n",
                    "    'return_rate', 'monthly_sales'\\n",
                    "]\\n\\n",
                    "# Filtrar colunas existentes\\n",
                    "colunas_normalizar = [col for col in colunas_normalizar if col in df.columns]\\n\\n",
                    "# Aplicar scaler\\n",
                    "scaler = StandardScaler()\\n",
                    "df[colunas_normalizar] = scaler.fit_transform(df[colunas_normalizar])\\n\\n",
                    "# Salvar scaler\\n",
                    "os.makedirs(\\\"models\\\", exist_ok=True)\\n",
                    "joblib.dump(scaler, \\\"models/scaler.pkl\\\")\\n\\n",
                    "print(f\\\"✅ Normalização aplicada em {len(colunas_normalizar)} colunas\\\")\\n",
                    "print(\\\"💾 Scaler salvo em 'models/scaler.pkl'\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "feature-engineering",
                "metadata": {},
                "source": [
                    "## 5. 🎯 FEATURE ENGINEERING"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "criar-features",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- FEATURE ENGINEERING ---\\n",
                    "print(\\\"🔍 Criando novas features...\\\")\\n\\n",
                    "df['marketing_eficiencia'] = df['monthly_sales'] / (df['marketing_spend'] + 1)\\n",
                    "df['traffic_conversion'] = df['website_traffic'] * df['conversion_rate']\\n",
                    "df['price_rating_ratio'] = df['avg_price'] / (df['avg_product_rating'] + 1)\\n",
                    "df['customer_value'] = df['monthly_sales'] / (df['customer_reviews'] + 1)\\n\\n",
                    "print(\\\"✅ 4 novas features criadas!\\\")\\n\\n",
                    "# Mostrar estatísticas das novas features\\n",
                    "novas_features = ['marketing_eficiencia', 'traffic_conversion', 'price_rating_ratio', 'customer_value']\\n",
                    "for feature in novas_features:\\n",
                    "    if feature in df.columns:\\n",
                    "        print(f\\\"📊 {feature}: mean={df[feature].mean():.4f}, std={df[feature].std():.4f}\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "salvamento",
                "metadata": {},
                "source": [
                    "## 6. 💾 SALVAMENTO FINAL"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "salvar-dataset",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- SALVAR DATASET FINAL ---\\n",
                    "df.to_csv(\\\"dataset_preprocessado.csv\\\", index=False)\\n\\n",
                    "print(\\\"💾 Dataset salvo como 'dataset_preprocessado.csv'\\\")\\n",
                    "print(f\\\"📊 Dimensões finais: {df.shape}\\\")\\n",
                    "print(f\\\"🔍 Valores faltantes finais: {df.isna().sum().sum()}\\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "verificacao-final",
                "metadata": {},
                "source": [
                    "## 7. ✅ VERIFICAÇÃO FINAL"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "verificacao-completa",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- VERIFICAÇÃO FINAL ---\\n",
                    "print(\\\"🎯 VERIFICAÇÃO FINAL:\\\")\\n",
                    "print(\\\"=\" * 50)\\n\\n",
                    "# 1. Valores faltantes\\n",
                    "missing_final = df.isna().sum().sum()\\n",
                    "print(f\\\"🔍 Valores faltantes: {'✅ ZERO' if missing_final == 0 else f'❌ {missing_final}'}\\\")\\n\\n",
                    "# 2. Dimensões\\n",
                    "print(f\\\"📊 Formato do dataset: {df.shape}\\\")\\n\\n",
                    "# 3. Normalização\\n",
                    "if colunas_normalizar:\\n",
                    "    mean_check = df[colunas_normalizar].mean().abs().max()\\n",
                    "    std_check = df[colunas_normalizar].std().mean()\\n",
                    "    print(f\\\"📈 Médias após scaling: {mean_check:.4f} (deve ser ~0)\\\")\\n",
                    "    print(f\\\"📈 Desvio padrão médio: {std_check:.4f} (deve ser ~1)\\\")\\n\\n",
                    "# 4. Arquivos salvos\\n",
                    "scaler_exists = os.path.exists(\\\"models/scaler.pkl\\\")\\n",
                    "dataset_exists = os.path.exists(\\\"dataset_preprocessado.csv\\\")\\n",
                    "print(f\\\"💾 Scaler salvo: {'✅ SIM' if scaler_exists else '❌ NÃO'}\\\")\\n",
                    "print(f\\\"💾 Dataset salvo: {'✅ SIM' if dataset_exists else '❌ NÃO'}\\\")\\n\\n",
                    "# 5. Novas features\\n",
                    "novas_features = ['marketing_eficiencia', 'traffic_conversion', 'price_rating_ratio', 'customer_value']\\n",
                    "features_count = sum([1 for f in novas_features if f in df.columns])\\n",
                    "print(f\\\"🆕 Features criadas: {features_count}/{len(novas_features)}\\\")\\n\\n",
                    "print(\\\"=\" * 50)\\n\\n",
                    "# Verificação de sucesso\\n",
                    "if missing_final == 0 and scaler_exists and dataset_exists and features_count == len(novas_features):\\n",
                    "    print(\\\"\\\\n🎊 PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!\\\")\\n",
                    "    print(\\\"🚀 O dataset está pronto para modelagem!\\\")\\n",
                    "else:\\n",
                    "    print(\\\"\\\\n⚠️  ALGUMAS VERIFICAÇÕES FALHARAM!\\\")\\n\\n",
                    "print(\\\"\\\\n📋 RESUMO FINAL:\\\")\\n",
                    "print(f\\\"   • Features finais: {len(df.columns)}\\\")\\n",
                    "print(f\\\"   • Linhas processadas: {df.shape[0]}\\\")\\n",
                    "print(f\\\"   • Novas features: +{features_count}\\\")\\n",
                    "print(\\\"   • Arquivo: dataset_preprocessado.csv\\\")"
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
    
    # Salvar como arquivo .ipynb
    with open('02_Preprocessamento.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    print("✅ NOTEBOOK CRIADO COM SUCESSO!")
    print("📓 Arquivo: 02_Preprocessamento.ipynb")
    print("🚀 Execute no Jupyter e siga as células!")

if __name__ == "__main__":
    create_preprocessing_notebook()