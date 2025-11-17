# fix_sale_id_issue.py
import json

def create_fixed_baseline():
    """Cria um notebook que remove a coluna sale_id antes do treinamento"""
    
    notebook = {
        "cells": [
            # CÉLULA 1 - INTRODUÇÃO
            {
                "cell_type": "markdown",
                "id": "introducao",
                "metadata": {},
                "source": [
                    "# ETAPA 3: MODELO BASELINE - REGRESSÃO LINEAR",
                    "",
                    "⚠️ **CORREÇÃO:** Removendo coluna 'sale_id' que causa erro no modelo",
                    "",
                    "## OBJETIVOS",
                    "- Criar primeiro modelo de Machine Learning",
                    "- Avaliar performance com métricas robustas", 
                    "- Identificar overfitting",
                    "- Analisar features mais importantes",
                    "- Estabelecer baseline para comparação futura"
                ]
            },
            # CÉLULA 2 - IMPORTAÇÕES
            {
                "cell_type": "markdown",
                "id": "importacoes",
                "metadata": {},
                "source": ["## 1. IMPORTAÇÕES E CONFIGURAÇÕES"]
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
                    "import seaborn as sns",
                    "from sklearn.model_selection import train_test_split",
                    "from sklearn.linear_model import LinearRegression",
                    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",
                    "import joblib",
                    "import os",
                    "",
                    "sns.set_style('whitegrid')",
                    "plt.rcParams['figure.figsize'] = (12, 6)",
                    "",
                    "print('✅ Bibliotecas importadas!')"
                ]
            },
            # CÉLULA 3 - CARREGAR DADOS
            {
                "cell_type": "markdown", 
                "id": "carregar-dados",
                "metadata": {},
                "source": ["## 2. CARREGAR DADOS PRÉ-PROCESSADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "carregar-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('📥 Carregando dataset...')",
                    "df = pd.read_csv('dataset_preprocessado.csv')",
                    "",
                    "print(f'✅ Dataset carregado: {df.shape}')",
                    "print('🔍 Primeiras linhas:')",
                    "display(df.head(2))",
                    "",
                    "print('📋 Colunas disponíveis:')",
                    "print(list(df.columns))"
                ]
            },
            # CÉLULA 4 - PREPARAR DADOS (COM CORREÇÃO)
            {
                "cell_type": "markdown",
                "id": "preparar-dados",
                "metadata": {}, 
                "source": ["## 3. PREPARAR DADOS PARA MODELAGEM"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "preparar-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('🎯 Preparando dados para modelagem...')",
                    "",
                    "target = 'monthly_sales'",
                    "",
                    "if target not in df.columns:",
                    "    print('❌ ERRO: Coluna target não encontrada!')",
                    "    print('Colunas disponíveis:', list(df.columns))",
                    "else:",
                    "    # REMOVER COLUNAS NÃO NUMÉRICAS (como sale_id)",
                    "    colunas_para_remover = ['sale_id']  # Adicione outras colunas não numéricas se necessário",
                    "    ",
                    "    # Verificar quais colunas existem no dataset",
                    "    colunas_existentes = [col for col in colunas_para_remover if col in df.columns]",
                    "    ",
                    "    if colunas_existentes:",
                    "        print(f'🚫 Removendo colunas não numéricas: {colunas_existentes}')",
                    "        X = df.drop(colunas_existentes + [target], axis=1)",
                    "    else:",
                    "        X = df.drop(target, axis=1)",
                    "    ",
                    "    y = df[target]",
                    "    ",
                    "    print('✅ Dados preparados:')",
                    "    print(f'   🎯 Target: {target}')",
                    "    print(f'   📈 Features: {X.shape[1]} colunas')",
                    "    print(f'   📊 Amostras: {X.shape[0]} linhas')",
                    "    print(f'   📐 Média target: {y.mean():.2f}')",
                    "    ",
                    "    # Verificar tipos de dados das features",
                    "    print(f'   🔍 Tipos de dados das features:')",
                    "    print(X.dtypes.value_counts())"
                ]
            },
            # CÉLULA 5 - DIVISÃO DOS DADOS
            {
                "cell_type": "markdown",
                "id": "divisao-dados",
                "metadata": {},
                "source": ["## 4. DIVISÃO DOS DADOS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "divisao-dados-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('📊 Dividindo dados...')",
                    "",
                    "X_temp, X_test, y_temp, y_test = train_test_split(",
                    "    X, y, test_size=0.2, random_state=42, shuffle=True",
                    ")",
                    "",
                    "X_train, X_val, y_train, y_val = train_test_split(",
                    "    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True",
                    ")",
                    "",
                    "print('✅ Dados divididos:')",
                    "print(f'   🟢 Treino: {X_train.shape[0]} amostras')",
                    "print(f'   🟡 Validação: {X_val.shape[0]} amostras')",
                    "print(f'   🔴 Teste: {X_test.shape[0]} amostras')",
                    "",
                    "# Verificar se todas as features são numéricas",
                    "print(f'   🔍 Verificando tipos de dados:')",
                    "print(f'      X_train dtypes: {X_train.dtypes.unique()}')",
                    "print(f'      Todas as features são numéricas: {X_train.select_dtypes(include=[\"number\"]).shape[1] == X_train.shape[1]}')"
                ]
            },
            # CÉLULA 6 - TREINAR MODELO (CORRIGIDO)
            {
                "cell_type": "markdown",
                "id": "treinar-modelo",
                "metadata": {},
                "source": ["## 5. TREINAR MODELO"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "treinar-modelo-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('🤖 Iniciando treinamento do modelo...')",
                    "",
                    "# VERIFICAÇÃO FINAL: garantir que todas as features são numéricas",
                    "non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns",
                    "if len(non_numeric_cols) > 0:",
                    "    print(f'❌ ERRO: Features não numéricas encontradas: {list(non_numeric_cols)}')",
                    "    print('💡 Remova essas colunas antes de continuar')",
                    "else:",
                    "    print('✅ Todas as features são numéricas!')",
                    "    ",
                    "    # Criar e treinar modelo",
                    "    modelo = LinearRegression()",
                    "    ",
                    "    print('📦 Treinando modelo LinearRegression...')",
                    "    modelo.fit(X_train, y_train)",
                    "    ",
                    "    print('✅ Modelo treinado com sucesso!')",
                    "    print(f'   📐 Coeficientes: {len(modelo.coef_)}')",
                    "    print(f'   📍 Intercept: {modelo.intercept_:.4f}')",
                    "    ",
                    "    # Fazer previsões",
                    "    y_pred_train = modelo.predict(X_train)",
                    "    y_pred_val = modelo.predict(X_val)",
                    "    ",
                    "    print('🎯 Previsões realizadas:')",
                    "    print(f'   📈 Treino: {len(y_pred_train)} previsões')",
                    "    print(f'   📊 Validação: {len(y_pred_val)} previsões')"
                ]
            },
            # CÉLULA 7 - MÉTRICAS
            {
                "cell_type": "markdown",
                "id": "metricas-avaliacao",
                "metadata": {},
                "source": ["## 6. CALCULAR MÉTRICAS"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "metricas-avaliacao-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('📈 Calculando métricas...')",
                    "",
                    "def calcular_metricas(y_real, y_pred, nome):",
                    "    mse = mean_squared_error(y_real, y_pred)",
                    "    rmse = np.sqrt(mse)",
                    "    mae = mean_absolute_error(y_real, y_pred)",
                    "    r2 = r2_score(y_real, y_pred)",
                    "    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}",
                    "",
                    "metricas_treino = calcular_metricas(y_train, y_pred_train, 'treino')",
                    "metricas_val = calcular_metricas(y_val, y_pred_val, 'validação')",
                    "",
                    "print('📊 MÉTRICAS - TREINO:')",
                    "print(f'   MSE:  {metricas_treino[\"MSE\"]:>10.4f}')",
                    "print(f'   RMSE: {metricas_treino[\"RMSE\"]:>10.4f}')",
                    "print(f'   MAE:  {metricas_treino[\"MAE\"]:>10.4f}')",
                    "print(f'   R²:   {metricas_treino[\"R2\"]:>10.4f}')",
                    "",
                    "print('📊 MÉTRICAS - VALIDAÇÃO:')",
                    "print(f'   MSE:  {metricas_val[\"MSE\"]:>10.4f}')",
                    "print(f'   RMSE: {metricas_val[\"RMSE\"]:>10.4f}')",
                    "print(f'   MAE:  {metricas_val[\"MAE\"]:>10.4f}')",
                    "print(f'   R²:   {metricas_val[\"R2\"]:>10.4f}')",
                    "",
                    "# Análise de overfitting",
                    "diferenca = abs(metricas_treino['R2'] - metricas_val['R2'])",
                    "print(f'🔍 DIFERENÇA R²: {diferenca:.4f}')",
                    "",
                    "if diferenca < 0.05:",
                    "    print('   ✅ EXCELENTE - Modelo generaliza bem')",
                    "elif diferenca < 0.10:",
                    "    print('   ⚠️  BOM - Pequeno overfitting')",
                    "elif diferenca < 0.15:",
                    "    print('   🔶 MODERADO - Overfitting presente')",
                    "else:",
                    "    print('   ❌ ALTO - Overfitting significativo')"
                ]
            },
            # CÉLULA 8 - FEATURE IMPORTANCE
            {
                "cell_type": "markdown",
                "id": "feature-importance",
                "metadata": {},
                "source": ["## 7. FEATURE IMPORTANCE"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "feature-importance-codigo",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('🎯 Analisando features mais importantes...')",
                    "",
                    "features_df = pd.DataFrame({",
                    "    'Feature': X.columns,",
                    "    'Coeficiente': modelo.coef_,",
                    "    'Impacto': abs(modelo.coef_)",
                    "}).sort_values('Impacto', ascending=False)",
                    "",
                    "print('📊 TOP 5 FEATURES MAIS IMPORTANTES:')",
                    "print('-' * 50)",
                    "for i, row in features_df.head().iterrows():",
                    "    sinal = '+' if row['Coeficiente'] > 0 else '-'",
                    "    print(f'   {sinal} {row[\"Feature\"]:<25} | {row[\"Coeficiente\"]:>8.4f}')",
                    "",
                    "# Visualização",
                    "plt.figure(figsize=(10, 6))",
                    "top_5 = features_df.head()",
                    "cores = ['#2E86AB' if x > 0 else '#A23B72' for x in top_5['Coeficiente']]",
                    "",
                    "plt.barh(top_5['Feature'], top_5['Impacto'], color=cores)",
                    "plt.xlabel('Importância Absoluta')",
                    "plt.title('Top 5 Features Mais Importantes')",
                    "plt.gca().invert_yaxis()",
                    "plt.grid(axis='x', alpha=0.3)",
                    "plt.tight_layout()",
                    "plt.show()"
                ]
            },
            # CÉLULA 9 - SALVAR MODELO
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
                    "print('💾 Salvando modelo...')",
                    "",
                    "os.makedirs('models', exist_ok=True)",
                    "joblib.dump(modelo, 'models/modelo_baseline.pkl')",
                    "features_df.to_csv('models/feature_importance_baseline.csv', index=False)",
                    "",
                    "print('✅ Modelo salvo: models/modelo_baseline.pkl')",
                    "print('✅ Features salvas: models/feature_importance_baseline.csv')",
                    "",
                    "print('')",
                    "print('🎉 ETAPA 3 CONCLUÍDA COM SUCESSO!')",
                    "print('🚀 Problema da coluna sale_id resolvido!')"
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
    
    # Salvar o notebook
    with open('03_Modelo_Baseline_CORRIGIDO.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ NOTEBOOK CRIADO: 03_Modelo_Baseline_CORRIGIDO.ipynb")
    print("🔧 CORREÇÃO: Removendo coluna 'sale_id' que causava erro")
    print("🚀 AGORA DEVE FUNCIONAR!")

if __name__ == "__main__":
    create_fixed_baseline()