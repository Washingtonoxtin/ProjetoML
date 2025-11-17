import nbformat as nbf

# Nome do arquivo de saída
notebook_name = "Notebook_Melhorado.ipynb"

# Criar estrutura base
nb = nbf.v4.new_notebook()
cells = []

# ------------------ INTRO ------------------
cells.append(nbf.v4.new_markdown_cell(
"# 📘 ETAPA 3 — Modelo Baseline (Versão Melhorada)\n"
"Este notebook foi gerado automaticamente com ajustes adicionais:\n"
"- Gráfico Valores Reais vs Previstos\n"
"- Top 3 features mais importantes\n"
"- Organização mais clara para apresentação\n"
))

# ------------------ IMPORTAÇÕES ------------------
cells.append(nbf.v4.new_markdown_cell("## 1. Importações"))
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import os
"""
))

# ------------------ CARREGAR DADOS ------------------
cells.append(nbf.v4.new_markdown_cell("## 2. Carregar Dataset"))
cells.append(nbf.v4.new_code_cell(
"""df = pd.read_csv('dataset_preprocessado.csv')
print('Shape:', df.shape)
df.head()"""
))

# ------------------ PREPARAR DADOS ------------------
cells.append(nbf.v4.new_markdown_cell("## 3. Preparar Dados (X e y)"))
cells.append(nbf.v4.new_code_cell(
"""# Remover colunas irrelevantes
if 'sale_id' in df.columns:
    df = df.drop(columns=['sale_id'])

# Selecionar variável target
y = df['monthly_sales']
X = df.drop(columns=['monthly_sales'])

# Garantir que tudo é numérico
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print('X shape:', X.shape)
print('y shape:', y.shape)
"""
))

# ------------------ 60/20/20 SPLIT ------------------
cells.append(nbf.v4.new_markdown_cell("## 4. Divisão 60/20/20"))
cells.append(nbf.v4.new_code_cell(
"""# 60% treino, 40% temporário
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)

# 20% val, 20% teste
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print("Treino:", X_train.shape)
print("Validação:", X_val.shape)
print("Teste:", X_test.shape)
"""
))

# ------------------ TREINAR MODELO ------------------
cells.append(nbf.v4.new_markdown_cell("## 5. Treinar Modelo Linear"))
cells.append(nbf.v4.new_code_cell(
"""modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred_train = modelo.predict(X_train)
y_pred_val = modelo.predict(X_val)

print("Modelo treinado!")"""
))

# ------------------ MÉTRICAS ------------------
cells.append(nbf.v4.new_markdown_cell("## 6. Métricas de Desempenho"))
cells.append(nbf.v4.new_code_cell(
"""mse_train = mean_squared_error(y_train, y_pred_train)
mse_val = mean_squared_error(y_val, y_pred_val)

rmse_train = np.sqrt(mse_train)
rmse_val = np.sqrt(mse_val)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)

r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)

print("=== Métricas Treino ===")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAE:", mae_train)
print("R²:", r2_train)
print("\n=== Métricas Validação ===")
print("MSE:", mse_val)
print("RMSE:", rmse_val)
print("MAE:", mae_val)
print("R²:", r2_val)
"""
))

# ------------------ VALORES REAIS VS PREVISTOS ------------------
cells.append(nbf.v4.new_markdown_cell("## 7. Gráfico: Valores Reais vs Previstos (VAL)"))
cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
plt.scatter(y_val, y_pred_val, alpha=0.6)
plt.xlabel("Valores reais")
plt.ylabel("Previsões")
plt.title("Valores Reais vs Previstos - Validação")
plt.grid(True)
plt.show()
"""
))

# ------------------ RESÍDUOS ------------------
cells.append(nbf.v4.new_markdown_cell("## 8. Gráfico de Resíduos"))
cells.append(nbf.v4.new_code_cell(
"""residuos = y_val - y_pred_val

plt.figure(figsize=(7,5))
plt.scatter(y_pred_val, residuos, alpha=0.6)
plt.axhline(0, color='red')
plt.xlabel("Previsões")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Previsões")
plt.grid(True)
plt.show()
"""
))

# ------------------ TOP 3 FEATURES ------------------
cells.append(nbf.v4.new_markdown_cell("## 9. Top 3 Features Mais Importantes"))
cells.append(nbf.v4.new_code_cell(
"""coeficientes = pd.Series(modelo.coef_, index=X_train.columns)
top3 = coeficientes.abs().sort_values(ascending=False).head(3)
print("Top 3 features mais importantes:")
print(top3)
"""
))

# ------------------ SALVAR MODELO ------------------
cells.append(nbf.v4.new_markdown_cell("## 10. Salvar Modelo"))
cells.append(nbf.v4.new_code_cell(
"""os.makedirs('models', exist_ok=True)
joblib.dump(modelo, 'models/modelo_baseline_melhorado.pkl')
print("Modelo salvo em models/modelo_baseline_melhorado.pkl")
"""
))

# Finalizar
nb['cells'] = cells

# Salvar notebook
with open(notebook_name, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_name}' criado com sucesso!")
