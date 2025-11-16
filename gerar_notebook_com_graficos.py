import nbformat as nbf

# Caminho do notebook original
input_path = "03_Modelo_Baseline.ipynb"

# Caminho de saída do novo notebook
output_path = "03_Modelo_Baseline_com_graficos.ipynb"

# Carregar notebook existente
nb = nbf.read(input_path, as_version=4)

# --- CÉLULAS DE GRÁFICOS A SEREM ADICIONADAS ---

cell_dist = nbf.v4.new_code_cell("""
# Gráfico: Distribuição da variável target
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(df['monthly_sales'], bins=30)
plt.title('Distribuição de monthly_sales')
plt.xlabel('monthly_sales')
plt.ylabel('Frequência')
plt.show()
""")

cell_coef = nbf.v4.new_code_cell("""
# Gráfico: Coeficientes do modelo linear
coef = modelo.coef_
features = X.columns
plt.figure(figsize=(8,6))
plt.barh(features, coef)
plt.title('Coeficientes do modelo linear')
plt.xlabel('Peso')
plt.ylabel('Feature')
plt.show()
""")

cell_resid = nbf.v4.new_code_cell("""
# Gráfico: Resíduos vs valores previstos
residuos = y_val - y_pred_val
plt.figure(figsize=(6,4))
plt.scatter(y_pred_val, residuos)
plt.axhline(0)
plt.title('Resíduos vs Previsões')
plt.xlabel('Previsões')
plt.ylabel('Resíduos')
plt.show()
""")

# Adicionar as novas células ao final do notebook
nb.cells.extend([cell_dist, cell_coef, cell_resid])

# Salvar novo notebook
nbf.write(nb, output_path)

print("Notebook gerado com sucesso:", output_path)
