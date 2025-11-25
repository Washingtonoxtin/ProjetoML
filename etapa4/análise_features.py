# análise_features.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

def analisar_features_importantes(modelo, feature_names, top_n=15):
    """Analisa as features mais importantes do modelo"""
    
    print("🔍 ANALISANDO FEATURES MAIS IMPORTANTES...")
    
    # Obter importância das features
    importancia = modelo.feature_importances_
    
    # Criar DataFrame
    df_importancia = pd.DataFrame({
        'feature': feature_names,
        'importance': importancia
    }).sort_values('importance', ascending=False)
    
    # Top N features
    top_features = df_importancia.head(top_n)
    
    print(f"\n🏆 TOP {top_n} FEATURES MAIS IMPORTANTES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:30} : {row['importance']:.4f}")
    
    # Gráfico
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Features Mais Importantes - XGBoost')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig('features_importantes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_features

# Carregar dados para obter nomes das features
print("📁 Carregando dados para análise...")
try:
    df = pd.read_csv('../dataset_preprocessado.csv')
    
    # Preparar dados como na Etapa 3
    if 'sale_id' in df.columns:
        df = df.drop(columns=['sale_id'])
    
    y = df['monthly_sales']
    X = df.drop(columns=['monthly_sales'])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    feature_names = X.columns.tolist()
    print(f"✅ {len(feature_names)} features carregadas")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    exit()

# Carregar modelo e analisar
print("\n🔧 Carregando modelo final...")
try:
    with open('modelo_final.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    print("✅ Modelo carregado com sucesso!")
    
    # Analisar features
    top_features = analisar_features_importantes(modelo, feature_names, top_n=15)
    
    print("\n💡 INSIGHTS DAS FEATURES:")
    print("   - Features no topo têm maior impacto nas previsões")
    print("   - Considere focar nessas features em análises futuras") 
    print("   - Features com baixa importância podem ser removidas")
    
    # Análise adicional: distribuição da importância
    print(f"\n📊 DISTRIBUIÇÃO DA IMPORTÂNCIA:")
    print(f"   Feature mais importante: {top_features.iloc[0]['feature']} ({top_features.iloc[0]['importance']:.4f})")
    print(f"   Soma top 5 features: {top_features.head(5)['importance'].sum():.4f}")
    print(f"   Soma todas as features: {top_features['importance'].sum():.4f}")
    
    # Salvar ranking completo
    top_features.to_csv('ranking_features.csv', index=False)
    print("💾 Ranking completo salvo como 'ranking_features.csv'")
    
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    print("💡 Certifique-se que 'modelo_final.pkl' existe na pasta")