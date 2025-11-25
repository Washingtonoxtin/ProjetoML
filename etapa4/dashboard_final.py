# dashboard_final.py
import streamlit as st
import requests
import pandas as pd

st.title("🚀 Dashboard de Previsão de Vendas")
st.markdown("Previsões em tempo real usando XGBRegressor")

# Input das features principais
st.sidebar.header("📊 Parâmetros de Entrada")

marketing_eficiencia = st.sidebar.slider("Eficiência do Marketing", 0.0, 1.0, 0.8)
marketing_spend = st.sidebar.slider("Investimento em Marketing", 0.0, 1.0, 0.7)
website_traffic = st.sidebar.slider("Tráfego do Website", 0.0, 1.0, 0.7)
customer_value = st.sidebar.slider("Valor do Cliente", 0.0, 1.0, 0.7)
competition_level = st.sidebar.slider("Nível de Competição", 0.0, 1.0, 0.3)
customer_reviews = st.sidebar.slider("Avaliações dos Clientes", 0.0, 1.0, 0.8)
discount_percentage = st.sidebar.slider("Percentual de Desconto", 0, 50, 10)
avg_product_rating = st.sidebar.slider("Avaliação Média do Produto", 0.0, 1.0, 0.85)
payment_bank_transfer = st.sidebar.selectbox("Transferência Bancária", [1, 0])

if st.sidebar.button("🔮 Fazer Previsão"):
    dados = {
        'marketing_eficiencia': marketing_eficiencia,
        'payment_methods_Bank Transfer': payment_bank_transfer,
        'marketing_spend': marketing_spend,
        'customer_value': customer_value,
        'competition_level': competition_level,
        'customer_reviews': customer_reviews,
        'discount_percentage': discount_percentage,
        'website_traffic': website_traffic,
        'avg_product_rating': avg_product_rating
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/prever", json=dados)
        if response.status_code == 200:
            resultado = response.json()
            
            st.success("✅ Previsão realizada com sucesso!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Vendas Previstas", f"R$ {resultado['previsao_real']:,.2f}")
            
            with col2:
                st.metric("🎯 Categoria", f"{resultado['categoria']} {resultado['emoji']}")
            
            with col3:
                st.metric("🔒 Confiança", f"{resultado['confianca']:.1%}")
            
            # Gráfico de faixas
            st.subheader("📈 Faixas de Vendas")
            faixas = pd.DataFrame({
                'Categoria': ['BAIXAS', 'MÉDIAS', 'ALTAS'],
                'Limite Inferior': [0, 10000, 25000],
                'Limite Superior': [10000, 25000, 50000]
            })
            st.dataframe(faixas)
            
        else:
            st.error("❌ Erro na previsão")
    except:
        st.error("❌ Erro de conexão com a API")

st.info("💡 Ajuste os parâmetros na sidebar e clique em 'Fazer Previsão'")