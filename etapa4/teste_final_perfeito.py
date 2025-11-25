# teste_final_perfeito.py
import requests

print("🚀 TESTE FINAL - API 100% FUNCIONAL!")
print("=" * 40)

dados = {
    'marketing_eficiencia': 0.8,
    'payment_methods_Bank Transfer': 1,
    'marketing_spend': 0.7,
    'customer_value': 0.7,
    'competition_level': 0.3,
    'customer_reviews': 0.8,
    'discount_percentage': 10,
    'website_traffic': 0.7,
    'avg_product_rating': 0.85
}

try:
    response = requests.post("http://127.0.0.1:5000/prever", json=dados)
    print(f"📤 Status: {response.status_code}")
    
    if response.status_code == 200:
        resultado = response.json()
        print("🎉 🎉 🎉 SUCESSO TOTAL! 🎉 🎉 🎉")
        print(f"📊 Valor normalizado: {resultado['previsao_normalizada']}")
        print(f"💰 Vendas (escala real): R$ {resultado['previsao_real']:,.2f}")
        print(f"🎯 Categoria: {resultado['categoria']} {resultado['emoji']}")
        print(f"🔒 Confiança: {resultado['confianca']:.1%}")
        print(f"📝 Tipo do modelo: {resultado['tipo_modelo']}")
        
        print("\n⭐" + "="*50 + "⭐")
        print("   SEU PROJETO DE MACHINE LEARNING ESTÁ PRONTO!")
        print("⭐" + "="*50 + "⭐")
        
        # Teste adicional com cenário ótimo
        print("\n🔍 TESTANDO CENÁRIO ÓTIMO:")
        dados_otimo = dados.copy()
        dados_otimo.update({
            'marketing_eficiencia': 0.95,
            'marketing_spend': 0.9,
            'website_traffic': 0.9,
            'customer_value': 0.9,
            'competition_level': 0.1,
            'customer_reviews': 0.9,
            'discount_percentage': 5,
            'avg_product_rating': 0.95
        })
        
        response_otimo = requests.post("http://127.0.0.1:5000/prever", json=dados_otimo)
        if response_otimo.status_code == 200:
            resultado_otimo = response_otimo.json()
            print(f"💰 Vendas Ótimas: R$ {resultado_otimo['previsao_real']:,.2f}")
            print(f"🎯 Categoria: {resultado_otimo['categoria']} {resultado_otimo['emoji']}")
            
    else:
        print("❌ Erro na API:")
        print(response.json())
        
except Exception as e:
    print(f"💥 Erro de conexão: {e}")