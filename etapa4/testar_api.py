# testar_api.py
import requests
import json

def testar_api():
    """Testa a API atualizada de previsão de vendas"""
    
    print("🧪 TESTANDO API DE PREVISÃO DE VENDAS")
    print("==================================================")
    
    base_url = "http://127.0.0.1:5000"
    
    # Primeiro, testar se a API está online (usando endpoint raiz)
    print("🔍 Verificando se a API está online...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ API está online e funcionando!")
        else:
            print("❌ API retornou erro:", response.status_code)
            # Mesmo com erro, continuamos o teste
    except requests.exceptions.ConnectionError:
        print("❌ ERRO: API não está rodando!")
        print("💡 Execute primeiro: python app.py")
        return
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        # Continuamos mesmo com erro
    
    # Cenários de teste
    cenarios = {
        "ÓTIMO": {
            'marketing_eficiencia': 0.95,
            'payment_methods_Bank Transfer': 1,
            'marketing_spend': 0.9,
            'customer_value': 0.9,
            'competition_level': 0.1,
            'customer_reviews': 0.9,
            'discount_percentage': 5,
            'website_traffic': 0.9,
            'avg_product_rating': 0.95
        },
        "MÉDIO": {
            'marketing_eficiencia': 0.6,
            'payment_methods_Bank Transfer': 1,
            'marketing_spend': 0.6,
            'customer_value': 0.6,
            'competition_level': 0.5,
            'customer_reviews': 0.6,
            'discount_percentage': 15,
            'website_traffic': 0.6,
            'avg_product_rating': 0.7
        },
        "RUIM": {
            'marketing_eficiencia': 0.3,
            'payment_methods_Bank Transfer': 1,
            'marketing_spend': 0.3,
            'customer_value': 0.3,
            'competition_level': 0.9,
            'customer_reviews': 0.3,
            'discount_percentage': 30,
            'website_traffic': 0.3,
            'avg_product_rating': 0.4
        }
    }
    
    resultados = {}
    
    for nome, dados in cenarios.items():
        print(f"\n📊 CENÁRIO {nome}:")
        print("-" * 25)
        
        # Mostrar alguns dados principais
        principais = ['marketing_eficiencia', 'marketing_spend', 'competition_level']
        for key in principais:
            print(f"   {key}: {dados[key]}")
        
        # Fazer requisição
        print("📤 Enviando para /prever...")
        try:
            response = requests.post(f"{base_url}/prever", json=dados, timeout=10)
            
            if response.status_code == 200:
                resultado = response.json()
                print("✅ Sucesso!")
                
                previsao_real = resultado.get('previsao_real', 'N/A')
                categoria = resultado.get('categoria', 'N/A')
                emoji = resultado.get('emoji', '')
                confianca = resultado.get('confianca', 0)
                
                print(f"   💰 Vendas: R$ {previsao_real:,.2f}")
                print(f"   📊 {emoji} {categoria}")
                print(f"   🎯 Confiança: {confianca:.1%}")
                
                resultados[nome] = resultado
            else:
                print(f"❌ Erro {response.status_code}")
                print(f"   Detalhes: {response.text}")
                
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
    
    # Resumo final
    if resultados:
        print("\n🎯 RESUMO DOS RESULTADOS:")
        print("=" * 45)
        for cenario, resultado in resultados.items():
            previsao_real = resultado.get('previsao_real', 0)
            categoria = resultado.get('categoria', 'ERRO')
            emoji = resultado.get('emoji', '')
            confianca = resultado.get('confianca', 0)
            
            print(f"   {emoji} {cenario}: R$ {previsao_real:,.2f} | {categoria} | {confianca:.1%}")

if __name__ == "__main__":
    testar_api()