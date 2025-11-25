# app_corrigido_json.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Carregar modelo
try:
    modelo = joblib.load('modelo_final.pkl')
    print("✅ Modelo carregado com sucesso!")
    print(f"📊 Tipo do modelo: {type(modelo).__name__}")
    print(f"🎯 Features do modelo: {len(modelo.feature_names_in_)}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    exit()

def criar_template_features():
    """Cria template com todas as features esperadas pelo modelo"""
    features_esperadas = modelo.feature_names_in_
    
    template = {}
    for feature in features_esperadas:
        if any(x in feature for x in ['spend', 'traffic', 'rate', 'rating', 'value', 'efficiency', 'conversion']):
            template[feature] = 0.5
        elif 'percentage' in feature:
            template[feature] = 10
        elif 'price' in feature:
            template[feature] = 0.7
        elif 'products' in feature:
            template[feature] = 50
        elif 'level' in feature:
            template[feature] = 0.5
        elif any(x in feature for x in ['category', 'payment']):
            template[feature] = 0
        else:
            template[feature] = 0.5
    return template

def converter_para_json_serializable(obj):
    """Converte tipos numpy para tipos Python nativos para JSON"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route('/prever', methods=['POST'])
def prever():
    try:
        dados_recebidos = request.get_json()
        
        if not dados_recebidos:
            return jsonify({'mensagem': 'Nenhum dado fornecido', 'status': 'erro'}), 400
        
        print(f"📥 Dados recebidos: {len(dados_recebidos)} features")
        
        # Criar template completo
        dados_completos = criar_template_features()
        
        # Atualizar com dados recebidos
        for key, value in dados_recebidos.items():
            if key in dados_completos:
                dados_completos[key] = value
        
        # Converter para DataFrame
        features_ordenadas = modelo.feature_names_in_
        df = pd.DataFrame([dados_completos])[features_ordenadas]
        
        # Fazer predição
        previsao_normalizada = modelo.predict(df)[0]
        
        # Converter para tipo Python nativo
        previsao_normalizada = float(previsao_normalizada)
        
        # Converter para escala real (ajuste este multiplicador conforme necessário)
        if previsao_normalizada < 10:  # Se estiver normalizado
            previsao_real = previsao_normalizada * 10000
        else:
            previsao_real = previsao_normalizada
        
        previsao_real = max(0, previsao_real)
        
        # Categorizar
        if previsao_real < 10000:
            categoria, emoji = 'BAIXAS', '🔴'
        elif previsao_real < 25000:
            categoria, emoji = 'MÉDIAS', '🟡'
        else:
            categoria, emoji = 'ALTAS', '🟢'
        
        # Calcular confiança
        confianca = 0.7 + min(0.3, previsao_real / 50000)
        
        # Garantir que tudo seja serializável em JSON
        resposta = {
            'previsao_normalizada': previsao_normalizada,
            'previsao_real': previsao_real,
            'categoria': categoria,
            'emoji': emoji,
            'confianca': float(confianca),
            'faixas': {
                'BAIXAS': 'Até R$ 10.000',
                'MÉDIAS': 'R$ 10.000 - R$ 25.000', 
                'ALTAS': 'Acima de R$ 25.000'
            },
            'status': 'sucesso',
            'tipo_modelo': 'XGBRegressor'
        }
        
        # Aplicar conversão JSON a toda a resposta
        resposta_serializavel = {}
        for key, value in resposta.items():
            resposta_serializavel[key] = converter_para_json_serializable(value)
        
        return jsonify(resposta_serializavel)
        
    except Exception as e:
        return jsonify({
            'mensagem': f'Erro: {str(e)}', 
            'status': 'erro'
        }), 400

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'modelo': type(modelo).__name__,
        'num_features': len(modelo.feature_names_in_),
        'tipo': 'regressor'
    })

@app.route('/exemplo', methods=['GET'])
def exemplo():
    exemplo_dados = criar_template_features()
    features_principais = [
        'marketing_eficiencia', 'marketing_spend', 'website_traffic',
        'customer_value', 'competition_level', 'customer_reviews', 
        'discount_percentage', 'avg_product_rating', 'payment_methods_Bank Transfer'
    ]
    exemplo_simplificado = {k: exemplo_dados[k] for k in features_principais if k in exemplo_dados}
    
    return jsonify({
        'dados_exemplo': exemplo_simplificado,
        'instrucoes': 'Envie POST para /prever'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'mensagem': 'API de Previsão de Vendas - Corrigida',
        'endpoints': ['GET /', 'GET /info', 'GET /exemplo', 'POST /prever']
    })

if __name__ == '__main__':
    print("🚀 API CORRIGIDA - SEM ERROS JSON")
    print("📍 http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)