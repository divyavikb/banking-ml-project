"""
Simple Flask API for Fraud Detection
Reuses prediction logic from predict.py
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add src to path so we can import
sys.path.append(str(Path(__file__).parent / "src" / "models"))

# Setup
app = Flask(__name__)

# Load model and scaler
MODEL_PATH = Path("models/best_model_tuned.joblib")
SCALER_PATH = Path("models/scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    scaler = None


def engineer_features(data_dict):
    """
    Compute engineered features and return in CORRECT ORDER
    """
    
    # Compute engineered features
    if 'TX_TIME_SECONDS' in data_dict:
        data_dict['hour'] = int((data_dict['TX_TIME_SECONDS'] // 3600) % 24)
        data_dict['is_night'] = 1 if (data_dict['hour'] >= 22 or data_dict['hour'] <= 6) else 0
    
    if 'TX_TIME_DAYS' in data_dict:
        data_dict['day'] = int(data_dict['TX_TIME_DAYS'] % 31) + 1
        data_dict['month'] = int((data_dict['TX_TIME_DAYS'] // 30) % 12) + 1
        data_dict['weekday'] = int(data_dict['TX_TIME_DAYS'] % 7)
        data_dict['is_weekend'] = 1 if data_dict['weekday'] >= 5 else 0
    
    if 'TX_AMOUNT' in data_dict and 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW' in data_dict:
        data_dict['amount_deviation'] = data_dict['TX_AMOUNT'] - data_dict['CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
    
    if 'CUSTOMER_ID_NB_TX_1DAY_WINDOW' in data_dict and 'CUSTOMER_ID_NB_TX_30DAY_WINDOW' in data_dict:
        data_dict['tx_velocity'] = data_dict['CUSTOMER_ID_NB_TX_1DAY_WINDOW'] / (data_dict['CUSTOMER_ID_NB_TX_30DAY_WINDOW'] + 1)
    
    if 'TX_AMOUNT' in data_dict:
        data_dict['high_amount'] = 1 if data_dict['TX_AMOUNT'] > 200 else 0
    
    # CRITICAL: Return features in EXACT order scaler expects
    feature_order = [
        'TX_AMOUNT',
        'TX_TIME_SECONDS',
        'TX_TIME_DAYS',
        'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
        'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
        'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
        'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
        'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
        'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
        'TERMINAL_ID_NB_TX_1DAY_WINDOW',
        'TERMINAL_ID_RISK_1DAY_WINDOW',
        'TERMINAL_ID_NB_TX_7DAY_WINDOW',
        'TERMINAL_ID_RISK_7DAY_WINDOW',
        'TERMINAL_ID_NB_TX_30DAY_WINDOW',
        'TERMINAL_ID_RISK_30DAY_WINDOW',
        'hour',
        'day',
        'month',
        'weekday',
        'is_weekend',
        'is_night',
        'amount_deviation',
        'high_amount',
        'tx_velocity'
    ]
    
    # Create ordered dict
    ordered_data = {key: data_dict.get(key, 0) for key in feature_order}
    
    return ordered_data


def make_prediction(data):
    """
    Simple prediction function
    Input: dict or list of dicts
    Output: predictions
    """
    if model is None or scaler is None:
        raise ValueError("Model not loaded")
    
    # Convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Scale
    df_scaled = scaler.transform(df)
    
    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    
    return predictions, probabilities


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fraud Detection API',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Single transaction',
            '/predict/batch': 'POST - Multiple transactions',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    """Single transaction prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if data is already engineered/scaled
        is_scaled = data.pop('scaled', False)
        
        # If raw data, compute engineered features
        if not is_scaled:
            data = engineer_features(data)
        
        # Make prediction
        predictions, probabilities = make_prediction(data)
        
        prediction = int(predictions[0])
        probability = float(probabilities[0])
        
        # Risk level
        if probability >= 0.8:
            risk_level = "high"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return jsonify({
            'fraud_prediction': prediction,
            'fraud_probability': probability,
            'risk_level': risk_level,
            'message': 'Fraudulent transaction detected!' if prediction == 1 else 'Transaction appears legitimate'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400





if __name__ == '__main__':
    print("🚀 Starting Fraud Detection API...")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📂 Scaler: {SCALER_PATH}")
    
    # Run
    app.run(debug=True, host='0.0.0.0', port=5000)