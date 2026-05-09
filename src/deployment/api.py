"""
Flask API for Fraud Detection
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None


def load_model_and_scaler():
    """Load trained model and scaler"""
    global model, scaler
    
    try:
        # Load model
        model_path = Path('models/best_model_tuned.joblib')
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        logger.info(f"   Model type: {type(model).__name__}")
        
        # Load scaler
        scaler_path = Path('models/scaler.joblib')
        scaler = joblib.load(scaler_path)
        logger.info(f"✅ Scaler loaded from {scaler_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model/scaler: {e}")
        return False


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fraud Detection API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Single transaction prediction',
            '/predict/batch': 'POST - Batch predictions',
            '/model/info': 'GET - Model information'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make fraud prediction for a single transaction
    
    Request body (JSON):
    {
        "TX_AMOUNT": 150.0,
        "TX_TIME_SECONDS": 82800,
        ... (all features)
        "scaled": false  # Optional: set to true if data is already scaled
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Check if data is already scaled
        is_scaled = data.pop('scaled', False)  # Default: False (raw data)
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Scale features only if not already scaled
        if is_scaled:
            logger.info("Data is already scaled, skipping scaling")
            df_processed = df
        else:
            logger.info("Scaling raw data")
            df_processed = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "high"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Response
        result = {
            'fraud_prediction': int(prediction),
            'fraud_probability': float(probability),
            'risk_level': risk_level,
            'message': 'Fraudulent transaction detected!' if prediction == 1 else 'Transaction appears legitimate',
            'recommendation': 'Block transaction' if prediction == 1 else 'Approve transaction'
        }
        
        logger.info(f"Prediction: fraud={prediction}, prob={probability:.4f}, risk={risk_level}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """
    Make batch predictions
    
    Request body (JSON):
    {
        "transactions": [
            {"TX_AMOUNT": 150.0, "TX_TIME_SECONDS": 82800, ...},
            {"TX_AMOUNT": 50.0, "TX_TIME_SECONDS": 50400, ...}
        ]
    }
    
    Response:
    {
        "predictions": [
            {"transaction_id": 0, "fraud_prediction": 1, ...},
            {"transaction_id": 1, "fraud_prediction": 0, ...}
        ],
        "summary": {
            "total": 2,
            "fraudulent": 1,
            "legitimate": 1,
            "fraud_rate": "50.00%"
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        transactions = data['transactions']
        
        if not transactions:
            return jsonify({'error': 'Empty transactions list'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]
        
        # Build response
        results = []
        fraud_count = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob >= 0.8:
                risk_level = "high"
            elif prob >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            results.append({
                'transaction_id': i,
                'fraud_prediction': int(pred),
                'fraud_probability': float(prob),
                'risk_level': risk_level
            })
            
            if pred == 1:
                fraud_count += 1
        
        response = {
            'predictions': results,
            'summary': {
                'total': len(transactions),
                'fraudulent': fraud_count,
                'legitimate': len(transactions) - fraud_count,
                'fraud_rate': f"{(fraud_count/len(transactions)*100):.2f}%"
            }
        }
        
        logger.info(f"Batch prediction: {len(transactions)} transactions, {fraud_count} fraudulent")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'model_name': 'Fraud Detection Model',
            'model_type': type(model).__name__ if model else 'Not loaded',
            'version': '1.0.0',
            'description': 'Real-time credit card fraud detection',
            'features_required': [
                'TX_AMOUNT',
                'TX_TIME_SECONDS',
                'TX_TIME_DAYS',
                'TX_DURING_NIGHT',
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
                'tx_velocity',
                'high_amount'
            ],
            'performance_metrics': {
                'f1_score': '0.9089',
                'roc_auc': '0.9845',
                'precision': '0.8912',
                'recall': '0.9203',
                'accuracy': '0.9923'
            },
            'risk_levels': {
                'low': 'probability < 0.5',
                'medium': '0.5 <= probability < 0.8',
                'high': 'probability >= 0.8'
            }
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting Fraud Detection API")
    logger.info("=" * 60)
    
    # Load model and scaler
    if load_model_and_scaler():
        logger.info("✅ Model and scaler loaded successfully")
        logger.info("=" * 60)
        logger.info("🌐 API running on: http://localhost:5000")
        logger.info("📖 Endpoints:")
        logger.info("   GET  /              - Home")
        logger.info("   GET  /health        - Health check")
        logger.info("   POST /predict       - Single prediction")
        logger.info("   POST /predict/batch - Batch predictions")
        logger.info("   GET  /model/info    - Model information")
        logger.info("=" * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("❌ Failed to load model/scaler. Exiting.")
        logger.error("\n💡 Make sure these files exist:")
        logger.error("   - models/best_model_tuned.joblib")
        logger.error("   - models/scaler.joblib")