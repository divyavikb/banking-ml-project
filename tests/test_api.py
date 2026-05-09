import pandas as pd
import joblib

# Load test data
test_df = pd.read_csv('data/final/test.csv')
model = joblib.load('models/best_model_tuned.joblib')

# Get a known fraud transaction
fraud_row = test_df[test_df['TX_FRAUD'] == 1].iloc[0]
X_fraud = fraud_row.drop('TX_FRAUD')

# Test WITHOUT scaling (data already scaled)
pred = model.predict([X_fraud])[0]
prob = model.predict_proba([X_fraud])[0][1]

print(f"Known fraud transaction:")
print(f"  Prediction: {pred}")
print(f"  Probability: {prob:.4f}")
print(f"  Should be: 1 (fraud) with high probability")

# Show first few values
print(f"\nFirst 5 feature values:")
print(X_fraud.head())