import pandas as pd
import joblib
import json

# Load scaler to get exact feature order
scaler = joblib.load('models/scaler.joblib')

print("="*70)
print("EXACT FEATURE ORDER REQUIRED")
print("="*70)

# Get feature names in correct order
if hasattr(scaler, 'feature_names_in_'):
    feature_names = scaler.feature_names_in_.tolist()
else:
    # Fallback: load from test.csv
    test_df = pd.read_csv('data/final/test.csv')
    feature_names = [col for col in test_df.columns if col != 'TX_FRAUD']

print(f"\nTotal features: {len(feature_names)}\n")
for i, feat in enumerate(feature_names, 1):
    print(f"{i:2d}. {feat}")

# Load a real fraud transaction
test_df = pd.read_csv('data/final/test.csv')
fraud_row = test_df[test_df['TX_FRAUD'] == 1].iloc[0]

# Create ordered dict
fraud_dict = {}
for feat in feature_names:
    fraud_dict[feat] = fraud_row[feat]

fraud_dict['scaled'] = False  # This is RAW data, needs scaling

print("\n" + "="*70)
print("POSTMAN JSON (CORRECT ORDER):")
print("="*70)
print(json.dumps(fraud_dict, indent=2))

# Save to file for easy copy
with open('postman_request.json', 'w') as f:
    json.dump(fraud_dict, f, indent=2)

print("\n✅ Also saved to: postman_request.json")