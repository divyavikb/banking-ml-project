"""
Fix test.csv - Regenerate without NaN values
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import set_config

print("="*70)
print("FIXING test.csv")
print("="*70)

# Enable pandas output from sklearn
set_config(transform_output="pandas")

# Paths
root_path = Path.cwd()
features_path = root_path / "data" / "processed" / "features_engineered.csv"
scaler_path = root_path / "models" / "scaler.joblib"
output_path = root_path / "data" / "final" / "test.csv"

print(f"\n📂 Loading data from: {features_path}")

# Load original data
df = pd.read_csv(features_path)
print(f"✅ Loaded: {df.shape}")

# Separate features and target
X = df.drop(columns=['TX_FRAUD', 'TX_FRAUD_SCENARIO'])
y = df['TX_FRAUD']

print(f"\n📊 Features: {X.shape[1]}")
print(f"📊 Samples: {len(X)}")
print(f"📊 Fraud rate: {y.mean()*100:.2f}%")

# Split (same as train.py)
print(f"\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,}")

# Load scaler
print(f"\n📂 Loading scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)
print(f"✅ Scaler loaded")

# Transform test data
print(f"\n🔧 Scaling test data...")
X_test_scaled = scaler.transform(X_test)

print(f"Type after scaling: {type(X_test_scaled)}")

# Convert to DataFrame if needed
if isinstance(X_test_scaled, pd.DataFrame):
    print("✅ Scaler returned DataFrame")
    test_data = X_test_scaled.copy()
else:
    print("⚠️  Scaler returned numpy array, converting to DataFrame...")
    test_data = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns,
        index=X_test.index
    )

# Add target
test_data['TX_FRAUD'] = y_test.values

# Check for NaN
nan_count = test_data.isnull().sum().sum()
print(f"\n🔍 Checking for NaN values: {nan_count}")

if nan_count > 0:
    print("❌ ERROR: Still have NaN values!")
    print(test_data.isnull().sum()[test_data.isnull().sum() > 0])
else:
    print("✅ No NaN values!")

# Save
output_path.parent.mkdir(parents=True, exist_ok=True)
test_data.to_csv(output_path, index=False)

print(f"\n💾 Saved to: {output_path}")
print(f"   Shape: {test_data.shape}")
print(f"   Fraud count: {test_data['TX_FRAUD'].sum()}")

# Verify
print(f"\n🔍 Verifying saved file...")
verify_df = pd.read_csv(output_path)
verify_nan = verify_df.isnull().sum().sum()

print(f"   Loaded shape: {verify_df.shape}")
print(f"   NaN values: {verify_nan}")

if verify_nan == 0:
    print("\n✅ test.csv successfully regenerated without NaN!")
    
    # Show first row
    print(f"\n📋 First fraud transaction:")
    fraud_row = verify_df[verify_df['TX_FRAUD'] == 1].iloc[0]
    print(fraud_row.head(10))
    
    # Test with model
    print(f"\n🧪 Testing with model...")
    model = joblib.load(root_path / "models" / "best_model_tuned.joblib")
    
    X_fraud = fraud_row.drop('TX_FRAUD').values.reshape(1, -1)
    pred = model.predict(X_fraud)[0]
    prob = model.predict_proba(X_fraud)[0][1]
    
    print(f"   Prediction: {pred}")
    print(f"   Probability: {prob:.4f}")
    
    if prob > 0.5:
        print(f"   ✅ Model correctly predicts fraud!")
    else:
        print(f"   ⚠️  Low probability - check data")
        
else:
    print("\n❌ ERROR: Saved file still has NaN values!")

print("\n" + "="*70)
print("DONE")
print("="*70)