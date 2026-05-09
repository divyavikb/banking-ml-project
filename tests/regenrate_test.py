import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import set_config

set_config(transform_output="pandas")

# Load original data
df = pd.read_csv('data/processed/features_engineered.csv')

# Split
X = df.drop(columns=['TX_FRAUD', 'TX_FRAUD_SCENARIO'])
y = df['TX_FRAUD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load scaler
scaler = joblib.load('models/scaler.joblib')

# Transform (returns DataFrame if set_config is set)
X_test_scaled = scaler.transform(X_test)

# Create test dataframe
if isinstance(X_test_scaled, pd.DataFrame):
    test_data = X_test_scaled.copy()
else:
    # If numpy array, convert properly
    test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)

test_data['TX_FRAUD'] = y_test.values

# Save
test_data.to_csv('data/external/test.csv', index=False)

print(f"✅ Test data regenerated!")
print(f"   Shape: {test_data.shape}")
print(f"   Columns: {test_data.columns.tolist()}")
print(f"   Missing values: {test_data.isnull().sum().sum()}")
print(f"\nFirst row:")
print(test_data.iloc[0])