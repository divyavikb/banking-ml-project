import pandas as pd
import joblib
import logging
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import os
import yaml
import mlflow
import mlflow.sklearn

set_config(transform_output="pandas")

test_size = yaml.safe_load(open("params.yaml"))["model_training"]["test_size"]

# ✅ Setup MLflow (2 lines added)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection-training")

# Create logger
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and log metrics, return ROC-AUC"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logger.info("\n" + "=" * 60)
    logger.info(f"MODEL EVALUATION — {model_name}")
    logger.info("=" * 60)
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info("=" * 60)

    return roc_auc


if __name__ == "__main__":
    
    # ✅ Start MLflow parent run (1 line added)
    with mlflow.start_run(run_name="model_training_comparison"):
        
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        data_path = root_path / "data" / "processed" / "features_engineered.csv"

        # Read data
        df = pd.read_csv(data_path)
        logger.info("Data read successfully")
        logger.info(f"Shape: {df.shape}")

        X = df.drop(columns=["TX_FRAUD", "TX_FRAUD_SCENARIO"])
        y = df["TX_FRAUD"]
        logger.info(f"Features: {X.shape[1]}, Target fraud rate: {y.mean() * 100:.2f}%")
        
        # ✅ Log dataset info (3 lines added)
        mlflow.log_param("total_samples", df.shape[0])
        mlflow.log_param("total_features", X.shape[1])
        mlflow.log_param("fraud_rate", y.mean())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=34, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # ✅ Log split info (2 lines added)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("train_samples", len(X_train))

        # ========================================
        # SCALING
        # ========================================
        logger.info("\n🔧 Applying StandardScaler...")
        from sklearn import set_config
        set_config(transform_output="pandas")  # Returns DataFrame with original column names

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("✅ Scaling complete")

        # Save scaler (required DVC output)
        scaler_save_path = root_path / "models" / "scaler.joblib"
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        logger.info(f"✅ Scaler saved: {scaler_save_path}")

        # ========================================
        # SMOTE (only on training data)
        # ========================================
        logger.info("\n🔄 Applying SMOTE to training data...")
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"✅ SMOTE applied: {len(X_train_balanced):,} samples")
        logger.info(f"   Fraud rate after SMOTE: {y_train_balanced.mean() * 100:.2f}%")
        
        # ✅ Log SMOTE info (1 line added)
        mlflow.log_param("smote_sampling_strategy", 0.5)

        # ========================================
        # SAVE TRAIN/TEST DATASETS (required DVC outputs)
        # ========================================
        logger.info("\n💾 Saving preprocessed datasets...")

        train_data = pd.DataFrame(X_train_balanced, columns=X_train_scaled.columns)
        train_data["TX_FRAUD"] = y_train_balanced.values
        train_save_path = root_path / "data" / "final" / "train.csv"
        train_save_path.parent.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(train_save_path, index=False)
        logger.info(f"✅ Train data saved: {train_save_path} | Shape: {train_data.shape}")

        test_data = pd.DataFrame(X_test_scaled, columns=X_test_scaled.columns)
        test_data["TX_FRAUD"] = y_test.values
        test_save_path = root_path / "data" / "final" / "test.csv"
        test_data.to_csv(test_save_path, index=False)
        logger.info(f"✅ Test data saved: {test_save_path} | Shape: {test_data.shape}")

        # ========================================
        # DEFINE MODELS
        # ========================================
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "XGBoost":             XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss"),
            "LightGBM":            LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        }

        # ========================================
        # TRAIN & EVALUATE ALL — LOG RESULTS ONLY
        # ========================================
        results = {}
        trained_models = {}

        for model_name, model in models.items():
            
            # ✅ Create nested run for each model (1 line added)
            with mlflow.start_run(run_name=model_name, nested=True):
                
                logger.info(f"\n🚀 Training {model_name}...")
                model.fit(X_train_balanced, y_train_balanced)
                logger.info(f"✅ {model_name} trained")
                
                roc_auc = evaluate_model(model, X_test_scaled, y_test, model_name)
                results[model_name] = roc_auc
                trained_models[model_name] = model
                
                # ✅ Log metrics (5 lines added)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
                
                # ✅ Log model (1 line added)
                mlflow.sklearn.log_model(model, "model")

        # ========================================
        # SUMMARY
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("📋 MODEL COMPARISON SUMMARY (ROC-AUC)")
        logger.info("=" * 60)
        for name, roc_auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {name:<25} ROC-AUC: {roc_auc:.4f}")

        best_name = max(results, key=results.get)
        logger.info(f"\n🏆 Best model: {best_name} (ROC-AUC: {results[best_name]:.4f})")
        logger.info("=" * 60)
        
        # ✅ Log best model info (2 lines added)
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_roc_auc", results[best_name])

        # Save best model as model.joblib (required DVC output)
        model_save_path = root_path / "models" / "model.joblib"
        joblib.dump(trained_models[best_name], model_save_path)
        logger.info(f"\n✅ Best model ({best_name}) saved to {model_save_path}")
        
        # ✅ Log artifact (1 line added)
        mlflow.log_artifact(str(model_save_path))
        
        logger.info("\n🎯 View results: mlflow ui")
        logger.info("   Then open: http://localhost:5000")