import pandas as pd
import joblib
import logging
from pathlib import Path
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Create logger
logger = logging.getLogger("hyperparameter_tuning")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


class ModelTuner:
    """Fast hyperparameter tuning - 2 models only"""
    
    def __init__(self, X_train, y_train):
        """Split for validation"""
        
        # Split train for validation
        logger.info("📊 Splitting train data for validation...")
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        logger.info(f"   Train: {len(self.X_train):,}")
        logger.info(f"   Validation: {len(self.X_val):,}")
         
        # Save full training data
        self.X_train_full = X_train
        self.y_train_full = y_train
    
    def objective_xgboost(self, trial):
        """XGBoost - simple params to prevent overfitting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),  # Reduced to prevent overfitting
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),  # Regularization
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # Prevent overfitting
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_val)
        return f1_score(self.y_val, y_pred)
    
    def objective_lightgbm(self, trial):
        """LightGBM - simple params to prevent overfitting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),  # Reduced
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),  # Reduced
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),  # Regularization
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_val)
        return f1_score(self.y_val, y_pred)
    
    def tune_model(self, model_name, objective_func, n_trials=10):
        """Tune a single model"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Tuning: {model_name}")
        logger.info(f"{'='*70}")
        
        study = optuna.create_study(direction='maximize', study_name=model_name)
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best F1-Score (validation): {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def train_final_model(self, model_name, best_params):
        """Train final model on full training data"""
        logger.info(f"\n📊 Training final {model_name}...")
        
        # Create model
        if model_name == 'XGBoost':
            model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
        elif model_name == 'LightGBM':
            model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        
        # Train on full training data
        model.fit(self.X_train_full, self.y_train_full)
        logger.info("✅ Model trained")
        
        # Show validation performance
        y_val_pred = model.predict(self.X_val)
        logger.info("\n📊 Validation Set Performance:")
        print(classification_report(self.y_val, y_val_pred, target_names=['Legitimate', 'Fraudulent']))
        
        return model


def run_hyperparameter_tuning(n_trials=10):
    """Fast tuning - 2 models, 10 trials each"""
    
    logger.info("="*70)
    logger.info("FAST HYPERPARAMETER TUNING")
    logger.info("="*70)
    logger.info("\n⚡ Settings:")
    logger.info("   Models: XGBoost, LightGBM")
    logger.info("   Trials: 10 per model (20 total)")
    logger.info("   Time: ~3-5 minutes")
    logger.info("   Focus: Prevent overfitting with regularization")
    
    # Paths
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    # Load train data
    logger.info("\n📂 Loading data...")
    train_path = root_path / "data" / "final" / "train.csv"
    scaler_path = root_path / "models" / "scaler.joblib"
    
    if not train_path.exists():
        logger.error(f"❌ {train_path} not found! Run train.py first")
        return
    
    if not scaler_path.exists():
        logger.error(f"❌ {scaler_path} not found! Run train.py first")
        return
    
    train_df = pd.read_csv(train_path)
    scaler = joblib.load(scaler_path)
    
    logger.info(f"✅ Train: {train_df.shape}")
    
    # Separate X and y
    X_train = train_df.drop(columns=['TX_FRAUD'])
    y_train = train_df['TX_FRAUD']
    
    logger.info(f"   Fraud rate: {y_train.mean()*100:.2f}%")
    
    # Initialize tuner
    tuner = ModelTuner(X_train, y_train)
    
    # Models to tune (only 2!)
    models_to_tune = {
        'XGBoost': tuner.objective_xgboost,
        'LightGBM': tuner.objective_lightgbm
    }
    
    # Results
    results = {}
    
    # Tune each model
    for model_name, objective_func in models_to_tune.items():
        best_params, best_val_score = tuner.tune_model(model_name, objective_func, n_trials=n_trials)
        final_model = tuner.train_final_model(model_name, best_params)
        
        results[model_name] = {
            'model': final_model,
            'best_params': best_params,
            'val_f1': best_val_score
        }
    
    # Compare
    logger.info("\n" + "="*70)
    logger.info("📊 MODEL COMPARISON (VALIDATION)")
    logger.info("="*70)
    
    logger.info(f"\n{'Model':<15} {'Val F1-Score':<15}")
    logger.info("-"*70)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name:<15} {metrics['val_f1']:<15.4f}")
    
    # Best model
    best_model_name = max(results, key=lambda x: results[x]['val_f1'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   Val F1: {results[best_model_name]['val_f1']:.4f}")
    
    # Save
    models_dir = root_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    best_model_path = models_dir / "best_model_tuned.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info(f"\n💾 Best model saved: {best_model_path}")
    
    # Save both
    for model_name, result in results.items():
        filename = model_name.lower().replace(' ', '_')
        model_path = models_dir / f"{filename}_tuned.joblib"
        joblib.dump(result['model'], model_path)
        logger.info(f"💾 Saved: {model_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TUNING COMPLETE!")
    logger.info("="*70)
    logger.info("\n🎯 Next: Run predict.py to evaluate on test.csv")
    
    return results, best_model_name


if __name__ == "__main__":
    results, best_model = run_hyperparameter_tuning(n_trials=10)
    
    if results:
        logger.info(f"\n💡 Use {best_model} for deployment")