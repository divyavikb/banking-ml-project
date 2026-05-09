# 🏦 Banking Fraud Detection - End-to-End ML System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-green)
![Flask](https://img.shields.io/badge/Flask-2.3.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade fraud detection system built on 1.75M credit card transactions, achieving **91% F1-score** and **98% ROC-AUC** with complete MLOps pipeline implementation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Development](#model-development)
- [API Deployment](#api-deployment)
- [MLOps Pipeline](#mlops-pipeline)
- [Monitoring](#monitoring)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an end-to-end machine learning system for detecting fraudulent credit card transactions. The system handles severe class imbalance (0.84% fraud rate), processes transactions in real-time (<100ms latency), and achieves production-grade performance metrics.

### Business Problem

- **Dataset**: 1.75M credit card transactions (April-September 2018)
- **Fraud Rate**: 0.84% (severe class imbalance - 1:118 ratio)
- **Challenge**: Detect fraud with high recall while minimizing false positives
- **Impact**: Potentially prevent $163K in fraud while flagging only 195 false alarms

### Technical Solution

- **Feature Engineering**: Created 9 behavioral features (tx_velocity, amount_deviation, temporal patterns)
- **Sampling**: SMOTE with sampling_strategy=0.5 to handle imbalance
- **Modeling**: Evaluated 4 algorithms, selected LightGBM after Optuna tuning
- **Deployment**: Flask REST API with MLflow tracking and DVC versioning

---

## ✨ Key Features

### 🔧 Advanced Feature Engineering
- **Temporal Features**: hour, day, month, weekday, is_weekend, is_night
- **Behavioral Features**: tx_velocity, amount_deviation, high_amount
- **Aggregated Features**: Customer and terminal risk scores over 1/7/30 day windows

### 🤖 Machine Learning Pipeline
- **4 Algorithms Evaluated**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Hyperparameter Optimization**: Optuna (20 trials)
- **Class Imbalance Handling**: SMOTE (sampling_strategy=0.5)
- **Cross-Validation**: 5-fold stratified CV

### 🚀 Production Deployment
- **REST API**: Flask with JSON request/response
- **Latency**: <100ms prediction time
- **Auto Feature Engineering**: Computes features from raw transaction data
- **Monitoring**: MLflow experiment tracking

### 📊 MLOps Implementation
- **Version Control**: Git for code, DVC for data/models
- **Experiment Tracking**: MLflow (20+ experiments logged)
- **Reproducibility**: Parameterized pipelines (params.yaml)
- **CI/CD Ready**: Containerization-ready architecture

---

## 📈 Results

### Model Performance

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **F1-Score** | 91% | 87-93% ✅ |
| **ROC-AUC** | 98% | 95-98% ✅ |
| **Precision** | 89% | 85-92% ✅ |
| **Recall** | 92% | 88-94% ✅ |
| **False Positive Rate** | 0.09% | 1-3% ✅ |
| **Prediction Latency** | <100ms | 50-200ms ✅ |

### Confusion Matrix (Test Set: 209,715 transactions)

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 207,750 (TN) | 195 (FP) |
| **Actual Positive** | 141 (FN) | 1,629 (TP) |

### Business Impact

- **Fraud Detected**: 1,629 out of 1,770 (92% recall)
- **Fraud Prevented**: ~$163K (assuming $100 avg transaction)
- **False Alarms**: 195 out of 210K transactions (0.09% FPR)
- **Improvement over Baseline**: 99.4% reduction in false positives vs. rule-based systems

---

## 📁 Project Structure

```
banking-ml-project/
│
├── data/
│   ├── raw/                    # Original transaction data
│   ├── processed/              # Feature-engineered data
│   ├── final/                  # Train/test splits
│   └── external/               # Test datasets for API
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py           # Load and validate data
│   │   └── data_transformation.py      # Feature engineering
│   │
│   ├── models/
│   │   ├── train.py                    # Model training pipeline
│   │   ├── hyperparameter_tuning.py    # Optuna optimization
│   │   └── predict.py                  # Batch predictions
│   │
│   └── utils/
│       └── logger.py                   # Logging utilities
│
├── models/
│   ├── best_model_tuned.joblib        # Trained LightGBM model
│   ├── scaler.joblib                  # StandardScaler
│   └── high_amount_threshold.joblib   # Feature threshold
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb   # Feature creation
│   └── 03_Model_Evaluation.ipynb      # Performance analysis
│
├── reports/
│   └── figures/                       # Visualization outputs
│       ├── 01_fraud_distribution.png
│       ├── 03_correlation_heatmap.png
│       └── 04_fraud_correlation.png
│
├── tests/
│   └── test_api.py                    # API testing scripts
│
├── app.py                             # Flask REST API
├── requirements.txt                   # Python dependencies
├── params.yaml                        # Hyperparameters
├── dvc.yaml                          # DVC pipeline definition
└── README.md                         # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Git
- (Optional) Docker for containerization

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/banking-fraud-detection.git
cd banking-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
```bash
# Place your transaction data in data/raw/
# Expected file: transactions.csv
```

5. **Initialize DVC (Optional)**
```bash
dvc init
dvc remote add -d myremote /path/to/remote/storage
```

---

## 💻 Usage

### 1. Data Preparation

```bash
# Run data ingestion
python src/data/data_ingestion.py

# Feature engineering
python src/data/data_transformation.py
```

### 2. Model Training

```bash
# Train models
python src/models/train.py

# Hyperparameter tuning (optional)
python src/models/hyperparameter_tuning.py
```

### 3. Model Evaluation

```bash
# Generate predictions and evaluation report
python src/models/predict.py
```

### 4. Start API Server

```bash
# Start Flask API
python app.py

# API will be available at: http://localhost:5000
```

### 5. Make Predictions

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TX_DATETIME": "2024-03-06 14:30:00",
    "TX_AMOUNT": 45.50,
    "CUSTOMER_ID_NB_TX_1DAY_WINDOW": 2,
    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": 40.25,
    "CUSTOMER_ID_NB_TX_7DAY_WINDOW": 8,
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": 42.30,
    "CUSTOMER_ID_NB_TX_30DAY_WINDOW": 25,
    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW": 43.20,
    "TERMINAL_ID_NB_TX_1DAY_WINDOW": 150,
    "TERMINAL_ID_RISK_1DAY_WINDOW": 0.02,
    "TERMINAL_ID_NB_TX_7DAY_WINDOW": 800,
    "TERMINAL_ID_RISK_7DAY_WINDOW": 0.03,
    "TERMINAL_ID_NB_TX_30DAY_WINDOW": 3000,
    "TERMINAL_ID_RISK_30DAY_WINDOW": 0.02
  }'
```

**Response:**
```json
{
  "fraud_prediction": 0,
  "fraud_probability": 0.05,
  "risk_level": "low",
  "message": "Transaction appears legitimate"
}
```

---

## 🔄 Data Pipeline

### Stage 1: Data Ingestion
- **Input**: Raw transaction CSV (1.75M records)
- **Process**: Load, validate schema, check data quality
- **Output**: `data/raw/transactions.csv`

### Stage 2: Feature Engineering
- **Input**: Raw transactions
- **Process**: 
  - Create temporal features (hour, day, month, weekday, is_weekend, is_night)
  - Create behavioral features (tx_velocity, amount_deviation, high_amount)
  - Calculate customer/terminal aggregations
- **Output**: `data/processed/features_engineered.csv`

### Stage 3: Data Transformation
- **Input**: Engineered features
- **Process**:
  - Train/test split (80/20, stratified)
  - StandardScaler normalization
  - SMOTE oversampling (training only)
- **Output**: `data/final/train.csv`, `data/final/test.csv`

---

## 🤖 Model Development

### Algorithms Evaluated

| Model | F1-Score | ROC-AUC | Precision | Recall | Training Time |
|-------|----------|---------|-----------|--------|---------------|
| **LightGBM** ✅ | **91%** | **98%** | **89%** | **92%** | 45s |
| XGBoost | 89% | 97% | 87% | 91% | 120s |
| Random Forest | 85% | 95% | 83% | 88% | 180s |
| Logistic Regression | 72% | 88% | 68% | 76% | 15s |

### Hyperparameter Tuning (Optuna)

**LightGBM Best Parameters:**
```yaml
n_estimators: 200
max_depth: 10
learning_rate: 0.05
num_leaves: 50
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
```

**Optimization Metric**: F1-Score (balances precision and recall)
**Trials**: 20
**Improvement**: +3% F1-score over default parameters

---

## 🌐 API Deployment

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home - API information |
| GET | `/health` | Health check - model status |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict/batch` | Batch predictions |

### Request Format

**Required Fields (16 raw features):**
- `TX_DATETIME` (string): Transaction timestamp
- `TX_AMOUNT` (float): Transaction amount
- `CUSTOMER_ID_NB_TX_*DAY_WINDOW` (int): Customer transaction counts
- `CUSTOMER_ID_AVG_AMOUNT_*DAY_WINDOW` (float): Customer average amounts
- `TERMINAL_ID_NB_TX_*DAY_WINDOW` (int): Terminal transaction counts
- `TERMINAL_ID_RISK_*DAY_WINDOW` (float): Terminal fraud rates

**Auto-Computed Features:**
- `hour`, `day`, `month`, `weekday`, `is_weekend`, `is_night`
- `amount_deviation`, `tx_velocity`, `high_amount`

### Response Format

```json
{
  "fraud_prediction": 1,
  "fraud_probability": 0.89,
  "risk_level": "high",
  "message": "Fraudulent transaction detected!",
  "recommendation": "Block transaction"
}
```

---

## 📊 MLOps Pipeline

### Version Control

**Code**: Git
```bash
git add .
git commit -m "Updated feature engineering"
git push origin main
```

**Data & Models**: DVC
```bash
dvc add data/raw/transactions.csv
dvc add models/best_model_tuned.joblib
dvc push
```

### Experiment Tracking

**MLflow**: All experiments logged automatically
```bash
# Start MLflow UI
mlflow ui

# View at: http://localhost:5000
```

**Tracked Metrics:**
- Training/validation accuracy, precision, recall, F1, ROC-AUC
- Hyperparameters
- Model artifacts
- Feature importance
- Confusion matrices

### Reproducibility

**DVC Pipeline:**
```bash
# Reproduce entire pipeline
dvc repro

# Run specific stage
dvc repro train
```

**Parameters** (`params.yaml`):
```yaml
test_size: 0.2
random_state: 42
smote_strategy: 0.5
n_estimators: 200
max_depth: 10
learning_rate: 0.05
```

---

## 📈 Monitoring

### Model Performance Metrics

Track in production:
- **Fraud Detection Rate**: % of actual frauds caught
- **False Positive Rate**: % of legitimate transactions flagged
- **Precision/Recall**: Updated weekly
- **Prediction Latency**: <100ms threshold

### Data Drift Detection

Monitor feature distributions:
- **TX_AMOUNT**: Compare monthly distributions
- **TERMINAL_RISK**: Track risk score trends
- **Transaction Counts**: Detect volume changes

### Alerting Thresholds

- ⚠️ FPR > 1%
- ⚠️ Recall < 90%
- ⚠️ Latency > 150ms
- ⚠️ Feature drift (KS-test p < 0.05)

---

## 🚧 Future Improvements

### Deployment & Infrastructure
1. **Docker Containerization** - Package entire application
2. **AWS ECS Deployment** - Scalable cloud hosting
3. **CI/CD Pipeline** - GitHub Actions for automated testing

### Monitoring & Operations
4. **Real-time Dashboard** - Grafana + Prometheus
5. **Automated Retraining** - Weekly/trigger-based
6. **A/B Testing** - Canary deployments for model updates

### Model Enhancement
7. **SHAP Explainability** - Show why transactions flagged
8. **Additional Features** - Device fingerprinting, geolocation, behavioral biometrics
9. **Ensemble Methods** - Combine LightGBM + Neural Network
10. **Online Learning** - Adapt to new patterns without full retraining

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Divya Balasubramanyam**

- 15+ years experience in banking software QA
- Transitioning to Data Science
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset inspired by credit card fraud detection research
- MLOps best practices from production ML systems
- Feature engineering techniques from banking domain expertise

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/)
- [DVC Documentation](https://dvc.org/)

---

**⭐ If you found this project helpful, please give it a star!**

---

## 📞 Contact

For questions or collaboration opportunities:
- Open an issue in this repository
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

*Last Updated: May 2026*
