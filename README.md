# 🔍 Fraud Detection — End-to-End ML Project

A production-style machine learning pipeline for detecting financial transaction fraud on a **6.3 million row** real-world dataset, featuring XGBoost, class imbalance handling, SHAP explainability, and a Streamlit web app.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | PaySim — synthetic mobile money transactions |
| **Rows** | 6,362,620 |
| **Fraud rate** | 0.13% (severely imbalanced) |
| **Model** | XGBoost with `scale_pos_weight` |
| **Deployment** | Streamlit web app |
| **Explainability** | SHAP TreeExplainer |

---

## 📊 Results

| Metric | Logistic Regression (baseline) | XGBoost (final) |
|--------|-------------------------------|-----------------|
| Fraud precision | 2% | ~80%+ |
| Fraud recall | 94% | ~85%+ |
| ROC-AUC | ~0.95 | ~0.99 |
| PR-AUC (primary) | ~0.04 | ~0.85+ |

> **Why PR-AUC?** Accuracy and ROC-AUC are misleading on imbalanced data. A model that predicts "no fraud" always achieves 99.87% accuracy. Precision-Recall AUC is the correct primary metric here.

---

## 🗂️ Project Structure

```
fraud-detection/
├── fraud_detection.ipynb         # Full analysis notebook (EDA → model → SHAP)
├── Fraud_detection.py            # Streamlit web application
├── fraud_detection_pipeline.pkl  # Saved model pipeline
├── AIML Dataset.csv              # Dataset (not included — see below)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit joblib imbalanced-learn
```

### 3. Get the dataset
Download the PaySim dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and save it as `AIML Dataset.csv` in the project root.

### 4. Train the model
Run all cells in `fraud_detection.ipynb`. This will generate `fraud_detection_pipeline.pkl`.

### 5. Launch the app
```bash
streamlit run Fraud_detection.py
```

---

## 🧠 Key Techniques

### Handling class imbalance
Fraud makes up only 0.13% of transactions. Three approaches combined:
- `scale_pos_weight = n_negatives / n_positives` in XGBoost (~775×)
- Engineered features that directly encode suspicious patterns
- Evaluated using PR-AUC instead of accuracy

### Feature Engineering
Three features were derived from raw balance columns:

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `balanceDiffOrig` | `oldbalanceOrg - newbalanceOrig` | Should equal `amount` in honest transactions |
| `balanceDiffDest` | `newbalanceDest - oldbalanceDest` | Checks if receiver balance actually increased |
| `zeroBalanceOrig` | `1 if oldBalance>0 and newBalance==0` | Accounts draining to zero are a strong fraud signal |

### Explainability (SHAP)
Every prediction comes with a SHAP waterfall plot showing which features contributed most to the fraud score. This is critical for real-world deployment where decisions must be auditable.

---

## 📷 Screenshots



---

## 📚 Dataset Citation

Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. EMSS 2016.

---

## 🪪 License

MIT License — free to use, modify, and distribute.
