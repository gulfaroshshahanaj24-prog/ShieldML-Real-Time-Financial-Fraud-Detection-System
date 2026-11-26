# ShieldML — Real-Time Financial Fraud Detection

A runnable starter implementation of the Code A Nova Project 4 brief. It includes:
- transaction data loading (or synthetic demo data generation)
- time-ordered train/validation/test split
- past-only velocity and merchant amount features
- train-only SMOTE (with a safe fallback if unavailable)
- Random Forest, XGBoost and LightGBM (optional dependencies)
- soft-voting ensemble, validation threshold selection using F2
- evaluation metrics and ROC / precision-recall plots
- SHAP explanations (when SHAP is installed)
- FastAPI `/predict` and `/metrics` endpoints
- optional MLflow tracking

> **Important:** The included synthetic data is for testing the software only. It is not evidence of real-world fraud detection performance. For your submission, replace it with the Kaggle credit-card fraud dataset linked in the project brief and report results from the held-out test set.

## 1. Setup

Python 3.10–3.12 recommended.

### Windows (PowerShell)

```powershell
cd ShieldML_Ready_to_Use
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installing XGBoost or LightGBM fails on your computer, install the core requirements first:

```powershell
pip install -r requirements-core.txt
```

The training script will still run with the available models.

## 2. Train

```powershell
python -m src.train
```

If no `data/transactions.csv` exists, a demo dataset is generated automatically. Model artifacts and reports are written to `artifacts/` and `reports/`.

## 3. Start the API

```powershell
uvicorn src.api:app --reload
```

Open:
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

Example prediction:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"amount\":125.5,\"merchant\":\"shop_12\",\"timestamp\":\"2025-01-15T10:30:00\",\"device_id\":\"device_4\",\"customer_id\":\"customer_9\"}"
```

The API expects the same input fields used by the training pipeline. A model is loaded once at application startup, not on each request.

## 4. Use a real dataset

Download the Kaggle dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The Kaggle dataset has columns `Time`, `Amount`, `V1`...`V28`, and `Class`. For this starter, the pipeline supports both that schema and a transaction-style schema containing:
`timestamp, amount, merchant, device_id, customer_id, is_fraud`.

For the Kaggle schema, `Class` is automatically renamed to `is_fraud`, and a timestamp is derived from `Time`. Since that dataset does not contain merchant/device/customer identifiers, those engineered features use neutral defaults. Do not claim those fields were present in the original dataset.

Place the CSV at `data/transactions.csv`.

## 5. Project structure

```text
ShieldML_Ready_to_Use/
├── data/                 # put transactions.csv here
├── artifacts/            # trained model and metadata
├── reports/              # metrics and plots
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── api.py
├── requirements.txt
├── requirements-core.txt
├── Dockerfile
└── README.md
```

## 6. What to submit / customize

1. Replace demo data with the real dataset.
2. Run training and inspect `reports/metrics.json`.
3. Confirm ROC-AUC exceeds the project rubric target (>0.90) on the held-out test set; do not assume it will.
4. Review the selected F2 threshold in `artifacts/metadata.json`.
5. Add SHAP plots after confirming the installed SHAP version supports your selected model.
6. Run the API and measure latency on your machine; the `<50ms` target is a benchmark, not guaranteed.
7. Add your own model card, architecture diagram, MLflow screenshots, and demo video.

## 7. Data leakage safeguards

- Rows are sorted by timestamp before splitting.
- Velocity features use only earlier transactions from the same customer.
- SMOTE is applied only to the training partition.
- The threshold is selected on validation data and evaluated once on test data.
- This starter keeps feature engineering simple and transparent; review it carefully before production use.

## 8. Limitations

This is an educational internship project starter, not a production banking system. It does not include PCI compliance, authentication, encryption/key management, distributed feature stores, robust drift monitoring, or a guaranteed sub-50ms SLA. Validate privacy, security, fairness, and regulatory requirements before any real deployment.
