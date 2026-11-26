import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, fbeta_score, precision_recall_curve,
                             roc_curve, classification_report)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from src.data import load_data
from src.features import add_features, model_matrix, NUMERIC_BASE

ART = Path("artifacts"); REP = Path("reports")
ART.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)

def build_models():
    models = [("random_forest", RandomForestClassifier(
        n_estimators=180, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42))]
    try:
        from xgboost import XGBClassifier
        models.append(("xgboost", XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            scale_pos_weight=8, n_jobs=-1, random_state=42)))
    except Exception as e:
        warnings.warn(f"XGBoost unavailable; skipping: {e}")
    try:
        from lightgbm import LGBMClassifier
        models.append(("lightgbm", LGBMClassifier(
            n_estimators=180, learning_rate=0.05, class_weight="balanced",
            verbosity=-1, random_state=42)))
    except Exception as e:
        warnings.warn(f"LightGBM unavailable; skipping: {e}")
    return models

def main():
    df = load_data()
    featured = add_features(df)
    X = model_matrix(featured)
    y = featured["is_fraud"].astype(int)
    # Chronological partitions: earliest 70%, next 15%, latest 15%.
    n = len(df); a, b = int(n * .70), int(n * .85)
    X_train, X_val, X_test = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
    y_train, y_val, y_test = y.iloc[:a], y.iloc[a:b], y.iloc[b:]
    if y_train.nunique() < 2:
        raise ValueError("Training partition must contain both fraud and non-fraud examples.")

    # SMOTE is restricted to the training partition. Fall back to class weights if unavailable.
    try:
        from imblearn.over_sampling import SMOTE
        minority = int(y_train.value_counts().min())
        if minority >= 2:
            X_fit, y_fit = SMOTE(random_state=42, k_neighbors=min(5, minority - 1)).fit_resample(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train
    except Exception as e:
        warnings.warn(f"SMOTE unavailable/not applicable; using original training rows: {e}")
        X_fit, y_fit = X_train, y_train

    fitted = []
    for name, model in build_models():
        print(f"Training {name}...")
        model.fit(X_fit, y_fit)
        fitted.append((name, model))

    # Equal-weight soft voting across available classifiers.
    class Ensemble:
        def __init__(self, estimators): self.estimators = estimators
        def predict_proba(self, X):
            probs = [m.predict_proba(X)[:, 1] for _, m in self.estimators]
            p = np.mean(probs, axis=0)
            return np.column_stack([1-p, p])
        def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    ensemble = Ensemble(fitted)
    val_probs = ensemble.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    # precision_recall_curve has one more precision/recall item than thresholds.
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-12)
    threshold = float(thresholds[int(np.nanargmax(f2))]) if len(thresholds) else 0.5

    test_probs = ensemble.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    metrics = {
        "threshold_selected_on_validation": threshold,
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)) if y_test.nunique() > 1 else None,
        "test_pr_auc": float(average_precision_score(y_test, test_probs)),
        "test_f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_rows": int(len(y_test)),
        "models": [name for name, _ in fitted],
        "feature_names": NUMERIC_BASE
    }
    (REP / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REP / "classification_report.txt").write_text(classification_report(y_test, test_pred, zero_division=0))

    if y_test.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
        plt.savefig(REP / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()
    p, r, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.savefig(REP / "precision_recall_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    joblib.dump({"estimators": fitted, "feature_names": NUMERIC_BASE}, ART / "model.joblib")
    metadata = {"threshold": threshold, "feature_names": NUMERIC_BASE, "models": metrics["models"]}
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2))

    try:
        import mlflow
        with mlflow.start_run(run_name="shieldml-ensemble"):
            mlflow.log_params({"split": "70/15/15 chronological", "threshold": threshold})
            mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v, (int,float)) and v is not None})
            mlflow.log_artifact(str(REP / "metrics.json"))
            mlflow.log_artifact(str(ART / "metadata.json"))
        print("Metrics logged to MLflow.")
    except Exception as e:
        print(f"MLflow tracking skipped (optional): {e}")

    print(json.dumps(metrics, indent=2))
    print("Saved artifacts/model.joblib and reports/")

if __name__ == "__main__":
    main()
