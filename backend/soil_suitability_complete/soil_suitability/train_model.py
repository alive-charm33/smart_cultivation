"""
Trains an XGBoost classifier for tomato soil-suitability prediction and
saves the model + a SHAP explainer alongside it.

Usage:
    python train_model.py

Requires: xgboost, shap, scikit-learn, pandas, joblib, matplotlib
    pip install -r requirements.txt
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/tomato_suitability.csv"
MODEL_PATH = "models/xgb_suitability.joblib"
FEATURE_ORDER_PATH = "models/feature_order.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["suitable"]
    X = df.drop(columns=["suitable"])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, preds))

    # Feature importance sanity check (should roughly match agronomic
    # intuition: pH and temperature should rank highly)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    print("\nFeature importances:\n", importances.sort_values(ascending=False))

    # Save model + feature order (needed at inference time to build the
    # input row in the exact order the model expects)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_ORDER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Precompute a SHAP explainer summary plot for the report/demo
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150)
    print("Saved SHAP summary plot to outputs/shap_summary.png")


if __name__ == "__main__":
    main()
