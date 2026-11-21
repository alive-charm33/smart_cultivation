"""
Tomato Soil Suitability — Full Training Pipeline (XGBoost + SHAP)
===================================================================
Runs end-to-end in ONE file: generates the training data, trains the
real XGBoost model, evaluates it, explains it with SHAP, and saves the
trained model.

HOW TO RUN ON GOOGLE COLAB:
  1. Create a new Colab notebook.
  2. Paste this entire file into a single cell.
  3. Run the cell (Shift+Enter). That's it — pip installs happen
     automatically at the top.
  4. At the end, the trained model files will be offered as a download.

HOW TO RUN ON YOUR LAPTOP:
  1. Save this file as train_pipeline.py
  2. In a terminal:  pip install xgboost shap scikit-learn pandas numpy matplotlib joblib
  3. Run:  python train_pipeline.py
  4. Trained model files appear in ./models/
"""

# ── Step 0: Install dependencies (safe to run multiple times) ──────────
import subprocess
import sys

def _pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

try:
    import xgboost  # noqa
    import shap  # noqa
except ImportError:
    print("Installing xgboost and shap (first run only)...")
    _pip_install(["xgboost", "shap"])

import os

import joblib
import matplotlib.pyplot as plt
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

RNG = np.random.default_rng(42)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Step 1: Generate the tomato-suitability training dataset ───────────
# Agronomic reference ranges for tomato (Solanum lycopersicum)
# (ideal_low, ideal_high, tolerable_low, tolerable_high)
#
# Sources (see the project README for full citations):
#   pH, temperature      -> tomato-specific extension/agronomy guidance
#   phosphorus, potassium -> UC Davis / California FREP tomato-specific
#                            soil-test research (Olsen-P, exchangeable K)
#   rainfall, humidity    -> tomato-specific crop-water-requirement and
#                            pollination-humidity research
#   nitrogen, organic carbon -> general agricultural benchmarks (no
#                            tomato-specific research found for these two
#                            — documented limitation, not an oversight)
RANGES = {
    "ph":              (6.0, 6.8, 5.5, 7.5),
    "nitrogen_ppm":    (1000, 2000, 300, 3000),   # mg/kg TOTAL N (SoilGrids scale)
    "phosphorus_ppm":  (10, 20, 5, 35),            # mg/kg Olsen-P (tomato-specific)
    "potassium_ppm":   (135, 250, 60, 350),        # mg/kg exchangeable K (tomato-specific)
    "organic_carbon_pct": (1.5, 3.0, 0.8, 4.0),
    "rainfall_mm":     (400, 600, 300, 800),       # growing-season total (tomato-specific)
    "temperature_c":   (20, 27, 15, 32),           # tomato-specific
    "humidity_pct":    (60, 70, 40, 85),           # tomato-specific
}
FEATURES = list(RANGES.keys())


def _feature_score(value, bounds):
    """1.0 inside the ideal range, smoothly fading to 0.0 outside tolerable."""
    ideal_low, ideal_high, tol_low, tol_high = bounds
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        if value <= tol_low:
            return 0.0
        return (value - tol_low) / (ideal_low - tol_low)
    else:
        if value >= tol_high:
            return 0.0
        return (tol_high - value) / (tol_high - ideal_high)


def _sample_feature(bounds):
    """Sample a value across a wide plausible range, clipped at 0."""
    _, _, tol_low, tol_high = bounds
    span = tol_high - tol_low
    low = max(0.0, tol_low - 0.4 * span)
    high = tol_high + 0.4 * span
    return RNG.uniform(low, high)


def generate_dataset(n_samples=4000):
    weights = {
        "ph": 0.22, "nitrogen_ppm": 0.13, "phosphorus_ppm": 0.10,
        "potassium_ppm": 0.10, "organic_carbon_pct": 0.10,
        "rainfall_mm": 0.13, "temperature_c": 0.15, "humidity_pct": 0.07,
    }
    rows = []
    for _ in range(n_samples):
        sample = {f: _sample_feature(RANGES[f]) for f in FEATURES}
        score = sum(weights[f] * _feature_score(sample[f], RANGES[f]) for f in FEATURES)
        noisy_score = np.clip(score + RNG.normal(0, 0.06), 0, 1)
        label = int(noisy_score >= 0.39)  # threshold ~ median score, keeps classes balanced
        sample["suitable"] = label
        rows.append(sample)
    df = pd.DataFrame(rows)
    for f in FEATURES:
        df[f] = df[f].round(2)
    return df


print("Step 1/4: Generating training dataset...")
df = generate_dataset(4000)
df.to_csv("data_tomato_suitability.csv", index=False)
print(f"  Generated {len(df)} rows, class balance:\n{df['suitable'].value_counts(normalize=True)}\n")


# ── Step 2: Train XGBoost ───────────────────────────────────────────────
print("Step 2/4: Training XGBoost model...")
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
print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"  F1 score: {f1_score(y_test, preds):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_test, preds)}")
print(f"  Classification report:\n{classification_report(y_test, preds)}")

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(f"  Feature importances:\n{importances}\n")


# ── Step 3: SHAP explainability ─────────────────────────────────────────
print("Step 3/4: Computing SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150)
plt.close()
print("  Saved SHAP summary plot to outputs/shap_summary.png\n")


# ── Step 4: Refit on full data and save ─────────────────────────────────
print("Step 4/4: Saving final model...")
model.fit(X, y)  # refit on all data now that test performance looks good
joblib.dump(model, "models/xgb_suitability.joblib")
joblib.dump(feature_names, "models/feature_order.joblib")
print("  Saved models/xgb_suitability.joblib and models/feature_order.joblib\n")


# ── Quick sanity check with example soils ───────────────────────────────
print("Sanity check with example soils:")
good_soil = {"ph": 6.4, "nitrogen_ppm": 1500, "phosphorus_ppm": 15, "potassium_ppm": 180,
             "organic_carbon_pct": 2.2, "rainfall_mm": 500, "temperature_c": 24, "humidity_pct": 65}
poor_soil = {"ph": 8.3, "nitrogen_ppm": 300, "phosphorus_ppm": 5, "potassium_ppm": 60,
             "organic_carbon_pct": 0.5, "rainfall_mm": 200, "temperature_c": 38, "humidity_pct": 30}

for label, soil in [("Good soil", good_soil), ("Poor soil", poor_soil)]:
    row = [[soil[f] for f in feature_names]]
    proba = model.predict_proba(row)[0][1]
    print(f"  {label}: P(suitable) = {proba:.3f}")

print("\nDone! Model is ready to use.")

# ── Optional: download the trained model if running on Colab ───────────
try:
    from google.colab import files  # noqa
    print("\nRunning on Colab — offering model files for download...")
    files.download("models/xgb_suitability.joblib")
    files.download("models/feature_order.joblib")
except ImportError:
    pass  # not running on Colab, skip
