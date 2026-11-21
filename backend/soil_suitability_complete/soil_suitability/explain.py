"""
Wraps SHAP so the API can return a human-readable explanation alongside
each prediction, e.g.:

    {"feature": "ph", "value": 7.9, "impact": -0.18, "direction": "lowered suitability"}

This is what powers the "SHAP (Explainable AI) is used to show which
soil factors have the most effect on the prediction" line in your
architecture diagram.
"""

from typing import Dict, List

import pandas as pd
import shap


class SuitabilityExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, features: Dict[str, float], top_k: int = 4) -> List[Dict]:
        """Returns the top_k most influential features for this single
        prediction, ranked by absolute SHAP value."""
        row = pd.DataFrame([[features[f] for f in self.feature_names]], columns=self.feature_names)
        shap_values = self.explainer.shap_values(row)

        # shap_values shape: (1, n_features) for binary XGBoost classifier
        values = shap_values[0]

        contributions = [
            {
                "feature": f,
                "value": round(float(features[f]), 2),
                "impact": round(float(v), 4),
                "direction": "increased suitability" if v > 0 else "decreased suitability",
            }
            for f, v in zip(self.feature_names, values)
        ]
        contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
        return contributions[:top_k]
