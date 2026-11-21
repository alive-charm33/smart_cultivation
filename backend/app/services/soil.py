"""Tomato suitability (your trained xgb_suitability.joblib + SHAP) and a soil-fertility
status table. Re-uses soil_suitability/explain.py, soil_client.py, weather_client.py
unchanged. The model is only LOADED here - never retrained.
"""
import sys
import threading

from .. import config

if str(config.SOIL_DIR) not in sys.path:
    sys.path.insert(0, str(config.SOIL_DIR))

_lock = threading.Lock()
_state = {"model": None, "features": None, "explainer": None, "error": None}

# Same agronomic ranges the suitability model was trained on
# (ideal_low, ideal_high, tolerable_low, tolerable_high) - see colab_train_pipeline.py
RANGES = {
    "ph": (6.0, 6.8, 5.5, 7.5),
    "nitrogen_ppm": (1000, 2000, 300, 3000),
    "phosphorus_ppm": (10, 20, 5, 35),
    "potassium_ppm": (135, 250, 60, 350),
    "organic_carbon_pct": (1.5, 3.0, 0.8, 4.0),
    "rainfall_mm": (400, 600, 300, 800),
    "temperature_c": (20, 27, 15, 32),
    "humidity_pct": (60, 70, 40, 85),
}
LABELS = {
    "ph": "Soil pH", "nitrogen_ppm": "Nitrogen (total, mg/kg)", "phosphorus_ppm": "Phosphorus (Olsen-P, mg/kg)",
    "potassium_ppm": "Potassium (mg/kg)", "organic_carbon_pct": "Organic carbon (%)",
    "rainfall_mm": "Season rainfall (mm)", "temperature_c": "Temperature (°C)", "humidity_pct": "Humidity (%)",
}
ADVICE_LOW = {
    "ph": "Soil is acidic for tomato: apply agricultural lime as per soil-test recommendation.",
    "nitrogen_ppm": "Low nitrogen: add well-decomposed FYM/compost and split N doses.",
    "phosphorus_ppm": "Low phosphorus: apply P at planting (basal) as per soil-test / Soil Health Card.",
    "potassium_ppm": "Low potassium: K is critical for fruit quality - add potash as recommended.",
    "organic_carbon_pct": "Low organic carbon: add FYM, compost or green manure before planting.",
    "rainfall_mm": "Low rainfall: plan drip irrigation for the whole season.",
    "temperature_c": "Cool climate: prefer the warmer season or protected cultivation.",
    "humidity_pct": "Dry air: mulch and maintain even irrigation; watch for spider mites.",
}
ADVICE_HIGH = {
    "ph": "Soil is alkaline: use gypsum/organic matter and acid-forming fertilizers; watch micronutrients.",
    "nitrogen_ppm": "High nitrogen: reduce N to avoid lush, disease-prone canopy.",
    "phosphorus_ppm": "High phosphorus: skip extra P this season.",
    "potassium_ppm": "High potassium: no extra K needed.",
    "organic_carbon_pct": "Organic carbon is high - good; keep it.",
    "rainfall_mm": "High rainfall: raised beds and good drainage; protective sprays against blights.",
    "temperature_c": "Hot climate: choose heat-tolerant hybrids; flower drop likely above ~32 °C.",
    "humidity_pct": "High humidity: wider spacing, pruning and preventive fungicide program.",
}


def _load():
    if _state["model"] is not None:
        return
    with _lock:
        if _state["model"] is not None:
            return
        if not config.SOIL_MODEL_PATH.exists() or not config.SOIL_FEATURES_PATH.exists():
            _state["error"] = (f"Soil model not found. Copy xgb_suitability.joblib and feature_order.joblib "
                               f"into {config.MODELS_DIR}")
            raise RuntimeError(_state["error"])
        import joblib
        from explain import SuitabilityExplainer  # soil_suitability/explain.py
        _state["model"] = joblib.load(config.SOIL_MODEL_PATH)
        _state["features"] = joblib.load(config.SOIL_FEATURES_PATH)
        _state["explainer"] = SuitabilityExplainer(_state["model"], _state["features"])
        _state["error"] = None


def status():
    return {"file": str(config.SOIL_MODEL_PATH), "exists": config.SOIL_MODEL_PATH.exists(),
            "loaded": _state["model"] is not None, "error": _state["error"]}


def fertility(features):
    """Rule-based band per parameter using the same thresholds the model was trained on."""
    rows = []
    for f, (il, ih, tl, th) in RANGES.items():
        v = features.get(f)
        if v is None:
            continue
        if il <= v <= ih:
            band, advice = "optimal", None
        elif v < il:
            band = "low" if v > tl else "very low"
            advice = ADVICE_LOW[f]
        else:
            band = "high" if v < th else "very high"
            advice = ADVICE_HIGH[f]
        rows.append({"feature": f, "label": LABELS[f], "value": round(float(v), 2),
                     "ideal": [il, ih], "band": band, "advice": advice})
    score = sum(1 for r in rows if r["band"] == "optimal") / max(len(rows), 1)
    overall = "Good" if score >= 0.75 else "Moderate" if score >= 0.5 else "Poor"
    return {"overall": overall, "optimal_count": sum(r["band"] == "optimal" for r in rows),
            "total": len(rows), "parameters": rows}


def predict(features):
    _load()
    names = _state["features"]
    model_features = {f: float(features[f]) for f in names}
    row = [[model_features[f] for f in names]]
    proba = float(_state["model"].predict_proba(row)[0][1])
    suitable = proba >= 0.5
    reasons = _state["explainer"].explain(model_features, top_k=len(names))
    for r in reasons:
        r["label"] = LABELS.get(r["feature"], r["feature"])
    return {"suitable": suitable, "probability_suitable": round(proba, 3),
            "confidence": round(proba if suitable else 1 - proba, 3),
            "soil_features_used": model_features, "shap": reasons,
            "top_reasons": reasons[:4], "fertility": fertility(model_features)}


def predict_by_location(lat, lon, overrides):
    from soil_client import get_soil_profile
    from weather_client import fetch_climatology
    soil = get_soil_profile(lat, lon)
    if soil.get("ph") is None:
        raise ValueError("No SoilGrids data for this location (water / built-up area?)")
    sources = dict(soil.pop("data_sources"))
    for k in ("phosphorus_ppm", "potassium_ppm"):
        if overrides.get(k) is not None:
            soil[k] = overrides[k]
            sources[k] = "farmer-entered (soil test)"
    climate = fetch_climatology(lat, lon)
    for k in ("rainfall_mm", "temperature_c", "humidity_pct"):
        if overrides.get(k) is not None:
            climate[k] = overrides[k]
            sources[k] = "farmer override"
        else:
            sources[k] = "NASA POWER (real)"
    result = predict({**soil, **climate})
    result["data_sources"] = sources
    return result
