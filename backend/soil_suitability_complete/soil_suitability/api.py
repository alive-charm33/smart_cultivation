"""
FastAPI service exposing the tomato soil-suitability model.

Run with:
    uvicorn api:app --reload --port 8000

Then either:
  POST /suitability/predict-by-location   {"lat": 16.85, "lon": 74.57}
  POST /suitability/predict-by-soil       {"ph": 6.4, "nitrogen_ppm": 210, ...}
"""

from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from explain import SuitabilityExplainer
from soil_client import get_soil_profile
from weather_client import fetch_climatology

app = FastAPI(title="Tomato Soil Suitability API")

MODEL_PATH = "models/xgb_suitability.joblib"
FEATURE_ORDER_PATH = "models/feature_order.joblib"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_ORDER_PATH)
explainer = SuitabilityExplainer(model, feature_names)


class LocationRequest(BaseModel):
    lat: float
    lon: float
    # Optional climate overrides — if not given, real values are fetched
    # from NASA POWER automatically (see weather_client.py). Only use
    # these if the frontend has its own more specific weather source.
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    # Optional farmer-entered soil-test values (from a Soil Health Card,
    # a handheld NPK probe, or a lab report). No free API provides P/K,
    # so these are the honest way to get real numbers — if omitted, a
    # clearly-flagged placeholder is used instead.
    phosphorus_ppm: Optional[float] = None
    potassium_ppm: Optional[float] = None


class SoilFeaturesRequest(BaseModel):
    ph: float
    nitrogen_ppm: float
    phosphorus_ppm: float
    potassium_ppm: float
    organic_carbon_pct: float
    rainfall_mm: float
    temperature_c: float
    humidity_pct: float


def _predict_and_explain(features: dict) -> dict:
    model_features = {f: features[f] for f in feature_names}
    row = [[model_features[f] for f in feature_names]]
    proba = model.predict_proba(row)[0][1]  # probability of class "suitable"
    suitable = bool(proba >= 0.5)
    reasons = explainer.explain(model_features)

    return {
        "suitable": suitable,
        "confidence": round(float(proba if suitable else 1 - proba), 3),
        "soil_features_used": model_features,
        "top_reasons": reasons,
    }


@app.post("/suitability/predict-by-location")
def predict_by_location(req: LocationRequest):
    try:
        soil = get_soil_profile(req.lat, req.lon)
    except Exception as e:
        # Should rarely trigger now — get_soil_profile() catches its own
        # failures and falls back internally — but kept as a safety net.
        raise HTTPException(status_code=502, detail=f"Soil data fetch failed: {e}")

    used_fallback_soil = soil.pop("used_fallback_soil_data", False)
    fallback_tier = soil.pop("fallback_tier", None)
    data_sources = dict(soil.pop("data_sources"))

    # Farmer-entered P/K (Soil Health Card / NPK probe / lab report)
    # overrides the placeholder, since it's real, field-specific data.
    if req.phosphorus_ppm is not None:
        soil["phosphorus_ppm"] = req.phosphorus_ppm
        data_sources["phosphorus_ppm"] = "farmer-entered (soil test)"
    if req.potassium_ppm is not None:
        soil["potassium_ppm"] = req.potassium_ppm
        data_sources["potassium_ppm"] = "farmer-entered (soil test)"

    # Climate: use real NASA POWER data by default, but allow the caller
    # to override any field (e.g. frontend already has a weather widget).
    climate_source = "NASA POWER (real)"
    try:
        live_climate = fetch_climatology(req.lat, req.lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {e}")

    climate = {}
    for field in ("rainfall_mm", "temperature_c", "humidity_pct"):
        provided = getattr(req, field)
        if provided is None:
            climate[field] = live_climate[field]
            data_sources[field] = climate_source
        else:
            climate[field] = provided
            data_sources[field] = "caller override"

    features = {**soil, **climate}
    result = _predict_and_explain(features)
    result["data_sources"] = data_sources
    result["used_fallback_soil_data"] = used_fallback_soil
    result["fallback_tier"] = fallback_tier  # None | "cache" | "constant"
    if fallback_tier == "cache":
        result["warning"] = (
            "SoilGrids was unavailable — this result used real soil data "
            "from the nearest pre-cached Maharashtra district, not this "
            "exact location. See data_sources.ph for which district."
        )
    elif fallback_tier == "constant":
        result["warning"] = (
            "SoilGrids was unavailable AND no soil cache was found — this "
            "result used a rough regional guess, not real data. Run "
            "build_soil_cache.py to enable the more accurate cache fallback."
        )
    return result


@app.post("/suitability/predict-by-soil")
def predict_by_soil(req: SoilFeaturesRequest):
    return _predict_and_explain(req.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}