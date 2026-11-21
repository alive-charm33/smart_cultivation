"""
Fetches real soil properties for a given lat/lon from ISRIC SoilGrids
(https://www.isric.org/explore/soilgrids), a free global soil-property API.

SoilGrids gives pH, organic carbon, AND total nitrogen. It does NOT give
phosphorus or potassium at all (those need lab soil tests or a national
soil-testing scheme dataset, e.g. India's Soil Health Card database) —
no free global API provides P/K, because P and K vary too sharply
field-to-field to map at 250m resolution the way pH/organic carbon can.

IMPORTANT CAVEAT ON NITROGEN: SoilGrids' "nitrogen" property is TOTAL
nitrogen (organic + mineral, everything locked in the soil), not
"available nitrogen" (the smaller, plant-usable portion that soil-test
labs report in mg/kg and that most crop-suitability thresholds are
written around). Total N is typically 5-15x larger than available N.
We convert it to mg/kg (ppm) units for consistency, but treat it as an
approximate signal, not a lab-equivalent value — flagged in the output
so this isn't silently misrepresented as a soil-test result.
"""

import json
import math
import os
from typing import Dict, Optional

import requests

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids properties we care about, at 0-5cm depth (topsoil, most
# relevant to a newly planted crop)
PROPERTIES = ["phh2o", "soc", "nitrogen"]
DEPTH = "0-5cm"

# FALLBACK, TIER 1: a cache of REAL SoilGrids data pre-fetched for every
# Maharashtra district (see build_soil_cache.py). If SoilGrids fails at
# request time, we use the geographically NEAREST cached district's real
# data instead of a single guessed constant. Build/refresh this by
# running: python build_soil_cache.py (whenever SoilGrids happens to be up)
SOIL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "soil_cache.json")

# FALLBACK, TIER 2: used only if the cache file doesn't exist yet (e.g.
# you haven't run build_soil_cache.py) or is empty. A single rough
# regional guess, NOT real data -- last resort only.
FALLBACK_SOIL_PROFILE = {
    "ph": 6.8,
    "organic_carbon_pct": 1.8,
    "total_nitrogen_ppm": 1200.0,
}


def _load_soil_cache() -> list:
    if not os.path.exists(SOIL_CACHE_PATH):
        return []
    try:
        with open(SOIL_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _find_nearest_cached(lat: float, lon: float) -> Optional[Dict]:
    """Returns the closest cached district's soil data, plus its distance
    in km, or None if the cache is empty/missing."""
    cache = _load_soil_cache()
    if not cache:
        return None
    nearest = min(cache, key=lambda entry: _haversine_km(lat, lon, entry["lat"], entry["lon"]))
    distance_km = _haversine_km(lat, lon, nearest["lat"], nearest["lon"])
    return {**nearest, "distance_km": round(distance_km, 1)}


def fetch_soilgrids(lat: float, lon: float) -> Dict:
    """Fetch pH, organic carbon, and total nitrogen from SoilGrids for a
    given coordinate.

    Raises requests.HTTPError / requests.RequestException if the API
    call fails (e.g. no network, SoilGrids outage, or point over water).
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": PROPERTIES,
        "depth": DEPTH,
        "value": "mean",
    }
    resp = requests.get(SOILGRIDS_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    layers = {
        layer["name"]: layer["depths"][0]["values"]["mean"]
        for layer in data["properties"]["layers"]
    }

    # Unit conversions per SoilGrids documentation:
    #   phh2o     -> mapped as pH*10          -> divide by 10 for real pH
    #   soc       -> mapped as dg/kg          -> divide by 100 for %
    #   nitrogen  -> mapped as cg/kg          -> divide by 100 for g/kg,
    #               then *1000 to express as mg/kg (ppm) — net: raw * 10
    ph = layers.get("phh2o")
    soc = layers.get("soc")
    nitrogen_raw = layers.get("nitrogen")

    total_nitrogen_ppm = round(nitrogen_raw * 10, 2) if nitrogen_raw is not None else None

    return {
        "ph": round(ph / 10, 2) if ph is not None else None,
        "organic_carbon_pct": round(soc / 100, 2) if soc is not None else None,
        "total_nitrogen_ppm": total_nitrogen_ppm,  # TOTAL N, see caveat above
    }


def _estimate_npk_placeholder() -> Dict:
    """Placeholder phosphorus/potassium values, used because no free API
    provides these. ALWAYS prefer real lab/Soil-Health-Card data over this
    in an actual deployment — P and K vary far more by field than pH does,
    so a placeholder here is a real accuracy risk, not just a formality."""
    return {
        "phosphorus_ppm": 30.0,
        "potassium_ppm": 200.0,
    }


def get_soil_profile(lat: float, lon: float) -> Dict:
    """Returns the full soil feature set needed by the suitability model:
    real pH + organic carbon + total nitrogen from SoilGrids, combined
    with placeholder P/K (no free source exists for those).

    If the live SoilGrids call fails (outage, timeout, etc.), falls back
    in two tiers:
      1. The geographically nearest pre-cached Maharashtra district's
         REAL SoilGrids data (see build_soil_cache.py) -- still real
         data, just not for this exact point.
      2. If no cache exists at all, a single rough regional guess (NOT
         real data) as a last resort.
    Whichever path is used is clearly flagged in data_sources so this is
    never silently mistaken for a live, location-specific reading.

    Note: `total_nitrogen_ppm` is passed through as `nitrogen_ppm` for
    the model, but it is TOTAL nitrogen, not available nitrogen — see
    module docstring. This is a known approximation, not a lab result.
    """
    fallback_tier = None  # None | "cache" | "constant"
    fallback_detail = None

    try:
        soil = fetch_soilgrids(lat, lon)
        if soil.get("ph") is None:
            raise ValueError("SoilGrids returned no data for this location")
    except Exception:
        cached = _find_nearest_cached(lat, lon)
        if cached is not None:
            soil = {
                "ph": cached["ph"],
                "organic_carbon_pct": cached["organic_carbon_pct"],
                "total_nitrogen_ppm": cached["total_nitrogen_ppm"],
            }
            fallback_tier = "cache"
            fallback_detail = f"{cached['name']} ({cached['distance_km']} km away)"
        else:
            soil = dict(FALLBACK_SOIL_PROFILE)
            fallback_tier = "constant"
            fallback_detail = "no cache available — used a single rough regional guess"

    npk_placeholder = _estimate_npk_placeholder()
    used_fallback = fallback_tier is not None

    if fallback_tier == "cache":
        soil_source_note = f"FALLBACK (cached real data) — SoilGrids unavailable, using nearest cached district: {fallback_detail}"
    elif fallback_tier == "constant":
        soil_source_note = f"FALLBACK (rough guess) — {fallback_detail}"
    else:
        soil_source_note = "SoilGrids (real)"

    return {
        "ph": soil["ph"],
        "organic_carbon_pct": soil["organic_carbon_pct"],
        "nitrogen_ppm": soil["total_nitrogen_ppm"],  # from SoilGrids (total N, approximation)
        "phosphorus_ppm": npk_placeholder["phosphorus_ppm"],  # placeholder
        "potassium_ppm": npk_placeholder["potassium_ppm"],  # placeholder
        "used_fallback_soil_data": used_fallback,
        "fallback_tier": fallback_tier,  # None | "cache" | "constant"
        "data_sources": {
            "ph": soil_source_note,
            "organic_carbon_pct": soil_source_note,
            "nitrogen_ppm": soil_source_note if used_fallback else "SoilGrids (real, but TOTAL N not available N — approximation)",
            "phosphorus_ppm": "placeholder — no free API available, needs lab/Soil Health Card data",
            "potassium_ppm": "placeholder — no free API available, needs lab/Soil Health Card data",
        },
    }


if __name__ == "__main__":
    # Example: Sangli, Maharashtra (approx coordinates)
    profile = get_soil_profile(lat=16.85, lon=74.57)
    print(profile)
