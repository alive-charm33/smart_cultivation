"""
Fetches real SoilGrids data for every Maharashtra TALUKA (360 of them,
vs. the coarser 36-district version) and saves it to soil_cache.json.

Since taluka-level coordinates aren't reliably available in one dataset,
this script first GEOCODES each taluka name (finds its lat/lon) using
OpenStreetMap's free Nominatim service, then fetches its SoilGrids data.

IMPORTANT — THIS TAKES A LONG TIME: with ~360 talukas, respecting both
Nominatim's usage policy (max 1 request/second) and SoilGrids' fair-use
policy (max 5 requests/minute), this takes roughly:
  - Geocoding: ~360 x 1.1s  =~ 7 minutes (more if retries are needed)
  - SoilGrids:  ~360 x 13s  =~ 78 minutes
  - Total: around 85-95 minutes

THIS SCRIPT IS RESUMABLE: it saves progress after every single taluka,
and on re-running, skips any taluka already successfully cached. So you
can safely stop it (Ctrl+C) and re-run later to pick up where it left
off -- you do NOT need one uninterrupted session.

NOTE ON SKIPPED TALUKAS: not every taluka will geocode successfully —
Nominatim (the free geocoding service) doesn't recognize every small
place name, especially with exact district-qualified phrasing. This
script tries 3 progressively broader queries per taluka before giving
up, which recovers most of these — but a small number of skips is
still normal and NOT a problem: the fallback system only needs
reasonably good statewide coverage, not all 360 points, to work well.

Usage:
    python build_soil_cache.py
"""

import json
import time

import requests

from maharashtra_talukas import MAHARASHTRA_TALUKAS
from soil_client import fetch_soilgrids

CACHE_PATH = "soil_cache.json"
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy requires a descriptive User-Agent and caps
# usage at 1 request/second: https://operations.osmfoundation.org/policies/nominatim/
GEOCODE_HEADERS = {"User-Agent": "TomatoSoilSuitabilityProject/1.0 (student project)"}
SECONDS_BETWEEN_GEOCODE_CALLS = 1.1
SECONDS_BETWEEN_SOILGRIDS_CALLS = 13  # stays under SoilGrids' 5-calls/minute limit


def geocode_taluka(taluka: str, district: str):
    """Looks up a taluka's approximate lat/lon via Nominatim. Returns
    (lat, lon) or None if not found.

    Tries a specific query first, then falls back to progressively
    broader, simpler queries if the specific one finds nothing -- this
    recovers many talukas that would otherwise be skipped just because
    Nominatim doesn't recognize the exact "X district" phrasing.
    """
    queries_to_try = [
        f"{taluka}, {district} district, Maharashtra, India",  # specific
        f"{taluka} taluka, Maharashtra, India",                 # medium
        f"{taluka}, Maharashtra, India",                        # broad
    ]

    for query in queries_to_try:
        params = {"q": query, "format": "json", "limit": 1}
        try:
            resp = requests.get(GEOCODE_URL, params=params, headers=GEOCODE_HEADERS, timeout=15)
            resp.raise_for_status()
            results = resp.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
        except Exception as e:
            print(f"    Geocoding attempt failed ({query}): {e}")
        time.sleep(SECONDS_BETWEEN_GEOCODE_CALLS)  # respect rate limit between retries too

    return None


def _load_existing_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def build_cache():
    cache = _load_existing_cache()
    already_done = {(entry["taluka"], entry["district"]) for entry in cache if "taluka" in entry}
    print(f"Resuming: {len(already_done)} talukas already cached from a previous run.\n")

    remaining = [(t, d) for t, d in MAHARASHTRA_TALUKAS if (t, d) not in already_done]
    print(f"{len(remaining)} talukas left to fetch (~{len(remaining) * 14 // 60} minutes estimated)\n")

    skipped = []
    for i, (taluka, district) in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {taluka}, {district}...")

        coords = geocode_taluka(taluka, district)
        if coords is None:
            print("  -> Could not geocode after all attempts, skipping")
            skipped.append(f"{taluka}, {district}")
            continue
        lat, lon = coords

        try:
            soil = fetch_soilgrids(lat, lon)
            if soil.get("ph") is None:
                print("  -> No SoilGrids data for this point, skipping")
                skipped.append(f"{taluka}, {district}")
                time.sleep(SECONDS_BETWEEN_SOILGRIDS_CALLS)
                continue
            cache.append({
                "name": f"{taluka}, {district}",
                "taluka": taluka,
                "district": district,
                "lat": lat,
                "lon": lon,
                **soil,
            })
            _save_cache(cache)  # save after EVERY success, so progress is never lost
            print(f"  -> OK: pH={soil['ph']}, organic_carbon={soil['organic_carbon_pct']}%, "
                  f"nitrogen={soil['total_nitrogen_ppm']}ppm  (saved, {len(cache)} total so far)")
        except Exception as e:
            print(f"  -> SoilGrids fetch failed: {e}")
            skipped.append(f"{taluka}, {district}")

        time.sleep(SECONDS_BETWEEN_SOILGRIDS_CALLS)

    print(f"\nDone. Cached {len(cache)}/{len(MAHARASHTRA_TALUKAS)} talukas to {CACHE_PATH}")
    if skipped:
        print(f"\n{len(skipped)} talukas were skipped (this is normal, not an error):")
        for s in skipped:
            print(f"  - {s}")
        print(
            "\nYou don't need to fix these by hand. The fallback system only "
            "needs reasonably good statewide coverage, not all 360 points. "
            "If you want to try recovering some, just re-run this script "
            "later -- Nominatim/SoilGrids may succeed on a retry, and "
            "already-cached talukas won't be re-fetched."
        )


if __name__ == "__main__":
    build_cache()