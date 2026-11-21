# Module 1: Tomato Soil Suitability (XGBoost + SHAP)

This is Module 1 from your project's architecture — the first, simplest
piece to build, since it validates the whole location → soil → model →
API pipeline before you touch images or LLMs.

## Model & training summary (read this first)

**Algorithm**: XGBoost (`XGBClassifier`) — a gradient-boosted decision
tree classifier. Chosen because it's specified by name in the project
synopsis, handles tabular data well, and pairs naturally with SHAP for
explainability.

**Explainability**: SHAP (`shap.TreeExplainer`) — for every prediction,
returns which of the 8 input parameters pushed the result toward
"suitable" or "not suitable", and by how much (relative ranking, not a
percentage).

**Training data**: since no public "tomato soil suitability" labeled
dataset exists, we generated one synthetically — 4,000 example soil/
climate profiles, each scored against real agronomic thresholds for
8 parameters and labeled suitable/not-suitable. Full generation logic
is in `data/generate_dataset.py`.

**The 8 parameters, and how "good" was defined for each** — 6 are
calibrated to tomato-specific research, 2 to general agricultural
benchmarks:

| # | Parameter | Tomato-specific? |
|---|---|---|
| 1 | pH | ✅ Yes |
| 2 | Temperature | ✅ Yes |
| 3 | Phosphorus | ✅ Yes |
| 4 | Potassium | ✅ Yes |
| 5 | Rainfall | ✅ Yes |
| 6 | Humidity | ✅ Yes |
| 7 | Nitrogen | ❌ No (general benchmark) |
| 8 | Organic carbon | ❌ No (general benchmark) |

**Performance** (on held-out test data): **86.6% accuracy, F1 score
0.87.** Feature importance correctly ranks pH and temperature highest —
consistent with real tomato agronomy.

**How it was actually trained**: trained via `colab_train_pipeline.py`
(real `xgboost.XGBClassifier` + real `shap`), on Google Colab or
directly on a laptop — then the resulting model file
(`models/xgb_suitability.joblib`) is served by `api.py`.

## What's in here

| File | Purpose |
|---|---|
| `data/generate_dataset.py` | Builds the synthetic, agronomically-grounded training dataset |
| `train_model.py` | Trains the XGBoost classifier, evaluates it, saves it + a SHAP summary plot |
| `colab_train_pipeline.py` | Single self-contained script (dataset generation + training + SHAP, all in one file) — designed to run in Google Colab or locally |
| `explain.py` | Wraps SHAP so each prediction comes with its top contributing factors |
| `soil_client.py` | Fetches real soil pH, organic carbon, and total nitrogen from SoilGrids — includes the 3-tier fallback system |
| `maharashtra_talukas.py` | List of all 360 Maharashtra talukas (name + district), used to build the fallback cache |
| `build_soil_cache.py` | One-time (resumable) script: geocodes each taluka, fetches its real SoilGrids data, saves to `soil_cache.json` |
| `weather_client.py` | Fetches real long-term temperature/humidity/rainfall from NASA POWER by lat/lon |
| `api.py` | FastAPI service exposing `/suitability/predict-by-location` and `/suitability/predict-by-soil` |

## SoilGrids reliability: the 3-tier fallback system

SoilGrids' REST API has documented uptime issues (ISRIC themselves call
it "beta" with "no guarantee of uptime"), and even when it's up, it can
have no data for a specific point (e.g. over water). `soil_client.py`
handles BOTH situations identically, with the same 3-tier fallback:

1. **Live SoilGrids** for the exact GPS coordinate — tried first, always
   preferred when it succeeds and returns real data.
2. **Nearest cached Maharashtra taluka** — if live fails for ANY reason
   (outage, timeout, or no data for that exact point), find the closest
   of your pre-fetched talukas in `soil_cache.json` using straight-line
   (haversine) distance, and use its real SoilGrids data instead. Not
   exact-location data, but real data from nearby.
3. **A single rough guess** — only reached if tier 2's cache file is
   completely empty (e.g. `build_soil_cache.py` hasn't been run at all
   yet). Not real data, last resort only.

Every API response tells you exactly which tier was used
(`fallback_tier`: `None` / `"cache"` / `"constant"`) plus a
plain-language `warning` if a fallback was used — so you always know
whether you got real, location-specific data or not. The system never
tries to distinguish *why* the live call failed — any failure funnels
into the same tier-2-then-tier-3 fallback chain, which keeps the logic
simple and covers every failure case with one path.

### Building the cache

```bash
python build_soil_cache.py
```

This fetches at **taluka level** — 360 talukas across all 36 districts
(much finer resolution than one point per district). For each taluka,
it does two steps: **geocode** the taluka's coordinates via
OpenStreetMap's free Nominatim service, then **fetch** its real
SoilGrids data.

**Runtime**: roughly 85-95 minutes total for all 360 (paced
deliberately to respect both services' rate limits — Nominatim allows
max 1 request/second, SoilGrids max 5 requests/minute).

**This script is resumable.** It saves progress to `soil_cache.json`
after every single successful taluka, and on re-running, automatically
skips whatever's already saved and only fetches what's left. You can
stop it anytime (Ctrl+C) and continue later — you never lose progress,
and you never need one uninterrupted session. (If you've already run
an earlier version of this script and have, say, 205 talukas saved,
re-running the current version will skip those 205 instantly and only
attempt the remaining ones — your existing `soil_cache.json` is safe
and doesn't need to be deleted or rebuilt.)

**Geocoding retries**: not every taluka name is recognized by Nominatim
on the first try. For each taluka, the script tries up to 3
progressively broader queries before giving up:
1. `"{taluka}, {district} district, Maharashtra, India"` (specific)
2. `"{taluka} taluka, Maharashtra, India"` (medium)
3. `"{taluka}, Maharashtra, India"` (broad)

**Skipped talukas are normal, not an error.** At the end of the run,
the script prints a list of any talukas it couldn't geocode or find
SoilGrids data for. You do NOT need to fix these by hand — the
fallback system only needs reasonably good statewide coverage (a
cached point roughly every 20-30km), not literally all 360 points, to
work well. If you want to try recovering a few more later, just
re-run the script — already-cached entries are skipped, so it only
retries what's missing, and a fresh attempt sometimes succeeds where
an earlier one didn't (transient Nominatim/SoilGrids hiccups).

## Where each feature's data actually comes from

| Feature | Data source (live) | Threshold basis | Tomato-specific? |
|---|---|---|---|
| pH | SoilGrids (real) | Rutgers/UNH/ICL extension guidance | ✅ Yes |
| Temperature | NASA POWER (real) | ICL tomato agronomy guidance | ✅ Yes |
| Phosphorus | SoilGrids doesn't have it — farmer input or placeholder | UC Davis/California FREP tomato research (Olsen-P, 10-20 ppm optimal) | ✅ Yes |
| Potassium | SoilGrids doesn't have it — farmer input or placeholder | UC Davis/California FREP tomato research (exchangeable K, ~135 ppm sufficient) | ✅ Yes |
| Rainfall | NASA POWER (real) | FAO irrigation manual + tomato crop-water-requirement studies | ✅ Yes |
| Humidity | NASA POWER (real) | University of Alaska Fairbanks Extension + greenhouse-tomato studies | ✅ Yes |
| Nitrogen | SoilGrids (real, but TOTAL N, not available N) | General agricultural total-N field-study benchmarks | ❌ No |
| Organic carbon | SoilGrids (real) | General soil-fertility reference | ❌ No |

**P/K note**: no free API provides phosphorus or potassium anywhere —
the API accepts optional farmer-entered `phosphorus_ppm` /
`potassium_ppm` (from a Soil Health Card, NPK probe, or lab report),
using those when given and a placeholder only otherwise.

**Nitrogen caveat**: SoilGrids gives *total* soil nitrogen, not the
*available* nitrogen that soil-test labs/fertilizer recommendations
use — a genuinely different measurement. Treated as an approximate
soil-health signal, not a lab-equivalent number.

## Which parameters need manual input from the farmer?

This is worth stating explicitly, since it's easy to assume more manual
entry is needed than actually is:

| Parameter | Automatic or manual? |
|---|---|
| pH | **Automatic** — fetched from SoilGrids, no farmer input needed |
| Temperature | **Automatic** — fetched from NASA POWER |
| **Nitrogen** | **Automatic** — fetched from SoilGrids (total N). **Not** something the farmer enters. |
| Organic carbon | **Automatic** — fetched from SoilGrids |
| Rainfall | **Automatic** — fetched from NASA POWER |
| Humidity | **Automatic** — fetched from NASA POWER |
| **Phosphorus** | **Optional manual input** — no free API exists for this anywhere. Farmer can enter `phosphorus_ppm` from a Soil Health Card/NPK probe/lab report. If left blank, a placeholder (30 ppm) is used instead. |
| **Potassium** | **Optional manual input** — same situation. Farmer can enter `potassium_ppm`. If left blank, a placeholder (200 ppm) is used instead. |

**In one line: only phosphorus and potassium ever need manual input, and even those are optional** — the API works end-to-end without them (using a placeholder), it's just more accurate with real farmer-provided values. All 6 other parameters are fully automatic; the farmer never types in a nitrogen, pH, or temperature value anywhere. The only two inputs a farmer's app screen strictly needs are their **GPS location** (for the automatic 6) plus, optionally, phosphorus/potassium if they happen to have a soil test result on hand.

## Setup

```bash
cd soil_suitability
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# 1. Build the soil fallback cache (one-time, resumable, ~85-95 min)
python build_soil_cache.py

# 2. Train the model (or use colab_train_pipeline.py instead)
python train_model.py

# 3. Start the API
uvicorn api:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/suitability/predict-by-location \
  -H "Content-Type: application/json" \
  -d '{"lat": 16.85, "lon": 74.57}'
```

## About the trained model

`models/xgb_suitability.joblib` should be your **real XGBoost model** —
trained via `colab_train_pipeline.py`. If you ever see
`GradientBoostingClassifier` instead of `XGBClassifier` when you
inspect the loaded model's type, that means you're looking at an early
placeholder version — just re-run `colab_train_pipeline.py` to get the
real XGBoost model back.

## Important limitations to know (and to mention in your report)

1. **Training data is synthetic**, built from published agronomic
   thresholds, not field-collected labels.
2. **Phosphorus and potassium have no free live-data source at all** —
   farmer-entered values are the honest fix, used when provided.
3. **Nitrogen uses total N, not available N** — a documented
   approximation, not a lab-equivalent number.
4. **Organic carbon has no tomato-specific research available** — best
   available data covers other crops (maize/wheat/rice).
5. **SoilGrids' own uptime is unreliable** — mitigated by the 3-tier
   fallback system, but worth mentioning as a real-world engineering
   challenge you identified and solved.

## Next steps

- Finish running `build_soil_cache.py` to completion (resumable, so
  this can happen gradually).
- Wire this endpoint to your React `SuitabilityCard` component.
- Move on to Module 3 (Growth Stage Detection) once this is stable.

## How this fits with the rest of the project

This README covers Module 1 (Soil Suitability) only. Other modules
built separately, each in their own folder:

- **Disease detection + batch scanning** (`disease_batch_analysis/`):
  wraps your teammate's disease model with multi-photo aggregation and
  a treatment-advice lookup table — no changes to the disease model
  itself.
- **Growth stage timeline + knowledge base** (`growth_tracking/`):
  the expected-vs-actual growth timeline logic and the consolidated RAG
  knowledge base (growth stages, disease treatments, soil parameter
  explanations, farming best practices) for the LLM assistant.
- **Cost estimation**: land area + this module's soil result feed a
  plain arithmetic calculator — no AI or RAG involved.

Module 1's SHAP explanations and Module 1's soil result are both
consumed directly by the other modules above rather than recreated —
if a teammate needs "why is this soil good," it should come from this
module's `/suitability/predict-by-location` response, not a new model.