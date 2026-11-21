"""
Generates a synthetic tomato-soil-suitability dataset grounded in real
agronomic thresholds for tomato cultivation.

WHY SYNTHETIC: there is no single public "tomato suitability" labeled
tabular dataset. This script encodes known agronomic rules (ideal/tolerable
ranges for pH, NPK, rainfall, temperature, humidity) into a scoring
function, then adds noise so the resulting dataset isn't perfectly
separable (matching real-world messiness) — that's what XGBoost trains on.

Replace this with real labeled data (extension officer records, field
trial outcomes, etc.) as soon as you have it — this is a credible
starting point for a student project, not a substitute for real data
in a production system.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Agronomic reference ranges for tomato (Solanum lycopersicum)
# (ideal_low, ideal_high, tolerable_low, tolerable_high)
#
# NOTE on nitrogen_ppm: calibrated to TOTAL soil nitrogen (what SoilGrids
# provides), NOT "available nitrogen". IMPORTANT HONESTY NOTE: the
# genuinely tomato-specific nitrogen threshold, from UC Davis/FREP
# research (Krueskopf et al.), is a soil NITRATE test (not total N):
# tomato yield response is unlikely when pre-sidedress soil nitrate-N
# in the top 2 feet exceeds ~16 ppm. That measurement is fundamentally
# different from SoilGrids' total-N value (different chemistry, different
# scale, different sampling protocol) and can't be substituted in without
# a live soil-nitrate test. So nitrogen remains the one feature NOT
# calibrated to a tomato-specific source -- flag this honestly if asked.
#
# NOTE on phosphorus_ppm / potassium_ppm: NOW calibrated to genuinely
# TOMATO-SPECIFIC research, not a generic multi-crop chart:
#   Source: UC Davis / California FREP fertilization guidelines for
#   processing tomatoes (Geisseler et al. 2020, citing Hartz & Hanson
#   2009 for P and Hartz et al. 1999/2005 for K):
#   http://geisseler.ucdavis.edu/Guidelines/Tomato.html
#   - Phosphorus (Olsen-P test): optimal/sufficient range is 10-20 ppm;
#     below 10 ppm is low (yield-limiting), above 20 ppm rarely gives
#     any further yield response.
#   - Potassium (exchangeable K test): ~135 ppm is "generally sufficient
#     for maximum yield" (paste tomatoes); higher for peeled/diced fruit
#     quality; 300 ppm shown as already fully sufficient (no added
#     benefit from more K).
#
# NOTE on rainfall_mm: this range (400-600mm ideal growing-season total)
# turns out to align well with genuinely tomato-specific crop-water-
# requirement research: FAO's irrigation manual works a 150-day tomato
# growing-season example totaling ~786mm, and field studies report
# tomato seasonal water requirements in the ~400-800mm range depending
# on climate and irrigation method. Tomato-specific, reasonably confident.
#
# NOTE on humidity_pct: genuinely tomato-specific. University of Alaska
# Fairbanks Cooperative Extension (and multiple corroborating greenhouse-
# tomato studies) puts 70% relative humidity as optimal specifically for
# pollination and fruit set -- too humid and pollen clumps and fails to
# transfer; too dry and stigmas dry out. General tomato growth range
# cited as roughly 60-85%.
#
# NOTE on organic_carbon_pct: NO tomato-specific number found despite
# searching -- available research (e.g. a Nature Geoscience global study)
# gives optimal soil organic carbon for maize, wheat, and rice, not
# tomato specifically. This range remains a general soil-fertility
# reference, not a tomato-specific one -- the one parameter besides
# nitrogen where that's the honest answer.
RANGES = {
    "ph":              (6.0, 6.8, 5.5, 7.5),
    "nitrogen_ppm":    (1000, 2000, 300, 3000),   # mg/kg TOTAL N (SoilGrids scale) -- NOT tomato-specific, see note
    "phosphorus_ppm":  (10, 20, 5, 35),            # mg/kg Olsen-P -- tomato-specific (UC Davis/FREP)
    "potassium_ppm":   (135, 250, 60, 350),        # mg/kg exchangeable K -- tomato-specific (UC Davis/FREP)
    "organic_carbon_pct": (1.5, 3.0, 0.8, 4.0),    # NOT tomato-specific -- general soil-fertility reference
    "rainfall_mm":     (400, 600, 300, 800),       # growing-season total -- tomato-specific (FAO, crop-water-requirement studies)
    "temperature_c":   (20, 27, 15, 32),           # tomato-specific (ICL agronomy guidance)
    "humidity_pct":    (60, 70, 40, 85),           # tomato-specific (UAF Extension, greenhouse-tomato studies)
}

FEATURES = list(RANGES.keys())


def _feature_score(value: float, bounds: tuple) -> float:
    """Returns 1.0 if within ideal range, decays smoothly to 0.0 outside
    the tolerable range. This is what makes the dataset agronomically
    meaningful rather than arbitrary."""
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


def _sample_feature(bounds: tuple) -> float:
    """Sample a value across a wide plausible range (wider than 'tolerable'
    so we get clear negative examples too). Clipped at 0 since none of
    these soil/climate quantities can physically be negative."""
    _, _, tol_low, tol_high = bounds
    span = tol_high - tol_low
    low = max(0.0, tol_low - 0.4 * span)
    high = tol_high + 0.4 * span
    return RNG.uniform(low, high)


def generate(n_samples: int = 4000) -> pd.DataFrame:
    rows = []
    for _ in range(n_samples):
        sample = {f: _sample_feature(RANGES[f]) for f in FEATURES}

        # Weighted agronomic suitability score (0-1).
        # pH and temperature matter most for tomato; weights reflect that.
        weights = {
            "ph": 0.22,
            "nitrogen_ppm": 0.13,
            "phosphorus_ppm": 0.10,
            "potassium_ppm": 0.10,
            "organic_carbon_pct": 0.10,
            "rainfall_mm": 0.13,
            "temperature_c": 0.15,
            "humidity_pct": 0.07,
        }
        score = sum(weights[f] * _feature_score(sample[f], RANGES[f]) for f in FEATURES)

        # Add real-world noise so the boundary isn't perfectly sharp
        noisy_score = np.clip(score + RNG.normal(0, 0.06), 0, 1)
        # Threshold picked at the empirical median of the scoring function
        # (see data exploration) so classes come out roughly balanced.
        label = int(noisy_score >= 0.39)

        sample["suitability_score_raw"] = round(score, 3)  # kept for inspection only
        sample["suitable"] = label
        rows.append(sample)

    df = pd.DataFrame(rows)
    # round for readability
    for f in FEATURES:
        df[f] = df[f].round(2)
    return df


if __name__ == "__main__":
    df = generate(4000)
    df.drop(columns=["suitability_score_raw"]).to_csv(
        "/home/claude/soil_suitability/data/tomato_suitability.csv", index=False
    )
    print(df["suitable"].value_counts(normalize=True))
    print(df.head())
    print(f"\nSaved {len(df)} rows to data/tomato_suitability.csv")
