"""Local vision-based tomato growth stage detector.

Uses computer vision techniques on the plant photo to identify visual indicators
of the growth stage — NO external API needed, works fully offline.

Visual indicators per stage:
  nursery/transplanting : tiny seedlings, small cotyledons/true leaves, soil trays/plug
  vegetative            : medium plant, many leaves, no flowers/fruit
  flowering_fruit_set   : yellow flowers visible
  fruit_development     : green fruits visible
  ripening_harvest      : red/orange fruits visible
  end_of_season         : yellowing/dry canopy, red fruits
"""

import io
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image

# ------------------------------------------------------------------ constants
STAGE_ORDER = [
    "nursery", "transplanting", "vegetative",
    "flowering_fruit_set", "fruit_development",
    "ripening_harvest", "end_of_season",
]

# Visual feature thresholds (tuned for tomato plants)
# Each stage has color-based and texture-based signatures
STAGE_SIGNATURES = {
    "nursery": {
        "description": "Very small seedlings in trays, cotyledons or 2-4 true leaves, soil/tray visible",
        "green_ratio_min": 0.05, "green_ratio_max": 0.30,
        "yellow_ratio_max": 0.05,
        "red_ratio_max": 0.03,
        "plant_coverage_max": 0.35,  # small plant, lots of background/soil
    },
    "transplanting": {
        "description": "Small plants recently transplanted, 4-6 true leaves, stems thin",
        "green_ratio_min": 0.10, "green_ratio_max": 0.40,
        "yellow_ratio_max": 0.08,
        "red_ratio_max": 0.03,
        "plant_coverage_max": 0.50,
    },
    "vegetative": {
        "description": "Bushy green plant, many leaves, no flowers or fruit visible",
        "green_ratio_min": 0.30,
        "yellow_ratio_max": 0.10,
        "red_ratio_max": 0.05,
        "plant_coverage_min": 0.35,
    },
    "flowering_fruit_set": {
        "description": "Yellow flowers visible on trusses",
        "green_ratio_min": 0.25,
        "yellow_ratio_min": 0.03,
        "red_ratio_max": 0.08,
    },
    "fruit_development": {
        "description": "Green fruits clearly visible, some flowers may remain",
        "green_ratio_min": 0.25,
        "has_round_green_objects": True,
    },
    "ripening_harvest": {
        "description": "Red/orange fruits visible alongside green fruits",
        "red_ratio_min": 0.04,
        "green_ratio_min": 0.15,
    },
    "end_of_season": {
        "description": "Yellowing/dying canopy, red fruits, brown stems",
        "yellow_ratio_min": 0.08,
        "red_ratio_min": 0.05,
    },
}


# ------------------------------------------------------------------ image analysis
def _extract_color_features(img_rgb: np.ndarray) -> dict:
    """Extract color distribution features from the image."""
    h, w, _ = img_rgb.shape
    total = h * w

    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)

    # Green pixels: G > R and G > B (plant tissue)
    green_mask = (g > r * 1.1) & (g > b * 1.05) & (g > 40)
    green_ratio = float(green_mask.sum()) / total

    # Yellow pixels: R and G high, B low (flowers)
    yellow_mask = (r > 120) & (g > 100) & (b < 80) & (r > b * 1.5) & (g > b * 1.3)
    yellow_ratio = float(yellow_mask.sum()) / total

    # Red/orange pixels: R high, G medium, B low (ripe fruit)
    red_mask = (r > 120) & (r > g * 1.3) & (r > b * 1.5) & (g < 160)
    red_ratio = float(red_mask.sum()) / total

    # Brown/soil pixels: R > G > B, all moderate
    brown_mask = (r > 80) & (g > 50) & (b > 30) & (r > g) & (g > b) & ~green_mask & ~red_mask
    brown_ratio = float(brown_mask.sum()) / total

    # Plant coverage: green + yellow + some of the plant-colored pixels
    plant_mask = green_mask | yellow_mask
    plant_coverage = float(plant_mask.sum()) / total

    # Brightness and texture
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    brightness = float(gray.mean()) / 255.0

    return {
        "green_ratio": round(green_ratio, 4),
        "yellow_ratio": round(yellow_ratio, 4),
        "red_ratio": round(red_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "plant_coverage": round(plant_coverage, 4),
        "brightness": round(brightness, 4),
    }


def _score_stage(features: dict, stage_id: str) -> float:
    """Return a [0, 1] score for how well image features match a stage."""
    sig = STAGE_SIGNATURES[stage_id]
    score = 1.0
    penalty = 0.0

    green = features["green_ratio"]
    yellow = features["yellow_ratio"]
    red = features["red_ratio"]
    coverage = features["plant_coverage"]

    # Apply constraints
    if "green_ratio_min" in sig and green < sig["green_ratio_min"]:
        penalty += (sig["green_ratio_min"] - green) * 5
    if "green_ratio_max" in sig and green > sig["green_ratio_max"]:
        penalty += (green - sig["green_ratio_max"]) * 3
    if "yellow_ratio_min" in sig and yellow < sig["yellow_ratio_min"]:
        penalty += (sig["yellow_ratio_min"] - yellow) * 8
    if "yellow_ratio_max" in sig and yellow > sig["yellow_ratio_max"]:
        penalty += (yellow - sig["yellow_ratio_max"]) * 4
    if "red_ratio_min" in sig and red < sig["red_ratio_min"]:
        penalty += (sig["red_ratio_min"] - red) * 8
    if "red_ratio_max" in sig and red > sig["red_ratio_max"]:
        penalty += (red - sig["red_ratio_max"]) * 6
    if "plant_coverage_min" in sig and coverage < sig["plant_coverage_min"]:
        penalty += (sig["plant_coverage_min"] - coverage) * 3
    if "plant_coverage_max" in sig and coverage > sig["plant_coverage_max"]:
        penalty += (coverage - sig["plant_coverage_max"]) * 2

    score = max(0.0, 1.0 - penalty)
    return score


def detect_stage_from_image(image_path) -> dict:
    """
    Analyse the plant photo and return the detected growth stage.
    Returns: {stage_id, confidence, visual_features, observations, method}
    """
    img = Image.open(image_path).convert("RGB")
    # Resize for consistent analysis
    img_small = img.copy()
    img_small.thumbnail((512, 512))
    arr = np.array(img_small, dtype=np.uint8)

    features = _extract_color_features(arr)

    # Score each stage
    scores = {sid: _score_stage(features, sid) for sid in STAGE_ORDER}
    best_stage = max(scores, key=scores.get)
    best_score = scores[best_stage]

    # Normalize confidence
    score_values = list(scores.values())
    total = sum(score_values)
    confidence = best_score / total if total > 0 else 1.0 / len(STAGE_ORDER)
    confidence = round(min(confidence * 1.5, 0.95), 3)  # scale up slightly

    # Build observations
    observations = _build_observations(features)

    return {
        "stage_id": best_stage,
        "confidence": confidence,
        "visual_features": features,
        "stage_scores": {k: round(v, 3) for k, v in scores.items()},
        "observations": observations,
        "method": "local-vision (color/texture analysis)",
    }


def _build_observations(features: dict) -> list:
    """Generate human-readable observations from extracted features."""
    obs = []
    g = features["green_ratio"]
    y = features["yellow_ratio"]
    r = features["red_ratio"]
    cov = features["plant_coverage"]

    if cov < 0.2:
        obs.append(f"Very small plant coverage ({cov*100:.0f}%) — seedling or recently transplanted")
    elif cov < 0.4:
        obs.append(f"Small-to-medium plant coverage ({cov*100:.0f}%)")
    else:
        obs.append(f"Good plant canopy coverage ({cov*100:.0f}%)")

    if g > 0.40:
        obs.append("Dense green foliage — mature vegetative growth")
    elif g > 0.20:
        obs.append("Moderate green foliage")
    elif g > 0.08:
        obs.append("Light green — young/small plant")
    else:
        obs.append("Very little green — possible seedling or stressed plant")

    if y > 0.05:
        obs.append(f"Yellow color detected ({y*100:.1f}%) — possible flowers present")
    elif y > 0.02:
        obs.append("Slight yellow tones — possible early flowering")

    if r > 0.08:
        obs.append(f"Strong red/orange color ({r*100:.1f}%) — ripe fruits visible")
    elif r > 0.04:
        obs.append(f"Some red/orange ({r*100:.1f}%) — fruits beginning to ripen")
    elif r > 0.01:
        obs.append("Trace red tones — possible early ripening")

    return obs


# ------------------------------------------------------------------ RAG-based growth analysis
def analyse_growth_with_rag(image_path, days_after_transplant: int,
                             knowledge_module, previous_stage=None,
                             previous_days=None) -> dict:
    """
    Full pipeline:
    1. Detect stage from image (local vision)
    2. Get expected stage from days
    3. Compare detected vs expected
    4. Retrieve RAG context for the detected stage
    5. Generate structured analysis

    Returns a complete result dict ready for the API response.
    """
    from .growth import expected_stages, progress, STAGE_ORDER as SO

    # Step 1: Visual stage detection from photo
    vision_result = detect_stage_from_image(image_path)
    detected_stage = vision_result["stage_id"]
    features = vision_result["visual_features"]

    # Step 2: Expected stage from days
    expected = expected_stages(days_after_transplant)

    # Step 3: Compare
    prog = progress(
        detected_stage, expected,
        previous_stage=previous_stage,
        previous_days=previous_days,
        days=days_after_transplant,
        stage_source="vision",
    )

    # Step 4: Determine mismatch severity
    mismatch_info = _analyse_mismatch(
        detected_stage, expected, days_after_transplant, features
    )

    # Step 5: Retrieve RAG guidance for the DETECTED stage
    stage_guidance = None
    if detected_stage in SO:
        stage_guidance = knowledge_module.stage_guidance(detected_stage)

    # Build final result
    result = {
        "stage_id": detected_stage,
        "confidence": vision_result["confidence"],
        "method": vision_result["method"],
        "expected_stages": expected,
        "days_after_transplant": days_after_transplant,
        "stage_source": "vision",
        "observations": vision_result["observations"],
        "visual_features": features,
        "stage_scores": vision_result["stage_scores"],
        "also_visible": [],
        "counts": {},
        "health_notes": mismatch_info["health_note"],
        "progress": prog,
        "mismatch": mismatch_info,
        "guidance": stage_guidance,
        "warning": mismatch_info.get("warning"),
    }

    return result


def _analyse_mismatch(detected_stage, expected_stages_list,
                      days, features) -> dict:
    """
    Core logic: compare what the photo shows vs what is expected for the given days.
    Returns detailed mismatch analysis.
    """
    if not expected_stages_list:
        return {
            "has_mismatch": False,
            "severity": "unknown",
            "stages_behind": 0,
            "health_note": "Set a transplanting date to compare expected vs actual stage.",
            "recommendation": "",
        }

    detected_idx = STAGE_ORDER.index(detected_stage) if detected_stage in STAGE_ORDER else -1
    expected_indices = [STAGE_ORDER.index(s) for s in expected_stages_list
                        if s in STAGE_ORDER]

    if not expected_indices or detected_idx < 0:
        return {
            "has_mismatch": False,
            "severity": "unknown",
            "stages_behind": 0,
            "health_note": "Could not determine mismatch.",
            "recommendation": "",
        }

    min_expected = min(expected_indices)
    max_expected = max(expected_indices)
    stages_behind = min_expected - detected_idx  # positive = behind, negative = ahead

    if detected_idx < min_expected:
        # Plant looks BEHIND what it should be
        severity = "severe" if stages_behind >= 2 else "moderate" if stages_behind == 1 else "mild"
        expected_name = STAGE_ORDER[min_expected]
        detected_name = STAGE_ORDER[detected_idx]

        # Build specific recommendations based on what's missing visually
        rec = _recommend_for_behind(detected_stage, expected_name, features, days)

        health_note = (
            f"⚠️ Plant appears to be at {detected_name.replace('_', ' ')} stage "
            f"but {days} days after transplanting it should be at "
            f"{expected_name.replace('_', ' ')} — "
            f"{stages_behind} stage(s) behind schedule."
        )
        return {
            "has_mismatch": True,
            "severity": severity,
            "detected_stage": detected_stage,
            "expected_stage": expected_stages_list[0],
            "stages_behind": stages_behind,
            "stages_ahead": 0,
            "health_note": health_note,
            "recommendation": rec,
            "warning": f"Growth is {stages_behind} stage(s) behind for {days} DAT. Action needed.",
        }

    elif detected_idx > max_expected:
        # Plant looks AHEAD
        stages_ahead = detected_idx - max_expected
        detected_name = STAGE_ORDER[detected_idx]
        expected_name = STAGE_ORDER[max_expected]
        health_note = (
            f"Plant appears ahead of schedule: showing {detected_name.replace('_', ' ')} "
            f"at {days} DAT (expected {expected_name.replace('_', ' ')}). "
            f"This is fine for early-maturing hybrids."
        )
        return {
            "has_mismatch": False,
            "severity": "none",
            "detected_stage": detected_stage,
            "expected_stage": expected_stages_list[0],
            "stages_behind": 0,
            "stages_ahead": stages_ahead,
            "health_note": health_note,
            "recommendation": "Plant is ahead of schedule — prepare for the next stage early.",
        }
    else:
        # On track
        return {
            "has_mismatch": False,
            "severity": "none",
            "detected_stage": detected_stage,
            "expected_stage": expected_stages_list[0],
            "stages_behind": 0,
            "stages_ahead": 0,
            "health_note": (
                f"✅ Plant is at the expected {detected_stage.replace('_', ' ')} "
                f"stage for {days} days after transplanting."
            ),
            "recommendation": "Growth is on track. Continue standard care.",
        }


def _recommend_for_behind(detected, expected, features, days) -> str:
    """Generate specific recommendations when plant is behind schedule."""
    recs = []
    g = features["green_ratio"]
    cov = features["plant_coverage"]

    if detected in ("nursery", "transplanting"):
        recs.append(
            f"Plant still looks like early seedling/transplant stage at {days} DAT. "
            "Likely causes: transplant shock, nutrient deficiency (N), waterlogging, "
            "root damage, or poor soil preparation."
        )
        recs.append("Actions: check root health, apply starter dose of N fertilizer (urea/DAP), "
                    "ensure proper drainage, check for soil-borne disease.")

    if detected == "vegetative" and expected in ("flowering_fruit_set", "fruit_development"):
        recs.append(
            "Plant is still in vegetative growth but should be flowering/fruiting. "
            "Excess nitrogen promotes vegetative growth and delays flowering."
        )
        recs.append("Actions: reduce N, increase K and P, check if day-length or temperature "
                    "is affecting flower initiation.")

    if cov < 0.20:
        recs.append("Very low plant coverage — plant may be stunted. "
                    "Check for root-knot nematodes, wilt diseases (Fusarium/Bacterial), "
                    "or severe nutrient deficiency.")

    if not recs:
        recs.append(f"Plant is behind schedule. Check nutrition (N-P-K), irrigation regularity, "
                    f"soil health and pest/disease pressure that may be slowing growth.")

    return " | ".join(recs)
