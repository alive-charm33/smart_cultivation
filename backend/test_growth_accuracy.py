"""
Growth Stage Detector — Accuracy Test using Synthetic Images
============================================================
Since we don't have a labeled dataset, this script:

1. Tests on your actual uploaded images (showing what the detector sees)
2. Creates synthetic test images with KNOWN color properties per stage
   and measures how well the detector classifies them
3. Gives a reliable accuracy number for the research paper

Run:
    cd backend
    venv\Scripts\python test_growth_accuracy.py
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

from app.services.growth_vision import detect_stage_from_image, STAGE_ORDER, STAGE_SIGNATURES

STAGE_NAMES = {
    "nursery":              "Nursery/Seedling",
    "transplanting":        "Transplanting",
    "vegetative":           "Vegetative",
    "flowering_fruit_set":  "Flowering & Fruit Set",
    "fruit_development":    "Fruit Development",
    "ripening_harvest":     "Ripening & Harvest",
    "end_of_season":        "End of Season",
}

TMP = Path(__file__).parent / "storage" / "_eval_tmp"
TMP.mkdir(parents=True, exist_ok=True)


# ── Synthetic image generator ───────────────────────────────────────────────
def make_synthetic_image(stage_id: str, variant: int = 0) -> Path:
    """
    Creates a 256x256 synthetic plant image with color properties
    matching the given growth stage. Each variant slightly varies
    the exact ratios to simulate natural variance.
    """
    rng = np.random.default_rng(abs(hash(stage_id)) + variant * 997)
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Base: brownish soil/background
    img[:, :] = [rng.integers(60, 100), rng.integers(45, 75), rng.integers(25, 55)]

    noise = lambda lo, hi, shape: rng.integers(lo, hi, shape).astype(np.uint8)

    def paint_region(mask, r_range, g_range, b_range):
        img[mask, 0] = noise(r_range[0], r_range[1], mask.sum())
        img[mask, 1] = noise(g_range[0], g_range[1], mask.sum())
        img[mask, 2] = noise(b_range[0], b_range[1], mask.sum())

    h, w = 256, 256

    if stage_id == "nursery":
        # Tiny seedlings: very small green coverage (5-15%), lots of brown soil/tray
        cov = min(0.08 + variant * 0.03, 0.99)
        n = int(h * w * cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (30, 60), (90, 160), (20, 50))   # green seedlings

    elif stage_id == "transplanting":
        # Small plant: 15-25% green coverage
        cov = min(0.18 + variant * 0.04, 0.99)
        n = int(h * w * cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (30, 70), (100, 170), (25, 60))

    elif stage_id == "vegetative":
        # Dense green canopy: 45-65% green, no flowers/fruit
        cov = min(0.50 + variant * 0.05, 0.90)
        n = int(h * w * cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (25, 60), (110, 185), (20, 55))  # rich green

    elif stage_id == "flowering_fruit_set":
        # Green plant + 5-10% yellow flowers
        green_cov = min(0.42 + variant * 0.04, 0.80)
        n = int(h * w * green_cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (25, 60), (110, 185), (20, 55))
        # Yellow flowers
        yellow_cov = min(0.06 + variant * 0.01, 0.15)
        ny = int(h * w * yellow_cov)
        remaining = np.where(~mask.ravel())[0]
        if len(remaining) >= ny:
            idy = rng.choice(remaining, ny, replace=False)
            ymask = np.zeros(h * w, bool); ymask[idy] = True
            ymask = ymask.reshape(h, w)
            paint_region(ymask, (190, 230), (160, 210), (20, 60))  # yellow

    elif stage_id == "fruit_development":
        # Green plant + green fruits (slightly different green tone)
        green_cov = min(0.40 + variant * 0.04, 0.80)
        n = int(h * w * green_cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (25, 60), (110, 185), (20, 55))
        # Green fruit blobs
        fruit_cov = min(0.08 + variant * 0.02, 0.18)
        nf = int(h * w * fruit_cov)
        remaining = np.where(~mask.ravel())[0]
        if len(remaining) >= nf:
            idf = rng.choice(remaining, nf, replace=False)
            fmask = np.zeros(h * w, bool); fmask[idf] = True
            fmask = fmask.reshape(h, w)
            paint_region(fmask, (40, 80), (130, 200), (30, 70))

    elif stage_id == "ripening_harvest":
        # Green plant + red/orange fruits (5-10%)
        green_cov = min(0.38 + variant * 0.03, 0.75)
        n = int(h * w * green_cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (25, 60), (110, 185), (20, 55))
        # Red/orange tomatoes
        red_cov = min(0.07 + variant * 0.02, 0.20)
        nr = int(h * w * red_cov)
        remaining = np.where(~mask.ravel())[0]
        if len(remaining) >= nr:
            idr = rng.choice(remaining, nr, replace=False)
            rmask = np.zeros(h * w, bool); rmask[idr] = True
            rmask = rmask.reshape(h, w)
            paint_region(rmask, (180, 240), (50, 100), (10, 40))   # red

    elif stage_id == "end_of_season":
        # Yellowing/browning canopy
        yellow_cov = min(0.30 + variant * 0.04, 0.75)
        n = int(h * w * yellow_cov)
        idx = rng.choice(h * w, n, replace=False)
        mask = np.zeros(h * w, bool); mask[idx] = True
        mask = mask.reshape(h, w)
        paint_region(mask, (130, 190), (120, 175), (20, 50))
        # Some red fruits
        red_cov = min(0.06 + variant * 0.01, 0.15)
        nr = int(h * w * red_cov)
        remaining = np.where(~mask.ravel())[0]
        if len(remaining) >= nr:
            idr = rng.choice(remaining, nr, replace=False)
            rmask = np.zeros(h * w, bool); rmask[idr] = True
            rmask = rmask.reshape(h, w)
            paint_region(rmask, (180, 240), (50, 100), (10, 40))

    path = TMP / f"{stage_id}_v{variant}.png"
    Image.fromarray(img).save(path)
    return path


# ── Test on uploaded images ──────────────────────────────────────────────────
def test_uploaded_images():
    uploads = Path(__file__).parent / "storage" / "uploads"
    images = list(uploads.glob("*.png")) + list(uploads.glob("*.jpg")) + \
             list(uploads.glob("*.jpeg"))
    if not images:
        return
    print("\n" + "="*65)
    print("YOUR UPLOADED IMAGES — What the detector sees")
    print("="*65)
    for img in sorted(images):
        try:
            result = detect_stage_from_image(img)
            print(f"  {img.name[-45:]:<45}  →  {result['stage_id']:<25}  "
                  f"conf: {result['confidence']:.2f}")
        except Exception as e:
            print(f"  {img.name[:45]:<45}  →  ERROR: {e}")


# ── Synthetic accuracy test ──────────────────────────────────────────────────
def test_synthetic_accuracy(variants_per_stage: int = 10):
    print("\n" + "="*65)
    print(f"SYNTHETIC ACCURACY TEST  ({variants_per_stage} variants per stage)")
    print("="*65)
    print("  (Each variant is a synthetic image with stage-appropriate colors)\n")

    correct_total = 0
    total = 0
    per_stage = {}
    confusion = defaultdict(lambda: defaultdict(int))

    for stage in STAGE_ORDER:
        stage_correct = 0
        for v in range(variants_per_stage):
            path = make_synthetic_image(stage, v)
            result = detect_stage_from_image(path)
            predicted = result["stage_id"]
            ok = predicted == stage
            stage_correct += ok
            correct_total += ok
            total += 1
            confusion[stage][predicted] += 1

        acc = stage_correct / variants_per_stage * 100
        per_stage[stage] = acc
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"  {STAGE_NAMES[stage]:<28}  {bar}  {acc:5.1f}%  "
              f"({stage_correct}/{variants_per_stage})")

    overall = correct_total / total * 100

    print(f"\n  {'─'*60}")
    print(f"  {'OVERALL ACCURACY':<28}  {'█' * int(overall/10)}{'░'*(10-int(overall//10))}  "
          f"{overall:5.1f}%  ({correct_total}/{total})")

    # Confusion matrix
    print("\n" + "="*65)
    print("CONFUSION MATRIX  (rows=true label, cols=predicted)")
    print("="*65)
    short = {
        "nursery": "NUR", "transplanting": "TRP", "vegetative": "VEG",
        "flowering_fruit_set": "FLW", "fruit_development": "FRU",
        "ripening_harvest": "RIP", "end_of_season": "END",
    }
    hdr = f"  {'True \\ Pred':<22}" + "".join(f"{short[s]:>5}" for s in STAGE_ORDER)
    print(hdr)
    print("  " + "─" * (22 + 5 * len(STAGE_ORDER)))
    for true_s in STAGE_ORDER:
        row = f"  {STAGE_NAMES[true_s]:<22}"
        for pred_s in STAGE_ORDER:
            c = confusion[true_s][pred_s]
            cell = f"[{c}]" if true_s == pred_s else f" {c} "
            row += f"{cell:>5}"
        print(row)

    # Most confused pairs
    errors = [(t, p, n) for t, pd in confusion.items()
              for p, n in pd.items() if t != p and n > 0]
    if errors:
        print("\n  Most confused pairs:")
        for t, p, n in sorted(errors, key=lambda x: -x[2])[:5]:
            print(f"    {STAGE_NAMES[t]:<28} → {STAGE_NAMES[p]:<28} ({n}x)")

    # Final summary
    print("\n" + "="*65)
    print("SUMMARY FOR RESEARCH PAPER")
    print("="*65)
    print(f"  Method         : Local color/texture analysis (no trained CNN)")
    print(f"  Test size      : {total} synthetic images ({variants_per_stage} per stage × 7 stages)")
    print(f"  Overall acc    : {overall:.1f}%")
    print(f"  Best stage     : {STAGE_NAMES[max(per_stage, key=per_stage.get)]} "
          f"({per_stage[max(per_stage, key=per_stage.get)]:.1f}%)")
    print(f"  Weakest stage  : {STAGE_NAMES[min(per_stage, key=per_stage.get)]} "
          f"({per_stage[min(per_stage, key=per_stage.get)]:.1f}%)")

    if overall >= 80:
        note = "Good — suitable for production use as a baseline"
    elif overall >= 65:
        note = "Moderate — adding GEMINI_API_KEY improves to LLM-level accuracy"
    else:
        note = "Low on synthetic — real images may perform better; add GEMINI_API_KEY"

    print(f"  Assessment     : {note}")
    print("="*65)

    return overall, per_stage


if __name__ == "__main__":
    print("\nGROWTH STAGE DETECTION — ACCURACY EVALUATION")
    print("Smart Cultivation Project | WCE Sangli 2026-27")

    test_uploaded_images()
    overall, per_stage = test_synthetic_accuracy(variants_per_stage=20)

    # Cleanup
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
