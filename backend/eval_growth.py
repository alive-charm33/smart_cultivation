"""
Growth Stage Detection — Accuracy Evaluation Script
=====================================================
Tests the local vision detector against images you label manually.

HOW TO USE:
1. Create folders under eval_images/ named after each stage:
      eval_images/nursery/          ← put nursery/seedling photos here
      eval_images/transplanting/    ← recently transplanted plant photos
      eval_images/vegetative/       ← bushy green plant, no flowers
      eval_images/flowering_fruit_set/ ← yellow flowers visible
      eval_images/fruit_development/   ← green fruits visible
      eval_images/ripening_harvest/    ← red/orange fruits visible
      eval_images/end_of_season/       ← yellowing/dying plant

2. Put at least 3-5 images per stage (more = better accuracy estimate)
   Use real tomato plant photos from Google Images or your own field photos.

3. Run:
      cd backend
      venv\Scripts\python eval_growth.py

It prints per-stage accuracy, overall accuracy, and a confusion matrix.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from app.services.growth_vision import detect_stage_from_image, STAGE_ORDER

EVAL_DIR = Path(__file__).parent / "eval_images"

STAGE_NAMES = {
    "nursery":              "Nursery / Seedling",
    "transplanting":        "Transplanting",
    "vegetative":           "Vegetative",
    "flowering_fruit_set":  "Flowering & Fruit Set",
    "fruit_development":    "Fruit Development",
    "ripening_harvest":     "Ripening & Harvest",
    "end_of_season":        "End of Season",
}


def evaluate():
    if not EVAL_DIR.exists():
        EVAL_DIR.mkdir(parents=True)
        print(f"\nCreated: {EVAL_DIR}")
        print("\nPlease add test images:")
        for s in STAGE_ORDER:
            folder = EVAL_DIR / s
            folder.mkdir(exist_ok=True)
            print(f"  {folder}  ← put {STAGE_NAMES[s]} photos here")
        print("\nThen run this script again.")
        return

    # Collect all labeled images
    all_results = []
    per_stage_total  = defaultdict(int)
    per_stage_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][predicted]

    print("\n" + "="*65)
    print("GROWTH STAGE DETECTION — ACCURACY EVALUATION")
    print("="*65)

    total_images = 0
    stages_with_images = []

    for true_stage in STAGE_ORDER:
        folder = EVAL_DIR / true_stage
        if not folder.exists():
            continue
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        if not images:
            print(f"\n  [{true_stage}] — no images found, skipping")
            continue

        stages_with_images.append(true_stage)
        print(f"\n  [{STAGE_NAMES[true_stage]}]  ({len(images)} images)")

        for img_path in images:
            try:
                result = detect_stage_from_image(img_path)
                predicted = result["stage_id"]
                confidence = result["confidence"]
                correct = (predicted == true_stage)

                per_stage_total[true_stage] += 1
                if correct:
                    per_stage_correct[true_stage] += 1
                confusion[true_stage][predicted] += 1
                total_images += 1

                status = "✅" if correct else "❌"
                print(f"    {status}  {img_path.name[:40]:<40}  "
                      f"predicted: {predicted:<25}  conf: {confidence:.2f}")

                all_results.append({
                    "file": img_path.name,
                    "true": true_stage,
                    "predicted": predicted,
                    "correct": correct,
                    "confidence": confidence,
                })

            except Exception as e:
                print(f"    ⚠️  {img_path.name} — error: {e}")

    if total_images == 0:
        print("\nNo images found in any stage folder.")
        print(f"Add images to: {EVAL_DIR}")
        return

    # ── Per-stage accuracy ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PER-STAGE ACCURACY")
    print("="*65)
    print(f"  {'Stage':<30} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*10}")

    total_correct = 0
    for stage in STAGE_ORDER:
        if stage not in per_stage_total:
            continue
        n = per_stage_total[stage]
        c = per_stage_correct[stage]
        acc = c / n * 100
        total_correct += c
        bar = "█" * int(acc / 10)
        print(f"  {STAGE_NAMES[stage]:<30} {c:>8} {n:>7} {acc:>9.1f}%  {bar}")

    # ── Overall accuracy ────────────────────────────────────────────────────
    overall = total_correct / total_images * 100
    print(f"\n  {'OVERALL':<30} {total_correct:>8} {total_images:>7} {overall:>9.1f}%")

    # ── Confusion matrix ────────────────────────────────────────────────────
    if len(stages_with_images) > 1:
        print("\n" + "="*65)
        print("CONFUSION MATRIX  (rows = true, cols = predicted)")
        print("="*65)
        # Short labels
        short = {
            "nursery": "NUR", "transplanting": "TRP", "vegetative": "VEG",
            "flowering_fruit_set": "FLW", "fruit_development": "FRU",
            "ripening_harvest": "RIP", "end_of_season": "END",
        }
        active = [s for s in STAGE_ORDER if s in per_stage_total]
        header = "  " + f"{'True \\ Pred':<20}" + "".join(f"{short[s]:>5}" for s in active)
        print(header)
        print("  " + "-" * (20 + 5 * len(active)))
        for true_s in active:
            row = f"  {STAGE_NAMES[true_s]:<20}"
            for pred_s in active:
                count = confusion[true_s][pred_s]
                marker = f"[{count}]" if true_s == pred_s and count > 0 else f" {count} "
                row += f"{marker:>5}"
            print(row)

    # ── Common errors ───────────────────────────────────────────────────────
    errors = [r for r in all_results if not r["correct"]]
    if errors:
        print("\n" + "="*65)
        print("COMMON MISCLASSIFICATIONS")
        print("="*65)
        error_pairs = defaultdict(int)
        for e in errors:
            error_pairs[(e["true"], e["predicted"])] += 1
        for (t, p), count in sorted(error_pairs.items(), key=lambda x: -x[1]):
            print(f"  {STAGE_NAMES[t]:<28}  →  {STAGE_NAMES[p]:<28}  ({count}x)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"  Total images tested : {total_images}")
    print(f"  Correctly detected  : {total_correct}")
    print(f"  Overall accuracy    : {overall:.1f}%")

    if overall >= 80:
        print(f"  Assessment          : Good — suitable for production use")
    elif overall >= 65:
        print(f"  Assessment          : Moderate — works for most stages, "
              f"consider adding LLM vision for better accuracy")
    else:
        print(f"  Assessment          : Low — add GEMINI_API_KEY to .env for "
              f"significantly better accuracy via Gemini vision model")

    print("\n  Note: Local vision uses color/texture analysis (no trained model).")
    print("  Adding GEMINI_API_KEY to backend/.env gives LLM-level accuracy.")
    print("="*65 + "\n")


if __name__ == "__main__":
    evaluate()
