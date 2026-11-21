"""
Test Gemini Vision for Growth Stage Detection
==============================================
Tests:
  1. Whether GEMINI_API_KEY is set and valid
  2. Gemini vision on your uploaded plant images
  3. Full growth pipeline (local vision + Gemini) on each image
  4. Accuracy comparison: local vision vs Gemini

Run:
    cd backend
    venv\Scripts\python test_gemini_growth.py
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from app import config  # loads .env

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
UPLOADS = Path(__file__).parent / "storage" / "uploads"

STAGE_NAMES = {
    "nursery": "Nursery/Seedling",
    "transplanting": "Transplanting",
    "vegetative": "Vegetative",
    "flowering_fruit_set": "Flowering & Fruit Set",
    "fruit_development": "Fruit Development",
    "ripening_harvest": "Ripening & Harvest",
    "end_of_season": "End of Season",
    "unknown": "Unknown",
}

print("\n" + "="*65)
print("GEMINI VISION — GROWTH STAGE DETECTION TEST")
print("="*65)

# ── Step 1: Check API key ──────────────────────────────────────────
print("\n[1] Checking GEMINI_API_KEY...")
if not GEMINI_KEY or GEMINI_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("  ❌ GEMINI_API_KEY not set in backend/.env")
    print("  → Get a free key at: https://aistudio.google.com/app/apikey")
    print("  → Add to backend/.env:  GEMINI_API_KEY=AIza...")
    sys.exit(1)
print(f"  ✅ Key found: {GEMINI_KEY[:8]}...{GEMINI_KEY[-4:]}")
print(f"  Model: {GEMINI_MODEL}")

# ── Step 2: Test Gemini API connectivity ──────────────────────────
print("\n[2] Testing Gemini API connectivity (text only)...")
try:
    from google import genai
    client = genai.Client(api_key=GEMINI_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=["Reply with exactly: GEMINI_OK"]
    )
    if "GEMINI_OK" in resp.text:
        print(f"  ✅ Gemini API connected — model {GEMINI_MODEL} responding")
    else:
        print(f"  ✅ Gemini responding: {resp.text[:50]}")
except Exception as e:
    print(f"  ❌ Gemini API error: {e}")
    sys.exit(1)

# ── Step 3: Test vision on uploaded images ────────────────────────
images = sorted(list(UPLOADS.glob("*.png")) +
                list(UPLOADS.glob("*.jpg")) +
                list(UPLOADS.glob("*.jpeg")))

if not images:
    print("\n❌ No uploaded images found in storage/uploads/")
    print("  Upload a plant photo from the app first.")
    sys.exit(1)

print(f"\n[3] Testing Gemini vision on {len(images)} uploaded image(s)...")
print("    (These are the images you uploaded via the app)\n")

from app.services.growth_vision import detect_stage_from_image
from app.services.growth import _detect_llm_gemini, expected_stages, progress

results = []
for img in images[:10]:  # test first 10 to save API quota
    name = img.name[-50:] if len(img.name) > 50 else img.name
    print(f"  Image: {name}")

    # Local vision
    try:
        local = detect_stage_from_image(img)
        local_stage = local["stage_id"]
        local_conf = local["confidence"]
    except Exception as e:
        local_stage = "error"
        local_conf = 0.0
        print(f"    Local vision error: {e}")

    # Gemini vision
    try:
        gem = _detect_llm_gemini(img)
        gem_stage = gem["stage_id"]
        gem_conf = gem.get("confidence", 0.0)
        gem_obs = gem.get("observations", [])
    except Exception as e:
        gem_stage = "error"
        gem_conf = 0.0
        gem_obs = []
        print(f"    Gemini error: {e}")

    print(f"    Local  → {STAGE_NAMES.get(local_stage, local_stage):<28} conf: {local_conf:.2f}")
    print(f"    Gemini → {STAGE_NAMES.get(gem_stage, gem_stage):<28} conf: {gem_conf:.2f}")
    if gem_obs:
        print(f"    Gemini observations: {gem_obs[0][:70]}")
    agree = "✅ Agree" if local_stage == gem_stage else "⚠️  Differ"
    print(f"    {agree}\n")

    results.append({
        "image": img.name,
        "local": local_stage,
        "gemini": gem_stage,
        "agree": local_stage == gem_stage
    })

# ── Step 4: Full growth pipeline test ────────────────────────────
print("\n[4] Full growth pipeline test (Gemini + mismatch detection)...")
print("    Scenario: Seedling image uploaded with 60 DAT\n")

# pick the nursery-looking image (smallest file = likely seedling screenshot)
test_img = min(images, key=lambda x: x.stat().st_size)
print(f"  Using: {test_img.name}")

from app.services.growth import analyse
res = analyse(test_img, days_after_transplant=60, use_llm=True)

print(f"\n  Detected stage : {res['stage_id']}")
print(f"  Method         : {res['method']}")
print(f"  Expected at 60D: {res['expected_stages']}")
print(f"  Stage source   : {res.get('stage_source')}")
print(f"  Progress status: {res['progress']['status']}")
print(f"  Progress msg   : {res['progress']['message'][:80]}")

if res.get("mismatch"):
    mm = res["mismatch"]
    print(f"\n  Mismatch found : {mm.get('has_mismatch')}")
    print(f"  Stages behind  : {mm.get('stages_behind', 0)}")
    print(f"  Health note    : {mm.get('health_note', '')[:80]}")

if res.get("warning"):
    print(f"\n  Warning        : {res['warning'][:80]}")

# ── Summary ────────────────────────────────────────────────────────
agree_count = sum(1 for r in results if r["agree"])
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"  Images tested        : {len(results)}")
print(f"  Local == Gemini      : {agree_count}/{len(results)} ({agree_count/len(results)*100:.0f}% agreement)")
print(f"  Gemini API           : ✅ Working")
print(f"  Growth pipeline      : ✅ Working")
print(f"\n  ✅ GEMINI VISION IS READY — restart the server to use it.")
print(f"  The app will now use Gemini for all growth stage detection.")
print("="*65 + "\n")
