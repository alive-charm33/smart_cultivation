"""Growth-stage detection: combines local image analysis + RAG + optional LLM.

Detection pipeline (in order):
  1. Local vision  — always runs, reads the actual photo (color/texture analysis)
  2. Groq LLM      — enhances if VISION_MODEL is set and Groq key exists
  3. Gemini LLM    — enhances if GEMINI_API_KEY is set
  4. Days fallback — only if no photo analysis is possible at all

The core comparison (photo stage vs expected stage for given days) always runs
using the local vision — it never falsely says "on track" when the photo shows
a seedling at 60 DAT.
"""
import base64
import io
import json
import os
import re

from .. import config
from . import knowledge
from .growth_vision import (
    detect_stage_from_image,
    analyse_growth_with_rag,
    STAGE_ORDER,
)

GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


def _stage_definitions():
    lines = []
    for sid in STAGE_ORDER:
        d = knowledge.stage_doc(sid)
        what = (d["sections"].get("What happens") or "").replace("\n", " ")
        lines.append(
            f"- {sid}: {d['meta'].get('stage_name', sid)} "
            f"(BBCH {d['meta'].get('bbch', '-')}). {what[:200]}"
        )
    return "\n".join(lines)


VISION_PROMPT = """You are an agronomist. Look at this tomato plant photo.
Identify the growth stage. Allowed stage_id values:
{defs}

Reply with JSON only:
{{"stage_id": "...", "confidence": 0.0-1.0, "observations": ["..."],
  "counts": {{"flower_clusters": null, "green_fruits": null, "ripe_fruits": null}},
  "plant_health_notes": "one sentence"}}"""


def _encode_image(image_path):
    img = __import__("PIL.Image", fromlist=["Image"]).Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


# ------------------------------------------------------------------ LLM vision (optional enhancement)
def _detect_llm_groq(image_path):
    vm = config.VISION_MODEL
    if not vm or vm.lower() in ("none", "disabled", ""):
        raise RuntimeError("Groq vision disabled")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("No GROQ_API_KEY")
    from groq import Groq
    b64, mime = _encode_image(image_path)
    resp = Groq().chat.completions.create(
        model=vm, temperature=0.1, max_tokens=400,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT.format(defs=_stage_definitions())},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
    )
    d = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0))
    if d.get("stage_id") not in STAGE_ORDER + ["unknown"]:
        d["stage_id"] = "unknown"
    d["method"] = f"llm/groq ({vm})"
    return d


def _detect_llm_gemini(image_path):
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("No GEMINI_API_KEY")
    from google import genai
    b64, mime = _encode_image(image_path)
    resp = genai.Client(api_key=key).models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            VISION_PROMPT.format(defs=_stage_definitions()),
            {"inline_data": {"mime_type": mime, "data": b64}},
        ],
    )
    d = json.loads(re.search(r"\{.*\}", resp.text, re.S).group(0))
    if d.get("stage_id") not in STAGE_ORDER + ["unknown"]:
        d["stage_id"] = "unknown"
    d["method"] = f"llm/gemini ({GEMINI_VISION_MODEL})"
    return d


def _try_llm_vision(image_path):
    """Try Groq then Gemini. Returns None if both unavailable."""
    try:
        return _detect_llm_groq(image_path)
    except Exception:
        pass
    try:
        return _detect_llm_gemini(image_path)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ public API
def expected_stages(days_after_transplant):
    if days_after_transplant is None:
        return []
    if days_after_transplant < 0:
        return ["nursery"]
    return knowledge.stages_from_days(days_after_transplant)


def analyse(image_path, days_after_transplant=None, use_llm=True):
    """
    Full analysis:
    1. Always run local vision on the photo → get detected stage
    2. Optionally enhance with LLM vision
    3. Compare detected stage vs expected for the given days
    4. Return progress with honest behind/on-track/ahead status
    """
    expected = expected_stages(days_after_transplant)
    result = {
        "expected_stages": expected,
        "days_after_transplant": days_after_transplant,
    }

    # --- Step 1: Local vision (always runs when a photo is provided) ---
    local = None
    try:
        local = detect_stage_from_image(image_path)
    except Exception as e:
        result["warning"] = f"Local image analysis failed: {e}"

    # --- Step 2: Optional LLM enhancement ---
    llm_result = None
    llm_warning = None
    if use_llm:
        llm_result = _try_llm_vision(image_path)
        if llm_result is None:
            llm_warning = (
                "No LLM vision model available (add GEMINI_API_KEY to backend/.env for free "
                "enhanced analysis). Using local image analysis."
            )

    # --- Step 3: Decide final stage ---
    # LLM result wins over local if it's confident and not "unknown"
    if llm_result and llm_result.get("stage_id") not in (None, "unknown"):
        stage_id = llm_result["stage_id"]
        confidence = llm_result.get("confidence", 0.7)
        method = llm_result["method"] + " + local vision"
        observations = llm_result.get("observations", [])
        if local:
            observations += local["observations"]
        counts = llm_result.get("counts", {})
        health_notes = llm_result.get("plant_health_notes", "")
        stage_source = "vision"
        if llm_warning:
            result["info"] = llm_warning

    elif local and local["stage_id"] != "unknown":
        # Local vision result
        stage_id = local["stage_id"]
        confidence = local["confidence"]
        method = local["method"]
        observations = local["observations"]
        counts = {}
        health_notes = ""
        stage_source = "vision"
        if llm_warning:
            result["info"] = llm_warning

    elif expected:
        # No vision at all — fall back to day estimate
        stage_id = expected[0]
        confidence = None
        method = "estimated from days after transplanting (no image analysis)"
        observations = []
        counts = {}
        health_notes = ""
        stage_source = "days"
        result["warning"] = (
            "Could not analyse the image. Stage estimated from days only — "
            "this cannot detect if actual plant is behind or ahead of schedule."
        )

    else:
        stage_id = "unknown"
        confidence = None
        method = "none"
        observations = []
        counts = {}
        health_notes = ""
        stage_source = "none"
        result["warning"] = "No transplanting date set and image analysis failed."

    result.update(
        stage_id=stage_id,
        confidence=confidence,
        method=method,
        observations=observations,
        also_visible=[],
        counts=counts,
        health_notes=health_notes,
        stage_source=stage_source,
    )

    # --- Step 4: Progress comparison (photo vs days) ---
    prog = progress(
        stage_id, expected,
        days=days_after_transplant,
        stage_source=stage_source,
    )
    result["progress"] = prog

    # --- Step 5: Mismatch detail (only for vision-based detection) ---
    if stage_source == "vision" and local:
        from .growth_vision import _analyse_mismatch
        mismatch = _analyse_mismatch(stage_id, expected, days_after_transplant,
                                     local["visual_features"])
        result["mismatch"] = mismatch
        if mismatch.get("has_mismatch") and not result.get("warning"):
            result["warning"] = mismatch.get("warning")

    return result


def progress(current_stage, expected, previous_stage=None,
             previous_days=None, days=None, stage_source="vision"):
    """
    Compare detected stage vs expected stage for given days.
    stage_source="days" → cannot claim on_track (same source).
    stage_source="vision" → real comparison.
    """
    if current_stage not in STAGE_ORDER:
        return {"status": "unknown",
                "message": "Stage could not be determined from this photo."}

    ci = STAGE_ORDER.index(current_stage)
    out = {
        "stage_index": ci,
        "stage_count": len(STAGE_ORDER),
        "percent_of_season": round(100 * ci / (len(STAGE_ORDER) - 1)),
        "stage_source": stage_source,
    }

    if stage_source == "days":
        out.update(
            status="estimated",
            message=(
                f"Stage estimated from {days} days — photo not analysed. "
                "Cannot compare actual plant with expected stage."
            ),
        )
        return out

    # Vision-based: real comparison
    if expected:
        idx = [STAGE_ORDER.index(s) for s in expected if s in STAGE_ORDER]
        if not idx:
            out.update(status="no_reference",
                       message="Could not resolve expected stage.")
            return out

        if ci < min(idx):
            stages_behind = min(idx) - ci
            out.update(
                status="behind",
                message=(
                    f"⚠️ Plant is {stages_behind} stage(s) BEHIND schedule. "
                    f"Photo shows {current_stage.replace('_',' ')} but at {days} DAT "
                    f"it should be {STAGE_ORDER[min(idx)].replace('_',' ')}. "
                    "Check nutrition, irrigation, root health and soil conditions."
                ),
            )
        elif ci > max(idx):
            stages_ahead = ci - max(idx)
            out.update(
                status="ahead",
                message=(
                    f"Plant is {stages_ahead} stage(s) ahead of schedule — "
                    "fine for early hybrids. Prepare for the next stage."
                ),
            )
        else:
            out.update(
                status="on_track",
                message=f"✅ Growth is on track for {days} DAT.",
            )
    else:
        out.update(status="no_reference",
                   message="Set a transplanting date to compare expected vs actual stage.")

    if previous_stage and previous_stage in STAGE_ORDER:
        pi = STAGE_ORDER.index(previous_stage)
        delta = (days - previous_days
                 if days is not None and previous_days is not None else None)
        out["since_last"] = {
            "previous_stage": previous_stage,
            "stages_advanced": ci - pi,
            "days_between": delta,
        }

    return out
