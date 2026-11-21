"""RAG layer: maps the predicted disease to its cure (and growth stage to its care),
re-using your tomato_rag pipeline (plan -> filtered Chroma retrieval -> Groq LLM).

The Chroma index and the embedding model are opened ONCE and cached.
If the LLM is unreachable (no key / no internet), a structured answer is built
directly from the knowledge-base markdown so the app still works offline.
"""
import re
import sys
import threading

import yaml

from .. import config

if str(config.RAG_DIR) not in sys.path:
    sys.path.insert(0, str(config.RAG_DIR))

import rag  # noqa: E402  (your tomato_rag/rag.py)
import stages as stage_lib  # noqa: E402  (your tomato_rag/stages.py)

STAGES = rag.STAGES
_lock = threading.Lock()
_cache = {"col": None, "emb": None, "error": None}


def collection():
    if _cache["col"] is None:
        with _lock:
            if _cache["col"] is None:
                try:
                    _cache["col"], _cache["emb"] = rag.open_collection()
                    _cache["error"] = None
                except Exception as e:  # e.g. embedding model not downloaded yet
                    _cache["error"] = str(e)
                    raise
    return _cache["col"], _cache["emb"]


def status():
    return {"chroma_db": (config.RAG_DIR / "chroma_db").exists(),
            "loaded": _cache["col"] is not None, "error": _cache["error"],
            "llm_key_set": bool(__import__("os").getenv("GROQ_API_KEY")
                                or __import__("os").getenv("ANTHROPIC_API_KEY"))}


# ------------------------------------------------------------ markdown helpers
def _read_doc(sub, doc_id):
    path = config.RAG_DIR / "docs" / sub / f"{doc_id}.md"
    if not path.exists():
        return {}, {}
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(encoding="utf-8"), re.S)
    meta, body = yaml.safe_load(m.group(1)), m.group(2)
    parts = re.split(r"^## +(.+)$", body, flags=re.M)
    sections = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}
    return meta, sections


def disease_doc(disease_id):
    meta, sections = _read_doc("diseases", disease_id)
    return {"meta": meta, "sections": sections}


def stage_doc(stage_id):
    meta, sections = _read_doc("stages", stage_id)
    return {"meta": meta, "sections": sections}


def stage_table():
    return stage_lib.load_stage_table()


def stages_from_days(days, days_from="transplanting", nursery_days=stage_lib.NURSERY_DAYS_DEFAULT):
    return stage_lib.resolve_stages(days, days_from, nursery_days)


STAGE_ROW_NAMES = {"nursery": "nursery", "transplanting": "transplanting", "vegetative": "vegetative",
                   "flowering_fruit_set": "flowering/fruit set", "fruit_development": "fruit development",
                   "ripening_harvest": "ripening/harvest", "end_of_season": "end of season"}


def _stage_row(sections, stage_id):
    """Pick the row of the disease's 'Stage-wise action' table for this stage."""
    want = STAGE_ROW_NAMES.get(stage_id, stage_id)
    for line in sections.get("Stage-wise action", "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].lower() == want:
            return {"stage": cells[0], "risk": cells[1], "action": cells[2]}
    return None


def offline_answer(preds, stages):
    """Answer built straight from the KB files (used when the LLM is unavailable)."""
    top = preds[0]
    doc = disease_doc(top[0])
    s, meta = doc["sections"], doc["meta"]
    out = [f"### {meta.get('common_name', top[0])} ({top[2]:.0%} confidence)"]
    if meta.get("pathogen"):
        out.append(f"*Cause:* {meta['pathogen']}  \n*Curable:* {meta.get('curable', '-')}")
    if top[2] < rag.CONF_MIN:
        out.append("> Low confidence - please retake the photo: one leaf, natural light, "
                   "show the spots clearly.")
    if len(preds) > 1 and top[2] < rag.CONF_SURE:
        out.append(f"> Also possible: {preds[1][1].replace('Tomato___', '').replace('_', ' ')} "
                   f"({preds[1][2]:.0%}). Compare the symptoms before acting.")
    for st in stages:
        row = _stage_row(s, st)
        if row:
            out.append(f"**At your growth stage ({row['stage']}) - risk {row['risk']}:** {row['action']}")
    for title, body in s.items():
        if title == "Stage-wise action":
            continue
        out.append(f"**{title}**\n{body}")
    out.append("_Doses are not given on purpose: follow the product label and check CIB&RC "
               "registration for tomato. Confirm with your local KVK / agriculture officer._")
    return "\n\n".join(out)


def advise_disease(top_preds, question=None, stage=None, days=None, days_from="transplanting",
                   use_llm=True):
    """top_preds: list of dicts from disease.predict(). Returns answer + sources + plan."""
    preds = [(p["disease_id"], p["label"], p["confidence"]) for p in top_preds]
    stages, days_info = rag.choose_stages(stage, days, days_from)
    result = {"stages": stages, "days_info": days_info, "sources": [], "mode": "offline"}
    pl = rag.plan(preds)
    result["plan"] = {k: v for k, v in pl.items() if k != "candidates"}
    result["plan"]["candidates"] = [c[0] for c in pl["candidates"]]
    try:
        col, emb = collection()
        hits = rag.retrieve(col, emb, pl, question, stages)
        context, sources = rag.format_context(hits)
        result["sources"] = sources
        if use_llm:
            msg = rag.build_user_message(preds, pl, question, stages, days_info, context)
            result["answer"] = rag.generate(msg)
            result["mode"] = "llm"
            return result
    except Exception as e:
        result["warning"] = f"LLM/retrieval unavailable, showing knowledge-base answer ({e.__class__.__name__}: {e})"
    result["answer"] = offline_answer(preds, stages)
    return result


def stage_guidance(stage_id):
    d = stage_doc(stage_id)
    s = d["sections"]
    return {"stage_id": stage_id, "stage_name": d["meta"].get("stage_name", stage_id),
            "bbch": d["meta"].get("bbch"), "timing": s.get("Timing"),
            "what_happens": s.get("What happens"),
            "water_nutrients": s.get("Water and nutrient needs"),
            "key_management": s.get("Key management"),
            "disease_risks": s.get("Disease and pest risks at this stage"),
            "diseases_at_risk": d["meta"].get("diseases_at_risk", [])}
