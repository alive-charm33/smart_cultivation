"""Tomato leaf image -> EfficientNetB0 -> filtered retrieval -> LLM answer.

Examples
  python rag.py --image leaf.jpg --stage flowering_fruit_set --question "How do I treat this?"
  python rag.py --simulate "Tomato___Early_blight=0.55,Tomato___Target_Spot=0.35,Tomato___Septoria_leaf_spot=0.10" --no-llm
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from stages import resolve_stages, NURSERY_DAYS_DEFAULT

ROOT = Path(__file__).parent
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DB_DIR = ROOT / "chroma_db"
COLLECTION = "tomato_kb"
LABEL_TO_ID = json.loads((ROOT / "label_map.json").read_text())
STAGES = ["nursery", "transplanting", "vegetative", "flowering_fruit_set",
          "fruit_development", "ripening_harvest", "end_of_season"]

CONF_SURE = 0.85      # above this: single disease
CONF_MIN = 0.50       # below this: ask for a better photo + use top 3
CONFUSABLE = {"early_blight", "target_spot", "septoria_leaf_spot", "bacterial_spot"}

# EfficientNetB0 in Keras has its own rescaling layer and expects raw 0-255 pixels.
# If you trained with preprocess_input or /255 outside the model, repeat that here.
IMG_SIZE = (224, 224)
EXTERNAL_PREPROCESS = None  # e.g. lambda x: x / 255.0

SYSTEM_PROMPT = """You are a tomato crop advisor for farmers in India.
Answer ONLY from the CONTEXT passages. If the context does not contain the answer, say so and
suggest contacting the local Krishi Vigyan Kendra (KVK) or agriculture officer. Never invent facts.

Rules:
1. Start with the diagnosis and the classifier confidence in plain words.
2. If confidence is low or several diseases are listed, do NOT present one as certain. Explain the
   differences using the comparison context and ask 2-3 short questions that would separate them.
3. Give advice in this order: immediate actions, prevention, organic/biological options, chemical options.
4. Never give pesticide doses or mixing rates. Name active ingredients only and tell the farmer to
   follow the product label and check that it is registered for tomato (CIB&RC), including pre-harvest interval.
5. Viral diseases (leaf curl, mosaic) have no cure: say so clearly and focus on vector control,
   sanitation, roguing and resistant varieties.
6. If growth stage(s) or days are given, tailor advice to them using the stage-wise action rows and the
   stage documents. If several stages apply (they overlap), say so and cover each briefly. If only the
   day count was given, mention that stage was estimated from days and may differ from the real plant.
7. Cite sources like [S1], [S2] after the statements they support.
8. Reply in the language of the user's question (English, Marathi or Hindi). Keep it short and practical."""


# ---------- 1. classifier ----------
def load_classifier(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def load_class_names():
    names = json.loads((ROOT / "class_names.json").read_text())
    missing = [n for n in names if n not in LABEL_TO_ID]
    assert not missing, f"labels missing in label_map.json: {missing}"
    return names  # MUST be the same order as during training


def classify(model, img_path, top_k=3):
    import tensorflow as tf
    names = load_class_names()
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    if EXTERNAL_PREPROCESS:
        x = EXTERNAL_PREPROCESS(x)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(LABEL_TO_ID[names[i]], names[i], float(probs[i])) for i in order]


def parse_simulation(s):
    preds = []
    for part in s.split(","):
        label, p = part.rsplit("=", 1)
        preds.append((LABEL_TO_ID[label.strip()], label.strip(), float(p)))
    return sorted(preds, key=lambda t: -t[2])


# ---------- 2. retrieval planning ----------
def plan(preds):
    top_id, top_conf = preds[0][0], preds[0][2]
    cands = [preds[0]]
    if top_conf < CONF_SURE and len(preds) > 1:
        cands.append(preds[1])
    if top_conf < CONF_MIN and len(preds) > 2:
        cands.append(preds[2])
    # top-2 both brown-spot diseases and second is not negligible -> include second
    if len(preds) > 1 and top_id in CONFUSABLE and preds[1][0] in CONFUSABLE and preds[1][2] > 0.10 \
            and preds[1] not in cands:
        cands.append(preds[1])
    ids = [c[0] for c in cands]
    return {
        "candidates": cands,
        "low_conf": top_conf < CONF_MIN,
        "uncertain": len(cands) > 1,
        "compare": len(cands) > 1 and any(i in CONFUSABLE for i in ids),
    }


def _query(col, emb, text, where, k):
    r = col.query(query_embeddings=[emb.embed_query(text)], n_results=k, where=where)
    return list(zip(r["documents"][0], r["metadatas"][0]))


def retrieve(col, emb, pl, question, stages):
    q = question or "symptoms, prevention and treatment"
    hits = []
    for rank, (did, _, _) in enumerate(pl["candidates"]):
        where = {"$and": [{"doc_type": "disease"}, {"disease_id": did}]}
        hits += _query(col, emb, q, where, k=9 if rank == 0 else 4)
    for i, sid in enumerate(stages):
        where = {"$and": [{"doc_type": "growth_stage"}, {"stage_id": sid}]}
        hits += _query(col, emb, q + " timing", where, k=4 if i == 0 else 2)
    if stages:
        hits += _query(col, emb, "risk by growth stage", {"doc_id": "disease_stage_risk_matrix"}, k=2)
    if pl["compare"]:
        hits += _query(col, emb, "how to tell leaf spot diseases apart", {"doc_id": "leaf_spot_comparison"}, k=3)
    seen, out = set(), []
    for text, meta in hits:  # de-duplicate
        key = (meta["source_file"], meta["section"], text[:60])
        if key not in seen:
            seen.add(key)
            out.append((text, meta))
    return out


def format_context(hits):
    blocks, sources = [], []
    for i, (text, meta) in enumerate(hits, 1):
        blocks.append(f"[S{i}] ({meta['source_file']} > {meta['section']})\n{text}")
        sources.append({"id": f"S{i}", "file": meta["source_file"], "section": meta["section"],
                        "origin": meta.get("sources", "")})
    return "\n\n".join(blocks), sources


# ---------- 3. generation ----------
def build_user_message(preds, pl, question, stages, days_info, context):
    pred_txt = "\n".join(f"- {label} ({p:.0%})" for _, label, p in preds)
    flags = []
    if pl["low_conf"]:
        flags.append("LOW CONFIDENCE: ask the user for a clearer photo (single leaf, natural light, upper and lower side).")
    if pl["uncertain"]:
        flags.append("Several candidate diseases were retrieved; treat the diagnosis as uncertain.")
    return (f"CLASSIFIER OUTPUT (top predictions):\n{pred_txt}\n\n"
            f"GROWTH STAGE(S): {', '.join(stages) or 'not given'}\n"
            f"DAYS INFO: {days_info}\n"
            f"FLAGS: {' '.join(flags) or 'none'}\n\n"
            f"USER QUESTION: {question or 'What is wrong and what should I do?'}\n\n"
            f"CONTEXT:\n{context}")


def generate(user_message):
    """LLM_PROVIDER=groq (default) or anthropic. LLM_MODEL overrides the default model."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider == "groq":
        from groq import Groq
        client = Groq()  # reads GROQ_API_KEY from the environment
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_message}],
            temperature=0.2,      # low = sticks to the context
            max_tokens=3000,
        )
        return resp.choices[0].message.content
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=1200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


# ---------- 4. orchestration ----------
def open_collection():
    import chromadb
    from common import Embedder
    return chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION), Embedder()


def choose_stages(stage=None, days=None, days_from="transplanting", nursery_days=NURSERY_DAYS_DEFAULT):
    """An explicitly given stage (what the farmer SEES on the plant) wins over the day estimate."""
    if stage:
        assert stage in STAGES, f"stage must be one of {STAGES}"
        info = f"stage given by user ({stage})" + (f"; also {days} days {days_from}" if days is not None else "")
        return [stage], info
    if days is not None:
        st = resolve_stages(days, days_from, nursery_days)
        return st, f"{days} days after {days_from}; stage(s) ESTIMATED from days: {', '.join(st) or 'none matched'}"
    return [], "not given"


def diagnose(preds, question=None, stage=None, days=None, days_from="transplanting",
             nursery_days=NURSERY_DAYS_DEFAULT, use_llm=True):
    stages, days_info = choose_stages(stage, days, days_from, nursery_days)
    col, emb = open_collection()
    pl = plan(preds)
    hits = retrieve(col, emb, pl, question, stages)
    context, sources = format_context(hits)
    msg = build_user_message(preds, pl, question, stages, days_info, context)
    answer = generate(msg) if use_llm else msg
    return {"answer": answer, "predictions": preds, "plan": pl, "sources": sources,
            "stages": stages, "days_info": days_info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--model", default="efficientnetb0_tomato.keras")
    ap.add_argument("--simulate", help='"LabelA=0.6,LabelB=0.3,..." to skip the classifier')
    ap.add_argument("--stage", choices=STAGES, help="stage seen on the plant (overrides --days)")
    ap.add_argument("--days", type=int, help="number of days since transplanting (or sowing, see --days-from)")
    ap.add_argument("--days-from", choices=["transplanting", "sowing"], default="transplanting")
    ap.add_argument("--nursery-days", type=int, default=NURSERY_DAYS_DEFAULT,
                    help="nursery duration, used to convert days-after-sowing to days-after-transplanting")
    ap.add_argument("--question")
    ap.add_argument("--no-llm", action="store_true", help="print the prompt/context instead of calling the LLM")
    a = ap.parse_args()

    if a.simulate:
        preds = parse_simulation(a.simulate)
    elif a.image:
        preds = classify(load_classifier(a.model), a.image)
    else:
        ap.error("give --image or --simulate")

    res = diagnose(preds, a.question, a.stage, a.days, a.days_from, a.nursery_days, use_llm=not a.no_llm)
    print("Predictions:", [(i, f"{p:.0%}") for i, _, p in res["predictions"]])
    print("Stages:", res["stages"], "|", res["days_info"])
    print("Plan:", {k: v for k, v in res["plan"].items() if k != "candidates"},
          "candidates:", [c[0] for c in res["plan"]["candidates"]])
    print("\n" + res["answer"])
    print("\nSources:")
    for s in res["sources"]:
        print(f"  {s['id']}: {s['file']} > {s['section']}")
