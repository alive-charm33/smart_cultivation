"""AI Based Smart Cultivation - FastAPI backend.

Run (from the backend folder):
    uvicorn app.main:app --port 8000
API docs: http://localhost:8000/docs

The trained models are only LOADED (once, at startup):
    models/best_tomato_model.h5     leaf disease (EfficientNet-B0)
    models/xgb_suitability.joblib   tomato suitability (XGBoost)
    models/feature_order.joblib
"""
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config
from .auth import (SignupIn, _user_out, create_token, get_current_user,
                   hash_password, require_user, verify_password)
from .db import DiseaseScan, Farm, GrowthRecord, SuitabilityResult, User, get_session, init_db
from .services import disease, growth, knowledge, planning, soil


def _warm_up():
    """Load the saved models in the background so the first scan is fast."""
    for fn in (disease.get_model, soil._load, knowledge.collection):
        try:
            fn()
        except Exception as e:
            print(f"[warm-up] {fn.__module__}.{fn.__name__}: {e}")
    print("[warm-up] done:", {"disease": disease.status()["loaded"], "soil": soil.status()["loaded"],
                              "rag": knowledge.status()["loaded"]})


@asynccontextmanager
async def lifespan(app):
    init_db()
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="AI Based Smart Cultivation API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS + ["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/media", StaticFiles(directory=str(config.STORAGE_DIR)), name="media")

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _media_url(path):
    return "/media/" + Path(path).resolve().relative_to(config.STORAGE_DIR.resolve()).as_posix() if path else None


def _save_upload(file: UploadFile, sub="uploads"):
    """Save with a unique prefix but KEEP the original file name, so every result
    can be traced back to exactly the image the farmer uploaded."""
    original = Path(file.filename or "image.jpg").name
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use JPG or PNG.")
    safe = re.sub(r"[^\w.\- ]", "_", original)
    dest = config.STORAGE_DIR / sub / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}_{safe}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file.file.read())
    return original, dest


def _farm_or_none(s: Session, farm_id):
    if farm_id is None:
        return None
    farm = s.get(Farm, farm_id)
    if not farm:
        raise HTTPException(404, f"Farm {farm_id} not found")
    return farm


def _days_after_transplant(farm, explicit=None):
    if explicit is not None:
        return explicit
    if farm and farm.transplant_date:
        return (date.today() - farm.transplant_date).days
    return None


# ================================================================ health
@app.get("/api/health")
def health():
    import os as _os
    groq_vision = config.VISION_MODEL
    groq_vision_ok = bool(groq_vision and groq_vision.lower() not in ("none", "disabled", ""))
    gemini_key = bool(_os.getenv("GEMINI_API_KEY") and
                      _os.getenv("GEMINI_API_KEY") != "YOUR_GEMINI_API_KEY_HERE")
    vision_available = groq_vision_ok or gemini_key
    vision_provider = ("groq" if groq_vision_ok else "gemini" if gemini_key else "none")
    return {
        "status": "ok",
        "disease_model": disease.status(),
        "soil_model": soil.status(),
        "rag": knowledge.status(),
        "database": config.DATABASE_URL.split("://")[0],
        "vision": {
            "available": vision_available,
            "provider": vision_provider,
            "model": groq_vision if groq_vision_ok else (
                _os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash") if gemini_key else "none"
            ),
        },
    }


# ================================================================ auth
@app.post("/api/auth/signup")
def signup(body: SignupIn, s: Session = Depends(get_session)):
    if s.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username already taken.")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    u = User(username=body.username, hashed_password=hash_password(body.password),
             full_name=body.full_name, mobile=body.mobile, village=body.village,
             district=body.district, state=body.state or "Maharashtra")
    s.add(u)
    s.commit()
    token = create_token(u.id, u.username)
    return {"token": token, "token_type": "bearer", "user": _user_out(u)}


@app.post("/api/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), s: Session = Depends(get_session)):
    u = s.query(User).filter(User.username == form.username).first()
    if not u or not verify_password(form.password, u.hashed_password):
        raise HTTPException(401, "Incorrect username or password.")
    token = create_token(u.id, u.username)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(u)}


@app.post("/api/auth/login/json")
def login_json(body: dict, s: Session = Depends(get_session)):
    """JSON login (for fetch-based clients that don't use form encoding)."""
    u = s.query(User).filter(User.username == body.get("username", "")).first()
    if not u or not verify_password(body.get("password", ""), u.hashed_password):
        raise HTTPException(401, "Incorrect username or password.")
    token = create_token(u.id, u.username)
    return {"token": token, "token_type": "bearer", "user": _user_out(u)}


@app.get("/api/auth/me")
def me(user: User = Depends(require_user)):
    return _user_out(user)


@app.put("/api/auth/profile")
def update_profile(body: dict, user: User = Depends(require_user), s: Session = Depends(get_session)):
    allowed = {"full_name", "mobile", "village", "district", "state"}
    for k, v in body.items():
        if k in allowed:
            setattr(user, k, v)
    s.add(user)
    s.commit()
    return _user_out(user)


@app.post("/api/auth/change-password")
def change_password(body: dict, user: User = Depends(require_user), s: Session = Depends(get_session)):
    if not verify_password(body.get("old_password", ""), user.hashed_password):
        raise HTTPException(400, "Current password is incorrect.")
    if len(body.get("new_password", "")) < 6:
        raise HTTPException(400, "New password must be at least 6 characters.")
    user.hashed_password = hash_password(body["new_password"])
    s.add(user)
    s.commit()
    return {"ok": True}


# ================================================================ farms
class FarmIn(BaseModel):
    name: str
    farmer_name: Optional[str] = None
    village: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    area_acres: float = 1.0
    variety: Optional[str] = None
    transplant_date: Optional[date] = None


def _farm_dict(f: Farm):
    return {"id": f.id, "name": f.name, "farmer_name": f.farmer_name, "village": f.village, "lat": f.lat,
            "lon": f.lon, "area_acres": f.area_acres, "variety": f.variety,
            "transplant_date": f.transplant_date.isoformat() if f.transplant_date else None,
            "days_after_transplant": _days_after_transplant(f)}


@app.get("/api/farms")
def list_farms(s: Session = Depends(get_session),
               user: Optional[User] = Depends(get_current_user)):
    q = select(Farm).order_by(Farm.id)
    if user:
        q = q.where(Farm.user_id == user.id)
    return [_farm_dict(f) for f in s.scalars(q)]


@app.post("/api/farms")
def create_farm(body: FarmIn, s: Session = Depends(get_session),
                user: Optional[User] = Depends(get_current_user)):
    f = Farm(**body.model_dump(), user_id=user.id if user else None)
    s.add(f)
    s.commit()
    return _farm_dict(f)


@app.put("/api/farms/{farm_id}")
def update_farm(farm_id: int, body: FarmIn, s: Session = Depends(get_session)):
    f = _farm_or_none(s, farm_id)
    for k, v in body.model_dump().items():
        setattr(f, k, v)
    s.commit()
    return _farm_dict(f)


@app.delete("/api/farms/{farm_id}")
def delete_farm(farm_id: int, s: Session = Depends(get_session)):
    s.delete(_farm_or_none(s, farm_id))
    s.commit()
    return {"deleted": farm_id}


# ================================================================ suitability (XGBoost + SHAP)
class LocationIn(BaseModel):
    lat: float
    lon: float
    farm_id: Optional[int] = None
    phosphorus_ppm: Optional[float] = None
    potassium_ppm: Optional[float] = None
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None


class SoilIn(BaseModel):
    ph: float
    nitrogen_ppm: float
    phosphorus_ppm: float
    potassium_ppm: float
    organic_carbon_pct: float
    rainfall_mm: float
    temperature_c: float
    humidity_pct: float
    farm_id: Optional[int] = None


def _store_suitability(s, farm_id, res):
    if farm_id is not None:
        _farm_or_none(s, farm_id)
        s.add(SuitabilityResult(farm_id=farm_id, suitable=res["suitable"],
                                probability=res["probability_suitable"], result=res))
        s.commit()
    return res


@app.post("/api/suitability/by-location")
def suitability_by_location(body: LocationIn, s: Session = Depends(get_session)):
    try:
        res = soil.predict_by_location(body.lat, body.lon, body.model_dump())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not fetch soil/climate data (SoilGrids / NASA POWER): {e}. "
                                 f"You can enter soil values manually instead.")
    return _store_suitability(s, body.farm_id, res)


@app.post("/api/suitability/by-soil")
def suitability_by_soil(body: SoilIn, s: Session = Depends(get_session)):
    try:
        res = soil.predict(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    res["data_sources"] = {k: "entered by farmer" for k in res["soil_features_used"]}
    return _store_suitability(s, body.farm_id, res)


# ================================================================ disease (EfficientNet-B0 + Grad-CAM + RAG)
@app.post("/api/disease/predict")
def disease_predict(file: UploadFile = File(...), farm_id: Optional[int] = Form(None),
                    stage: Optional[str] = Form(None), days: Optional[int] = Form(None),
                    question: Optional[str] = Form(None), use_llm: bool = Form(True),
                    s: Session = Depends(get_session)):
    farm = _farm_or_none(s, farm_id)
    original, path = _save_upload(file)
    try:
        top, all_probs, idx, img, x = disease.predict(path)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not read image '{original}': {e}")

    cam_info, overlay_url, heat_url = None, None, None
    try:
        cam = disease.gradcam(x, idx)
        over, heat = disease.overlay(img, cam)
        stem = Path(path).stem
        op, hp = config.GRADCAM_DIR / f"{stem}_gradcam.png", config.GRADCAM_DIR / f"{stem}_heatmap.png"
        over.save(op)
        heat.save(hp)
        overlay_url, heat_url = _media_url(op), _media_url(hp)
        cam_info = disease.describe_cam(cam)
    except Exception as e:
        cam_info = {"error": f"Grad-CAM failed: {e}"}

    stage = stage or None
    if stage and stage not in knowledge.STAGES:
        raise HTTPException(400, f"stage must be one of {knowledge.STAGES}")
    d = _days_after_transplant(farm, days)
    advice = knowledge.advise_disease(top, question, stage, d, use_llm=use_llm)
    doc = knowledge.disease_doc(top[0]["disease_id"])["meta"]

    scan = DiseaseScan(farm_id=farm_id, original_filename=original, image_path=str(path),
                       gradcam_path=str(config.GRADCAM_DIR / f"{Path(path).stem}_gradcam.png") if overlay_url else None,
                       disease_id=top[0]["disease_id"], confidence=top[0]["confidence"],
                       predictions={"top": top, "all": all_probs}, advice=advice["answer"],
                       stage=",".join(advice["stages"]) or None)
    s.add(scan)
    s.commit()
    return {"scan_id": scan.id, "filename": original, "image_url": _media_url(path),
            "prediction": top[0], "top_predictions": top, "all_probabilities": all_probs,
            "disease_info": {"common_name": doc.get("common_name"), "pathogen": doc.get("pathogen"),
                             "category": doc.get("category"), "curable": doc.get("curable")},
            "is_healthy": top[0]["disease_id"] == "healthy",
            "gradcam": {"overlay_url": overlay_url, "heatmap_url": heat_url, **(cam_info or {})},
            "advice": advice, "preprocess": disease.status()["preprocess"]}


@app.get("/api/disease/history")
def disease_history(farm_id: Optional[int] = None, s: Session = Depends(get_session)):
    q = select(DiseaseScan).order_by(DiseaseScan.created_at.desc())
    if farm_id is not None:
        q = q.where(DiseaseScan.farm_id == farm_id)
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "filename": r.original_filename,
             "disease_id": r.disease_id, "confidence": r.confidence, "image_url": _media_url(r.image_path),
             "gradcam_url": _media_url(r.gradcam_path), "stage": r.stage} for r in s.scalars(q.limit(100))]


@app.get("/api/disease/{disease_id}")
def disease_info(disease_id: str):
    d = knowledge.disease_doc(disease_id)
    if not d["meta"]:
        raise HTTPException(404, "Unknown disease")
    return d


# ================================================================ growth stage + progress
@app.post("/api/growth/analyze")
def growth_analyze(file: UploadFile = File(...), farm_id: Optional[int] = Form(None),
                   days: Optional[int] = Form(None), use_llm: bool = Form(True),
                   s: Session = Depends(get_session)):
    farm = _farm_or_none(s, farm_id)
    original, path = _save_upload(file)
    d = _days_after_transplant(farm, days)
    # analyse() now runs local vision on the photo — always compares photo vs days
    res = growth.analyse(path, d, use_llm=use_llm)
    # Add since_last comparison if a previous record exists for this farm
    prev = None
    if farm_id is not None:
        prev = s.scalars(select(GrowthRecord).where(GrowthRecord.farm_id == farm_id)
                         .order_by(GrowthRecord.created_at.desc()).limit(1)).first()
    if prev and prev.stage_id != res["stage_id"]:
        # Update progress with since_last info
        res["progress"] = growth.progress(
            res["stage_id"], res["expected_stages"],
            prev.stage_id, prev.days_after_transplant, d,
            stage_source=res.get("stage_source", "vision"),
        )
    res["guidance"] = knowledge.stage_guidance(res["stage_id"]) if res["stage_id"] in knowledge.STAGES else None
    rec = GrowthRecord(farm_id=farm_id, original_filename=original, image_path=str(path),
                       days_after_transplant=d, stage_id=res["stage_id"], method=res.get("method", ""),
                       details={k: v for k, v in res.items() if k != "guidance"})
    s.add(rec)
    s.commit()
    return {"record_id": rec.id, "filename": original, "image_url": _media_url(path), **res}


@app.get("/api/growth/history")
def growth_history(farm_id: Optional[int] = None, s: Session = Depends(get_session)):
    q = select(GrowthRecord).order_by(GrowthRecord.created_at)
    if farm_id is not None:
        q = q.where(GrowthRecord.farm_id == farm_id)
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "filename": r.original_filename,
             "image_url": _media_url(r.image_path), "days_after_transplant": r.days_after_transplant,
             "stage_id": r.stage_id, "stage_index": knowledge.STAGES.index(r.stage_id)
             if r.stage_id in knowledge.STAGES else None, "method": r.method,
             "status": (r.details.get("progress") or {}).get("status")} for r in s.scalars(q)]


@app.get("/api/growth/stages")
def growth_stages(days: Optional[int] = None):
    table = knowledge.stage_table()
    return {"stages": [{**r, **{"stage_name": knowledge.stage_doc(r["stage_id"])["meta"].get("stage_name")}}
                       for r in table],
            "expected": growth.expected_stages(days) if days is not None else None}


@app.get("/api/growth/stage/{stage_id}")
def stage_info(stage_id: str):
    if stage_id not in knowledge.STAGES:
        raise HTTPException(404, "Unknown stage")
    return knowledge.stage_guidance(stage_id)


# ================================================================ planning
class CostIn(BaseModel):
    area_acres: float = 1.0
    overrides: dict = {}
    yield_t_per_acre: Optional[float] = None
    price_per_kg: Optional[float] = None
    contingency_pct: float = planning.CONTINGENCY_PCT


@app.post("/api/planning/cost")
def planning_cost(body: CostIn):
    return planning.cost_estimate(body.area_acres, body.overrides, body.yield_t_per_acre,
                                  body.price_per_kg, body.contingency_pct)


@app.get("/api/planning/roadmap")
def planning_roadmap(farm_id: Optional[int] = None, transplant_date: Optional[date] = None,
                     s: Session = Depends(get_session)):
    farm = _farm_or_none(s, farm_id)
    td = transplant_date or (farm.transplant_date if farm else None)
    fert = None
    if farm:
        last = s.scalars(select(SuitabilityResult).where(SuitabilityResult.farm_id == farm.id)
                         .order_by(SuitabilityResult.created_at.desc()).limit(1)).first()
        fert = last.result.get("fertility") if last else None
    return {"transplant_date": td.isoformat() if td else None, "steps": planning.roadmap(td, fertility=fert)}


# ================================================================ dashboard
@app.get("/api/dashboard/{farm_id}")
def dashboard(farm_id: int, s: Session = Depends(get_session)):
    farm = _farm_or_none(s, farm_id)
    d = _days_after_transplant(farm)
    suit = s.scalars(select(SuitabilityResult).where(SuitabilityResult.farm_id == farm_id)
                     .order_by(SuitabilityResult.created_at.desc()).limit(1)).first()
    scans = list(s.scalars(select(DiseaseScan).where(DiseaseScan.farm_id == farm_id)
                           .order_by(DiseaseScan.created_at.desc()).limit(5)))
    grow = list(s.scalars(select(GrowthRecord).where(GrowthRecord.farm_id == farm_id)
                          .order_by(GrowthRecord.created_at)))
    expected = growth.expected_stages(d) if d is not None else []
    steps = planning.roadmap(farm.transplant_date, fertility=suit.result.get("fertility") if suit else None)
    today = date.today().isoformat()
    current_step = next((st for st in steps if st.get("start") and st["start"] <= today <= st.get("end", today)), None)
    return {
        "farm": _farm_dict(farm),
        "suitability": ({"suitable": suit.suitable, "probability": suit.probability,
                         "top_reasons": suit.result.get("top_reasons"),
                         "fertility": suit.result.get("fertility"),
                         "created_at": suit.created_at.isoformat()} if suit else None),
        "expected_stages": expected,
        "latest_growth": ({"stage_id": grow[-1].stage_id, "created_at": grow[-1].created_at.isoformat(),
                           "image_url": _media_url(grow[-1].image_path), "method": grow[-1].method,
                           "progress": grow[-1].details.get("progress")} if grow else None),
        "growth_timeline": [{"date": g.created_at.date().isoformat(), "days": g.days_after_transplant,
                             "stage_id": g.stage_id, "stage_index": knowledge.STAGES.index(g.stage_id)
                             if g.stage_id in knowledge.STAGES else None} for g in grow],
        "recent_scans": [{"id": r.id, "created_at": r.created_at.isoformat(), "filename": r.original_filename,
                          "disease_id": r.disease_id, "confidence": r.confidence,
                          "gradcam_url": _media_url(r.gradcam_path)} for r in scans],
        "cost": planning.cost_estimate(farm.area_acres or 1.0),
        "current_tasks": current_step,
    }


# ================================================================ serve the built React app (optional)
# After `npm run build` in frontend/, the whole app is served from http://localhost:8000
_dist = config.BACKEND_DIR.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
