"""Central configuration. Every path can be overridden with an environment variable
(put them in backend/.env). Nothing in this project trains a model: the backend only
LOADS the files you already trained and saved.
"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

# ---- load backend/.env (GROQ_API_KEY etc.) before anything else reads os.environ ----
_env = BACKEND_DIR / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", BACKEND_DIR / "models"))

# ---- Leaf disease model (EfficientNet-B0) ----
# The file name must stay exactly "best_tomato_model.h5" (the file you saved after training).
DISEASE_MODEL_FILENAME = "best_tomato_model.h5"
DISEASE_MODEL_PATH = Path(os.getenv("DISEASE_MODEL_PATH", MODELS_DIR / DISEASE_MODEL_FILENAME))

# How the image is scaled before model.predict():
#   auto      -> raw 0-255 if the model has its own Rescaling/Normalization layer
#                (Keras EfficientNetB0 does), otherwise divide by 255
#   raw       -> always raw 0-255 pixels
#   divide255 -> always x / 255.0
# Run `python check_model.py <your leaf image>` once to see which one gives sensible output.
DISEASE_PREPROCESS = os.getenv("DISEASE_PREPROCESS", "auto")

# ---- Soil suitability model (XGBoost + SHAP) ----
SOIL_MODEL_PATH = Path(os.getenv("SOIL_MODEL_PATH", MODELS_DIR / "xgb_suitability.joblib"))
SOIL_FEATURES_PATH = Path(os.getenv("SOIL_FEATURES_PATH", MODELS_DIR / "feature_order.joblib"))

# ---- RAG knowledge base (your existing tomato_rag folder, with its built chroma_db) ----
RAG_DIR = Path(os.getenv("RAG_DIR", BACKEND_DIR / "tomato_rag"))
SOIL_DIR = BACKEND_DIR / "soil_suitability_complete" / "soil_suitability"

# ---- Vision LLM for whole-plant growth-stage detection (Groq, same key as RAG) ----
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---- Storage + database ----
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BACKEND_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
GRADCAM_DIR = STORAGE_DIR / "gradcam"
for _d in (UPLOAD_DIR, GRADCAM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SQLite by default so it runs with zero setup; set DATABASE_URL for PostgreSQL, e.g.
#   DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/smart_cultivation
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'smart_cultivation.db').as_posix()}")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
