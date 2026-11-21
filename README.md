# AI Based Smart Cultivation: Tomato Farming Assistant

Full-stack implementation of the project synopsis (WCE Sangli, 2026-27):
**React frontend + FastAPI backend + your two trained models + RAG + database**.

> **No model is trained here.** The backend only *loads* the files you already trained:
> `best_tomato_model.h5` (leaf disease) and `xgb_suitability.joblib` + `feature_order.joblib` (soil).
> They are loaded **once** when the server starts and kept in memory, so you never wait the
> ~1.5 hours of retraining again.

## How the synopsis maps to the code

| Synopsis step | What runs | Where |
|---|---|---|
| 1-2 Enter location, soil data | GPS / lat-lon → ISRIC SoilGrids (pH, OC, N) + NASA POWER (temp, humidity, rain) | `soil_suitability/soil_client.py`, `weather_client.py` |
| 3 Suitability + SHAP | your XGBoost model + `shap.TreeExplainer`, plus soil-fertility bands | `backend/app/services/soil.py` |
| 4 Planning | cost estimate (editable rates, profit, break-even) + dated stage roadmap | `backend/app/services/planning.py` |
| 5-6 Crop monitoring, growth stage | whole-plant photo → Groq vision LLM → stage; compared with expected stage for the plant's age and with the previous photo | `backend/app/services/growth.py` |
| 7 Disease detection | `best_tomato_model.h5` (EfficientNet-B0), top-3 with confidence | `backend/app/services/disease.py` |
| 8 Disease explanation | **Grad-CAM** heat-map + overlay (works for flat and nested EfficientNet models) | `backend/app/services/disease.py` |
| RAG cure mapping | your `tomato_rag` pipeline: disease → filtered Chroma retrieval → Groq LLM, tailored to the growth stage; falls back to the knowledge-base text when offline | `backend/app/services/knowledge.py` + `tomato_rag/` |
| 9 Dashboard | suitability, stage + progress timeline, latest scans, cost, this week's tasks | `frontend/src/pages/Dashboard.jsx` |
| Database | SQLite by default, PostgreSQL via `DATABASE_URL` | `backend/app/db.py` |

## Folder layout

```
smart_cultivation/
├─ backend/
│  ├─ models/                 ← put best_tomato_model.h5, xgb_suitability.joblib, feature_order.joblib here
│  ├─ app/                    ← FastAPI app (main.py, config.py, db.py, services/)
│  ├─ tomato_rag/             ← your RAG folder, unchanged (docs/, chroma_db/, rag.py, stages.py ...)
│  ├─ soil_suitability/       ← your soil module, unchanged
│  ├─ check_model.py          ← one-time check of the .h5 preprocessing
│  ├─ predict_image.py        ← command-line test: classify + Grad-CAM + RAG on one image
│  ├─ .env                    ← GROQ_API_KEY (and optional settings)
│  └─ storage/                ← uploaded images, Grad-CAM outputs, SQLite DB (created automatically)
└─ frontend/                  ← React (Vite) app; `dist/` is already built
```

## Setup (Windows)

1. Copy your trained files into `backend\models\` with **exactly these names**:
   `best_tomato_model.h5`, `xgb_suitability.joblib`, `feature_order.joblib`.
2. Double-click `setup_windows.bat` once (creates a venv, installs packages, builds the frontend).
3. Double-click `start_windows.bat`, then open **http://localhost:8000**.

The sidebar shows a green dot for each model once it has loaded.

Manual setup (any OS):
```bash
cd backend
python -m venv venv && venv\Scripts\activate        # Linux/mac: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000           # API docs at http://localhost:8000/docs
# frontend (only if you change the React code):
cd ../frontend && npm install && npm run build       # or `npm run dev` → http://localhost:5173
```

### TensorFlow version
Install the same TensorFlow version you trained with if `best_tomato_model.h5` fails to load
(for example, `pip install tensorflow==2.15.*` for a Keras 2 model). The model is loaded with `compile=False`.

## First run: check the image preprocessing (one time)

Your old `debug.py` compared "raw 0-255" with "divided by 255". The same check is now:

```bash
cd backend
python check_model.py "C:\Users\Disha B\Downloads\tomato-leaf.jpg"
```

It prints the top-4 classes both ways. The backend picks the right one automatically
(raw pixels if the model has its own Rescaling layer, as Keras EfficientNetB0 does). If the
check shows the other option is correct, add `DISEASE_PREPROCESS=divide255` (or `raw`) to `backend/.env`.

To test the whole pipeline from the command line:
```bash
python predict_image.py "C:\Users\Disha B\Downloads\tomato-leaf.jpg" --days 40
```
The Grad-CAM image is saved next to the input as `tomato-leaf_gradcam.png`.

**File names are preserved.** Every upload is stored as `<timestamp>_<id>_<your original file name>`,
and each API result returns `"filename"` with the exact name you uploaded. The UI shows
"Input file: …" on every result, so you can always see which photo a prediction came from.

## Configuration (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (yours) | LLM for RAG answers + vision growth-stage detection |
| `LLM_MODEL` | `openai/gpt-oss-120b` | RAG answer model (from your rag.py) |
| `VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model for whole-plant photos. Check console.groq.com/docs/models if Groq renames it |
| `DISEASE_PREPROCESS` | `auto` | `auto` / `raw` / `divide255` |
| `DISEASE_MODEL_PATH` | `backend/models/best_tomato_model.h5` | Point to the model elsewhere if you don't want to copy it |
| `DATABASE_URL` | SQLite file in `storage/` | e.g. `postgresql+psycopg2://postgres:pass@localhost:5432/smart_cultivation` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the model your `chroma_db` was built with |

If you use PostgreSQL, create an empty database first. The tables are created automatically on startup.

## Offline behaviour
- **No internet / no Groq key:** disease advice is built directly from the knowledge-base markdown,
  including the growth-stage row from the "Stage-wise action" table. The growth stage is estimated from
  days after transplanting.
- **SoilGrids/NASA unreachable:** use the "Manual soil test values" tab.
- The first run downloads the `bge-small` embedding model (~130 MB) once; after that the RAG works offline.

## Main API endpoints (see /docs)
`POST /api/disease/predict` (image + farm_id, stage, question) · `POST /api/growth/analyze` (image + days) ·
`POST /api/suitability/by-location` · `POST /api/suitability/by-soil` · `POST /api/planning/cost` ·
`GET /api/planning/roadmap` · `GET /api/dashboard/{farm_id}` · `GET/POST /api/farms` · `GET /api/health`

## Known limitations (for the report / viva)
- The soil training data is synthetic, built from agronomic thresholds (see `soil_suitability/README.md`). P and K have no free API, so they are placeholders unless the farmer enters them.
- Soil fertility status uses rule-based bands from the same thresholds the model was trained on. It is not a second trained model.
- Growth-stage detection depends on the vision LLM. Without it, the stage is only estimated from dates.
- Cost figures are indicative defaults. Every quantity and rate can be edited in the UI.
- Chemical names come from the knowledge base. Doses are deliberately not given. Verify everything with CIB&RC and your local KVK.
