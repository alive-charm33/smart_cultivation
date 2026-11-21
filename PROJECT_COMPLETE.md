# AI Based Smart Cultivation — Complete Project Documentation
### Walchand College of Engineering, Sangli | 2026-27
---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Folder Structure](#3-folder-structure)
4. [How to Run](#4-how-to-run)
5. [Configuration (.env)](#5-configuration-env)
6. [Database](#6-database)
7. [Backend — Every File Explained](#7-backend--every-file-explained)
8. [Frontend — Every File Explained](#8-frontend--every-file-explained)
9. [All API Endpoints](#9-all-api-endpoints)
10. [Module 1 — Authentication](#10-module-1--authentication)
11. [Module 2 — Farm & Soil Suitability](#11-module-2--farm--soil-suitability)
12. [Module 3 — Leaf Disease Detection](#12-module-3--leaf-disease-detection)
13. [Module 4 — Growth Stage Monitoring](#13-module-4--growth-stage-monitoring)
14. [Module 5 — Planning & Cost Estimation](#14-module-5--planning--cost-estimation)
15. [Module 6 — Dashboard](#15-module-6--dashboard)
16. [RAG Pipeline](#16-rag-pipeline)
17. [Trained Models](#17-trained-models)
18. [Knowledge Base Documents](#18-knowledge-base-documents)
19. [External APIs Used](#19-external-apis-used)
20. [Error Handling & Offline Behaviour](#20-error-handling--offline-behaviour)
21. [Security](#21-security)
22. [Known Limitations](#22-known-limitations)
23. [Future Work](#23-future-work)
24. [Full Data Flow — End to End](#24-full-data-flow--end-to-end)

---

## 1. PROJECT OVERVIEW

**Project Name:** AI Based Smart Cultivation — Tomato Farming Assistant  
**Type:** Full-stack web application  
**Target users:** Tomato farmers in Maharashtra, India  
**Language support:** English, Marathi, Hindi (in RAG advice)

### What it does

This system is an end-to-end AI assistant that guides a tomato farmer through the entire crop cycle — from choosing a field to harvesting. It integrates five AI modules:

| # | Feature | AI/ML Used |
|---|---|---|
| 1 | Soil suitability check | XGBoost classifier + SHAP explainability |
| 2 | Leaf disease detection | EfficientNet-B0 CNN + Grad-CAM heatmap |
| 3 | Growth stage monitoring | Local computer vision + optional LLM |
| 4 | Disease cure advisory | RAG (ChromaDB + Groq LLM) |
| 5 | Cultivation planning | Rule-based cost engine + roadmap |

### What it does NOT do

- It does NOT train any model. All models are loaded from pre-trained files.
- It does NOT require an internet connection for core features (SoilGrids/LLM degrade gracefully).
- It does NOT store any images on a remote server — everything is local.

---

## 2. TECHNOLOGY STACK

### Backend
| Component | Technology | Version |
|---|---|---|
| Web framework | FastAPI | ≥0.110 |
| Server | Uvicorn (ASGI) | ≥0.29 |
| Database ORM | SQLAlchemy | ≥2.0 |
| Database | SQLite (default) / PostgreSQL | — |
| Deep learning | TensorFlow + Keras | 2.19.1 / 3.15.1 |
| ML suitability | XGBoost | ≥2.0 |
| Explainability | SHAP TreeExplainer | ≥0.45 |
| Vector store | ChromaDB | ≥0.5 |
| Embeddings | sentence-transformers (BAAI/bge-small-en-v1.5) | ≥3.0 |
| LLM provider | Groq API (openai/gpt-oss-120b) | — |
| Vision LLM | Google Gemini (gemini-2.5-flash) — optional | — |
| Auth | python-jose (JWT) + bcrypt | — |
| Image processing | Pillow | — |
| Grad-CAM | matplotlib + numpy | — |

### Frontend
| Component | Technology | Version |
|---|---|---|
| Framework | React | 18.3.1 |
| Build tool | Vite | 5.4 |
| Markdown rendering | react-markdown + remark-gfm | 9.0 / 4.0 |
| Styling | Plain CSS (no framework) | — |
| HTTP client | native fetch API | — |

---

## 3. FOLDER STRUCTURE

```
smart_cultivation/
│
├── README.md                        ← Quick start guide
├── PROJECT_COMPLETE.md              ← THIS FILE
├── setup_windows.bat                ← One-time setup (creates venv, installs packages)
├── start_windows.bat                ← Start the app (opens http://localhost:8000)
├── start_dev_frontend.bat           ← Start React dev server on :5173 (optional)
├── start.sh                         ← Linux/Mac start script
│
├── backend/
│   ├── .env                         ← API keys and config (GROQ_API_KEY, GEMINI_API_KEY, etc.)
│   ├── requirements.txt             ← All Python dependencies
│   ├── check_model.py               ← One-time test: verify .h5 preprocessing mode
│   ├── predict_image.py             ← CLI test: classify leaf + Grad-CAM + RAG
│   │
│   ├── models/                      ← Put your trained model files here
│   │   ├── best_tomato_model.h5     ← EfficientNet-B0 leaf disease model (10 classes)
│   │   ├── xgb_suitability.joblib   ← XGBoost soil suitability model
│   │   └── feature_order.joblib     ← Feature names in the order XGBoost expects
│   │
│   ├── app/                         ← FastAPI application
│   │   ├── __init__.py
│   │   ├── config.py                ← All paths and env-var settings
│   │   ├── db.py                    ← SQLAlchemy models (User, Farm, DiseaseScan, etc.)
│   │   ├── auth.py                  ← JWT auth: signup, login, token verify
│   │   ├── main.py                  ← All FastAPI routes (15+ endpoints)
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── disease.py           ← EfficientNet-B0 loader + predict + Grad-CAM
│   │       ├── growth.py            ← Growth analysis orchestrator
│   │       ├── growth_vision.py     ← Local computer vision stage detector
│   │       ├── knowledge.py         ← RAG retrieval + LLM advice + stage guidance
│   │       ├── planning.py          ← Cost estimate + farming roadmap
│   │       └── soil.py              ← XGBoost predict + SHAP + SoilGrids + NASA POWER
│   │
│   ├── tomato_rag/                  ← RAG pipeline (unchanged from original)
│   │   ├── rag.py                   ← Main RAG orchestrator
│   │   ├── stages.py                ← Day-to-stage mapping from docs front matter
│   │   ├── common.py                ← Embedding helper (Embedder class)
│   │   ├── build_index.py           ← Build ChromaDB from docs/ (run once)
│   │   ├── class_names.json         ← 10 class labels in training order
│   │   ├── label_map.json           ← Maps "Tomato___Early_blight" → "early_blight"
│   │   ├── chroma_db/               ← Built vector index (already included)
│   │   └── docs/
│   │       ├── diseases/            ← 10 disease markdown files with YAML front matter
│   │       │   ├── bacterial_spot.md
│   │       │   ├── early_blight.md
│   │       │   ├── late_blight.md
│   │       │   ├── leaf_mold.md
│   │       │   ├── septoria_leaf_spot.md
│   │       │   ├── spider_mites.md
│   │       │   ├── target_spot.md
│   │       │   ├── tomato_mosaic_virus.md
│   │       │   ├── tylcv.md
│   │       │   └── healthy.md
│   │       ├── stages/              ← 7 growth stage markdown files
│   │       │   ├── nursery.md
│   │       │   ├── transplanting.md
│   │       │   ├── vegetative.md
│   │       │   ├── flowering_fruit_set.md
│   │       │   ├── fruit_development.md
│   │       │   ├── ripening_harvest.md
│   │       │   └── end_of_season.md
│   │       └── cross/               ← Cross-reference docs
│   │           ├── disease_stage_risk_matrix.md
│   │           └── leaf_spot_comparison.md
│   │
│   ├── soil_suitability_complete/
│   │   └── soil_suitability/        ← Soil module (imported by app/services/soil.py)
│   │       ├── explain.py           ← SuitabilityExplainer (SHAP wrapper)
│   │       ├── soil_client.py       ← ISRIC SoilGrids API client
│   │       ├── weather_client.py    ← NASA POWER climate API client
│   │       ├── train_model.py       ← Training script (not run in production)
│   │       ├── colab_train_pipeline.py ← Colab training pipeline
│   │       ├── data/
│   │       │   ├── generate_dataset.py ← Synthetic dataset generator
│   │       │   └── tomato_suitability.csv ← Generated training data
│   │       └── models/              ← Source location of joblib files
│   │
│   └── storage/                     ← Created automatically at startup
│       ├── uploads/                 ← Uploaded leaf/plant photos
│       ├── gradcam/                 ← Grad-CAM overlay images
│       └── smart_cultivation.db    ← SQLite database
│
└── frontend/
    ├── index.html                   ← HTML entry point
    ├── package.json                 ← npm dependencies
    ├── vite.config.js               ← Vite config + dev proxy to :8000
    ├── dist/                        ← Built static files (served by FastAPI)
    └── src/
        ├── main.jsx                 ← React entry point
        ├── App.jsx                  ← Root component, routing, sidebar, auth gate
        ├── api.js                   ← fetch wrapper + auth token + constants
        ├── styles.css               ← All CSS (no framework)
        ├── components/
        │   └── ui.jsx               ← Reusable components (Card, Bar, ImagePicker, etc.)
        └── pages/
            ├── Login.jsx            ← Signup + login form
            ├── Profile.jsx          ← Farmer profile + change password + logout
            ├── Dashboard.jsx        ← Summary KPIs + timeline + recent scans
            ├── FarmSoil.jsx         ← Farm CRUD + soil suitability check
            ├── DiseaseScan.jsx      ← Leaf upload + disease result + Grad-CAM + RAG
            ├── GrowthMonitor.jsx    ← Plant upload + stage + mismatch analysis
            └── Planning.jsx         ← Cost estimate table + farming roadmap
```

---

## 4. HOW TO RUN

### First time setup
```bat
cd smart_cultivation
setup_windows.bat
```
This creates a Python venv, installs all packages, and builds the React frontend.

### Every time after that
```bat
cd smart_cultivation
.\start_windows.bat
```
Open **http://localhost:8000** — the React app is served directly by FastAPI.

### Development mode (hot reload for frontend)
```bat
REM Terminal 1 — backend
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --port 8000 --reload

REM Terminal 2 — frontend
cd frontend
npm run dev
```
Open **http://localhost:5173** — Vite proxies `/api` and `/media` to port 8000.

### API documentation
Go to **http://localhost:8000/docs** — Swagger UI with all endpoints.

---

## 5. CONFIGURATION (.env)

File location: `backend/.env`

```env
# ── LLM (disease RAG advice) ────────────────────────────────────────────────
GROQ_API_KEY=gsk_...                  # Your Groq API key (required for LLM advice)
LLM_MODEL=openai/gpt-oss-120b         # Text LLM via Groq

# ── Vision model (growth stage from photo) ──────────────────────────────────
VISION_MODEL=none                     # Groq vision model — set to "none" if unavailable
GEMINI_API_KEY=AIza...                # Google Gemini free key (aistudio.google.com)
GEMINI_VISION_MODEL=gemini-2.5-flash  # Free vision model

# ── Disease model preprocessing ─────────────────────────────────────────────
DISEASE_PREPROCESS=auto               # auto / raw / divide255

# ── Database ─────────────────────────────────────────────────────────────────
# Leave blank for SQLite (default). For PostgreSQL:
# DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/smart_cultivation

# ── JWT Auth ─────────────────────────────────────────────────────────────────
JWT_SECRET=change_this_in_production  # Secret key for signing JWT tokens
JWT_EXPIRE_MINUTES=10080              # Token expiry (7 days default)

# ── TensorFlow noise suppression ────────────────────────────────────────────
TF_CPP_MIN_LOG_LEVEL=3
TF_ENABLE_ONEDNN_OPTS=0
```

All variables have safe defaults so the app runs without any `.env` file — only the Groq key is needed for LLM features.

---

## 6. DATABASE

### Engine
SQLite by default — file at `backend/storage/smart_cultivation.db`.  
Switch to PostgreSQL by setting `DATABASE_URL` in `.env`.

### Tables

#### `users`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| username | VARCHAR(80) UNIQUE | Login name |
| hashed_password | VARCHAR(200) | bcrypt hash |
| full_name | VARCHAR(120) | Display name |
| mobile | VARCHAR(20) | Phone number |
| village | VARCHAR(120) | Village/taluka |
| district | VARCHAR(120) | District |
| state | VARCHAR(80) | State (default: Maharashtra) |
| is_active | BOOLEAN | Account status |
| created_at | DATETIME | Registration time |

#### `farms`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| user_id | FK → users.id | Owner (nullable for legacy) |
| name | VARCHAR(120) | Farm name |
| farmer_name | VARCHAR(120) | Farmer's name |
| village | VARCHAR(120) | Farm village |
| lat / lon | FLOAT | GPS coordinates |
| area_acres | FLOAT | Farm size |
| variety | VARCHAR(120) | Tomato hybrid name |
| transplant_date | DATE | Date of transplanting |
| created_at | DATETIME | Record creation |

#### `suitability_results`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | — |
| farm_id | FK → farms.id | Linked farm |
| suitable | BOOLEAN | Model prediction |
| probability | FLOAT | P(suitable) from XGBoost |
| result | JSON | Full result including SHAP, fertility |
| created_at | DATETIME | When check was run |

#### `disease_scans`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | — |
| farm_id | FK → farms.id | Linked farm (nullable) |
| original_filename | VARCHAR(255) | User's file name |
| image_path | VARCHAR(500) | Server storage path |
| gradcam_path | VARCHAR(500) | Grad-CAM overlay path |
| disease_id | VARCHAR(80) | e.g. "late_blight" |
| confidence | FLOAT | Top prediction confidence |
| predictions | JSON | All 10 class probabilities |
| advice | TEXT | RAG/LLM advice text |
| stage | VARCHAR(60) | Growth stage at scan time |
| created_at | DATETIME | Scan timestamp |

#### `growth_records`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | — |
| farm_id | FK → farms.id | Linked farm (nullable) |
| original_filename | VARCHAR(255) | User's file name |
| image_path | VARCHAR(500) | Server storage path |
| days_after_transplant | INTEGER | DAT input |
| stage_id | VARCHAR(60) | Detected/estimated stage |
| method | VARCHAR(120) | How stage was detected |
| details | JSON | Full analysis result |
| created_at | DATETIME | Record timestamp |

---

## 7. BACKEND — EVERY FILE EXPLAINED

### `app/config.py`
Central configuration. Reads `backend/.env` at import time using `os.environ.setdefault`.  
Key paths:
- `BACKEND_DIR` — absolute path to the `backend/` folder
- `MODELS_DIR` — where `.h5` and `.joblib` files live
- `RAG_DIR` — `backend/tomato_rag/`
- `SOIL_DIR` — `backend/soil_suitability_complete/soil_suitability/`
- `STORAGE_DIR` — uploaded images + SQLite database

### `app/db.py`
SQLAlchemy ORM definitions. Five tables: User, Farm, SuitabilityResult, DiseaseScan, GrowthRecord.  
`init_db()` creates tables if they don't exist (called at startup).  
`get_session()` is a FastAPI dependency that yields a session and closes it after the request.

### `app/auth.py`
JWT authentication using `python-jose` and `bcrypt`.  
- `hash_password(password)` → bcrypt hash
- `verify_password(plain, hashed)` → bool
- `create_token(user_id, username)` → JWT string (7 days)
- `get_current_user(token)` → User object or None (optional auth)
- `require_user(user)` → User object or raises 401 (mandatory auth)

### `app/main.py`
The FastAPI application. 300+ lines. Registers all routes, mounts static file serving, adds CORS middleware, and runs model warm-up in a background thread at startup.

### `app/services/disease.py`
Leaf disease pipeline:
1. `get_model()` — loads `best_tomato_model.h5` once using thread-safe locking
2. `_decide_preprocess()` — auto-detects if model has internal Rescaling layer (Keras 3)
3. `predict(image_path)` — returns top-3 predictions with confidence scores
4. `gradcam(x, class_index)` — Keras 3 compatible Grad-CAM using GradientTape
5. `overlay(original, cam)` — blends jet colormap heatmap over original image
6. `describe_cam(cam)` — returns highlighted percentage and region (upper left, centre, etc.)

### `app/services/growth.py`
Growth stage analysis orchestrator:
1. Runs **local vision** (`growth_vision.py`) on the uploaded photo — always works offline
2. Optionally enhances with **Groq LLM** or **Gemini LLM** if keys are set
3. LLM result beats local if confident; otherwise local vision is used
4. `progress()` — compares detected stage vs expected stage for given DAT
   - `stage_source="vision"` → can return behind/on_track/ahead
   - `stage_source="days"` → returns "estimated" (never falsely claims on_track)

### `app/services/growth_vision.py`
Local computer vision pipeline — no API, no model training, fully offline:
1. `detect_stage_from_image(path)` — extracts color features from photo
   - Green ratio (plant tissue)
   - Yellow ratio (flowers)
   - Red/orange ratio (ripe fruit)
   - Plant coverage (fraction of image that is plant)
2. Scores each of 7 stages against visual signatures
3. `_analyse_mismatch()` — compares detected stage vs expected stage, computes stages_behind/ahead
4. `_recommend_for_behind()` — generates agronomic recommendations when behind schedule

**Example:** 4-day seedling uploaded with 60 DAT input:
- Photo: low coverage (seedling), dense green, no fruit → detected as `nursery/transplanting`
- Expected at 60 DAT: `fruit_development`
- Result: **2 stages behind**, recommendation: check nutrition, check for transplant shock

### `app/services/knowledge.py`
RAG layer:
- `collection()` — opens ChromaDB + loads embedding model (cached)
- `advise_disease(top_preds, question, stage, days)` — full RAG pipeline: plan → retrieve → generate
- `offline_answer(preds, stages)` — fallback: builds answer directly from markdown KB
- `stage_guidance(stage_id)` — returns structured guidance from the stage markdown doc
- `stage_table()` — loads day ranges from all stage docs for roadmap calculation

### `app/services/soil.py`
Soil suitability pipeline:
- `_load()` — loads XGBoost model + feature list + SHAP explainer (once)
- `predict(features)` — XGBoost prediction + SHAP values + fertility bands
- `predict_by_location(lat, lon)` — fetches from SoilGrids + NASA POWER then runs predict
- `fertility(features)` — rule-based soil health bands (optimal/low/high/very low/very high)

### `app/services/planning.py`
Planning engine:
- `cost_estimate(area, overrides)` — 11 cost items × area_acres + contingency
- `roadmap(transplant_date, fertility)` — generates dated stage-by-stage plan
  - Each step has: start/end dates, tasks, water/nutrients, disease watch-list

---

## 8. FRONTEND — EVERY FILE EXPLAINED

### `api.js`
- `auth` object — `getToken()`, `setToken()`, `clearToken()`, `getUser()`, `setUser()` via localStorage
- `api.get/post/put/upload` — fetch wrappers that automatically attach `Authorization: Bearer <token>`
- `STAGES` — array of [id, name] for all 7 growth stages
- `DISEASE_NAMES` — human-readable names for all 10 disease IDs
- `rupees(n)` — formats number as ₹1,23,456 (Indian locale)
- `stageName(id)` / `diseaseName(id)` — lookup helpers

### `App.jsx`
Root component:
- Reads stored token/user from localStorage — shows Login if not authenticated
- Loads farm list on login, polls `/api/health` every 15 seconds
- Sidebar: logo, nav buttons, farm picker dropdown, user badge, model status dots
- Routes to one of 6 pages based on active tab

### `components/ui.jsx`
Reusable components:
- `Card` — white panel with header, icon, optional right-side button
- `PageHeader` — title + subtitle
- `Alert` — info/success/warn/error banner
- `Spinner` — animated loading indicator
- `Markdown` — renders markdown with GFM tables/blockquotes
- `Bar` — horizontal bar chart row (supports signed SHAP bars)
- `ImagePicker` — drag-and-drop image upload with preview
- `StageTrack` — 7-stage progress track with current/done/expected markers
- `Field` — form field wrapper with label and hint
- `Empty` — centered placeholder text

### `pages/Login.jsx`
Two-mode form: Sign in / Create account. On success, stores JWT token + user object in localStorage, calls `onLogin(user)` to update App state. Validates password match on signup client-side before sending.

### `pages/Profile.jsx`
- View mode: shows all profile fields in a definition list
- Edit mode: inline form with save/cancel
- Change password: separate form with old password verification
- Sign out: clears token + user from localStorage, resets app

### `pages/FarmSoil.jsx`
Two-panel layout:
- Left: Farm details form (create/edit/view) with GPS button
- Right: Suitability component
  - Tab 1: By GPS — sends lat/lon to `/api/suitability/by-location`
  - Tab 2: Manual — 8 soil parameter inputs, sends to `/api/suitability/by-soil`
  - Result: verdict badge + SHAP bar chart + fertility table + improvement advice

### `pages/DiseaseScan.jsx`
- ImagePicker for leaf photo
- Stage selector + optional question input + LLM toggle
- Calls `POST /api/disease/predict` via FormData upload
- Result panel: disease verdict, Grad-CAM overlay/side-by-side toggle, confidence bars
- RAG advice card: mode badge (LLM/KB), stage info, markdown answer, collapsible sources
- History grid: thumbnails of all past scans

### `pages/GrowthMonitor.jsx`
- ImagePicker for whole-plant photo
- Days after transplanting input
- Vision LLM checkbox (disabled with badge if no vision model configured)
- Calls `POST /api/growth/analyze`
- Result: stage verdict, StageTrack, progress box (behind/on_track/ahead/estimated)
- **Mismatch box** (new): red box if photo stage ≠ expected stage, with specific recommendations
- SVG Timeline chart: x = days after transplanting, y = stage index
- Thumbnails of past growth records

### `pages/Planning.jsx`
- Cost estimate: live-updating table as area/yield/price inputs change
- Each row has editable quantity and rate fields
- Stats bar: total cost, cost/acre, revenue, profit (green/red), break-even price
- Roadmap: chronological list of stages with dates, tasks, watch-for diseases
- Current stage highlighted in green ("Now")

### `pages/Dashboard.jsx`
- Calls `GET /api/dashboard/{farm_id}`
- 4 KPI cards: suitability, growth stage, latest disease, estimated cost
- Crop progress StageTrack
- Current week's tasks card
- SHAP reasons card
- Recent disease scan thumbnails
- Growth timeline SVG chart

---

## 9. ALL API ENDPOINTS

### Health
| Method | URL | Description |
|---|---|---|
| GET | `/api/health` | Model load status, vision availability, DB type |

### Auth
| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/signup` | No | Create account, returns JWT |
| POST | `/api/auth/login` | No | OAuth2 form login |
| POST | `/api/auth/login/json` | No | JSON login, returns JWT |
| GET | `/api/auth/me` | Required | Get current user profile |
| PUT | `/api/auth/profile` | Required | Update profile fields |
| POST | `/api/auth/change-password` | Required | Change password |

### Farms
| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/api/farms` | Optional | List farms (filtered by user if authenticated) |
| POST | `/api/farms` | Optional | Create farm |
| PUT | `/api/farms/{id}` | Optional | Update farm |
| DELETE | `/api/farms/{id}` | Optional | Delete farm |

### Soil Suitability
| Method | URL | Description |
|---|---|---|
| POST | `/api/suitability/by-location` | Fetch soil+climate from APIs, run XGBoost |
| POST | `/api/suitability/by-soil` | Use manually entered values, run XGBoost |

### Leaf Disease
| Method | URL | Description |
|---|---|---|
| POST | `/api/disease/predict` | Upload image → classify → Grad-CAM → RAG advice |
| GET | `/api/disease/history` | Past scans (filter by farm_id) |
| GET | `/api/disease/{disease_id}` | Full KB doc for a disease |

### Growth
| Method | URL | Description |
|---|---|---|
| POST | `/api/growth/analyze` | Upload image + days → detect stage → compare |
| GET | `/api/growth/history` | Past records (filter by farm_id) |
| GET | `/api/growth/stages` | Stage table with day ranges |
| GET | `/api/growth/stage/{stage_id}` | Full guidance for a stage |

### Planning
| Method | URL | Description |
|---|---|---|
| POST | `/api/planning/cost` | Cost estimate with overrides |
| GET | `/api/planning/roadmap` | Dated stage roadmap |

### Dashboard
| Method | URL | Description |
|---|---|---|
| GET | `/api/dashboard/{farm_id}` | Full aggregated farm summary |

### Static Files
| URL | Description |
|---|---|
| `/media/*` | Uploaded images and Grad-CAM outputs |
| `/*` | Built React app (from `frontend/dist/`) |

---

## 10. MODULE 1 — AUTHENTICATION

### How it works
1. Farmer fills signup form with username + password + optional profile
2. Backend: bcrypt hashes password, stores User in DB, returns JWT token (7-day expiry)
3. Frontend: stores token in localStorage, attaches as `Authorization: Bearer <token>` on every request
4. Login: sends username+password → verifies bcrypt → returns new JWT
5. All farm data is scoped to `user_id` — each user sees only their own farms

### JWT structure
```
Header: { alg: HS256 }
Payload: { sub: "user_id", username: "ramesh_patil", exp: <timestamp> }
Signature: HMAC-SHA256(secret_key)
```

### Security notes
- Passwords never stored in plain text — bcrypt with random salt
- Token expiry: 7 days (configurable via `JWT_EXPIRE_MINUTES` in `.env`)
- Secret key: set `JWT_SECRET` in `.env` for production

---

## 11. MODULE 2 — FARM & SOIL SUITABILITY

### Data sources
| Parameter | Source | Method |
|---|---|---|
| pH | ISRIC SoilGrids REST API | GET request with lat/lon |
| Organic carbon | ISRIC SoilGrids REST API | GET request with lat/lon |
| Nitrogen (total) | ISRIC SoilGrids REST API | GET request with lat/lon |
| Temperature | NASA POWER API (climatology) | GET request with lat/lon |
| Rainfall | NASA POWER API (climatology) | GET request with lat/lon |
| Humidity | NASA POWER API (climatology) | GET request with lat/lon |
| Phosphorus | Farmer input (Soil Health Card) | Manual entry |
| Potassium | Farmer input (Soil Health Card) | Manual entry |

### XGBoost model
- Trained on synthetic dataset of 10,000 samples generated from agronomic thresholds
- 8 input features: pH, N, P, K, organic_carbon, rainfall, temperature, humidity
- Output: probability of suitability (≥0.5 = suitable)
- Explainability: SHAP TreeExplainer gives per-feature impact values

### Fertility bands
Each parameter is classified into: `optimal`, `low`, `high`, `very low`, `very high`  
Based on the same thresholds used to generate training data.

### SHAP output
Returns a ranked list of features with:
- `impact` (signed float): positive = pushed towards suitable, negative = against
- `direction`: "increases suitability" or "reduces suitability"
- `label`: human-readable parameter name

---

## 12. MODULE 3 — LEAF DISEASE DETECTION

### Model
- Architecture: EfficientNet-B0 (transfer learning from ImageNet)
- Input: 224×224 RGB image
- Output: 10-class softmax probabilities
- Training: PlantVillage dataset
- File: `backend/models/best_tomato_model.h5`
- Keras version: 3.15.1 (TF 2.19.1 backend)

### 10 Disease Classes
| ID | Common Name | Type |
|---|---|---|
| bacterial_spot | Bacterial spot | Bacterial |
| early_blight | Early blight | Fungal |
| late_blight | Late blight | Oomycete |
| leaf_mold | Leaf mold | Fungal |
| septoria_leaf_spot | Septoria leaf spot | Fungal |
| spider_mites | Spider mites | Pest |
| target_spot | Target spot | Fungal |
| tylcv | Yellow leaf curl virus | Viral |
| tomato_mosaic_virus | Mosaic virus | Viral |
| healthy | Healthy | — |

### Preprocessing (auto-detected)
Keras 3 EfficientNet-B0 includes an internal `Rescaling` layer — model expects raw 0-255 pixels.  
The system detects this automatically (`_decide_preprocess()`).  
Override with `DISEASE_PREPROCESS=raw` or `divide255` in `.env` if needed.

### Grad-CAM
1. Finds the last Conv2D layer in the model (handles nested sub-models in Keras 3)
2. Builds a gradient model from input → (conv_output, predictions)
3. Uses `tf.GradientTape` to compute gradients of target class w.r.t. conv output
4. Global average pools gradients → per-channel weights
5. Weighted sum of conv feature maps → ReLU → normalize to [0,1]
6. Upsamples to original image size and blends with jet colormap at 45% opacity

### RAG advice flow
1. Determines growth stages from farmer input (explicit stage or days-based estimate)
2. Plans which disease docs to retrieve (top-1 always, top-2/3 if confidence < thresholds)
3. Retrieves relevant passages from ChromaDB using cosine similarity
4. Sends retrieved context + predictions to Groq LLM
5. LLM generates stage-specific advice following strict rules (no doses, verify CIB&RC, etc.)
6. Falls back to offline KB answer if LLM unavailable

---

## 13. MODULE 4 — GROWTH STAGE MONITORING

### 7 Growth Stages
| Stage ID | Name | Day Range (DAT) |
|---|---|---|
| nursery | Nursery / seedling | DAS 0-30 |
| transplanting | Transplanting | DAT 0-14 |
| vegetative | Vegetative growth | DAT 15-35 |
| flowering_fruit_set | Flowering & fruit set | DAT 30-50 |
| fruit_development | Fruit development | DAT 45-65 |
| ripening_harvest | Ripening & harvest | DAT 65-110 |
| end_of_season | End of season | DAT 100+ |

### Local Vision Detection (always runs)
Color feature extraction from the uploaded photo:

| Feature | Detection Method | Indicates |
|---|---|---|
| Green ratio | G > R×1.1 AND G > B×1.05 | Plant tissue |
| Yellow ratio | R>120, G>100, B<80 | Flowers |
| Red/orange ratio | R>120, R>G×1.3, R>B×1.5 | Ripe fruit |
| Plant coverage | (green + yellow pixels) / total | Plant size / age |

Each stage has a signature (expected ranges of these features). The stage with highest matching score is selected.

### Mismatch Logic
```
detected_stage_index vs expected_stage_index:

detected < expected → BEHIND schedule
  → compute stages_behind = expected_index - detected_index
  → generate recommendations (check N, irrigation, root health)

detected > expected → AHEAD of schedule
  → fine for early hybrids

detected == expected → ON TRACK ✅
```

**Critical fix:** When stage_source = "days" (no photo analysis), the system returns status="estimated" — it never claims "on track" because it is literally comparing the day-estimate against itself.

### Optional LLM Enhancement
If `GEMINI_API_KEY` is set in `.env`:
- Sends base64-encoded image to `gemini-2.5-flash`
- Gets structured JSON with stage_id, confidence, observations, counts
- LLM result overrides local vision if confident and not "unknown"

---

## 14. MODULE 5 — PLANNING & COST ESTIMATION

### Cost Items (per acre, Maharashtra defaults)
| Item | Default Qty | Unit | Default Rate (₹) |
|---|---|---|---|
| Land preparation | 1 | acre | 6,000 |
| Hybrid seedlings | 8,000 | plants | 1.80 each |
| FYM / compost | 8 | tonnes | 1,000/tonne |
| Fertilizers | 1 | season | 15,000 |
| Plant protection | 1 | season | 12,000 |
| Mulching film | 1 | acre | 8,000 |
| Drip irrigation | 1 | season | 10,000 |
| Staking | 1 | acre | 15,000 |
| Labour | 120 | person-days | 350/day |
| Electricity/water | 1 | season | 4,000 |
| Crates & transport | 1 | season | 10,000 |
| **Contingency** | 5% | — | — |

All quantities and rates are editable by the farmer.

### Calculations
```
subtotal = sum(qty × rate × area_acres)
contingency = subtotal × contingency_pct / 100
total_cost = subtotal + contingency
expected_revenue = yield_t_per_acre × 1000 × price_per_kg × area_acres
expected_profit = expected_revenue - total_cost
breakeven_price = total_cost / (yield_t_per_acre × 1000 × area_acres)
```

### Farming Roadmap
Generated from transplanting date + stage day ranges from knowledge base:
- Each step gets calendar start/end dates = transplant_date + day_from/day_to
- Tasks extracted from "Key management" section of each stage doc
- Disease watch-list from `diseases_at_risk` in stage doc front matter
- Current stage highlighted as "Now" if today falls within its date range
- Fertilizer recommendations injected from last suitability result's fertility advice

---

## 15. MODULE 6 — DASHBOARD

Calls `GET /api/dashboard/{farm_id}` which aggregates:
- Farm details with days after transplanting
- Latest suitability result (suitable/not, probability, SHAP top reasons)
- Expected stages for current days
- Latest growth record (stage, progress, image)
- Full growth timeline (all records, sorted by date)
- 5 most recent disease scans
- Cost estimate for farm area
- Current farming tasks (step from roadmap matching today's date)

---

## 16. RAG PIPELINE

### Architecture
```
User question + disease predictions + growth stage
        ↓
plan()  → which disease docs to fetch (1, 2, or 3 based on confidence)
        ↓
retrieve() → ChromaDB cosine similarity search
  - disease doc passages (top-9 for primary, top-4 for secondary)
  - growth stage passages (top-4)
  - disease-stage risk matrix (top-2)
  - leaf spot comparison (top-3, only for confusable diseases)
        ↓
format_context() → numbered passages [S1], [S2], ...
        ↓
generate() → Groq LLM with SYSTEM_PROMPT rules
        ↓
Answer in English / Marathi / Hindi (matches user's question language)
```

### ChromaDB Collection
- Collection name: `tomato_kb`
- Embedding model: `BAAI/bge-small-en-v1.5` (~130 MB, downloaded once)
- Metadata per chunk: `doc_type`, `disease_id`/`stage_id`, `source_file`, `section`
- Built with: `python build_index.py` (run once, index already included)

### Offline fallback
When Groq is unreachable: `offline_answer()` builds a structured response directly from the markdown files including stage-wise action table row for the detected stage.

---

## 17. TRAINED MODELS

### 1. best_tomato_model.h5
| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (transfer learning) |
| Framework | Keras 3.15.1 + TensorFlow 2.19.1 |
| Input shape | (None, 224, 224, 3) |
| Output shape | (None, 10) — 10-class softmax |
| Dataset | PlantVillage (tomato subset) |
| Preprocessing | Internal Rescaling layer (raw 0-255 input) |
| File size | ~20 MB |

### 2. xgb_suitability.joblib
| Property | Value |
|---|---|
| Algorithm | XGBoost classifier |
| Features | 8 (pH, N, P, K, OC, rainfall, temperature, humidity) |
| Training data | Synthetic dataset (10,000 samples) |
| Target | binary: suitable / not suitable |
| Explainability | SHAP TreeExplainer (loaded via explain.py) |

### 3. feature_order.joblib
A Python list of 8 feature names in the exact order the XGBoost model expects.  
Used to construct the feature vector correctly from the input dictionary.

---

## 18. KNOWLEDGE BASE DOCUMENTS

All documents use YAML front matter + markdown sections.

### Disease Documents (10 files)
Each disease doc contains:
- `## Symptoms` — visual identification
- `## Favorable conditions` — when it spreads
- `## Prevention` — cultural practices
- `## Cultural control when disease is seen`
- `## Organic / biological control`
- `## Chemical control` — active ingredients only (no doses)
- `## Stage-wise action` — risk and action per growth stage (table)

### Stage Documents (7 files)
Each stage doc contains:
- YAML: `day_from`, `day_to`, `day_reference` (DAT/DAS), `diseases_at_risk`
- `## Timing` — when this stage occurs
- `## What happens` — plant biology
- `## Water and nutrient needs`
- `## Key management` — bullet-point tasks
- `## Disease and pest risks at this stage` — table with risk level and notes

### Cross-Reference Documents (2 files)
- `disease_stage_risk_matrix.md` — all diseases × all stages risk matrix
- `leaf_spot_comparison.md` — how to tell apart confusable brown-spot diseases

---

## 19. EXTERNAL APIS USED

| API | Purpose | Free? | Fallback |
|---|---|---|---|
| ISRIC SoilGrids REST | pH, organic carbon, nitrogen from lat/lon | Yes | Manual entry tab |
| NASA POWER | Temperature, rainfall, humidity from lat/lon | Yes | Manual entry tab |
| Groq API | LLM for disease advice (openai/gpt-oss-120b) | Free tier | KB offline answer |
| Google Gemini | Vision LLM for growth stage (gemini-2.5-flash) | Free tier | Local vision analysis |

### Groq API limits (free tier)
- Model: `openai/gpt-oss-120b`
- 250K tokens per minute
- 1K requests per minute

### Google Gemini API limits (free tier)
- Model: `gemini-2.5-flash`
- Get key at: https://aistudio.google.com/app/apikey
- Free with generous rate limits

---

## 20. ERROR HANDLING & OFFLINE BEHAVIOUR

| Scenario | System Behaviour |
|---|---|
| SoilGrids API unreachable | Shows error; farmer uses "Manual soil test values" tab |
| NASA POWER unreachable | Same — manual fallback available |
| Groq LLM unavailable | Disease advice from KB markdown (offline_answer) |
| Gemini vision unavailable | Local color-based vision analysis used |
| No vision model at all | Stage estimated from days; clearly labeled as "estimated" |
| Model file missing | Clear error: "Copy best_tomato_model.h5 to backend/models/" |
| Low disease confidence (<50%) | Warning shown: "Retake photo — single leaf, natural light" |
| Unknown stage in photo | Warning: "Not a tomato plant — upload full-plant photo" |
| Image wrong format | 400 error: "Unsupported file type. Use JPG or PNG." |
| Farm not found | 404 error |
| Wrong password | 401 Unauthorized |
| Username taken | 400: "Username already taken" |

---

## 21. SECURITY

| Concern | Implementation |
|---|---|
| Password storage | bcrypt hash with random salt — never stored plain |
| API authentication | JWT Bearer token — 7-day expiry |
| Farm data isolation | `user_id` FK — each user only sees their own farms |
| File uploads | Extension whitelist (jpg/jpeg/png/webp/bmp), UUID prefix on saved name |
| CORS | Configured in FastAPI middleware (development: all origins) |
| SQL injection | SQLAlchemy ORM with parameterized queries — no raw SQL |
| Secrets | All keys in `.env`, never in source code |

---

## 22. KNOWN LIMITATIONS

1. **Soil P and K have no free API** — must be entered from the farmer's Soil Health Card
2. **XGBoost training data is synthetic** — generated from agronomic rules, not real soil survey data
3. **Growth vision accuracy** — local color analysis works well but can be confused by unusual lighting or backgrounds
4. **Vision LLM** — requires `GEMINI_API_KEY` for enhanced accuracy; without it, local vision still works
5. **Disease model performance on field photos** — PlantVillage images are lab-quality; field photos with mixed lighting/shadows may reduce accuracy
6. **Single crop** — entire system is tomato-specific. Expansion to other crops requires retraining and new KB docs
7. **No weather forecast** — climate data is historical climatology (30-year average), not real-time forecast
8. **Chemical doses deliberately omitted** — must be verified against CIB&RC registration and product labels

---

## 23. FUTURE WORK

1. **Mobile app** — React Native frontend using the same FastAPI backend
2. **Weather forecast integration** — IMD or OpenWeatherMap for real-time alerts
3. **Multi-crop support** — expand to chilli, onion, grape with new training data and KB
4. **Offline mobile model** — TFLite quantized model for disease detection without internet
5. **Regional language UI** — full Marathi/Hindi interface (currently advice is multilingual but UI is English)
6. **Push notifications** — alert farmer when disease risk is high based on current weather
7. **Marketplace integration** — connect break-even price with APMC mandi rates
8. **Drone/satellite imagery** — replace single-photo growth monitoring with field-scale analysis
9. **Farmer community features** — share scan results with local KVK/agriculture officer

---

## 24. FULL DATA FLOW — END TO END

### Flow 1: Soil Suitability Check
```
Farmer enters lat/lon  →  POST /api/suitability/by-location
    ↓
soil_client.py: GET https://rest.isric.org/soilgrids/v2.0/... (pH, OC, N)
weather_client.py: GET https://power.larc.nasa.gov/api/... (temp, rain, humidity)
    ↓
XGBoost: predict_proba([pH, N, P, K, OC, rain, temp, humidity])
SHAP: TreeExplainer.shap_values(row) → per-feature impacts
fertility(): rule-based bands for each parameter
    ↓
Store SuitabilityResult in DB
Return: {suitable, probability, shap[], fertility{}, data_sources{}}
    ↓
Frontend: renders verdict badge + signed SHAP bar chart + fertility table
```

### Flow 2: Disease Detection
```
Farmer uploads leaf photo  →  POST /api/disease/predict (multipart)
    ↓
_save_upload(): UUID-prefixed file saved to storage/uploads/
EfficientNet-B0: predict(image) → 10 probabilities → top-3
    ↓
Grad-CAM: GradientTape on last conv layer → heatmap
overlay(): jet colormap + blend → saved to storage/gradcam/
describe_cam(): highlighted % and region
    ↓
knowledge.advise_disease():
  plan() → how many docs to fetch
  retrieve() → ChromaDB cosine search
  generate() → Groq LLM with context → advice text
    ↓
Store DiseaseScan in DB
Return: {scan_id, prediction, gradcam_urls, advice, sources}
    ↓
Frontend: renders verdict + Grad-CAM overlay + RAG advice + probability bars
```

### Flow 3: Growth Monitoring
```
Farmer uploads plant photo + enters days  →  POST /api/growth/analyze
    ↓
growth_vision.detect_stage_from_image():
  extract_color_features(): green/yellow/red ratios, plant coverage
  score each of 7 stages → pick best match
    ↓
(Optional) Gemini API: send base64 image → get LLM stage prediction
  If LLM confident → use LLM result
  Else → use local vision result
    ↓
expected_stages(days): look up stage day ranges → what should plant be?
    ↓
progress(): compare detected_stage vs expected_stage
  detected < expected → BEHIND + recommendations
  detected > expected → AHEAD
  detected == expected → ON TRACK
    ↓
_analyse_mismatch(): generate specific agronomic recommendations
stage_guidance(): fetch RAG care advice for detected stage
    ↓
Store GrowthRecord in DB
Return: {stage_id, confidence, method, progress, mismatch, guidance}
    ↓
Frontend: renders stage badge + StageTrack + progress box + mismatch detail box
```

### Flow 4: Dashboard Load
```
Tab switch to Dashboard  →  GET /api/dashboard/{farm_id}
    ↓
Single query aggregates:
  Farm details + days after transplanting
  Latest SuitabilityResult
  Latest GrowthRecord
  All GrowthRecords (for timeline)
  Latest 5 DiseaseScan records
  cost_estimate(farm.area_acres)
  planning.roadmap(farm.transplant_date)
  → find current_step (today's date within step start/end)
    ↓
Return single JSON with everything
    ↓
Frontend renders:
  4 KPI cards (suitability / growth / disease / cost)
  StageTrack with expected stages
  Timeline SVG chart (stage_index vs days)
  Current tasks card
  SHAP reasons card
  Disease scan thumbnails
```

---

*Last updated: October 2026*  
*Institution: Walchand College of Engineering, Sangli*  
*Project: AI Based Smart Cultivation — Tomato Farming Assistant*
