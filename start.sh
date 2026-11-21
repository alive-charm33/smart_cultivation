#!/usr/bin/env bash
# Linux / macOS: starts the backend (also serves the built React app) on http://localhost:8000
cd "$(dirname "$0")/backend"
[ -d venv ] && source venv/bin/activate
python -m uvicorn app.main:app --port 8000
