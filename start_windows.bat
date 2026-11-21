@echo off
REM Starts the backend (which also serves the built React app) on http://localhost:8000
cd backend
call venv\Scripts\activate
start "" http://localhost:8000
python -m uvicorn app.main:app --port 8000
