@echo off
REM One-time setup (Windows). Run from the smart_cultivation folder.
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
cd ..\frontend
call npm install
call npm run build
cd ..
echo.
echo Setup done. Copy best_tomato_model.h5, xgb_suitability.joblib and feature_order.joblib into backend\models\
echo Then run start_windows.bat
pause
