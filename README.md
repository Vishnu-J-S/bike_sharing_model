## README.md

```
# Bike Sharing Demand Prediction API [attached_file:1]

Predicts hourly bike rental demand using **XGBoost** trained on Kaggle Bike Sharing Dataset (RMSLE ~0.35).

## Features
- Feature engineering: hour, day, month, weather, temp, etc.
- Compares LinearRegression, RandomForest, **XGBoost** (best)
- FastAPI endpoint: `POST /predict`
- Auto-generated interactive docs: `/docs`

## Quick Demo

uvicorn main:app --reload
# Visit: http://localhost:8000/docs


**Test Prediction**:

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"season":3,"weather":1,"temp":25.0,"atemp":26.0,"humidity":65.0,"windspeed":10.0,"hour":12,"day":15,"month":6,"year":2012,"weekday":2}'

**Response**: `{"predicted_count":234}`

## Setup

pip install -r requirements.txt
uvicorn main:app --reload
```
