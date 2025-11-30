from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Bike Sharing Demand API")


model = joblib.load("bike_model.pkl")

class BikeFeatures(BaseModel):
    season: int
    weather: int
    temp: float
    atemp: float
    humidity: float
    windspeed: float
    hour: int
    day: int
    month: int
    year: int
    weekday: int

@app.post("/predict")
async def predict_demand(features: BikeFeatures):
    df = pd.DataFrame([features.dict()])
    log_pred = model.predict(df)[0]
    count_pred = np.expm1(log_pred)  # Reverse log transform
    return {"predicted_count": int(count_pred)}

@app.get("/")
async def root():
    return {"message": "Bike Sharing Demand Prediction API"}
