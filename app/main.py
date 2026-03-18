from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Welcome to the Iris Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        },
        "deployed_on": "Render",
        "status": "live"
    }

@app.get("/predict")
def predict_get(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        return PredictionOutput(
            species=CLASS_NAMES[prediction],
            species_id=int(prediction),
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    species: str
    species_id: int
    confidence: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):

    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        return PredictionOutput(
            species=CLASS_NAMES[pred],
            species_id=int(pred),
            confidence=float(prob[pred])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))