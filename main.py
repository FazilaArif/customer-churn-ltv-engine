from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)

# Load trained model
model = joblib.load("Model/Churn_pipeline.pkl")


class CustomerData(BaseModel):
    tenure_months: float
    monthly_charges: float
    contract: str
    internet_service: str
    payment_method: str
    senior_citizen: int
    partner: str
    dependents: str


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0][1]
    else:
        probability = None

    return {
        "churn_prediction": int(prediction),
        "churn_probability": (
            round(float(probability), 4)
            if probability is not None
            else None
        )
    }


@app.post("/predict_batch")
def predict_batch(customers: list[CustomerData]):

    if not customers:
        raise HTTPException(
            status_code=400,
            detail="Customer list cannot be empty"
        )

    data = pd.DataFrame(
        [customer.model_dump() for customer in customers]
    )

    predictions = model.predict(data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[:, 1]
    else:
        probabilities = [None] * len(predictions)

    results = []

    for prediction, probability in zip(predictions, probabilities):
        results.append({
            "churn_prediction": int(prediction),
            "churn_probability": (
                round(float(probability), 4)
                if probability is not None
                else None
            )
        })

    return {
        "count": len(results),
        "predictions": results
    }