from fastapi import FastAPI

app = FastAPI(
    title="Customer Churn LTV Engine API",
    description="API for customer churn prediction",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn LTV Engine API is running"
    }


@app.post("/predict")
def predict():
    return {
        "message": "Prediction endpoint is working"
    }


@app.post("/predict_batch")
def predict_batch():
    return {
        "message": "Batch prediction endpoint is working"
    }