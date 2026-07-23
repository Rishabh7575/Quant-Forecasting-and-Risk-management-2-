from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for serving customer churn predictions.",
    version="1.0.0"
)

class CustomerData(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    # Add other relevant features here

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    Predict the churn probability for a given customer.
    """
    # In a real scenario, we would load the trained model and make a prediction here.
    # For now, this is a mock implementation.
    
    # Mock prediction logic based on a simple threshold (for demonstration)
    risk_score = min(data.monthly_charges / max(data.tenure, 1) * 0.1, 0.99)
    will_churn = risk_score > 0.5
    
    return {
        "customer_id": data.customer_id,
        "churn_prediction": will_churn,
        "risk_score": round(risk_score * 100, 2)
    }
