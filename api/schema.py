from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=100, description="Months with the company")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly bill in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charged to date")
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    InternetService: str = Field(..., description="DSL, Fiber optic, No")
    PaymentMethod: str = Field(..., description="Payment method used")
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, No phone service")
    OnlineSecurity: str = Field(..., description="Yes, No, No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, No internet service")
    TechSupport: str = Field(..., description="Yes, No, No internet service")
    StreamingTV: str = Field(..., description="Yes, No, No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, No internet service")
    PaperlessBilling: str = Field(..., description="Yes or No")

class ChurnPrediction(BaseModel):
    churn_probability: float = Field(..., description="0-1 probability of churn")
    churn_prediction: str = Field(..., description="High risk or Low risk")
    risk_score: str = Field(..., description="Low, Medium, High, or Critical")
    confidence: str = Field(..., description="Model confidence level")