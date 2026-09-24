import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schema import CustomerFeatures, ChurnPrediction

MODEL_PATH    = Path(__file__).parent.parent / 'models' / 'churn_model.pkl'
FEATURES_PATH = Path(__file__).parent.parent / 'models' / 'feature_names.pkl'

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost.",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

def preprocess(customer: CustomerFeatures) -> pd.DataFrame:
    d = customer.dict()
    
    # Binary conversions
    d['Partner']          = 1 if d['Partner'] == 'Yes' else 0
    d['Dependents']       = 1 if d['Dependents'] == 'Yes' else 0
    d['PhoneService']     = 1 if d['PhoneService'] == 'Yes' else 0
    d['PaperlessBilling'] = 1 if d['PaperlessBilling'] == 'Yes' else 0
    d['gender']           = 1 if d['gender'] == 'Male' else 0
    
    # Engineered features
    service_fields = ['PhoneService','MultipleLines','InternetService',
                      'OnlineSecurity','OnlineBackup','DeviceProtection',
                      'TechSupport','StreamingTV','StreamingMovies']
    d['num_services']      = sum(1 for f in service_fields if d.get(f) not in ['No', 0])
    d['charges_per_month'] = round(d['TotalCharges'] / max(d['tenure'], 1), 2)
    d['is_monthly']        = 1 if d['Contract'] == 'Month-to-month' else 0
    d['is_new_customer']   = 1 if d['tenure'] <= 12 else 0
    d['above_avg_charges'] = 1 if d['MonthlyCharges'] > 64.76 else 0

    # One-hot encode multi-value categoricals
    multi_cols = {
        'MultipleLines':    ['No phone service', 'Yes'],
        'InternetService':  ['Fiber optic', 'No'],
        'OnlineSecurity':   ['No internet service', 'Yes'],
        'OnlineBackup':     ['No internet service', 'Yes'],
        'DeviceProtection': ['No internet service', 'Yes'],
        'TechSupport':      ['No internet service', 'Yes'],
        'StreamingTV':      ['No internet service', 'Yes'],
        'StreamingMovies':  ['No internet service', 'Yes'],
        'Contract':         ['One year', 'Two year'],
        'PaymentMethod':    ['Credit card (automatic)',
                             'Electronic check', 'Mailed check']
    }
    for col, values in multi_cols.items():
        for val in values:
            key = f"{col}_{val}"
            d[key] = 1 if d.get(col) == val else 0
        d.pop(col, None)
    
    df = pd.DataFrame([d])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df

def risk_label(prob: float) -> str:
    if prob < 0.25: return "Low"
    if prob < 0.50: return "Medium"
    if prob < 0.75: return "High"
    return "Critical"

def confidence_label(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist > 0.35: return "High"
    if dist > 0.15: return "Medium"
    return "Low"

@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost churn classifier v1.0"}

@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures):
    try:
        X    = preprocess(customer)
        prob = float(model.predict_proba(X)[0][1])
        return ChurnPrediction(
            churn_probability=round(prob, 4),
            churn_prediction="High risk" if prob >= 0.5 else "Low risk",
            risk_score=risk_label(prob),
            confidence=confidence_label(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))