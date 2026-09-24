import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data' / 'telco_churn.csv'
OUT_PATH  = Path(__file__).parent.parent / 'data' / 'churn_features.csv'

def load_raw():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df

def clean(df):
    # TotalCharges has some blank strings — convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill the ~11 missing TotalCharges with MonthlyCharges * tenure
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = (
        df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
    )
    
    # Drop customerID — it's just an identifier, not a feature
    df = df.drop(columns=['customerID'])
    
    # Convert target to binary: Yes=1, No=0
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

def engineer_features(df):
    # How many services does the customer use?
    # More services = more embedded = less likely to churn
    service_cols = ['PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    df['num_services'] = (df[service_cols] != 'No').sum(axis=1)
    
    # Charge per month relative to tenure
    df['charges_per_month'] = (
        df['TotalCharges'] / df['tenure'].replace(0, 1)
    ).round(2)
    
    # Is the customer on a month-to-month contract?
    # This is the single strongest predictor of churn
    df['is_monthly'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # Has the customer been with us less than 1 year?
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    
    # Is the customer paying more than average monthly?
    avg_charge = df['MonthlyCharges'].mean()
    df['above_avg_charges'] = (df['MonthlyCharges'] > avg_charge).astype(int)
    
    return df

def encode_categoricals(df):
    # Binary yes/no columns to 1/0
    binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Gender to 1/0
    df['gender'] = (df['gender'] == 'Male').astype(int)
    
    # Multi-value categoricals to one-hot encoded dummy variables
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity',
                  'OnlineBackup','DeviceProtection','TechSupport',
                  'StreamingTV','StreamingMovies','Contract',
                  'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    return df

def prepare():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    
    print(f"Final shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    return df

if __name__ == '__main__':
    prepare()