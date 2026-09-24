import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, RocCurveDisplay)
from xgboost import XGBClassifier

DATA_PATH   = Path(__file__).parent.parent / 'data' / 'churn_features.csv'
MODEL_PATH  = Path(__file__).parent.parent / 'models' / 'churn_model.pkl'
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
METRICS_PATH= Path(__file__).parent.parent / 'models' / 'metrics.json'
DOCS_PATH   = Path(__file__).parent.parent / 'docs'

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    print(f"Features: {X.shape[1]}, Samples: {len(X):,}")
    return X, y

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, evaluate, and print results for one model."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model, X_train, y_train,
                          cv=StratifiedKFold(5), scoring='roc_auc')
    
    print(f"{'='*40}")
    print(f"{name}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    print(f"  CV ROC-AUC:    {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")
    
    return auc, model

def plot_feature_importance(model, feature_names):
    """Plot top 15 most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances — Churn Prediction",
              fontsize=14, fontweight='bold')
    plt.barh(range(15),
             importances[indices][::-1],
             color='#185FA5', alpha=0.8)
    plt.yticks(range(15),
               [feature_names[i] for i in indices][::-1],
               fontsize=11)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(DOCS_PATH / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved feature_importance.png")

def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve — shows model performance visually."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                    color='#185FA5', lw=2)
    ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
    ax.set_title("ROC Curve — Churn Prediction Model",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(DOCS_PATH / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved roc_curve.png")

def train():
    X, y = load_data()
    
    # Split: 80% train, 20% test — stratified keeps churn ratio equal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    
    results = {}
    
    # 1. Baseline: logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    auc, _ = evaluate_model("Logistic Regression (baseline)",
                             lr, X_train_sc, X_test_sc, y_train, y_test)
    results['logistic_regression'] = round(auc, 4)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    auc, _ = evaluate_model("Random Forest",
                             rf, X_train, X_test, y_train, y_test)
    results['random_forest'] = round(auc, 4)
    
    # 3. XGBoost — usually the best on tabular data
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05,
                         max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=42,
                         eval_metric='logloss', verbosity=0,
                         scale_pos_weight=3)  # handles class imbalance
    auc, xgb_trained = evaluate_model("XGBoost",
                                       xgb, X_train, X_test, y_train, y_test)
    results['xgboost'] = round(auc, 4)
    
    print(f"{'='*40}")
    print("MODEL COMPARISON:")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    # Save the best model (XGBoost almost always wins)
    print("Saving XGBoost model...")
    joblib.dump(xgb_trained, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), 
                Path(__file__).parent.parent / 'models' / 'feature_names.pkl')
    
    # Save metrics to JSON for the model card
    metrics = {
        'model': 'XGBoost',
        'test_roc_auc': results['xgboost'],
        'comparison': results,
        'n_features': X.shape[1],
        'n_train_samples': len(X_train),
        'churn_rate': round(float(y.mean()), 4)
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate charts for README
    plot_feature_importance(xgb_trained, list(X.columns))
    plot_roc_curve(xgb_trained, X_test, y_test)
    
    print(f"Done. Best model saved to {MODEL_PATH}")
    return xgb_trained

if __name__ == '__main__':
    train()