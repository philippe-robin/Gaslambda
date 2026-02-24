"""
Train XGBoost model for gas thermal conductivity prediction.

Pipeline:
1. Load reference dataset
2. Compute molecular descriptors
3. Train XGBoost with cross-validation
4. Evaluate and save model + metrics

Model choice rationale:
- XGBoost handles small datasets well (better than deep learning for n < 500)
- Built-in feature importance for interpretability
- Handles mixed feature types (continuous + discrete)
- Good generalization with proper regularization

References:
- Gharagheizi et al., Ind. Eng. Chem. Res., 2013, 52, 7165-7174
- Chen & Guestrin, "XGBoost", KDD 2016
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add parent to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.descriptors import compute_descriptors_batch, FEATURE_NAMES


def load_and_prepare_data(data_path: str = "data/reference_thermal_conductivity.csv"):
    """Load dataset and compute descriptors."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points")
    
    # Compute descriptors
    desc_df = compute_descriptors_batch(df)
    
    # Merge
    X = desc_df[FEATURE_NAMES].values
    y = df.loc[desc_df.index, "thermal_conductivity_W_mK"].values
    
    # Log-transform target (λ spans ~1 order of magnitude, log helps)
    y_log = np.log(y)
    
    return X, y, y_log, df.loc[desc_df.index].reset_index(drop=True), desc_df.reset_index(drop=True)


def train_model(X, y_log, n_splits=5):
    """
    Train XGBoost with K-fold cross-validation.
    
    Returns trained model, scaler, and cross-validation predictions.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # XGBoost parameters (tuned for small dataset)
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,      # L1 regularization
        "reg_lambda": 1.0,     # L2 regularization
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Cross-validation predictions (for honest evaluation)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_scaled, y_log, cv=kf)
    
    # Train final model on all data
    model.fit(X_scaled, y_log)
    
    return model, scaler, y_pred_cv


def evaluate(y_true, y_pred_log, y_true_log, y_pred_cv_log):
    """Compute evaluation metrics."""
    # Back-transform from log
    y_pred_cv = np.exp(y_pred_cv_log)
    
    # Metrics on original scale
    mae = mean_absolute_error(y_true, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_cv))
    r2 = r2_score(y_true, y_pred_cv)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred_cv) / y_true)) * 100
    
    # Max error
    max_err = np.max(np.abs(y_true - y_pred_cv))
    max_pct = np.max(np.abs((y_true - y_pred_cv) / y_true)) * 100
    
    metrics = {
        "MAE_W_mK": float(f"{mae:.6f}"),
        "RMSE_W_mK": float(f"{rmse:.6f}"),
        "R2": float(f"{r2:.4f}"),
        "MAPE_percent": float(f"{mape:.2f}"),
        "Max_error_W_mK": float(f"{max_err:.6f}"),
        "Max_error_percent": float(f"{max_pct:.1f}"),
        "n_samples": len(y_true),
    }
    
    return metrics, y_pred_cv


def get_feature_importance(model, feature_names):
    """Extract and rank feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    result = []
    for i in indices:
        if importances[i] > 0.001:
            result.append({
                "feature": feature_names[i],
                "importance": float(f"{importances[i]:.4f}"),
            })
    return result


def main():
    """Full training pipeline."""
    print("=" * 60)
    print("Gas Thermal Conductivity Predictor - Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading data and computing descriptors...")
    X, y, y_log, df, desc_df = load_and_prepare_data()
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # 2. Train
    print("\n[2/4] Training XGBoost with 5-fold CV...")
    model, scaler, y_pred_cv_log = train_model(X, y_log)
    
    # 3. Evaluate
    print("\n[3/4] Evaluating...")
    metrics, y_pred_cv = evaluate(y, y_pred_cv_log, y_log, y_pred_cv_log)
    
    print(f"  Cross-validation results (5-fold):")
    print(f"    R²   = {metrics['R2']:.4f}")
    print(f"    MAE  = {metrics['MAE_W_mK']:.6f} W/(m·K)")
    print(f"    RMSE = {metrics['RMSE_W_mK']:.6f} W/(m·K)")
    print(f"    MAPE = {metrics['MAPE_percent']:.2f}%")
    print(f"    Max error = {metrics['Max_error_percent']:.1f}%")
    
    # Feature importance
    feat_imp = get_feature_importance(model, FEATURE_NAMES)
    print(f"\n  Top 10 features:")
    for fi in feat_imp[:10]:
        print(f"    {fi['feature']:25s} {fi['importance']:.4f}")
    
    # 4. Save
    print("\n[4/4] Saving model and artifacts...")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/xgb_thermal_conductivity.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open("models/feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)
    
    with open("models/feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    
    # Save predictions for analysis
    results_df = df.copy()
    results_df["predicted_W_mK"] = y_pred_cv
    results_df["error_pct"] = (
        np.abs(y - y_pred_cv) / y * 100
    )
    results_df.to_csv("models/cv_predictions.csv", index=False)
    
    print("  Saved: models/xgb_thermal_conductivity.joblib")
    print("  Saved: models/scaler.joblib")
    print("  Saved: models/metrics.json")
    print("  Saved: models/feature_importance.json")
    print("  Saved: models/cv_predictions.csv")
    print("\n✓ Training complete!")
    
    return model, scaler, metrics


if __name__ == "__main__":
    main()
