"""
Train XGBoost model for gas thermal conductivity prediction.

Pipeline:
1. Load reference dataset
2. Compute molecular descriptors
3. Train XGBoost with GroupKFold cross-validation (grouped by compound)
4. Evaluate and save model + metrics + training statistics

Model choice rationale:
- XGBoost handles small datasets well (better than deep learning for n < 500)
- Built-in feature importance for interpretability
- Handles mixed feature types (continuous + discrete)
- Good generalization with proper regularization

Key design decisions:
- GroupKFold by SMILES: avoids data leakage when the same compound appears
  at multiple temperatures. This gives honest generalization estimates for
  *unseen compounds*, not just unseen (compound, T) pairs.
- Log-transform target: λ spans ~1 order of magnitude, log stabilizes variance.
- Bias correction: exp(log-pred) underestimates E[y] due to Jensen's inequality;
  we apply the Duan smearing correction using CV residuals.

References:
- Gharagheizi et al., Ind. Eng. Chem. Res., 2013, 52, 7165-7174
- Chen & Guestrin, "XGBoost", KDD 2016
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
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

    # Groups for GroupKFold: all temperatures of the same compound stay together
    groups = df.loc[desc_df.index, "smiles"].values

    return X, y, y_log, groups, df.loc[desc_df.index].reset_index(drop=True), desc_df.reset_index(drop=True)


def train_model(X, y_log, groups, n_splits=5):
    """
    Train XGBoost with GroupKFold cross-validation.

    GroupKFold ensures that all data points for the same compound (same SMILES)
    are in the same fold. This prevents data leakage and gives a realistic
    estimate of performance on *unseen compounds*.

    Also runs standard KFold for comparison, so users can see the difference.

    Returns trained model, scaler, CV predictions (GroupKFold), and comparison metrics.
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

    # --- GroupKFold CV (honest evaluation for unseen compounds) ---
    n_unique = len(set(groups))
    actual_splits = min(n_splits, n_unique)
    gkf = GroupKFold(n_splits=actual_splits)
    y_pred_group_cv = cross_val_predict(model, X_scaled, y_log, cv=gkf, groups=groups)

    # --- Standard KFold CV (for comparison — shows data leakage effect) ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_kfold_cv = cross_val_predict(model, X_scaled, y_log, cv=kf)

    # Train final model on all data
    model.fit(X_scaled, y_log)

    return model, scaler, y_pred_group_cv, y_pred_kfold_cv


def compute_bias_correction(y_true_log, y_pred_cv_log):
    """
    Compute Duan smearing factor for log-transform bias correction.

    When we predict log(λ) and back-transform with exp(), we get the
    *median* prediction, not the *mean*. For unbiased mean estimates:
        E[λ] = exp(log_pred) * smearing_factor

    The Duan smearing estimator is: mean(exp(residuals))
    This is more robust than the parametric correction exp(σ²/2).
    """
    residuals = y_true_log - y_pred_cv_log
    smearing_factor = float(np.mean(np.exp(residuals)))
    parametric_correction = float(np.exp(np.var(residuals) / 2.0))

    return smearing_factor, parametric_correction


def evaluate(y_true, y_pred_cv_log, label=""):
    """Compute evaluation metrics from log-scale CV predictions."""
    # Back-transform from log (no bias correction for CV evaluation — fair comparison)
    y_pred_cv = np.exp(y_pred_cv_log)

    # Metrics on original scale
    mae = mean_absolute_error(y_true, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_cv))
    r2 = r2_score(y_true, y_pred_cv)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred_cv) / y_true)) * 100

    # Median APE (more robust than MAPE to outliers)
    mdape = np.median(np.abs((y_true - y_pred_cv) / y_true)) * 100

    # Max error
    max_err = np.max(np.abs(y_true - y_pred_cv))
    max_pct = np.max(np.abs((y_true - y_pred_cv) / y_true)) * 100

    # 90th percentile error
    p90_pct = np.percentile(np.abs((y_true - y_pred_cv) / y_true) * 100, 90)

    metrics = {
        "MAE_W_mK": float(f"{mae:.6f}"),
        "RMSE_W_mK": float(f"{rmse:.6f}"),
        "R2": float(f"{r2:.4f}"),
        "MAPE_percent": float(f"{mape:.2f}"),
        "MdAPE_percent": float(f"{mdape:.2f}"),
        "P90_error_percent": float(f"{p90_pct:.1f}"),
        "Max_error_W_mK": float(f"{max_err:.6f}"),
        "Max_error_percent": float(f"{max_pct:.1f}"),
        "n_samples": len(y_true),
    }

    return metrics, y_pred_cv


def compute_training_stats(X_scaled, feature_names):
    """
    Compute and return training set statistics for applicability domain.

    Includes:
    - Feature means and standard deviations (for z-score check)
    - Leverage threshold (for Williams plot)
    - Covariance inverse (for Mahalanobis distance)
    """
    n, p = X_scaled.shape

    stats = {
        "train_mean": X_scaled.mean(axis=0).tolist(),
        "train_std": (X_scaled.std(axis=0) + 1e-10).tolist(),
        "n_train": n,
        "n_features": p,
        "hat_warning": float(3 * p / n),  # Williams plot threshold h* = 3p/n
    }

    # Regularized covariance for numerical stability (Ledoit-Wolf shrinkage)
    from sklearn.covariance import LedoitWolf
    try:
        lw = LedoitWolf().fit(X_scaled)
        cov_reg = lw.covariance_
        cov_inv = np.linalg.inv(cov_reg)
        stats["cov_inv"] = cov_inv.tolist()
        stats["mahalanobis_available"] = True
        stats["ledoit_wolf_shrinkage"] = float(lw.shrinkage_)
    except (np.linalg.LinAlgError, ValueError):
        stats["mahalanobis_available"] = False

    # Mahalanobis distance threshold: chi2(p, 0.99)
    from scipy import stats as scipy_stats
    stats["mahalanobis_threshold"] = float(scipy_stats.chi2.ppf(0.99, df=p))

    # Hat matrix: H = X (X'X)^{-1} X'
    # Use regularized (X'X + εI)^{-1} for stability
    try:
        XtX = X_scaled.T @ X_scaled
        XtX_reg = XtX + 1e-6 * np.eye(p)  # Tikhonov regularization
        XtX_inv = np.linalg.inv(XtX_reg)
        leverages = np.diag(X_scaled @ XtX_inv @ X_scaled.T)
        stats["train_leverages_max"] = float(np.max(leverages))
        stats["train_leverages_mean"] = float(np.mean(leverages))
        stats["XtX_inv"] = XtX_inv.tolist()
        stats["leverage_available"] = True
    except np.linalg.LinAlgError:
        stats["leverage_available"] = False

    return stats


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
    print("\n[1/5] Loading data and computing descriptors...")
    X, y, y_log, groups, df, desc_df = load_and_prepare_data()
    n_unique_compounds = len(set(groups))
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Unique compounds: {n_unique_compounds}")

    # 2. Train with both CV strategies
    print("\n[2/5] Training XGBoost with GroupKFold + KFold CV...")
    model, scaler, y_pred_group_log, y_pred_kfold_log = train_model(X, y_log, groups)

    # 3. Evaluate both strategies
    print("\n[3/5] Evaluating...")

    metrics_group, y_pred_group = evaluate(y, y_pred_group_log, "GroupKFold")
    metrics_kfold, y_pred_kfold = evaluate(y, y_pred_kfold_log, "KFold")

    print(f"\n  {'Metric':<20s} {'GroupKFold':>12s} {'KFold':>12s}  {'Note':s}")
    print(f"  {'-'*65}")
    print(f"  {'R²':<20s} {metrics_group['R2']:>12.4f} {metrics_kfold['R2']:>12.4f}  ← GroupKFold = honest")
    print(f"  {'MAE (W/(m·K))':<20s} {metrics_group['MAE_W_mK']:>12.6f} {metrics_kfold['MAE_W_mK']:>12.6f}")
    print(f"  {'RMSE (W/(m·K))':<20s} {metrics_group['RMSE_W_mK']:>12.6f} {metrics_kfold['RMSE_W_mK']:>12.6f}")
    print(f"  {'MAPE (%)':<20s} {metrics_group['MAPE_percent']:>12.2f} {metrics_kfold['MAPE_percent']:>12.2f}")
    print(f"  {'MdAPE (%)':<20s} {metrics_group['MdAPE_percent']:>12.2f} {metrics_kfold['MdAPE_percent']:>12.2f}")
    print(f"  {'P90 error (%)':<20s} {metrics_group['P90_error_percent']:>12.1f} {metrics_kfold['P90_error_percent']:>12.1f}")
    print(f"  {'Max error (%)':<20s} {metrics_group['Max_error_percent']:>12.1f} {metrics_kfold['Max_error_percent']:>12.1f}")

    if metrics_group["R2"] < metrics_kfold["R2"] - 0.05:
        print(f"\n  ⚠️  GroupKFold R² is significantly lower than KFold R².")
        print(f"      This confirms data leakage in standard KFold: the model was")
        print(f"      seeing the same compound at different T in both train and test.")
        print(f"      GroupKFold gives the true generalization performance for unseen compounds.")

    # Bias correction
    smearing, parametric = compute_bias_correction(y_log, y_pred_group_log)
    print(f"\n  Log-transform bias correction:")
    print(f"    Duan smearing factor:     {smearing:.4f}")
    print(f"    Parametric exp(σ²/2):     {parametric:.4f}")
    print(f"    (values close to 1.0 = minimal bias)")

    # Feature importance
    feat_imp = get_feature_importance(model, FEATURE_NAMES)
    print(f"\n  Top 10 features:")
    for fi in feat_imp[:10]:
        print(f"    {fi['feature']:25s} {fi['importance']:.4f}")

    # 4. Compute and save training statistics for applicability domain
    print("\n[4/5] Computing training statistics for applicability domain...")
    X_scaled = scaler.transform(X)
    train_stats = compute_training_stats(X_scaled, FEATURE_NAMES)
    train_stats["smearing_factor"] = smearing
    train_stats["parametric_bias_correction"] = parametric

    # 5. Save everything
    print("\n[5/5] Saving model and artifacts...")
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/xgb_thermal_conductivity.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Save GroupKFold metrics as primary (honest evaluation)
    metrics_group["cv_method"] = "GroupKFold"
    metrics_group["n_unique_compounds"] = n_unique_compounds
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_group, f, indent=2)

    # Save comparison metrics
    comparison = {
        "GroupKFold": metrics_group,
        "KFold": metrics_kfold,
        "note": "GroupKFold groups by SMILES to avoid data leakage. "
                "KFold shown for comparison only.",
    }
    with open("models/metrics_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open("models/feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    with open("models/feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    # Save training stats for applicability domain
    with open("models/training_stats.json", "w") as f:
        json.dump(train_stats, f)

    # Save predictions for analysis (GroupKFold)
    results_df = df.copy()
    results_df["predicted_W_mK"] = y_pred_group
    results_df["error_pct"] = np.abs(y - y_pred_group) / y * 100
    results_df.to_csv("models/cv_predictions.csv", index=False)

    print("  Saved: models/xgb_thermal_conductivity.joblib")
    print("  Saved: models/scaler.joblib")
    print("  Saved: models/metrics.json              (GroupKFold — primary)")
    print("  Saved: models/metrics_comparison.json   (GroupKFold vs KFold)")
    print("  Saved: models/feature_importance.json")
    print("  Saved: models/training_stats.json       (for applicability domain)")
    print("  Saved: models/cv_predictions.csv")
    print("\n✓ Training complete!")

    return model, scaler, metrics_group


if __name__ == "__main__":
    main()
