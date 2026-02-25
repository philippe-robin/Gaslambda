"""
Prediction module for gas thermal conductivity.

Provides:
- Single compound prediction from SMILES + T
- Batch prediction
- Temperature sweep for a compound
- Applicability domain check (z-score + leverage + Mahalanobis)

Applicability domain methods:
1. Z-score: detects individual features far from training range
2. Leverage (Williams plot): detects unusual feature combinations
3. Mahalanobis distance: full multivariate distance from training centroid
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.descriptors import compute_descriptors, FEATURE_NAMES


class ThermalConductivityPredictor:
    """
    QSPR model for predicting gas-phase thermal conductivity of organic compounds.

    Based on XGBoost trained on curated DIPPR/literature data with RDKit descriptors.
    Valid for organic compounds at 1 atm, typically 250-600 K.

    Usage
    -----
    >>> predictor = ThermalConductivityPredictor()
    >>> result = predictor.predict("CCO", temperature_K=400)
    >>> print(f"λ = {result['thermal_conductivity_W_mK']:.4f} W/(m·K)")
    """

    def __init__(self, model_dir: str = "models"):
        """Load trained model and artifacts."""
        self.model = joblib.load(os.path.join(model_dir, "xgb_thermal_conductivity.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

        with open(os.path.join(model_dir, "feature_names.json")) as f:
            self.feature_names = json.load(f)

        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.metrics = json.load(f)

        # Load pre-computed training statistics for applicability domain
        self._load_training_stats(model_dir)

    def _load_training_stats(self, model_dir: str):
        """Load training data stats for domain of applicability."""
        stats_path = os.path.join(model_dir, "training_stats.json")

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)

            self.train_mean = np.array(stats["train_mean"])
            self.train_std = np.array(stats["train_std"])
            self.hat_warning = stats["hat_warning"]
            self.n_train = stats["n_train"]
            self.n_features = stats["n_features"]
            self.smearing_factor = stats.get("smearing_factor", 1.0)

            # Leverage (Williams plot)
            self.leverage_available = stats.get("leverage_available", False)
            if self.leverage_available:
                self.XtX_inv = np.array(stats["XtX_inv"])

            # Mahalanobis distance
            self.mahalanobis_available = stats.get("mahalanobis_available", False)
            if self.mahalanobis_available:
                self.cov_inv = np.array(stats["cov_inv"])
                self.mahalanobis_threshold = stats["mahalanobis_threshold"]

            self.domain_available = True
        else:
            # Fallback: recompute from CV predictions (legacy mode)
            self._load_training_stats_legacy(model_dir)

    def _load_training_stats_legacy(self, model_dir: str):
        """Legacy: recompute stats from CV predictions (for backward compatibility)."""
        try:
            cv_df = pd.read_csv(os.path.join(model_dir, "cv_predictions.csv"))
            from src.descriptors import compute_descriptors_batch
            desc_df = compute_descriptors_batch(cv_df)
            X_train = desc_df[self.feature_names].values
            X_scaled = self.scaler.transform(X_train)

            self.train_mean = np.mean(X_scaled, axis=0)
            self.train_std = np.std(X_scaled, axis=0) + 1e-10
            self.n_train, self.n_features = X_scaled.shape
            self.hat_warning = 3 * self.n_features / self.n_train
            self.smearing_factor = 1.0

            self.leverage_available = False
            self.mahalanobis_available = False
            self.domain_available = True
        except Exception as e:
            print(f"Warning: could not load training stats: {e}")
            self.domain_available = False

    def predict(
        self,
        smiles: str,
        temperature_K: float = 300.0,
        check_domain: bool = True,
        apply_bias_correction: bool = False,
    ) -> dict:
        """
        Predict gas thermal conductivity for a single compound.

        Parameters
        ----------
        smiles : str
            SMILES of the organic compound.
        temperature_K : float
            Temperature in Kelvin (typical range: 250-600 K).
        check_domain : bool
            Whether to check applicability domain.
        apply_bias_correction : bool
            Whether to apply Duan smearing correction for log-transform bias.
            Default False for backward compatibility; the correction is usually
            small (< 1%) for well-behaved models.

        Returns
        -------
        dict with keys:
            - thermal_conductivity_W_mK: predicted λ in W/(m·K)
            - temperature_K: input temperature
            - smiles: input SMILES
            - in_domain: bool, whether prediction is within model domain
            - domain_warning: str or None
            - domain_details: dict with z-score, leverage, Mahalanobis info
            - model_uncertainty_pct: estimated prediction uncertainty
        """
        # Compute descriptors
        desc = compute_descriptors(smiles, temperature_K)
        X = np.array([[desc[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # Predict (model predicts log(λ))
        y_log_pred = self.model.predict(X_scaled)[0]
        y_pred = np.exp(y_log_pred)

        # Apply bias correction if requested
        if apply_bias_correction and hasattr(self, "smearing_factor"):
            y_pred *= self.smearing_factor

        # Domain check
        in_domain = True
        domain_warning = None
        domain_details = {}

        if check_domain and self.domain_available:
            in_domain, domain_warning, domain_details = self._check_domain(X_scaled[0])

        # Uncertainty estimate
        base_uncertainty = self.metrics.get("MAPE_percent", 5.0)
        if not in_domain:
            # Scale uncertainty by how far out of domain we are
            severity = domain_details.get("severity", 2.0)
            base_uncertainty *= severity

        return {
            "thermal_conductivity_W_mK": float(y_pred),
            "temperature_K": temperature_K,
            "smiles": smiles,
            "in_domain": in_domain,
            "domain_warning": domain_warning,
            "domain_details": domain_details,
            "model_uncertainty_pct": float(base_uncertainty),
        }

    def predict_temperature_sweep(
        self,
        smiles: str,
        T_min: float = 250.0,
        T_max: float = 600.0,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """Predict λ over a temperature range."""
        temperatures = np.linspace(T_min, T_max, n_points)
        results = []

        for T in temperatures:
            res = self.predict(smiles, T, check_domain=True)
            results.append(res)

        return pd.DataFrame(results)

    def predict_batch(
        self,
        smiles_list: List[str],
        temperatures: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Predict for multiple compounds."""
        if temperatures is None:
            temperatures = [300.0] * len(smiles_list)

        results = []
        for smi, T in zip(smiles_list, temperatures):
            try:
                res = self.predict(smi, T)
                res["error"] = None
            except Exception as e:
                res = {
                    "thermal_conductivity_W_mK": None,
                    "temperature_K": T,
                    "smiles": smi,
                    "in_domain": False,
                    "domain_warning": str(e),
                    "domain_details": {},
                    "model_uncertainty_pct": None,
                    "error": str(e),
                }
            results.append(res)

        return pd.DataFrame(results)

    def _check_domain(self, x_scaled: np.ndarray) -> Tuple[bool, Optional[str], dict]:
        """
        Check if a sample is within the model's applicability domain.

        Uses three complementary methods:
        1. Z-score: detects individual features far from training range
        2. Leverage (Williams plot): detects unusual feature *combinations*
           relative to the training design matrix
        3. Mahalanobis distance: full multivariate distance accounting for
           feature correlations

        Returns
        -------
        in_domain : bool
        warning : str or None
        details : dict with diagnostic info
        """
        warnings = []
        details = {}
        severity = 1.0  # multiplier for uncertainty

        # --- 1. Z-score check (univariate) ---
        z_scores = np.abs((x_scaled - self.train_mean) / self.train_std)
        max_z = float(np.max(z_scores))
        max_z_idx = int(np.argmax(z_scores))
        max_z_feature = self.feature_names[max_z_idx]

        details["max_z_score"] = max_z
        details["max_z_feature"] = max_z_feature

        if max_z > 3.0:
            warnings.append(
                f"Z-score: feature '{max_z_feature}' = {max_z:.1f} (threshold: 3.0)"
            )
            severity = max(severity, 1.5 + (max_z - 3.0) * 0.5)

        # --- 2. Leverage / Williams plot (multivariate) ---
        if self.leverage_available:
            # h_new = x' (X'X)^{-1} x
            leverage = float(x_scaled @ self.XtX_inv @ x_scaled.T)
            details["leverage"] = leverage
            details["leverage_threshold"] = self.hat_warning

            if leverage > self.hat_warning:
                warnings.append(
                    f"Leverage: h = {leverage:.3f} (threshold h* = {self.hat_warning:.3f})"
                )
                severity = max(severity, 1.5 + (leverage / self.hat_warning - 1.0))

        # --- 3. Mahalanobis distance (multivariate, correlation-aware) ---
        if self.mahalanobis_available:
            diff = x_scaled - self.train_mean
            mahal_sq = float(diff @ self.cov_inv @ diff.T)
            mahal = np.sqrt(max(mahal_sq, 0.0))
            mahal_threshold = np.sqrt(self.mahalanobis_threshold)

            details["mahalanobis_distance"] = mahal
            details["mahalanobis_threshold"] = mahal_threshold

            if mahal > mahal_threshold:
                warnings.append(
                    f"Mahalanobis: D = {mahal:.1f} (threshold: {mahal_threshold:.1f})"
                )
                severity = max(severity, 1.5 + (mahal / mahal_threshold - 1.0) * 0.5)

        # --- Aggregate ---
        in_domain = len(warnings) == 0
        details["severity"] = float(severity)

        if warnings:
            domain_warning = "Outside applicability domain. " + "; ".join(warnings)
        else:
            domain_warning = None

        return in_domain, domain_warning, details


# Convenience function
def predict_lambda(smiles: str, temperature_K: float = 300.0, model_dir: str = "models") -> float:
    """
    Quick prediction of gas thermal conductivity.

    Returns λ in W/(m·K).
    """
    predictor = ThermalConductivityPredictor(model_dir)
    result = predictor.predict(smiles, temperature_K)
    return result["thermal_conductivity_W_mK"]


if __name__ == "__main__":
    # Demo predictions
    predictor = ThermalConductivityPredictor()

    test_cases = [
        ("Methane", "C", 300),
        ("Ethanol", "CCO", 400),
        ("Benzene", "c1ccccc1", 450),
        ("Acetone", "CC(=O)C", 350),
        ("Chloroform", "ClC(Cl)Cl", 300),
    ]

    print("Gas Thermal Conductivity Predictions")
    print("=" * 80)
    print(f"{'Compound':<15} {'T (K)':<8} {'λ pred':>10} {'Domain':>8} {'±%':>6}  {'Details':s}")
    print("-" * 80)

    for name, smi, T in test_cases:
        r = predictor.predict(smi, T)
        domain = "✓" if r["in_domain"] else "✗"
        details = ""
        if r["domain_details"]:
            d = r["domain_details"]
            details = f"z={d.get('max_z_score', 0):.1f}"
            if "leverage" in d:
                details += f" h={d['leverage']:.3f}"
            if "mahalanobis_distance" in d:
                details += f" D={d['mahalanobis_distance']:.1f}"
        print(f"{name:<15} {T:<8} {r['thermal_conductivity_W_mK']:>10.4f} {domain:>8} "
              f"{r['model_uncertainty_pct']:>5.1f}%  {details}")
