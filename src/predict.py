"""
Prediction module for gas thermal conductivity.

Provides:
- Single compound prediction from SMILES + T
- Batch prediction
- Temperature sweep for a compound
- Applicability domain check
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
        
        # Training data statistics for applicability domain
        self._load_training_stats(model_dir)
    
    def _load_training_stats(self, model_dir: str):
        """Load training data stats for domain of applicability."""
        try:
            cv_df = pd.read_csv(os.path.join(model_dir, "cv_predictions.csv"))
            from src.descriptors import compute_descriptors_batch
            desc_df = compute_descriptors_batch(cv_df)
            X_train = desc_df[self.feature_names].values
            X_scaled = self.scaler.transform(X_train)
            
            self.train_mean = np.mean(X_scaled, axis=0)
            self.train_std = np.std(X_scaled, axis=0) + 1e-10
            
            # Leverage threshold (Williams plot)
            n, p = X_scaled.shape
            self.hat_warning = 3 * p / n
            
            self.domain_available = True
        except Exception:
            self.domain_available = False
    
    def predict(
        self, 
        smiles: str, 
        temperature_K: float = 300.0,
        check_domain: bool = True,
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
        
        Returns
        -------
        dict with keys:
            - thermal_conductivity_W_mK: predicted λ in W/(m·K)
            - temperature_K: input temperature
            - smiles: input SMILES
            - in_domain: bool, whether prediction is within model domain
            - domain_warning: str or None
            - model_uncertainty_pct: estimated prediction uncertainty
        """
        # Compute descriptors
        desc = compute_descriptors(smiles, temperature_K)
        X = np.array([[desc[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Predict (model predicts log(λ))
        y_log_pred = self.model.predict(X_scaled)[0]
        y_pred = np.exp(y_log_pred)
        
        # Domain check
        in_domain = True
        domain_warning = None
        
        if check_domain and self.domain_available:
            in_domain, domain_warning = self._check_domain(X_scaled[0])
        
        # Uncertainty estimate (based on CV MAPE + domain distance)
        base_uncertainty = self.metrics.get("MAPE_percent", 5.0)
        if not in_domain:
            base_uncertainty *= 2.0  # double uncertainty for out-of-domain
        
        return {
            "thermal_conductivity_W_mK": float(y_pred),
            "temperature_K": temperature_K,
            "smiles": smiles,
            "in_domain": in_domain,
            "domain_warning": domain_warning,
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
                    "model_uncertainty_pct": None,
                    "error": str(e),
                }
            results.append(res)
        
        return pd.DataFrame(results)
    
    def _check_domain(self, x_scaled: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check if a sample is within the model's applicability domain.
        
        Uses standardized distance from training centroid.
        """
        # Z-score distance
        z_scores = np.abs((x_scaled - self.train_mean) / self.train_std)
        max_z = np.max(z_scores)
        max_z_feature = self.feature_names[np.argmax(z_scores)]
        
        if max_z > 3.0:
            return False, (
                f"Out of domain: feature '{max_z_feature}' has z-score {max_z:.1f} "
                f"(threshold: 3.0). Prediction reliability is reduced."
            )
        
        return True, None


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
    print("=" * 70)
    print(f"{'Compound':<15} {'T (K)':<8} {'λ pred':>10} {'Domain':>8} {'±%':>6}")
    print("-" * 70)
    
    for name, smi, T in test_cases:
        r = predictor.predict(smi, T)
        domain = "✓" if r["in_domain"] else "✗"
        print(f"{name:<15} {T:<8} {r['thermal_conductivity_W_mK']:>10.4f} {domain:>8} "
              f"{r['model_uncertainty_pct']:>5.1f}%")
