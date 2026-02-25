# 🔬 GasLambda — Gas Thermal Conductivity Predictor

**QSPR/ML model for predicting gas-phase thermal conductivity (λ) of organic compounds from molecular structure.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gaslambda.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9+-green.svg)](https://www.rdkit.org/)

---

## Overview

GasLambda predicts the thermal conductivity of organic gases at low pressure (≈1 atm) as a function of temperature, using only the molecular structure (SMILES) as input. It combines:

- **RDKit molecular descriptors** — physics-informed features (degrees of freedom, polarity, molecular shape, Eucken factor proxy)
- **XGBoost gradient boosting** — robust ML for small datasets with built-in regularization
- **Applicability domain check** — 3-method detection (z-score, Williams plot, Mahalanobis distance)

### Performance (GroupKFold CV by compound)

| Metric | Value |
|--------|-------|
| R² | 0.857 |
| MAPE | 7.3% |
| MdAPE | 4.8% |
| MAE | 0.0016 W/(m·K) |
| Training compounds | 90 unique, 135 data points |
| Temperature range | 250–600 K |

> GroupKFold groups by SMILES to prevent data leakage — all temperatures of the same compound stay in the same fold. This gives honest generalization estimates for *unseen compounds*.

### Key features

- Single prediction, temperature sweep, and batch modes
- Interactive Streamlit web interface with Plotly visualizations
- Parity plot, error distribution, and feature importance analysis
- CSV export for all results
- Applicability domain detection (z-score + leverage + Mahalanobis)
- Duan smearing correction for log-transform bias

---

## Live Demo

👉 **[gaslambda.streamlit.app](https://gaslambda.streamlit.app)**

---

## Quick Start

### Installation

```bash
git clone https://github.com/philippe-robin/Gaslambda.git
cd Gaslambda

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Train the model

```bash
python data/build_dataset.py   # Build reference dataset
python src/train.py             # Train XGBoost model
```

### Run the web app

```bash
streamlit run app.py
```

### Python API

```python
from src.predict import ThermalConductivityPredictor

predictor = ThermalConductivityPredictor()

# Single prediction
result = predictor.predict("CCO", temperature_K=400)
print(f"λ(ethanol, 400K) = {result['thermal_conductivity_W_mK']:.4f} W/(m·K)")

# Temperature sweep
df = predictor.predict_temperature_sweep("c1ccccc1", T_min=300, T_max=600)

# Batch
results = predictor.predict_batch(
    ["C", "CC", "CCC", "CCCC"],
    temperatures=[300, 300, 300, 300]
)
```

---

## Project Structure

```
gaslambda/
├── app.py                          # Streamlit web application
├── requirements.txt
├── setup.py
├── LICENSE
├── data/
│   ├── build_dataset.py            # Reference data from DIPPR/Yaws/NIST
│   └── reference_thermal_conductivity.csv
├── src/
│   ├── __init__.py
│   ├── descriptors.py              # RDKit molecular descriptor computation
│   ├── train.py                    # Model training pipeline (GroupKFold)
│   └── predict.py                  # Prediction + applicability domain (3 methods)
└── models/
    ├── xgb_thermal_conductivity.joblib
    ├── scaler.joblib
    ├── metrics.json                # GroupKFold metrics (primary)
    ├── metrics_comparison.json     # GroupKFold vs KFold comparison
    ├── training_stats.json         # Covariance, leverage, thresholds
    ├── feature_importance.json
    ├── feature_names.json
    └── cv_predictions.csv
```

---

## Scientific Background

### Why QSPR for gas thermal conductivity?

Classical methods (Eucken, Chung, Stiel-Thodos) rely on critical properties and acentric factors that may not be available for novel compounds. QSPR predicts directly from molecular structure.

### Descriptor rationale

The 41 descriptors are chosen based on the physics of gas thermal conductivity:

| Category | Descriptors | Physical basis |
|----------|------------|----------------|
| **Temperature** | T, T^0.7, T/MW | λ ∝ T^n for polyatomic gases |
| **Molecular size** | MW, heavy atoms, 1/√MW | Kinetic theory: λ ∝ 1/√M |
| **Degrees of freedom** | DOF_vib, Cv_R, Eucken factor | Eucken correction: λ = η·(Cv + 9R/4M) |
| **Polarity** | TPSA, LogP, MolMR, H-bond donors/acceptors | Intermolecular forces affect mean free path |
| **Topology** | Kappa, Chi, Wiener index, BertzCT | Molecular shape → collision cross-section |
| **Atom types** | C, H, O, N, S, F, Cl, Br counts | Element-specific contributions |
| **Functionality** | Rings, aromaticity, sp3 fraction, bond types | Rigidity affects vibrational modes |

### Data sources

- **DIPPR 801** — AIChE recommended values (primary source)
- **Yaws' Handbook** — C.L. Yaws, *Transport Properties of Chemicals and Hydrocarbons*
- **NIST WebBook** — webbook.nist.gov

### Limitations

- **Low-pressure only** (~1 atm). High-pressure correction not included.
- **Organic compounds only**. Not validated for inorganics, metals, or ionic species.
- **Small training set** (135 points / 90 compounds). Expand with DIPPR 801 or TDE licensed data for production use.
- **Temperature range**: best within 250–600 K.
- **Applicability domain**: always check the domain flag before using predictions.

---

## Extending the Model

### Adding data

Add entries to `REFERENCE_DATA` in `data/build_dataset.py`, then retrain:

```python
("MyCompound", "C(=O)OCC", 300, 0.0142, "LITERATURE"),
```

### Adding descriptors

Extend `compute_descriptors()` in `src/descriptors.py` and add the new name to `FEATURE_NAMES`.

### Alternative models

Replace XGBoost in `src/train.py` with any scikit-learn compatible regressor (Random Forest, SVR, neural network).

---

## References

1. Poling, B.E., Prausnitz, J.M. & O'Connell, J.P. *The Properties of Gases and Liquids*, 5th ed., McGraw-Hill, 2001. Chapters 10–11.
2. Gharagheizi, F. et al. *Ind. Eng. Chem. Res.*, 2013, 52, 7165–7174. DOI: 10.1021/ie4005008
3. Chen, T. & Guestrin, C. "XGBoost: A Scalable Tree Boosting System", *KDD*, 2016.
4. DIPPR 801 Database, AIChE/BYU.
5. Yaws, C.L. *Transport Properties of Chemicals and Hydrocarbons*, 2nd ed., Elsevier, 2014.

---

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Alysophil SAS** — AI-driven flow chemistry
[www.alysophil.com](https://www.alysophil.com)
