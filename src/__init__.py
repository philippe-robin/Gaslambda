# GasLambda - Gas Thermal Conductivity Predictor
# QSPR/ML model for organic compounds

from src.predict import ThermalConductivityPredictor, predict_lambda
from src.descriptors import compute_descriptors

__version__ = "1.0.0"
__all__ = ["ThermalConductivityPredictor", "predict_lambda", "compute_descriptors"]
