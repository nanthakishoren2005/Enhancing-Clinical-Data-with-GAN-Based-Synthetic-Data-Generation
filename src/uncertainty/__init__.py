"""
Uncertainty and Out-of-Distribution (OOD) detection package.
"""
from .detector import UncertaintyDetector, assess_prediction_uncertainty

__all__ = [
    "UncertaintyDetector",
    "assess_prediction_uncertainty",
]
