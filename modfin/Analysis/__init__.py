"""
Modules for multi-asset analysis of finalcial time series.

    - ReturnAnalysis
    - CovMatrix
"""
from .return_analysis import ReturnAnalysis

del return_analysis  # noqa: F821

__all__ = [
    "ReturnAnalysis",
]
