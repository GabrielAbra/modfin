"""
The ModFin project aims to provide users with modules for Quantitative Financial Analysis
"""

import modfin.analysis as analysis
import modfin.metrics as metrics
import modfin.portfolio_opt as portfolio_opt
from modfin.analysis.risk_matrix import RiskMatrix

__version__ = "0.1.5"

__all__ = [
    'analysis',
    'metrics',
    'portfolio_opt',
    'RiskMatrix'
]
