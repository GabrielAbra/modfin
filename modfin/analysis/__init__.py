"""
Modules for Metric Analysy time series.
"""

from modfin.analysis.return_analysis import (calculate_returns, calculate_logreturns,
                                             calculate_cummreturns, total_return_from_returns)

__all__ = [
    'calculate_returns',
    'calculate_logreturns',
    'calculate_cummreturns',
    'total_return_from_returns']
