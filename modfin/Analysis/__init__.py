# flake8: noqa
"""
Modules for multi-asset analysis of finalcial time series.
"""

from modfin.analysis.return_analysis import (annualized_return, expected_return,
                                             exponencial_return, total_return)

__all__ = [
    'annualized_return',
    'expected_return',
    'exponencial_return',
    'total_return']
