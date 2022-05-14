"""
Modules for Metric Analysy time series.
"""
from modfin.metrics.return_metrics import (
    annualized_return, expected_return, exponencial_return)

from modfin.metrics.risk_metrics import (
    volatility, downside_risk, upside_risk, volatility_skewness, tracking_error,
    information_disc, information_disc_mag, information_disc_rel, var_gaussian,
    var_historical, conditional_var_gaussian, entropic_var_historical,
    entropic_var_gaussian, conditional_drawndown_at_risk, max_drawdown, alpha_capm,
    beta_capm, beta_downside, beta_upside, beta_quotient, beta_convexity,
    rsquare_score, autocorr_score, lower_partial_moment, higher_partial_moment)

from modfin.metrics.ratio_metrics import (
    sharpe_ratio, sortino_ratio, treynor_ratio, information_ratio, calmar_ratio,
    omega_ratio, tail_ratio, mm_ratio, hurst_exponent)

__all__ = [
    'annualized_return', 'expected_return', 'exponencial_return',
    'volatility', 'downside_risk', 'upside_risk', 'volatility_skewness',
    'tracking_error', 'information_disc', 'information_disc_mag',
    'information_disc_rel', 'var_gaussian', 'var_historical',
    'conditional_var_gaussian', 'entropic_var_historical', 'entropic_var_gaussian',
    'conditional_drawndown_at_risk', 'max_drawdown', 'alpha_capm', 'beta_capm',
    'beta_downside', 'beta_upside', 'beta_quotient', 'beta_convexity', 'rsquare_score',
    'autocorr_score', 'lower_partial_moment', 'higher_partial_moment', 'sharpe_ratio',
    'sortino_ratio', 'treynor_ratio', 'information_ratio', 'calmar_ratio',
    'omega_ratio', 'tail_ratio', 'mm_ratio', 'hurst_exponent']
