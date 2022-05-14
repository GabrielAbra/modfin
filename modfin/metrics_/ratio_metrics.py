import numpy as np
import pandas as pd
from ..numba_funcs import nb_ratio_metrics


def sharpe_ratio(asset_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Function to calculate the Sharpe Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    risk_free : `float`, (optional)
        Annual Risk-Free Rate (Default: 0.02)

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------
    sharpe_ratio : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    num_years = asset_returns.shape[0] / freq

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1

    ann_riskfree = risk_free = (1 + risk_free) ** (1 / num_years) - 1

    ann_volatility = np.std(asset_returns) * np.sqrt(freq)

    return (ann_return - ann_riskfree) / ann_volatility


def sortino_ratio(asset_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Function to calculate the Sortino Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    risk_free : `float`, (optional)
        Annual Risk-Free Rate (Default: 0.02)

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------

    sortino_ratio : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    num_years = asset_returns.shape[0] / freq

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1

    ann_riskfree = risk_free = (1 + risk_free) ** (1 / num_years) - 1

    ann_downside = asset_returns[asset_returns < 0].std() * np.sqrt(252)

    return (ann_return - ann_riskfree) / ann_downside


def treynor_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Function to calculate the Treynor Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    risk_free : `float`, (optional)
        Annual Risk-Free Rate (Default: 0.02)

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Returns
    ----------
    treynor_ratio: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError(
            'asset_returns and benchmark_returns must have the same length')

    num_years = asset_returns.shape[0] / freq

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1

    cov = np.cov(asset_returns, benchmark_returns)
    beta = cov[0][1] / cov[0][0]

    return (ann_return - risk_free) / beta


def information_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Information Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------
    information_ratio: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError(
            'asset_returns and benchmark_returns must have the same length')

    num_years = asset_returns.shape[0] / freq

    te = np.sum((asset_returns - benchmark_returns)**2)

    te /= len(asset_returns) ** (1 / 2)

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1
    ann_return_b = np.prod(1 + benchmark_returns) ** (1 / num_years) - 1

    return (ann_return - ann_return_b) / te


def omega_ratio(asset_returns: np.ndarray, threshold: float = 0, freq: int = 252) -> float:
    """
    Calculate the Omega Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    threshold : `float`, (optional)
        Annual minimum profit an investor expects to make (Default: 0)

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------
    OmegaRatio : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(threshold, (int, float)):
        raise ValueError('threshold must be a number')

    f_threshold = (threshold + 1) ** (1 / freq) - 1

    pos_returns = asset_returns[asset_returns >= f_threshold].sum()
    neg_returns = abs(asset_returns[asset_returns < f_threshold].sum())

    if neg_returns == 0:
        neg_returns = f_threshold

    return pos_returns / neg_returns


def calmar_ratio(asset_returns: np.ndarray) -> float:
    """
    Calculate the Calmar Ratio of a given asset.

    The ratio can be used to measure the risk-adjusted return of an investment.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    Return
    ----------

    CalmarRatio : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    return nb_ratio_metrics.numba_calmar_ratio(asset_returns)


def tail_ratio(asset_returns: np.ndarray, alpha=0.05) -> float:
    """
    Calculate the proportion of the two tails from the asset price distribution.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    Return
    ----------
    TailRatio : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    return np.quantile(asset_returns, 1 - alpha) / np.quantile(asset_returns, alpha)


def mm_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Calculate the M2 Ratio of a given asset.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    risk_free : `float`, (optional)
        Annual Risk-Free Rate (Default: 0.02)

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------

    M2Ratio : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    num_years = asset_returns.shape[0] / freq

    asset_ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1

    asset_vol = np.std(asset_returns) * np.sqrt(freq)

    benchmark_vol = np.std(benchmark_returns) * np.sqrt(freq)

    return risk_free + ((asset_ann_return - risk_free) / asset_vol) * benchmark_vol


def hurst_exponent(asset_returns: pd.Series, max_lag: int = 21) -> float:
    """
    Calculate the Hurst Exponent of a given asset.

    The Hurst Exponent is a measure of long term autocorrelation in the time series, based on Harold Edwin Hurst Studies.

    The Hurst exponent shall be interpreted as including the following:

        If greater than 0.5,  indicates a time series with positive autocorrelation over the periods (e.g. trending returns).

        If less than 0.5, indicates a time series with negative autocorrelation over the periods (e.g. mean reverting returns).

        If equal to 0.5, indicates a time series with no autocorrelation over the periods (e.g. random walk returns).

    The other way around if Hurst Exponent is less than 0.5, indicate that the time series returns are anti-correlated over the periods, which is a sign of a mean reverting returns.


    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    max_lag : `int`
        Maximum lag to calculate the Hurst Exponent (Default: 21)

    Return
    ----------
    Hurst Exponent : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    log_lag, log_rs = nb_ratio_metrics.numba_hurst(asset_returns)
    return np.polyfit(log_lag, log_rs, 1)[0]


def autocorr_score(asset_returns: pd.Series, max_lag: int = 21) -> float:
    """
    Calculate the auto-correlation score of a given asset returns

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    max_lag : `int`
        Maximum lag window to calculate the autocorrelation (Default: 21)
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(max_lag, int):
        raise ValueError('max_lag must be an integer')

    asset_prices = np.cumprod(1 + asset_returns)

    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(asset_prices[lag:], asset_prices[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
