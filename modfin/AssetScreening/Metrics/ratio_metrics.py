import numpy as np
import pandas as pd


class RiskMetrics():


def sharpe_ratio(asset_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Calculate the Sharpe Ratio of a given asset or portfolio

    Parameters
    ----------

    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    risk_free : :py:class:`float` return of the risk-free rate on the same period. (Default: 0)

    freq : :py:class:`int` number of days in a year. (Default: 252)

    Return
    ----------
    SharpeRatio : :py:class:`float`
    """
    assert isinstance(freq, (int, float))  # Freq must be an integer or float

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    num_years = asset_returns.shape[0] / freq
    asset_returns = asset_returns[~np.isnan(asset_returns)]

    if len(asset_returns) < 1:
        raise ValueError(
            'asset_returns must contain at least one valid periods')

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1
    ann_riskfree = risk_free = (1 + risk_free) ** (1 / num_years) - 1
    ann_volatility = np.std(asset_returns) * np.sqrt(252)
    return (ann_return - ann_riskfree) / ann_volatility


def sortino_ratio(asset_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Calculate the Sortino Ratio of a given asset or portfolio

    Parameters
    ----------

    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    risk_free : :py:class:`float` return of the risk-free rate on the same period. (Default: 0)

    Return
    ----------

    sortino_ratio : :py:class:`float`
    """
    assert isinstance(freq, (int, float))  # Freq must be an integer or float

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    num_years = asset_returns.shape[0] / freq
    asset_returns = asset_returns[~np.isnan(asset_returns)]

    if len(asset_returns) < 1:
        raise ValueError(
            'asset_returns must contain at least one valid periods')

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1
    ann_riskfree = risk_free = (1 + risk_free) ** (1 / num_years) - 1
    ann_downside = asset_returns[asset_returns < 0].std() * np.sqrt(252)

    return (ann_return - ann_riskfree) / ann_downside


def treynor_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free: float = 0.02, freq: int = 252) -> float:
    """
    Calculate the Treynor Ratio of a given asset or portfolio

    Parameters
    ----------

    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    benchmark : :py:class:`pandas.Series` daily prices of the benchmark.

    risk_free : :py:class:`float` return of the risk-free rate on the same period. (default: 0)

    Return
    ----------

    TreynorRatio: :py:class:`float`
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

    market_beta = np.cov(asset_returns, benchmark_returns)[0, 1] / \
        np.var(benchmark_returns)

    return (ann_return - risk_free) / market_beta


def information_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Information Ratio of a given asset or portfolio
    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    benchmark : :py:class:`pandas.Series` daily prices of the benchmark.

    Return
    ----------
    TreynorRatio: :py:class:`float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError(
            'asset_returns and benchmark_returns must have the same length')

    num_years = asset_returns.shape[0] / freq

    tracking_e = ((asset_returns - benchmark_returns)**2).sum() / \
        (len(asset_returns) - 1) ** (1 / 2)

    ann_return = np.prod(1 + asset_returns) ** (1 / num_years) - 1
    ann_return_benchmark = np.prod(
        1 + benchmark_returns) ** (1 / num_years) - 1

    return (ann_return - ann_return_benchmark) / tracking_e


def omega_ratio(asset_returns: np.ndarray, threshold: float = 0) -> float:
    """
    Calculate the Omega Ratio of a given asset or portfolio
    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    threshold : :py:class:`float` minimum annual return required. (Default: 0)

    Return
    ----------
    OmegaRatio : :py:class:`float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(threshold, (int, float)):
        raise ValueError('threshold must be a number')

    daily_threshold = (threshold + 1) ** (1 / 252) - 1

    positive_returns = asset_returns[asset_returns >= daily_threshold].sum()
    negative_returns = abs(
        asset_returns[asset_returns < daily_threshold].sum())

    if negative_returns == 0:
        negative_returns = daily_threshold

    return positive_returns / negative_returns


def calmar_ratio(asset_returns: np.ndarray) -> float:
    """
    Calculate the Calmar Ratio of a given asset or portfolio

    Parameters
    ----------

    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    Return
    ----------

    CalmarRatio : :py:class:`float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    returns_cum = np.insert(asset_returns + 1, 0, 1).cumprod()
    roll_max = np.maximum.accumulate(returns_cum)
    drawdown = (returns_cum - roll_max) / roll_max
    max_drawdown = np.abs(np.min(drawdown))
    return returns_cum[-1] / max_drawdown


def tail_ratio(asset_returns: np.ndarray, alpha=0.05) -> float:
    """
    Calculate the proportion of the two tails from the asset price distribution.

    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

    Alpha : :py:class:`float` significance level.

    Return
    ----------
    TailRatio : :py:class:`float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    return np.quantile(asset_returns, 1 - alpha) / np.quantile(asset_returns, alpha)


def M2Ratio():
    """
    ** Under construction **"""
    pass


def hurst_exponent(asset_returns: pd.Series, window: int = 21) -> float:
    """
    Calculate the Hurst Exponent of a given asset or portfolio

    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series`

    window : :py:class:`int` window size. (Default: 21)

    Analysis
    ----------
    H < 0.5 — Mean reverting

    H = 0.5 — Geometric Brownian Motion

    H > 0.5 — Monotonic Trend

    Return
    ----------
    H: :py:class:`float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if not isinstance(window, int):
        raise ValueError('window must be an integer')

    asset_prices = (1 + asset_returns).cumprod()

    lags = range(2, window)
    tau = [np.sqrt(np.std(np.subtract(asset_prices[lag:], asset_prices[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
