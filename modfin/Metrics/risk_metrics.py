
"""
Module containing functions to calculate risk metrics
"""

import numpy as np
from scipy import stats, optimize


def volatility(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the realized close-to-close volatility of a given asset
    over a given period.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Returns
    ----------
    volatility: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    return asset_returns.std() * np.sqrt(freq)


def downside_risk(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Downside Risk of a given asset
    over a given period.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------
    downside_risk: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    negative_returns = asset_returns[asset_returns < 0]

    return negative_returns.std() * np.sqrt(freq)


def upside_risk(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Upside Risk of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Returns
    ----------
    upside_risk: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    positive_returns = asset_returns[asset_returns > 0]

    return positive_returns.std() * np.sqrt(freq)


def volatility_skewness(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Volatility Skewness of a given asset

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset


    Return
    ----------
    volatility_skewness : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    upside_vol = asset_returns[asset_returns > 0].std() * np.sqrt(freq)
    downside_vol = asset_returns[asset_returns < 0].std() * np.sqrt(freq)

    return upside_vol / downside_vol


def tracking_error(asset_returns: np.ndarray, benchmark_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Tracking Error of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset


    benchmark_returns : `numpy.ndarray`
        Daily returns of a benchmark asset

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)

    Return
    ----------
    tracking_error: `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError(
            'asset_returns and benchmark_returns must have the same length')

    return (asset_returns - benchmark_returns).std() * np.sqrt(freq)


def information_disc(asset_returns: np.ndarray) -> float:
    """
    Calculate the Information Discretness of a given asset

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    Return
    ----------
    information_disc : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    sgn_ret = np.sign((1 + asset_returns).prod() - 1)

    pos_pct = len(asset_returns[asset_returns > 0.0]) / len(asset_returns)
    neg_pct = len(asset_returns[asset_returns < 0.0]) / len(asset_returns)

    return sgn_ret * (neg_pct - pos_pct)


def information_disc_mag(asset_returns: np.ndarray, bins: int = 4) -> float:
    """
    Calculate the pondered Information Discretness of a given asset

    The ponderation is based on the magnitude of the absolute value of the return.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    bins : `int`, (optional)
        Magnitude factor of the return (Default: 4)

    Returns
    ----------
    information_disc_mag : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    sgn_ret = np.sign((1 + asset_returns).prod() - 1) / \
        asset_returns.shape[0]

    mag_ret = np.arange(1 / bins, 1, 1 / bins)
    abs_ret = np.abs(asset_returns)
    bins_ret = np.quantile(abs_ret, mag_ret)
    weights = (np.digitize(abs_ret, bins_ret) + 1) / bins

    return sgn_ret * np.dot(asset_returns, weights)


def information_disc_rel(asset_returns: np.ndarray) -> float:
    """
    Calculate the Relative Information Discretness of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    Returns
    ----------
    information_disc_rel : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    sgn_ret = np.sign((1 + asset_returns).prod() - 1)

    pos_pct = len(asset_returns[asset_returns > 0.0])
    neg_pct = len(asset_returns[asset_returns < 0.0])

    if sgn_ret > 0:
        return sgn_ret * (pos_pct - neg_pct) / len(asset_returns)

    return sgn_ret * (neg_pct - pos_pct) / len(asset_returns)


def var_gaussian(asset_returns: np.ndarray, alpha: float = 0.05, CFE: bool = False) -> float:
    """
    Calculate the Gaussian Value at Risk (VaR) of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given a

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    CFE : `bool`, (optional)
        If True, the VaR is computed using the Cornish-Fisher Expation (Default: False)

    Return
    ----------

    Gaussian VaR : `float`
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be ~(0,1)')

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    mu = asset_returns.mean()
    sigma = asset_returns.std()
    z = stats.norm.ppf(1 - alpha)

    if CFE:
        s = stats.skew(asset_returns)
        k = stats.kurtosis(asset_returns)
        z = (z + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24      # noqa
                - (2 * z**3 - 5 * z) * (s**2) / 36)  # noqa

    return mu - z * sigma


def var_historical(asset_returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the Historical VaR of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given a

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    Return
    ----------

    Historical VaR : `float`
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be ~(0,1)')

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    return np.quantile(asset_returns, alpha)


def conditional_var_gaussian(asset_returns: np.ndarray, alpha: float = 0.05, CFE: bool = False) -> float:
    """
    Calculate the Gaussian Conditional Value at Risk of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given a

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    CFE : `bool`, (optional)
        If True, the VaR is computed using the Cornish-Fisher Expation (Default: False)

    Return
    ----------
    conditional_var : `float`
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be ~(0,1)')

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    mu = asset_returns.mean()
    sigma = asset_returns.std()
    z = stats.norm.ppf(1 - alpha)
    inv_alpha = alpha ** -1

    if CFE:
        s = stats.skew(asset_returns)
        k = stats.kurtosis(asset_returns)
        z = (
            z + (z**2 - 1) * s / 6               # noqa: W503
            + (z**3 - 3 * z) * (k - 3) / 24      # noqa: W503
            - (2 * z**3 - 5 * z) * (s**2) / 36)  # noqa: W503

    return mu - inv_alpha * stats.norm.pdf(z) * sigma


def _entropy(z, ret, alpha=0.05):
    ret_dim = np.array(ret, ndmin=2)
    if ret_dim.shape[0] == 1 and ret_dim.shape[1] > 1:
        ret_dim = ret_dim.T
    if ret_dim.shape[0] > 1 and ret_dim.shape[1] > 1:
        raise ValueError
    value = np.mean(np.exp(-1 / z * ret_dim), axis=0)
    value = z * (np.log(value) + np.log(1 / alpha))
    value = np.array(value).item()
    return value


def entropic_var_historical(asset_returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the Historical Entropic Value at Risk of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    Returns
    ----------
    evar_historical : `float`
    """

    ret_dim = np.array(asset_returns, ndmin=2)

    if ret_dim.shape[0] == 1 and ret_dim.shape[1] > 1:
        ret_dim = ret_dim.T
    if ret_dim.shape[0] > 1 and ret_dim.shape[1] > 1:
        raise ValueError

    bnd = optimize.Bounds([1e-12], [np.inf])
    min = optimize.minimize(
        _entropy,
        [1],
        args=(asset_returns, alpha),
        method="SLSQP",
        bounds=bnd,
        tol=1e-12).x.item()
    value = _entropy(min, asset_returns, alpha)

    return value


def entropic_var_gaussian(asset_returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the Gaussian Entropic Value at Risk of a given asset

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    Returns
    ----------
    entropic_var_gaussian : `float`
    """

    mu = asset_returns.mean()
    sigma = asset_returns.std()
    z = np.sqrt(-2 * np.log(alpha))

    return mu - z * sigma


def conditional_drawndown_at_risk(asset_returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the Conditional Drawdown at Risk de um AssetPrice ou portfolio

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    alpha : `float`, (optional)
        Significance level (Default: 0.05)

    Returns
    ----------
    conditional_drawndown_at_risk : `float`
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be ~(0,1)')

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    drawdown = np.maximum.accumulate(asset_returns) - asset_returns
    max_dd = np.maximum.accumulate(drawdown)
    max_dd_alpha = np.quantile(max_dd, 1 - alpha, interpolation='higher')
    return np.mean(max_dd[max_dd >= max_dd_alpha])


def max_drawdown(asset_returns: np.ndarray) -> float:
    """
    Calculate the Maximum Drawdown of a given asset

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset


    Return
    ----------
    MaxDrawdown : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    synthectic_price = np.insert(asset_returns + 1, 0, 1).cumprod()
    rolling_max = np.maximum.accumulate(synthectic_price)
    drawdown = (synthectic_price - rolling_max) / rolling_max
    return np.min(drawdown)


def alpha_capm(asset_returns: np.ndarray, benchmark_returns: np.ndarray, freq: int = 252):
    """
    Calculate the Asset Alpha (CAPM) of a given asset

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
    alpha : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    num_years = float(len(asset_returns)) / freq

    cov = np.cov(asset_returns, benchmark_returns)
    beta = cov[0][1] / cov[0][0]

    ann_asset_total_return = (
        1 + asset_returns).prod() ** (1 / num_years) - 1

    ann_benchmark_total_return = (
        1 + benchmark_returns).prod() ** (1 / num_years) - 1

    return ann_asset_total_return - beta * ann_benchmark_total_return


def beta_capm(asset_returns: np.ndarray, benchmark_returns: np.ndarray):
    """
    Calculate the Asset Beta (CAPM) of a given asset

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    Return
    ----------
    beta_capm : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    cov = np.cov(asset_returns, benchmark_returns)

    return cov[0][1] / cov[0][0]


def beta_downside(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Asset Beta (CAPM) of a given asset when the refferd benchmark returns are negative

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    Returns
    ----------
    beta_downside : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    cov = np.cov(asset_returns[benchmark_returns < 0],
                 benchmark_returns[benchmark_returns < 0])
    return cov[0][1] / cov[0][0]


def beta_upside(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Asset Beta (CAPM) of a given asset when the refferd benchmark returns are positive

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    Returns
    ----------
    beta_upside : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    cov = np.cov(asset_returns[benchmark_returns > 0],
                 benchmark_returns[benchmark_returns > 0])
    return cov[0][1] / cov[0][0]


def beta_quotient(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Beta Quotient of a given asset

    The Beta Quotient is the Upside Beta divided by the Downside Beta

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    Returns
    ----------
    beta_quotient : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    pos_beta = np.cov(
        asset_returns[benchmark_returns > 0], benchmark_returns[benchmark_returns > 0])
    pos_beta = pos_beta[0][1] / pos_beta[0][0]

    neg_beta = np.cov(
        asset_returns[benchmark_returns < 0], benchmark_returns[benchmark_returns < 0])
    neg_beta = neg_beta[0][1] / neg_beta[0][0]

    return pos_beta / neg_beta


def beta_convexity(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Beta Convexity of a given asset.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    benchmark_returns : `numpy.ndarray`
        Daily returns of the asset's benchmark

    Returns
    ----------
    beta_convexity : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    if len(asset_returns) != len(benchmark_returns):
        raise ValueError(
            "asset_returns and benchmark_returns must have the same length")

    pos_beta = np.cov(
        asset_returns[benchmark_returns > 0], benchmark_returns[benchmark_returns > 0])
    pos_beta = pos_beta[0][1] / pos_beta[0][0]

    neg_beta = np.cov(
        asset_returns[benchmark_returns < 0], benchmark_returns[benchmark_returns < 0])
    neg_beta = neg_beta[0][1] / neg_beta[0][0]

    diff_beta = pos_beta - neg_beta

    diff_sign = np.sign(diff_beta)

    return diff_beta ** 2 / (diff_sign * (pos_beta ** 2 + neg_beta ** 2))


def rsquare_score(asset_returns: np.ndarray) -> float:
    """
    Calculate the R-Squared Score of a given asset.

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    Return
    ----------
    rsquare_score : `float`
    """

    cum_log_returns = np.log1p(asset_returns).cumsum()

    score = stats.linregress(
        np.arange(len(cum_log_returns)), cum_log_returns)[2]

    return score ** 2


def autocorr_score(asset_returns: np.ndarray, max_lag: int = 5) -> float:
    """
    Calculate the returns autocorrelation score.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    max_lag : `int`, (optional)
        Maximum lag to use in the autocorrelation calculation

    Returns
    ----------
    autocorr_score : `float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    lags = np.arange(1, max_lag)

    ACF = [np.corrcoef(asset_returns[:-lag], asset_returns[lag:])[0, 1]
           for lag in lags]

    return np.log(np.abs(ACF)).mean() / np.log(max_lag)


def lower_partial_moment(asset_returns: np.ndarray, threshold: float = 0, order: int = 1):
    """
    Calculate the Lower Partial Moment of Returns (LPM) from a given set of returns.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    threshold : float

    order : int

    Returns
    -------

    LPM : `python:float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    lower_part = np.abs(asset_returns[asset_returns < threshold] - threshold)
    return np.sum(lower_part ** order) / len(asset_returns)


def higher_partial_moment(asset_returns: np.ndarray, threshold: float = 0, order: int = 1, freq: int = 252):
    """
    Calculate the Higher Partial Moment of Returns (HPM) from a given set of returns.

    Parameters
    ----------

    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    threshold : `float`
        Annual minimum profit an investor expects to make (Default: 0)

    order : `int`
        Order of the partial moment (Default: 1)

    freq : `int`
        Number of trading periods in a year (Default: 252)

    Returns
    -------

    higher_partial_moment : `float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    f_threshold = (1 + threshold) ** (1 / freq) - 1

    higher_part = np.abs(
        asset_returns[asset_returns > f_threshold] - f_threshold)

    return np.sum(higher_part ** order) / len(asset_returns)
