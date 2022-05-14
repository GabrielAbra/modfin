import numpy as np


def annualized_return(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the annualized return of a given asset.

    The annualized return is the geometric mean of the returns relative to the given frequency.

    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series` daily prices of a given asset

    freq : :py:class:`int` number of days in a year. (Default: 252)

    Return
    ----------
    AnnualizedReturn: :py:class:`float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    num_years = asset_returns.shape[0] / freq

    return np.prod(1 + asset_returns) ** (1 / num_years) - 1


def expected_return(asset_returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate the Expected Return of a given asset.

    The Expected Return is essentially the average return across a time series of asset returns.

    Parameters
    ----------
    asset_returns : `numpy.ndarray`
        Daily returns of a given asset

    freq : `int`, (optional)
        Number of trading periods in a year (Default: 252)
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    # Calculate the expected return
    return np.mean(asset_returns) * freq


def exponencial_return(asset_returns: np.ndarray) -> float:
    """
    Calculate the total exponencial returns of a given asset.

    The exponencialization works with geometric weighted returns, giving higher weightings for recent ones.

    This function plays an essential role when it is assumed that recent results are more relevant than older ones.

    Parameters
    ----------
    asset_prices : :py:class:`pandas.Series` daily prices of a given asset

    Return
    -------
    ReturnTotal: :py:class:`float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    asset_returns = asset_returns[~np.isnan(asset_returns)]

    if len(asset_returns) < 1:
        raise ValueError(
            'asset_returns must contain at least one valid periods')

    k = np.array(2.0 / (1 + asset_returns.shape[0]))

    exp = (
        1 - np.repeat(k, asset_returns.shape[0])) ** np.arange(asset_returns.shape[0])
    return asset_returns.dot(exp)
