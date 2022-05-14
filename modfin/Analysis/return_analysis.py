import numpy as np
import pandas as pd
from ..utils import series_tools
from ..numba_funcs import nb_series


def calculate_returns(asset_prices: pd.Series, as_pandas: bool = False):
    """
    Create a numpy array or a pandas Series with the returns
    from a given asset or portfolio.

    Parameters
    -----------
    asset_prices : `numpy.ndarray` or `pandas.Series`
        with the daily prices of a given asset or portfolio.

    as_pandas : `bool`
        If True, the return will be a pandas Series,
        otherwise it will be a numpy array.
        (Default: False)

    Return
    -------
    returns: `numpy.ndarray` or `pandas.Series`
    """

    asset_prices_array = np.array(asset_prices)

    asset_prices_array = asset_prices_array[np.logical_not(
        np.isnan(asset_prices_array))]

    asset_returns = (asset_prices_array[1:] / asset_prices_array[:-1]) - 1

    if len(asset_returns) < 1:
        raise ValueError('asset_prices must contain at least two periods')

    if as_pandas:
        _index_ = series_tools.get_index(asset_prices)
        _columns_ = series_tools.get_names(asset_prices)

        if _columns_.shape[0] > 1:
            asset_returns = pd.DataFrame(
                asset_returns, index=_index_, columns=_columns_)

        if _columns_.shape[0] == 1:
            asset_returns = pd.Series(
                asset_returns, index=_index_, name=_columns_)

    return asset_returns


def calculate_logreturns(asset_prices, as_pandas: bool = False):
    """
    Create a numpy array or a pandas Series with the log returns
    from a given asset or portfolio.

    Parameters
    -----------
    asset_prices : `numpy.ndarray` or `pandas.Series`
        with the daily prices of a given asset or portfolio.

    as_pandas : `bool`
        If True, the return will be a pandas Series,
        otherwise it will be a numpy array.
        (Default: False)

    Return
    -------
    returns: `numpy.ndarray` or `pandas.Series`
    """

    asset_prices_array = np.array(asset_prices)

    asset_prices_array = asset_prices_array[np.logical_not(
        np.isnan(asset_prices_array))]

    asset_returns = np.log(asset_prices_array[1:] / asset_prices_array[:-1])

    if len(asset_returns) < 1:
        raise ValueError('asset_prices must contain at least two periods')

    if as_pandas:
        _index_ = series_tools.get_index(asset_prices)
        _columns_ = series_tools.get_names(asset_prices)

        if _columns_.shape[0] > 1:
            asset_returns = pd.DataFrame(
                asset_returns, index=_index_, columns=_columns_)

        if _columns_.shape[0] == 1:
            asset_returns = pd.Series(
                asset_returns, index=_index_, name=_columns_)

    return asset_returns


def calculate_cummreturns(asset_returns: pd.Series, as_pandas: bool = False):
    """
    Create a numpy array or a pandas Series with the cummulative returns
    from a given asset or portfolio.

    Parameters
    ----------
    asset_returns : `pandas.Series`

    as_pandas : `bool`
        If True, the return will be a pandas Series,
        otherwise it will be a numpy array.
        (Default: False)

    Return
    -------
    CummulativeReturnSerie: `pandas.Series`
    """

    if isinstance(asset_returns, np.ndarray):
        cummreturns = nb_series.synthetic_prices(asset_returns) - 1
    else:
        cummreturns = nb_series.synthetic_prices(np.ndarray(asset_returns)) - 1

    if as_pandas:
        _index_ = series_tools.get_index(asset_returns)
        _columns_ = series_tools.get_names(asset_returns)

        if _columns_.shape[0] > 1:
            cummreturns = pd.DataFrame(
                cummreturns, index=_index_, columns=_columns_)

        if _columns_.shape[0] == 1:
            cummreturns = pd.Series(
                cummreturns, index=_index_, name=_columns_)

    return cummreturns


def adjust_return(asset_returns: pd.Series, factor=1, operation="subtract") -> pd.Series:
    """
    Calculate the adjusted return from a given serie of returns.

    Parameters
    ----------
    asset_returns : `pandas.Series`

    Factor : `float`

    operation : `str`

    Return
    -------
    adjust_return: `pandas.Series`
    """
    if not isinstance(factor, (float, int)):
        raise TypeError("factor must be a float or int")

    if operation == "subtract":
        asset_returns = asset_returns - factor

    elif operation == "add":
        asset_returns = asset_returns + factor

    elif operation == "multiply":
        asset_returns = asset_returns * factor

    elif operation == "divide":
        asset_returns = asset_returns / factor

    else:
        raise ValueError(
            "operation must be 'subtract', 'add', 'multiply' or 'divide'")
    return asset_returns


def total_return_from_returns(asset_returns: np.ndarray) -> float:
    """
    Calculate the Return total of a given asset or portfolio

    Parameters
    ----------
    asset_returns : :py:class:`pandas.Series` asset prices.

    Return
    -------
    ReturnTotal: :py:class:`float`
    """
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns, dtype=np.float64)

    asset_returns = asset_returns[~np.isnan(asset_returns)]

    if len(asset_returns) < 2:
        raise ValueError('asset_returns must contain at least two periods')

    return nb_series.synthetic_prices(asset_returns)[-1] - 1
