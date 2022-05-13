import numpy as np
import pandas as pd


def get_returns(asset_prices: pd.Series, as_pandas: bool = False):
    """
    Create a numpy array or a pandas Series with the returns
    from a given asset or portfolio.

    Parameters
    -----------
    AssetPrice : :py:class:`numpy.ndarray` or :py:class:`pandas.Series`
    with the daily prices of a given asset or portfolio.

    as_pandas : :py:class:`bool`
    If True, the return will be a pandas Series.
    Otherwise, it will be a numpy array, default is False.

    Return
    -------
    returns: :py:class:`numpy.ndarray` or :py:class:`pandas.Series`
    """
    _index_ = asset_prices.index[1:]

    if not isinstance(asset_prices, np.ndarray):
        asset_prices = np.array(asset_prices)

    asset_prices = asset_prices[np.logical_not(np.isnan(asset_prices))]
    returns = (asset_prices[1:] / asset_prices[:-1]) - 1

    if len(returns) < 2:
        raise ValueError('asset_prices must contain at least two periods')

    if as_pandas:
        returns = pd.Series(returns, index=_index_)
    return returns


def get_cumreturns(AssetPrice: pd.Series, initial_k: float = 1) -> pd.Series:
    """
    Create a numpy array or a pandas Series with the cummulative returns
    from a given asset or portfolio.

    Parameters
    ----------
    AssetPrice : :py:class:`pandas.Series` asset prices.

    initial_k : :py:class:`float` initial capital of the cummulative returns.

    Return
    -------
    CummulativeReturnSerie: :py:class:`pandas.Series`
    """
    index_c = AssetPrice.index
    if not isinstance(AssetPrice, np.ndarray):
        AssetPrice = np.array(AssetPrice)

    AssetPrice = AssetPrice[~np.isnan(AssetPrice)]
    AssetPrice = AssetPrice[1:] / AssetPrice[:-1]

    if len(AssetPrice) < 2:
        return np.nan

    AssetPrice = np.insert(AssetPrice, 0, initial_k).cumprod()

    return pd.Series(AssetPrice, index=index_c)


def total_returns_from_returns(asset_returns: pd.Series) -> float:
    """
    Calculate the total return from a given serie of returns.

    Parameters
    ----------
    Returns : :py:class:`pandas.Series` returns.

    Return
    -------
    ReturnTotalReturn: :py:class:`float`
    """

    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)
    return (1 + asset_returns).prod() - 1


def adjust_return(asset_returns: pd.Series, factor=1, operation="subtract") -> pd.Series:
    """
    Calculate the adjusted return from a given serie of returns.

    Parameters
    ----------
    AssetPrice : :py:class:`pandas.Series` asset prices.

    Factor : :py:class:`float` factor to adjust the return.

    Return
    -------
    ReturnAdjusted: :py:class:`pandas.Series`
    """

    assert isinstance(factor, (float, int))
    assert isinstance(operation, str)

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
