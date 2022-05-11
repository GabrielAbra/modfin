import numpy as np
import pandas as pd


class ReturnMetrics():
    """
    Module containing functions to calculate return metrics.
    """

    def __init__(self):
        pass

    @staticmethod
    def annualized_return(asset_returns: np.ndarray, freq: int = 252) -> float:
        """
        Calculate the annualized return of a given asset or portfolio.

        Parameters
        ----------
        asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        freq : :py:class:`int` number of days in a year. (Default: 252)

        Return
        ----------
        AnnualizedReturn: :py:class:`float`
        """

        if not isinstance(asset_returns, np.ndarray):
            asset_returns = np.array(asset_returns)

        # Calculate the annualized return
        num_years = asset_returns.shape[0] / freq
        asset_returns = asset_returns[~np.isnan(asset_returns)]

        if len(asset_returns) < 1:
            raise ValueError(
                'asset_returns must contain at least one valid periods')

        return np.prod(1 + asset_returns) ** (1 / num_years) - 1

    @staticmethod
    def expected_return(asset_returns: np.ndarray, freq: int = 252) -> float:
        """
        Calculate the expected return of a given asset or portfolio.
        """

        if not isinstance(asset_returns, np.ndarray):
            asset_returns = np.array(asset_returns)

        # Calculate the expected return
        return np.mean(asset_returns) * freq

    @staticmethod
    def exponencial_return(asset_returns: np.ndarray) -> float:
        """
        Calculate the total exponencial returns of a given asset or portfolio.

        The exponencialization works with geometric weightings assigned for the daily returns, giving higher weightings for recent ones.

        Parameters
        ----------
        asset_prices : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Return
        -------
        ReturnTotal: :py:class:`float`
        """

        if not isinstance(asset_returns, np.ndarray):
            asset_returns = np.array(asset_returns)

        asset_returns = asset_returns[~np.isnan(asset_returns)]

        if len(asset_returns) < 1:
            raise ValueError(
                'asset_returns must contain at least one valid periods')

        k = np.array(2.0 / (1 + asset_returns.shape[0]))

        exp = (
            1 - np.repeat(k, asset_returns.shape[0])) ** np.arange(asset_returns.shape[0])
        return asset_returns.dot(exp)

    @staticmethod
    def total_return(asset_returns: pd.Series) -> float:
        """
        Calculate the Return total of a given asset or portfolio

        Parameters
        ----------
        asset_prices : :py:class:`pandas.Series` asset prices.

        Return
        -------
        ReturnTotal: :py:class:`float`
        """
        if not isinstance(asset_returns, np.ndarray):
            asset_returns = np.array(asset_returns)

        asset_returns = asset_returns[~np.isnan(asset_returns)]

        if len(asset_returns) < 2:
            raise ValueError('asset_prices must contain at least two periods')

        return np.prod(1 + asset_returns) - 1
