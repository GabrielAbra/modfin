import numpy as np
import pandas as pd


class ReturnMetrics():
    """
    Module containing functions to calculate return metrics.
    """
    @staticmethod
    def AnnualizedReturn(AssetPrice: pd.Series, Period: str = 'days') -> float:
        """
        Calculate the annualized return of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Period : :py:class:`str` period of the asset prices.
            - 'days' for daily prices
            - 'weeks' for weekly prices
            - 'months' for monthly prices
            - 'years' for annual prices

        Return
        -------
        AnnualizedReturn: :py:class:`float`
        """

        period_param = {
            'days': 252,
            'weeks': 52,
            'months': 12,
            'years': 1}

        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]

        if len(AssetPrice) < 2:
            raise ValueError('AssetPrice must contain at least two periods')

        num_year = float(len(AssetPrice)) / period_param[Period]

        total_return = ReturnMetrics.TotalReturn(AssetPrice)
        return_year = (1 + total_return) ** (1 / num_year) - 1
        return return_year

    @staticmethod
    def ExponencialReturns(AssetPrice: pd.Series) -> float:
        """
        Calculate the total exponencial returns of a given asset or portfolio.

        The exponencialization works with geometric weightings assigned for the daily returns, giving higher weightings for recent ones.

        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Return
        -------
        ReturnTotal: :py:class:`float`
        """

        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]

        if len(AssetPrice) < 2:
            raise ValueError('AssetPrice must contain at least two periods')

        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1

        k = np.array(2.0 / (1 + returns.shape[0]))

        exp_ar = (
            1 - np.repeat(k, returns.shape[0])) ** np.arange(returns.shape[0])
        exp_return = returns.dot(exp_ar)
        return exp_return

    @staticmethod
    def TotalReturn(AssetPrice: pd.Series) -> float:
        """
        Calculate the Return total of a given asset or portfolio

        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` asset prices.

        Return
        -------
        ReturnTotal: :py:class:`float`
        """
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]

        if len(AssetPrice) < 2:
            raise ValueError('AssetPrice must contain at least two periods')

        Return = (AssetPrice[-1] / AssetPrice[0]) - 1

        if np.isinf(Return):
            return np.nan
        return Return
