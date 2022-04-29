import numpy as np
import pandas as pd


class ReturnAnalysis():

    @staticmethod
    def SerieReturn(AssetPrice: pd.Series) -> pd.Series:
        """
        Create a pandas Series with the returns of a given asset or portfolio.

        Parameters
        -----------
        AssetPrice : :py:class:`pandas.Series` asset prices.

        Return
        -------
        SerieReturn: :py:class:`pandas.Series`
        """
        index_c = AssetPrice.index[1:]
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return np.nan

        return pd.Series(returns, index=index_c)

    @staticmethod
    def VetorizedReturns(AssetPrice: pd.Series) -> np.ndarray:
        """
        Create a numpy array with the returns of a given asset or portfolio.

        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` asset prices.

        Return
        -------
        VetorizedReturns: :py:class:`numpy.ndarray`
        """
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)
        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]

        if len(AssetPrice) < 2:
            return np.array([np.nan])

        AssetPrice = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        return AssetPrice

    @staticmethod
    def CummulativeReturnSerie(AssetPrice: pd.Series, initial_k: float = 1) -> pd.Series:
        """
        Create a pandas Series with the cummulative returns of a given asset or portfolio.

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

    @staticmethod
    def ReturnTotalReturn(Returns: pd.Series) -> float:
        """
        Calculate the total return from a given serie of returns.

        Parameters
        ----------
        Returns : :py:class:`pandas.Series` returns.

        Return
        -------
        ReturnTotalReturn: :py:class:`float`
        """

        Returns = np.array(Returns)
        return (1 + Returns).prod() - 1

    @staticmethod
    def ReturnAdjusted(AssetPrice: pd.Series, Factor: float = 2) -> pd.Series:
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
        Returns = ReturnAnalysis.SerieReturn(AssetPrice)
        if isinstance(Factor, (float, int)):
            if Factor == 0:
                return Returns
            Returns /= Factor
        return Returns - Factor
