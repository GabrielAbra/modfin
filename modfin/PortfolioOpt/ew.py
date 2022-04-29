import numpy as np
import pandas as pd
from .base import PortifolioOptBase


class EqualWeight(PortifolioOptBase):
    """
    Equal Weight Portfolio

    Parameters
    ----------
    AssetPrices : :py:class:`pandas.DataFrame` with the asset prices

    Returns
    -------
    EqualWeight : :py:class:`pandas.DataFrame` with the equal weight portfolio
    """

    def __init__(self, RiskMatrix: pd.DataFrame):
        # Get asset names
        self._names = EqualWeight._get_asset_names(RiskMatrix)
        self._num = len(self._names)

    def optimize(self) -> pd.DataFrame:
        """
        Calculate asset weights using equal weight algorithm.

        Parameters
        ----------
        AssetPrices : py:class:`pandas.DataFrame` with the daily asset prices

        Returns
        -------
        Portifolio : :py:class:`pandas.DataFrame` with the weights of the portfolio
        """

        # Portfolio "construction" LMAO :D
        value = np.repeat([1 / self._num], self._num, axis=0)
        Portifolio = pd.Series(value, index=self._names)
        Portifolio = Portifolio.to_frame().T.dropna(axis="columns")
        return Portifolio
