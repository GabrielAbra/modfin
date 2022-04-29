import numpy as np
import pandas as pd
from .base import PortifolioOptBase


class InverseVariance(PortifolioOptBase):

    """
    Inverse Variance Portfolio Optimization

    Parameters
    ----------

    RiskMatrix : :py:class:`pd.DataFrame` a matrix of the risk implied by the returns of the assets.
    If possible from Modfin RiskMatrix Module.


    Functions
    ---------

    optimize() : Returns the optimal weights using inverse variance algorithm.
    """

    def __init__(self, RiskMatrix: pd.DataFrame):

        # Check if the RiskMatrix is valid.
        self._RiskMatrix = self._check_rm(RiskMatrix)

        # Get Asset names
        self._asset_names = self._get_asset_names(RiskMatrix)

    def optimize(self, AssetPrices: pd.DataFrame):
        """
       Calculate the optimal portfolio allocation using the Inverse Variance algorithm.

        Parameters
        ----------
        AssetPrices : py:class:`pandas.DataFrame` with the daily asset prices

        Returns
        -------
        Portifolio : :py:class:`pandas.DataFrame` with the weights of the portfolio
        """

        # Diagolainize the RiskMatrix
        Weight = 1. / np.diag(self._RiskMatrix)

        # Create and return the portfolio
        return self._pandas_portifolio(Weight, self._asset_names)
