import numpy as np
import pandas as pd
from modfin.Analysis import CovMatrix


class InverseVariance():

    """
    Inverse Variance Portfolio Optimization

    Parameters
    ----------

    RiskMetric : :py:class:`str`, optional.
    - ``variance`` : Sample Covariance of the returns of the assets as the risk metric. (Default)
    - ``semivariance`` : Sample Semi-Covariance of the returns of the assets as the risk metric.
    - ``shrinkage`` : Shrinked Covariance as the risk metric.
    - ``letoidwolf`` : Shrinked Covariance by the Letoid-Wolf method as the risk metric.
    - ``oas`` : Shrinked Covariance by the Oracle Approximating method as the risk metric.


    Functions
    ---------
    optimize() : Returns the optimal weights using inverse variance algorithm.
    """

    def __init__(self, RiskMetrics: str = "Variance"):

        # Define the metric to be employed as risk
        if RiskMetrics.lower() not in ["variance", "semivariance", "shrinkage", "letoidwolf", "oas"]:
            raise ValueError(
                "RiskMetrics must be one of the following: 'Variance', 'Semivariance', 'Shrinkage', 'LetoidWolf' or 'OAS'")

        self._RiskMetrics = RiskMetrics.lower()

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

        # Check if the asset prices are valid
        if AssetPrices is not None:
            if not isinstance(AssetPrices, pd.DataFrame):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame!")
            if not isinstance(AssetPrices.index, pd.DatetimeIndex):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame index by datetime!")

        # Calculate asset returns
        asset_returns = AssetPrices.pct_change()
        asset_returns = asset_returns.dropna(
            axis="index", how='all').dropna(axis="columns", how='all')

        if self._RiskMetrics == "variance":
            risk_matrix = CovMatrix.Sample(asset_returns)

        elif self._RiskMetrics == "semivariance":
            risk_matrix = CovMatrix.SampleSemi(asset_returns)

        elif self._RiskMetrics == "shrinkage":
            risk_matrix = CovMatrix.BasicShrinkage(asset_returns)

        elif self._RiskMetrics == "letoidwolf":
            risk_matrix = CovMatrix.LedoitWolf(asset_returns)

        elif self._RiskMetrics == "oas":
            risk_matrix = CovMatrix.Oracle(asset_returns)

        # Remove the weights of the assets that are >1-e8.
        Portifolio = 1. / np.diag(risk_matrix)
        Portifolio = Portifolio.round(8)
        Portifolio /= Portifolio.sum()
        Portifolio[Portifolio == 0] = 'nan'
        Portifolio = pd.Series(Portifolio, index=asset_returns.columns)
        Portifolio = Portifolio.to_frame().T.dropna(axis="columns")
        return Portifolio
