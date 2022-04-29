import pandas as pd
import numpy as np
import scipy.optimize as so
from modfin.Analysis import CovMatrix


class RiskParity():
    """
    Risk Parity Portfolio Optimization

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
    optimize() : Returns the optimal weights using risk parity algorithm.
    """

    def __init__(self, RiskMetrics: str = "Variance"):

        # Define the metric to be employed as risk
        if RiskMetrics.lower() not in ["variance", "semivariance", "shrinkage", "letoidwolf", "oas"]:
            raise ValueError(
                "RiskMetrics must be one of the following: 'Variance', 'Semivariance', 'Shrinkage', 'LetoidWolf' or 'OAS'")
        self._RiskMetrics = RiskMetrics.lower()

    @staticmethod
    def _PortfolioRisk(Weights, RiskMatrix):

        # Calculate the portfolio risk
        portfolio_risk = np.sqrt((Weights * RiskMatrix * Weights.T))[0, 0]
        return portfolio_risk

    @staticmethod
    def _AssetRiskContribution(Weights, RiskMatrix):

        # Calculate the portfolio riskn
        portfolio_risk = RiskParity._PortfolioRisk(Weights, RiskMatrix)

        # Calculate the risk contribution of each asset
        assets_risk_contribution = np.multiply(
            Weights.T, RiskMatrix * Weights.T) / portfolio_risk
        return assets_risk_contribution

    @staticmethod
    def _RiskContributionError(Weights, args):

        RiskMatrix = args[0]

        RiskBudget = args[1]

        Weights = np.matrix(Weights)

        portfolio_risk = RiskParity._PortfolioRisk(Weights, RiskMatrix)

        assets_risk_contribution = RiskParity._AssetRiskContribution(
            Weights, RiskMatrix)

        assets_risk_target = np.asmatrix(
            np.multiply(portfolio_risk, RiskBudget))

        error = sum(np.square(
            assets_risk_contribution - assets_risk_target.T))[0, 0]

        # It returns the calculated error
        return error

    @staticmethod
    def _RiskParityWeights(RiskMatrix):

        # Restrictions to consider in the optimisation: only long positions whose
        # sum equals 100%
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                       {'type': 'ineq', 'fun': lambda x: x})

        InicialWeights = [1 / RiskMatrix.shape[1]] * RiskMatrix.shape[1]
        RiskBudget = [1 / RiskMatrix.shape[1]] * RiskMatrix.shape[1]

        # Optimisation process in scipy
        optimize_result = so.minimize(fun=RiskParity._RiskContributionError,
                                      x0=InicialWeights,
                                      args=[RiskMatrix, RiskBudget],
                                      method='SLSQP',
                                      constraints=constraints,
                                      tol=1e-10,
                                      options={'disp': False})

        # Recover the Weights from the optimised object
        Weights = optimize_result.x

        # It returns the optimised Weights
        return Weights

    def optimize(self, AssetPrices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the optimal portfolio allocation using the Risk Parity algorithm

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
        asset_returns = AssetPrices.pct_change().dropna(
            axis="index", how='all').dropna(axis="columns", how='all')

        asset_names = asset_returns.columns

        # Calculate the risk matrix
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

        # Vetorize the risk matrix and annualize it
        if isinstance(risk_matrix, pd.DataFrame):
            risk_matrix = risk_matrix.values * 252.0

        else:
            risk_matrix = risk_matrix * 252.0

        # Get the weights of the portfolio using Risk Parity algorithm
        Portifolio = RiskParity._RiskParityWeights(risk_matrix)

        Portifolio = pd.Series(Portifolio, index=asset_names)
        Portifolio = Portifolio.round(8)
        Portifolio /= Portifolio.sum()
        Portifolio[Portifolio == 0] = 'nan'
        Portifolio = Portifolio.to_frame().T.dropna(axis="columns")
        return Portifolio
