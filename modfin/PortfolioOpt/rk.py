import pandas as pd
import numpy as np
import scipy.optimize as so
from .base import PortifolioOptBase


class RiskParity(PortifolioOptBase):
    """
    Risk Parity Portfolio Optimization

    Parameters
    ----------

    RiskMatrix : :py:class:`pd.DataFrame` a matrix of the risk implied by the returns of the assets.
    If possible from Modfin RiskMatrix Module.

    Functions
    ---------

    optimize() : Returns the optimal weights using risk parity algorithm.
    """

    def __init__(self, RiskMatrix: pd.DataFrame):

        # Check if the RiskMatrix is valid.
        self._RiskMatrix = self._check_rm(RiskMatrix)

        # Get Asset names
        self._asset_names = self._get_asset_names(RiskMatrix)

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

    def optimize(self) -> pd.DataFrame:
        """
        Calculate the optimal portfolio allocation using the Risk Parity algorithm

        Parameters
        ----------
        AssetPrices : py:class:`pandas.DataFrame` with the daily asset prices

        Returns
        -------
        Portifolio : :py:class:`pandas.DataFrame` with the weights of the portfolio
        """

        Weight = RiskParity._RiskParityWeights(self._RiskMatrix)

        # Create and return the portfolio
        return self._pandas_portifolio(Weight, self._asset_names)
