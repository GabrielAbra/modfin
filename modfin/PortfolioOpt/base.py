import numpy as np
import pandas as pd


class PortifolioOptBase():

    @staticmethod
    def _check_rm(RiskMatrix):
        """
        Generic function to check if the risk matrix is valid.
        """
        # Ensure that RiskMatrix is not null
        if RiskMatrix.shape[0] == 0:
            raise ValueError("RiskMatrix cannot be empty")

        # Check if the risk matrix a square matrix
        if RiskMatrix.shape[0] != RiskMatrix.shape[1]:
            raise ValueError("RiskMatrix must be a square matrix")

        # Check if the risk matrix is symmetric
        if not np.allclose(RiskMatrix, RiskMatrix.T):
            raise ValueError("RiskMatrix must be symmetric")

        # Check if the risk matrix is positive definite
        if not np.all(np.linalg.eigvals(RiskMatrix) > 0):
            raise ValueError("RiskMatrix must be positive definite")

        return RiskMatrix

    @staticmethod
    def _check_rm_ret(RiskMatrix, ExpectedReturn):
        """
        Generic function to check if the RiskMatrix and the ExpectedReturns are symmetric.
        """
        # Ensure that RiskMatrix is not null
        if RiskMatrix.shape[0] == 0:
            raise ValueError("RiskMatrix cannot be empty")

        # Ensure that ExpectedReturn is not null
        if ExpectedReturn.shape[0] == 0:
            raise ValueError("ExpectedReturn cannot be empty")

        # Check if shape of the RM and the ER are the same
        if RiskMatrix.shape[0] != ExpectedReturn.shape[0]:
            raise ValueError(
                "RiskMatrix and ExpectedReturn must have the same number of assets")

        return RiskMatrix, ExpectedReturn

    @staticmethod
    def _get_asset_names(RiskMatrix=None, ExpectedReturn=None):
        """
        Generic function to get the asset names.
        """
        if RiskMatrix is not None and isinstance(RiskMatrix, pd.DataFrame):
            return RiskMatrix.columns.values

        elif ExpectedReturn is not None and isinstance(ExpectedReturn, pd.Series):
            return ExpectedReturn.index.values

        else:
            return np.arange(1, RiskMatrix.shape[0] + 1)

    @staticmethod
    def _pandas_portifolio(Weights, AssetNames):
        """
        Create Portifolio in a pandas DataFrame format.
        """
        Portifolio = pd.Series(Weights, index=AssetNames)
        # Remove weights with lower than 1e-12
        Portifolio = Portifolio.round(12)
        Portifolio /= Portifolio.sum()
        Portifolio[Portifolio == -0] = 0
        Portifolio = Portifolio.to_frame().T
        return Portifolio
