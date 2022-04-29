import numpy as np


class PortfolioMetrics():
    @staticmethod
    def ExpectedVariance(CovarianceMatrix, Weights):
        """
        Calculate the expected variance of a portfolio.

        Parameters
        ----------
        CovarianceMatrix : :py:class:`numpy.ndarray`, Covariance matrix of the assets.

        Weights : :py:class:`numpy.ndarray`, Weights of the assets.

        Returns
        -------
        Variance : :py:class:`float`, Expected variance of the portfolio.
        """
        return np.dot(np.dot(Weights, CovarianceMatrix), Weights)
