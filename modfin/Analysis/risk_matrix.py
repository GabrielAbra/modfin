import numpy as np
import pandas as pd
import sklearn.covariance as skcov


class RiskMatrix():
    """
    The Risk Matrix Module provides multiple functions for analyzing time series data and generating risk matrices.
    There are two different types of algorithms that can be distinguished as `Sample`, `Estimator` and `Shrinkage` algorithms.

    ### Sample Algorithms
    - Covariance
    - Semicovariance [1]

    ### Estimator Algorithms

    - Minimum Covariance Determinant
    - Empirical Covariance

    ### Shrinkage Algorithms

    - Shrinkage (Basic Shrinkage)
    - LedoitWolf (Ledoit-Wolf Shrinkage Method)
    - Oracle (Oracle Approximating Shrinkage)

    ### References

    - [1] Estrada, Javier. "Mean-semivariance optimization: A heuristic approach."
        Journal of Applied Finance (Formerly Financial Practice and Education) 18.1 (2008).
    """
    @staticmethod
    def Covariance(AssetReturns: pd.DataFrame, Pandas=True):
        """
        Calculate the basic sample covariance matrix of the asset returns.

        Parameters
        ----------

        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Returns
        -------

        Matrix : :py:class:`numpy.ndarray`, Covariance matrix.
        """
        if Pandas:
            return AssetReturns.cov() * np.sqrt(252)

        return AssetReturns.cov().values * np.sqrt(252)

    @staticmethod
    def Semicovariance(AssetReturns: pd.DataFrame, threshold: float = 0, Pandas=True):
        """
        Calculate the covariance matrix of the from the asset returns below a certain threshold.

        Parameters
        ----------

        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        threshold : :py:class:`float`, Threshold for the semi-covariance matrix. (Default: 0)

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Returns
        -------
        SemiCovariance : :py:class:`numpy.ndarray`, Semi-covariance matrix.
        """
        lower_threshold = AssetReturns - threshold < 0
        min_AssetReturns = ((AssetReturns - threshold) * lower_threshold)
        min_AssetReturns = min_AssetReturns.values
        semi_cov_matrix = AssetReturns.cov().values

        for row in range(semi_cov_matrix.shape[0]):
            for col in range(semi_cov_matrix.shape[1]):
                row_values = min_AssetReturns[:, row]
                col_values = min_AssetReturns[:, col]
                cramer_cov = row_values * col_values
                semi_cov_matrix[row,
                                col] = cramer_cov.sum() / min_AssetReturns.size

        semi_cov_matrix *= np.sqrt(252)

        if Pandas:
            semi_cov_matrix = pd.DataFrame(
                semi_cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return semi_cov_matrix

    @staticmethod
    def LedoitWolf(AssetReturns: pd.DataFrame, Pandas=True, Centralized=True):
        """
        Calculate the covariance matrix of the asset returns using the Ledoit-Wolf estimator.

        Parameters
        ----------

        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Centralized : :py:class:`bool`, If True, the covariance matrix is centralized. (Default: True)

        Returns
        -------

        LedoitWolfCov : :py:class:`numpy.ndarray`, Ledoit-Wolf covariance matrix.

        References
        ----------
        O. Ledoit and M. Wolf, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices”, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411.
        """
        cov_matrix = skcov.LedoitWolf(
            assume_centered=Centralized).fit(AssetReturns).covariance_

        cov_matrix *= np.sqrt(252)

        if Pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return cov_matrix

    @staticmethod
    def Shrinkage(AssetReturns: pd.DataFrame, Alpha=0.1, Pandas=True, Centralized=True):
        """
        Calculate the covariance matrix of the asset returns using Basic Shrinkage method.

        Parameters
        ----------
        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Alpha : :py:class:`float`, Coefficient in the convex combination used for the computation of the shrunk estimate, ranging between 0 and 1. (Default: 0.1)

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Centralized : :py:class:`bool`, If True, the covariance matrix is centralized. (Default: True)

        Returns
        -------
        ShrunkedCov : :py:class:`numpy.ndarray`, Shrunk covariance matrix.
        """
        cov_matrix = skcov.ShrunkCovariance(
            assume_centered=Centralized, shrinkage=Alpha).fit(AssetReturns).covariance_

        cov_matrix *= np.sqrt(252)

        if Pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return cov_matrix

    @staticmethod
    def Oracle(AssetReturns: pd.DataFrame, Pandas=True, Centralized=True):
        """
        Calculate the covariance matrix of the asset returns using the Oracle Approximating Shrinkage estimator;

        Which takes in the assumption that the assets return are normally distributed.

        Parameters
        ----------
        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Centralized : :py:class:`bool`, If True, the covariance matrix is centralized. (Default: True)

        Returns
        -------
        OASCov : :py:class:`numpy.ndarray`, OAS covariance matrix.

        References
        ----------
        Chen, Yilun, et al. "Shrinkage algorithms for MMSE covariance estimation." IEEE Transactions on Signal Processing 58.10 (2010): 5016-5029.
        """

        cov_matrix = skcov.OAS(assume_centered=Centralized).fit(
            AssetReturns).covariance_

        cov_matrix *= np.sqrt(252)

        if Pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return cov_matrix

    @staticmethod
    def MinimumDet(AssetReturns: pd.DataFrame, Pandas=True, Centralized=True):
        """
        Calculate the covariance matrix of the asset returns using the Minimum Covariance Determinant (robust estimator of covariance);

        A robust estimar for covariance matrices introduces by P.J. Rousseeuw.

        Parameters
        ----------
        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Centralized : :py:class:`bool`, If True, the covariance matrix is centralized. (Default: True)

        Returns
        -------
        MinCovDet : :py:class:`numpy.ndarray`, Minimum Covariance Determinant covariance matrix.

        References
        ----------
        P.J. Rousseeuw, “Least median of squares regression,” Journal of the American Statistical Association, Vol. 74, No. 353, pp. 714–716, 1984.
        """
        cov_matrix = skcov.MinCovDet(
            assume_centered=Centralized).fit(AssetReturns).covariance_

        cov_matrix *= np.sqrt(252)

        if Pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return cov_matrix

    @staticmethod
    def Empirical(AssetReturns: pd.DataFrame, Pandas=True, Centralized=True):
        """
        Calculate the covariance matrix of the asset returns using the Maximum likelihood covariance estimator.

        Parameters
        ----------
        AssetReturns : :py:class:`pandas.DataFrame` Daily returns of the assets.

        Pandas : :py:class:`bool`, If True, the covariance matrix is returned as a pandas DataFrame with asset names (Default: False)

        Centralized : :py:class:`bool`, If True, the covariance matrix is centralized. (Default: True)

        Returns
        -------
        EmpiricalCov : :py:class:`numpy.ndarray`, Empirical covariance matrix.
        """

        cov_matrix = skcov.EmpiricalCovariance(
            assume_centered=Centralized).fit(AssetReturns).covariance_

        cov_matrix *= np.sqrt(252)

        if Pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix, index=AssetReturns.columns, columns=AssetReturns.columns)

        return cov_matrix
