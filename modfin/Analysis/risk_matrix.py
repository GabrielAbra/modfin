import numpy as np
import pandas as pd
import sklearn.covariance as skcov


class RiskMatrix():

    """
    The Risk Matrix Module provides multiple functions for analyzing time series data
    and generating risk matrices.

    There are two different types of algorithms that can
    be distinguished as `Sample`, `Estimator` and `Shrinkage` algorithms.

    ### Sample Algorithms
        sample_covariance
        semi_covariance [1]

    ### Estimator Algorithms
        mindet_covariance (Minimum Determinant Covariance)
        empirical_covariance (Maximum Likelihood Covariance)

    ### Shrinkage Algorithms

        shrinkage_covariance (Basic Shrinkage)
        ledoitwolf_covariance (Ledoit-Wolf Shrinkage Method)
        oracle_covariance (Oracle Approximating Shrinkage)

    #### References

    [1] Estrada, Javier. "Mean-semivariance optimization: A heuristic approach."
    Journal of Applied Finance (Formerly Financial Practice and Education) 18.1 (2008)
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def sample_covariance(returns_frame: pd.DataFrame, as_pandas=True):
        """
        Calculate the basic sample covariance matrix of the asset returns.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------

        sample_covariance : `pd.DataFrame` or `numpy.ndarray`
            Sample covariance matrix.
        """
        if as_pandas:
            return returns_frame.cov() * np.sqrt(252)

        return returns_frame.cov().values * np.sqrt(252)

    @staticmethod
    def semi_covariance(returns_frame: pd.DataFrame, threshold: float = 0, as_pandas=True):
        """
        Calculate the covariance matrix of the from the asset returns below a
        given threshold.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        threshold : `float`, (optional)
            Threshold for the semivariance computation.

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------
        semi_covariance : `pd.DataFrame` or `numpy.ndarray`
            Semi-covariance matrix.
        """

        lower_threshold = returns_frame - threshold < 0
        min_returns_frame = ((returns_frame - threshold) * lower_threshold)
        min_returns_frame = min_returns_frame.values
        semi_cov_matrix = returns_frame.cov().values

        for row in range(semi_cov_matrix.shape[0]):
            for col in range(semi_cov_matrix.shape[1]):
                row_values = min_returns_frame[:, row]
                col_values = min_returns_frame[:, col]
                cramer_cov = row_values * col_values
                semi_cov_matrix[row,
                                col] = cramer_cov.sum() / min_returns_frame.size

        semi_cov_matrix *= np.sqrt(252)

        if as_pandas:
            semi_cov_matrix = pd.DataFrame(
                semi_cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return semi_cov_matrix

    @staticmethod
    def shrinkage_covariance(returns_frame: pd.DataFrame, alpha=0.1, as_pandas=True):
        """
        Calculate the covariance matrix of the asset returns using
        the basic shrinkage method.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        alpha : `float`, (optional)
            Shrinkage parameter coefficient in the convex combination of the
            sample covariance matrix and the shrinkage estimator.
            alpha~[0,1] (Default: 0.1)

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------
        shrinkage_covariance : `pd.DataFrame` or `numpy.ndarray`
            Shrunk covariance matrix.
        """
        cov_matrix = skcov.ShrunkCovariance(
            assume_centered=True, shrinkage=alpha).fit(returns_frame).covariance_

        cov_matrix *= np.sqrt(252)

        if as_pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return cov_matrix

    @staticmethod
    def ledoitwolf_covariance(returns_frame: pd.DataFrame, as_pandas=True):
        """
        Calculate the covariance matrix of the asset returns using the Ledoit-Wolf estimator.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------

        ledoitwolf_covariance : `pd.DataFrame` or `numpy.ndarray`
            Ledoit-Wolf covariance matrix.

        References
        ---------
        O. Ledoit and M. Wolf, “A Well-Conditioned Estimator for Large-Dimensional
        Covariance Matrices”, Journal of Multivariate Analysis,
        Volume 88, Issue 2, February 2004, pages 365-411.
        """
        cov_matrix = skcov.LedoitWolf(
            assume_centered=True).fit(returns_frame).covariance_

        cov_matrix *= np.sqrt(252)

        if as_pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return cov_matrix

    @staticmethod
    def oracle_covariance(returns_frame: pd.DataFrame, as_pandas=True):
        """
        Calculate the covariance matrix of the asset returns using the Oracle
        Approximating Shrinkage estimator. Which takes in the assumption that
        the assets return are normally distributed.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------
        oracle_covariance : `pd.DataFrame` or `numpy.ndarray`
            OAS covariance matrix.

        References
        ---------
        Chen, Yilun, et al. "Shrinkage algorithms for MMSE covariance estimation."
        IEEE Transactions on Signal Processing 58.10 (2010): 5016-5029.
        """

        cov_matrix = skcov.OAS(assume_centered=True).fit(
            returns_frame).covariance_

        cov_matrix *= np.sqrt(252)

        if as_pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return cov_matrix

    @staticmethod
    def mindet_covariance(returns_frame: pd.DataFrame, as_pandas=True):
        """
        Calculate the covariance matrix of the asset returns using the Minimum
        Covariance Determinant (robust estimator of covariance).

        A robust estimar for covariance matrices introduces by P.J. Rousseeuw.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------
        mindet_covariance : `pd.DataFrame` or `numpy.ndarray`
            Minimum Covariance Determinant covariance matrix.

        References
        ---------
        P.J. Rousseeuw, “Least median of squares regression,” Journal of the American Statistical Association, Vol. 74, No. 353, pp. 714-716, 1984.
        """
        cov_matrix = skcov.MinCovDet(
            assume_centered=True).fit(returns_frame).covariance_

        cov_matrix *= np.sqrt(252)

        if as_pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return cov_matrix

    @staticmethod
    def empirical_covariance(returns_frame: pd.DataFrame, as_pandas=True):
        """
        Calculate the covariance matrix of the asset returns using the
        Maximum likelihood covariance estimator.

        Parameters
        ---------

        returns_frame : `pd.DataFrame`
            DataFrame with asset returns with asset names as columns and dates as index

        as_pandas : `bool`, (optional)
            If True, the covariance matrix is returned as a pandas.DataFrame,
            otherwise the return is a numpy.ndarray (Default: True)

        Returns
        ------
        empirical_covariance : `pd.DataFrame` or `numpy.ndarray`
            Empirical covariance matrix.
        """

        cov_matrix = skcov.EmpiricalCovariance(
            assume_centered=True).fit(returns_frame).covariance_

        cov_matrix *= np.sqrt(252)

        if as_pandas:
            cov_matrix = pd.DataFrame(
                cov_matrix,
                index=returns_frame.columns,
                columns=returns_frame.columns)

        return cov_matrix
