import numpy as np


def corr_to_cov(corr, std) -> np.ndarray:
    """
    Convert a correlation matrix to a covariance matrix.

    Parameters
    ----------
    corr : :py:class:`numpy.ndarray`, Correlation matrix to be converted.

    std : :py:class:`numpy.ndarray` Standard deviation of the assets.

    Returns
    -------
    cov : :py:class:`numpy.ndarray`, Covariance matrix.
    """
    cov = corr * np.outer(std, std)
    return cov


def cov_to_corr(cov) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : :py:class:`numpy.ndarray`, Covariance matrix to be converted.

    Returns
    -------
    corr : :py:class:`numpy.ndarray`, Correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr
