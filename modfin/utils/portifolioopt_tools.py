import numpy as np


def expected_variance(covariance_matrix, weights):
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
    return np.dot(np.dot(weights, covariance_matrix), weights)


def expected_return(returns, weights):
    """
    Calculate the expected return of a portfolio.

    Parameters
    ----------
    Returns : :py:class:`numpy.ndarray`, Returns of the assets.

    Weights : :py:class:`numpy.ndarray`, Weights of the assets.

    Returns
    -------
    Return : :py:class:`float`, Expected return of the portfolio.
    """
    return np.dot(returns, weights)
