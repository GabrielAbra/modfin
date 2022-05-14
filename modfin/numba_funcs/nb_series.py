"""
This module contains JIT Numba functions employed in the series module.
"""

import numba
import numpy as np


@numba.njit(fastmath=True)
def synthetic_prices(returns_series):
    """
    Calculate the synthetic prices of a series of returns.

    Parameters
    ----------
    returns_series : numpy.ndarray
        A series of returns.

    Returns
    -------
    numpy.ndarray
        A series of synthetic prices.
    """
    n = returns_series.shape[0] + 1

    price_series = np.zeros(n)
    price_series[0] = 1
    for i in range(1, n):
        price_series[i] = price_series[i - 1] * (1 + returns_series[i - 1])
    return price_series
