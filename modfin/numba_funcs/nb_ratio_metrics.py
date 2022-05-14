import numpy as np
import numba

from modfin.numba_funcs.nb_series import synthetic_prices


@numba.njit
def numba_calmar_ratio(asset_returns):
    """
    Numba Jit Compiled Calmar ratio.
    """
    prices = synthetic_prices(asset_returns)

    max_price = np.ones(prices.shape[0])
    max_price[0] = prices[0]

    for i in range(1, prices.shape[0]):
        max_price[i] = max(max_price[i - 1], prices[i])

    drawdown = (prices / max_price) - 1
    total_return = prices[-1] / prices[0] - 1

    return total_return / np.min(drawdown) * -1


@numba.njit(fastmath=True)
def numba_hurst(returns_array: np.ndarray):
    """
    Numba Jit Compiled ln(R/S) and ln(n) calculation
    """

    # Get returns size
    N = len(returns_array)

    # Calculate the min two power available
    max_base2 = int(np.floor(np.log(N) / np.log(2)))
    N = 2**max_base2

    # Calculate the sample sizes
    samples_size = np.array([2**i for i in range(1, max_base2)])

    # Assert that the sample size is sufficient
    if 2 ** max_base2 < 32:
        raise ValueError("returns_array must be greater than 32")

    # Restrict the sample sizes
    returns_array = returns_array[-N:]
    # Create arrays to store results
    log_rs = np.zeros(len(samples_size))
    log_lag = np.zeros(len(samples_size))

    # Initial positions in sample sizing
    location = 0

    # Loop through sample sizes
    for lag in samples_size:
        R = 0
        S = 0

        sub_samples = [returns_array[i:i + lag] for i in range(0, N, lag)]

        for sample in sub_samples:
            cumsum_sample = np.cumsum(sample - sample.mean())
            R += max(cumsum_sample) - min(cumsum_sample)
            S += np.std(sample)

        R /= len(sub_samples)
        S /= len(sub_samples)
        log_rs[location] = np.log(R / S)
        log_lag[location] = np.log(lag)
        location += 1

    return log_lag, log_rs
