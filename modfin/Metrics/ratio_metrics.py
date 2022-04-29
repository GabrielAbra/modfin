import numpy as np
import pandas as pd
from modfin.Analysis import ReturnAnalysis
from modfin.Metrics import ReturnMetrics
from modfin.Metrics import RiskMetrics


class RatioMetrics():
    """
    Module containing functions to calculate risk-return ratios.
    """

    @staticmethod
    def SharpeRatio(AssetPrice: pd.Series, RiskFree: float = 0, **kwargs) -> float:
        """
        Calculate the Sharpe Ratio of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        RiskFree : :py:class:`float` return of the risk-free rate on the same period. (Default: 0)

        Period : :py:class:`str` period of the asset prices.
            - 'days' for daily prices (Default)
            - 'weeks' for weekly prices
            - 'months' for monthly prices
            - 'years' for annual prices

        Return
        ----------
        SharpeRatio : :py:class:`float`
        """
        RiskFree = (1 + RiskFree) ** (252 / len(AssetPrice)) - 1

        return (ReturnMetrics.AnnualizedReturn(AssetPrice, **kwargs) - RiskFree) / RiskMetrics.Volatility(AssetPrice)

    @staticmethod
    def SortinoRatio(AssetPrice: pd.Series, RiskFree: float = 0, **kwargs) -> float:
        """
        Calculate the Sortino Ratio of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        RiskFree : :py:class:`float` return of the risk-free rate on the same period. (Default: 0)

        Return
        ----------
        SortinoRatio : :py:class:`float`
        """
        RiskFree = (1 + RiskFree) ** (252 / len(AssetPrice)) - 1
        return (ReturnMetrics.AnnualizedReturn(AssetPrice, **kwargs) - RiskFree) / RiskMetrics.DownsideRisk(AssetPrice)

    @staticmethod
    def TreynorRatio(AssetPrice: pd.Series, Benchmark: pd.Series, RiskFree: float, **kwargs) -> float:
        """
        Calculate the Treynor Ratio of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.

        RiskFree : :py:class:`float` return of the risk-free rate on the same period. (default: 0)

        Return
        ----------
        TreynorRatio: :py:class:`float`
        """
        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])

        AssetPrice = AssetPrice[stard_date:end_date]
        Benchmark = Benchmark[stard_date:end_date]

        RiskFree = (1 + RiskFree) ** (252 / len(AssetPrice)) - 1

        return (ReturnMetrics.AnnualizedReturn(AssetPrice, **kwargs) - RiskFree) / RiskMetrics.Beta(AssetPrice, Benchmark)

    @staticmethod
    def InformationRatio(AssetPrice: pd.Series, Benchmark: pd.Series) -> float:
        """
        Calculate the Treynor Ratio of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.

        Return
        ----------
        TreynorRatio: :py:class:`float`
        """
        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])

        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])

        te = ((AssetPrice - Benchmark)**2).sum() / (len(AssetPrice) - 1)
        te = np.sqrt(te)

        num_anos = float(len(AssetPrice)) / 252
        AssetPrice = np.insert(AssetPrice, 0, 1).prod() ** (1 / num_anos) - 1
        Benchmark = np.insert(Benchmark, 0, 1).prod() ** (1 / num_anos) - 1
        return (AssetPrice - Benchmark) / te

    @staticmethod
    def OmegaRatio(AssetPrice: pd.Series, threshold: float = 0) -> float:
        """
        Calculate the Omega Ratio of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        threshold : :py:class:`float` minimum annual return required. (Default: 0)

        Return
        ----------
        OmegaRatio : :py:class:`float`
        """
        threshold = (threshold + 1) ** (1 / 252) - 1
        returns = ReturnAnalysis.VetorizedReturns(AssetPrice)
        pos = returns[returns >= threshold].sum()
        neg = abs(returns[returns < threshold].sum())

        if pos == neg:
            neg += threshold

        if neg == 0:
            return (pos + 1e-4) / (1e-4)
        return pos / neg

    @staticmethod
    def CalmarRatio(AssetPrice: pd.DataFrame) -> float:
        ""

        daily_return = ReturnAnalysis.VetorizedReturns(AssetPrice)

        if len(daily_return) < 2:
            return np.nan

        returns_cum = np.insert(daily_return + 1, 0, 1).cumprod()
        roll_max = np.maximum.accumulate(returns_cum)
        drawdown = (returns_cum - roll_max) / roll_max
        max_drawdown = np.abs(np.min(drawdown))

        return returns_cum[-1] / max_drawdown

    @staticmethod
    def TailRatio(AssetPrice: pd.Series, Alpha=0.05) -> float:
        """ Calculate the proportion of the two tails of the asset price distribution.

        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.

        Alpha : :py:class:`float` significance level.

        Return
        ----------
        TailRatio : :py:class:`float`
        """
        daily_return = ReturnAnalysis.VetorizedReturns(AssetPrice)

        return np.quantile(daily_return, 1 - Alpha) / np.quantile(daily_return, Alpha)

    # @staticmethod
    # def M2Ratio():
    #     """
    #     ** Under construction **"""
    #     pass

    @staticmethod
    def HurstExponent(AssetPrice: pd.Series, max_lag: int = 21) -> float:
        """
        Calculate the Hurst Exponent of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series`

        max_lag: :py:class:`int` Tamanho das janelas 'lag' analizadas (i.e., tamanho das analises).

        Interoperability
        ----------
        H < 0.5 — Mean reverting

        H = 0.5 — Geometric Brownian Motion

        H > 0.5 — Monotonic Trend

        Return
        ----------
        H: :py:class:`float`
        """

        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(AssetPrice[lag:], AssetPrice[:-lag])))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
