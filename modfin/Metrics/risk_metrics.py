import numpy as np
import pandas as pd
from scipy import stats, optimize
from modfin.Analysis import ReturnAnalysis


class RiskMetrics():
    """
    Module containing functions to calculate risk metrics.
    """

    @staticmethod
    def Volatility(AssetPrice: pd.Series, Period: str = 'days') -> float:
        """
        Calculate the realized volatility of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Period : :py:class:`str` period of the asset prices.
            - 'days' for daily prices (Default)
            - 'weeks' for weekly prices
            - 'months' for monthly prices
            - 'years' for annual prices
        Return
        ----------
        Volatility: :py:class:`float`
        """
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]

        if len(AssetPrice) < 2:
            return np.nan

        Period_params = {
            'days': 252,
            'weeks': 52,
            'months': 12,
            'years': 1}

        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        return returns.std(ddof=0) * np.sqrt(Period_params[Period])

    @staticmethod
    def DownsideRisk(AssetPrice: pd.Series) -> float:
        """
        Calculate the Downside Risk of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        DownsideRisk: :py:class:`float`
        """

        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]
        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        return returns[returns < 0].std(ddof=0) * np.sqrt(252)

    @staticmethod
    def UpsideRisk(AssetPrice: pd.Series) -> float:
        """
        Calculate the Upside Risk of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        UpsideRisk: :py:class:`float`
        """
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]
        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        return returns[returns > 0].std(ddof=0) * np.sqrt(252)

    @staticmethod
    def VolSkew(AssetPrice: pd.Series) -> float:
        """
        Calculate the Volatility Skewness of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        VolSkew : :py:class:`float`
        """
        if not isinstance(AssetPrice, np.ndarray):
            AssetPrice = np.array(AssetPrice)

        AssetPrice[AssetPrice == 0] = 'nan'
        AssetPrice = AssetPrice[~np.isnan(AssetPrice)]
        returns = (AssetPrice[1:] / AssetPrice[:-1]) - 1
        StdNeg = returns[returns < 0].std(ddof=0) * np.sqrt(252)
        StdPos = returns[returns > 0].std(ddof=0) * np.sqrt(252)
        return StdPos / StdNeg

    @staticmethod
    def TrackingError(AssetPrice: pd.Series, Benchmark: pd.Series) -> float:
        """
        Calculate the Tracking Error of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Benchmark: :py:class:`pandas.Series` daily prices of the benchmark.
        Return
        ----------
        TrackingError: :py:class:`float`
        """
        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])

        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])

        return np.sqrt(((AssetPrice - Benchmark)**2).sum() / (len(AssetPrice) - 1))

    @staticmethod
    def InfoDisk(AssetPrice: pd.Series) -> float:
        """
        Calculate the Information Discretness of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        InfoDisk : :py:class:`float`
        """

        returns = ReturnAnalysis.VetorizedReturns(AssetPrice)
        tot_ret_sign = np.sign((1 + returns).prod() - 1)
        pos_pct = len(returns[returns > 0.0]) / len(returns)
        neg_pct = len(returns[returns < 0.0]) / len(returns)

        if neg_pct == pos_pct:
            neg_pct += 1 / len(returns)

        return tot_ret_sign * (neg_pct - pos_pct)

    @staticmethod
    def MagInfoDisk(AssetPrice: pd.Series, Mag=4) -> float:
        """
        Calculate the pondered Information Discretness of a given asset or portfolio.
        The ponderation is based on the magnitude of the absolute value of the return.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        MagInfoDisk : :py:class:`float`
        """

        Return = ReturnAnalysis.VetorizedReturns(AssetPrice)
        mag = np.array([num / Mag for num in range(1, Mag)])
        tot_ret_sign = np.sign((1 + Return).prod() - 1)
        abs_ret = np.abs(Return)

        bins = np.quantile(abs_ret, [0, 0.25, 0.5, .75, 1])
        weights = (np.digitize(abs_ret, bins) + 1) / (10 * (mag.sum() - 0.5))

        idmag = (np.sign(Return) * weights).sum() / len(Return)
        idmag *= tot_ret_sign
        return idmag

    @staticmethod
    def RelativeInfoDisk(AssetPrice: pd.Series) -> float:
        """
        Calculate the relative Information Discretness of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        RelativeInfoDisk : :py:class:`float`
        """
        returns = ReturnAnalysis.VetorizedReturns(AssetPrice)
        tot_ret_sign = np.sign((1 + returns).prod() - 1)

        pos_pct = len(returns[returns > 0.0])
        neg_pct = len(returns[returns < 0.0])

        if tot_ret_sign > 0:
            return tot_ret_sign * (pos_pct - neg_pct) / len(returns)
        return tot_ret_sign * (neg_pct - pos_pct) / len(returns)

    @staticmethod
    def VaR(AssetPrice: pd.Series, Alpha: float = 0.05, Days: int = 1) -> float:
        """
        Calculate the Value at Risk of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Alpha : :py:class:`float` significance level.
        Days :py:class:`int` days of consecutive losses.

        Return
        ----------
        VaR : :py:class:`float`
        """

        Return = ReturnAnalysis.VetorizedReturns(AssetPrice)
        vol = Return.std(ddof=0) * np.sqrt(Days)
        ret = ((1 + np.mean(Return))**Days) - 1
        return vol * stats.norm.ppf(1 - Alpha) - ret

    @staticmethod
    def CVaR(AssetPrice: pd.Series, Alpha: float = 0.05, Days: int = 1) -> float:
        """
        Calculate the Conditional Value at Risk of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Alpha: :py:class:`float` significance level.
        Days :py:class:`int` days of consecutive losses.

        Return
        ----------
        CVaR: :py:class:`float`
        """
        Return = ReturnAnalysis.VetorizedReturns(AssetPrice)
        vol = Return.std(ddof=0) * np.sqrt(Days)
        ret = ((1 + np.mean(Return))**Days) - 1
        return Alpha**-1 * stats.norm.pdf(stats.norm.ppf(Alpha)) * vol - ret

    @staticmethod
    def _Ent(z, ret, Alpha=0.05):
        ret_dim = np.array(ret, ndmin=2)
        if ret_dim.shape[0] == 1 and ret_dim.shape[1] > 1:
            ret_dim = ret_dim.T
        if ret_dim.shape[0] > 1 and ret_dim.shape[1] > 1:
            raise ValueError
        value = np.mean(np.exp(-1 / z * ret_dim), axis=0)
        value = z * (np.log(value) + np.log(1 / Alpha))
        value = np.array(value).item()
        return value

    @staticmethod
    def EVaR(AssetPrice, Alpha: float = 0.05) -> float:
        """
        Calculate the Entropic Value at Risk of a given asset or portfolio    ----------
        # Parameters
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Alpha : :py:class:`float` significance level.
        # Return
        EVaR : :py:class:`float`
        """

        ret = ReturnAnalysis.VetorizedReturns(AssetPrice)
        ret_dim = np.array(ret, ndmin=2)

        if ret_dim.shape[0] == 1 and ret_dim.shape[1] > 1:
            ret_dim = ret_dim.T
        if ret_dim.shape[0] > 1 and ret_dim.shape[1] > 1:
            raise ValueError

        bnd = optimize.Bounds([1e-12], [np.inf])
        min = optimize.minimize(
            RiskMetrics._Ent, [1], args=(ret, Alpha), method="SLSQP", bounds=bnd, tol=1e-12).x.item()
        value = RiskMetrics._Ent(min, ret, Alpha)
        return value

    @staticmethod
    def EVaR_Normal(AssetPrice: pd.Series, Alpha: float = 0.05, Days: int = 1) -> float:
        """
        A funcao `EVaR_Normal` é um `Proxy` para o Entropic Value at Risk of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Alpha : :py:class:`float` significance level.
        Days :py:class:`int` days of consecutive losses.
        Return
        ----------
        EVaR: :py:class:`float`
        Obs.: O proxy foi adicionado ao modulo, em razao da complexidade dos calculos, que podem demorar muito em algumas maquinas.
        """
        Return = ReturnAnalysis.VetorizedReturns(AssetPrice)
        vol = Return.std(ddof=0) * np.sqrt(Days)
        ret = ((1 + np.mean(Return))**Days) - 1
        return ret + np.sqrt(-2 * np.log(Alpha)) * vol

    @staticmethod
    def CDaR(AssetPrice: pd.Series, Alpha: float = 0.05) -> float:
        """
        Calculate the Conditional Drawdown at Risk de um AssetPrice ou portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        """

        Returndiario = ReturnAnalysis.VetorizedReturns(AssetPrice)

        if len(Returndiario) < 2:
            return np.nan

        drawdown = np.maximum.accumulate(Returndiario) - Returndiario
        max_dd = np.maximum.accumulate(drawdown)
        max_dd_Alpha = np.quantile(max_dd, 1 - Alpha, interpolation='higher')
        cond_dd = np.nanmean(max_dd[max_dd >= max_dd_Alpha])
        return cond_dd

    @staticmethod
    def MaxDrawdown(AssetPrice: pd.Series) -> float:
        """
        Calculate the Maximum Drawdown of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        MaxDrawdown : :py:class:`float`
        """

        Returndiario = ReturnAnalysis.VetorizedReturns(AssetPrice)

        if len(Returndiario) < 2:
            return np.nan

        acumulado = np.insert(Returndiario + 1, 0, 1).cumprod()
        roll_max = np.maximum.accumulate(acumulado)
        drawdown = (acumulado - roll_max) / roll_max
        return np.abs(np.min(drawdown))

    @staticmethod
    def Alpha(AssetPrice: pd.Series, Benchmark: pd.Series):
        """
        Calculate the Alpha from Capital Market Model (CAPM) of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.
        Return
        ----------
        Alpha : :py:class:`float`
        """

        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])
        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])
        num_anos = float(len(AssetPrice)) / 252

        cov = np.cov(AssetPrice, Benchmark)
        beta = cov[0][1] / cov[0][0]

        AssetPrice = (np.insert(AssetPrice + 1, 0, 1).prod()
                      ) ** (1 / num_anos) - 1
        Benchmark = (np.insert(Benchmark + 1, 0, 1).prod()
                     ) ** (1 / num_anos) - 1
        return AssetPrice - (beta * Benchmark)

    @staticmethod
    def Beta(AssetPrice: pd.Series, Benchmark: pd.Series):
        """
        Calculate the Beta from Capital Market Model (CAPM) of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.
        Return
        ----------
        Beta : :py:class:`float`
        """

        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])
        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])

        cov = np.cov(AssetPrice, Benchmark)
        _beta = cov[0][1] / cov[0][0]

        return _beta

    @staticmethod
    def AutoCorrScore(AssetPrice: pd.Series, MaxLag: int = 21) -> float:
        """
        Calculate the returns autocorrelation score.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        MaxLag : :py:class:`int` maximum window lag to be considered.
        Return
        ----------
        AutoCorrScore : :py:class:`float`
        """

        daily_return = ReturnAnalysis.VetorizedReturns(AssetPrice)
        daily_return = daily_return[1:]
        lags = np.arange(1, MaxLag)
        ACF = [np.corrcoef(daily_return[:-lag], daily_return[lag:])[0, 1]
               for lag in lags]
        return np.log(np.abs(ACF)).mean() / np.log(lags)

    @staticmethod
    def DownsideBeta(AssetPrice: pd.Series, Benchmark: pd.Series) -> float:
        """
        Calculate the Negative Beta of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.
        Return
        ----------
        NegativeBeta : :py:class:`float`
        """

        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])
        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])

        cov = np.cov(AssetPrice, Benchmark)
        _beta = cov[0][1] / cov[0][0]

        neg_cov = np.cov(AssetPrice[Benchmark < 0], Benchmark[Benchmark < 0])
        _neg_beta = neg_cov[0][1] / neg_cov[0][0]

        return _neg_beta / _beta

    @staticmethod
    def BetaQuotient(AssetPrice: pd.Series, Benchmark: pd.Series) -> float:
        """
        ** Under construction **
        Calculate the Beta Quotient of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Benchmark : :py:class:`pandas.Series` daily prices of the benchmark.
        Return
        ----------
        BetaQuotient : :py:class:`float`
        """

        stard_date = max(AssetPrice.index[0], Benchmark.index[0])
        end_date = min(AssetPrice.index[-1], Benchmark.index[-1])
        AssetPrice = ReturnAnalysis.VetorizedReturns(
            AssetPrice[stard_date:end_date])
        Benchmark = ReturnAnalysis.VetorizedReturns(
            Benchmark[stard_date:end_date])

        cov = np.cov(AssetPrice, Benchmark)
        _beta = cov[0][1] / cov[0][0]

        return _beta

    @staticmethod
    def RSquaredScore(AssetPrice: pd.Series) -> float:
        """
        Calculate the R-Squared Score of a given asset or portfolio.
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        Return
        ----------
        RSquaredScore : :py:class:`float`
        """

        returns = ReturnAnalysis.VetorizedReturns(AssetPrice)

        cum_log_returns = np.log1p(returns).cumsum()

        score = stats.linregress(
            np.arange(len(cum_log_returns)), cum_log_returns)[2]

        return score * score

    @staticmethod
    def LPM(returns, threshold=0, order=1):
        """
        Calculate the Lower Partial Moment of a given asset or portfolio
        Parameters
        ----------
        AssetPrice : :py:class:`pandas.Series` daily prices of a given asset or portfolio.
        threshold : :py:class:`float` minimum annual return required. (Default: 0)
        order : :py:class:`int` degress of the polynomial. (Default: 1)
        Return
        -------
        LPM: :py:class:`float`
        """
        threshold_array = np.empty(len(returns))
        threshold_array.fill(threshold)
        diff = threshold_array - returns
        diff = diff.clip(min=0)
        return np.sum(diff ** order) / len(returns)
