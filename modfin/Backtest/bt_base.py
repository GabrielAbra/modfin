import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class BacktestBase():

    @staticmethod
    def valid_asset_prices(AssetPrices):
        # Check if the asset prices are valid
        if AssetPrices is None:
            raise ValueError("Asset Prices must be a Pandas DataFrame!")

        if AssetPrices is not None:
            if not isinstance(AssetPrices, pd.DataFrame):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame!")

            if not isinstance(AssetPrices.index, pd.DatetimeIndex):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame index by datetime!")
        return True

    @staticmethod
    def NominalizeAssetPrices(AssetPrices):
        if isinstance(AssetPrices, pd.DataFrame):
            return AssetPrices.apply(lambda x: x / x.dropna(axis="index", how='all').iloc[0])

        raise ValueError("AssetPrices must be a pandas DataFrame")

    @staticmethod
    def _rebalance_freq(frequency):
        """
        Get the frequency information from the rebalance frequency string.
        Returns
        -------
        freq_num : :py:class:`int` Integer with the frequency number.
        freq_type : :py:class:`str` String with the frequency type.
        """

        freq_num = filter(str.isdigit, frequency)
        freq_num = ''.join(freq_num)

        if freq_num == '':
            freq_num = 1
        else:
            freq_num = int(freq_num)

        freq_type = filter(str.isalpha, frequency.lower())
        freq_type = ''.join(freq_type)

        if freq_type not in ['d', 'w', 'm', 'y']:
            raise ValueError(
                "Period type must be one of the following: d (for days), w (for weeks), m (for months), y (for years).")

        return freq_num, freq_type

    @staticmethod
    def DatesRange(AssetPricesIndex, Frequency: str = "1m", adjusted: bool = True):
        """
        Create the data range for the backtest.
        Obs.: Dont support frequency lower than days.
        Parameters
        ----------
        AssetPricesIndex : :py:class:`pandas.DatetimeIndex` Index of the asset prices (DateTimeIndex).
        Frequency : :py:class:`str` String with the frequency of the rebalance.
        Adjusted : :py:class:`bool` Boolean indicating if the dates are adjusted to the first period available for the related frequency.
        Returns
        -------
        DatesRange : :py:class:`list` Index of the dates range.
        """
        if not isinstance(AssetPricesIndex, pd.DatetimeIndex):
            raise ValueError(
                "AssetPricesIndex must be a Pandas DataFrame Index")

        if not isinstance(Frequency, str):
            raise ValueError("frequency must be a string")

        if len(AssetPricesIndex) < 2:
            raise ValueError("AssetPricesIndex must have at least 2 dates")

        start_date = AssetPricesIndex[0]
        end_date = AssetPricesIndex[-1]

        if adjusted:
            date_range = pd.date_range(start_date, end_date, freq=Frequency)
            date_range = [date.date() for date in date_range]

        else:
            start_date = start_date.date()
            end_date = end_date.date()
            freq_num, freq_type = BacktestBase._rebalance_freq(Frequency)

            if freq_type == 'd':
                date_range = []
                while start_date < end_date:
                    date_range.append(start_date)
                    start_date += relativedelta(days=freq_num)

            elif freq_type == 'w':
                date_range = []
                while start_date < end_date:
                    date_range.append(start_date)
                    start_date += relativedelta(weeks=freq_num)

            elif freq_type == 'm':
                date_range = []
                while start_date < end_date:
                    date_range.append(start_date)
                    start_date += relativedelta(months=freq_num)

            elif freq_type == 'y':
                date_range = []
                while start_date < end_date:
                    date_range.append(start_date)
                    start_date += relativedelta(years=freq_num)

        # Check if the date frequency returns a valid date range
        if len(date_range) < 2:
            raise ValueError(
                "Frequency must be lower for the especified AssetPricesIndex")

        return date_range

    @staticmethod
    def ApplyWeights(AssetPrices, Weights, ReturnType="returns"):
        """
        Apply the weights to the asset prices.
        Parameters
        ----------
        AssetPrices : :py:class:`pandas.DataFrame` DataFrame with the daily asset prices.
        Weights : :py:class:`pandas.DataFrame` DataFrame with the weights to be applied on the asset prices.
        ReturnType : :py:class:`str` String with the type of return to be returned.
        - "returns" : Returns the asset prices multiplied by the weights.
        - "total" : Returns the total return of the asset prices multiplied by the weights.
        - "both" : Returns 'returns' and 'total'
        Returns
        -------
        portfolio: :py:class:`pandas.DataFrame` DataFrame with the portfolio values.
        """

        # Check if the ReturnType is valid
        if ReturnType not in ["returns", "total", "both"]:
            raise ValueError("ReturnType must be 'returns', 'total' or 'both'")

        # Check if the asset prices are valid
        if AssetPrices is not None:
            if not isinstance(AssetPrices, pd.DataFrame):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame!")
            if not isinstance(AssetPrices.index, pd.DatetimeIndex):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame index by datetime!")

        # Check if the weights are valid
        if not isinstance(Weights, pd.DataFrame):
            if isinstance(Weights, np.ndarray):
                UserWarning(
                    "Weights are a numpy array, the result suposed they are in the same order as the Asset Prices columns")
                Weights = pd.DataFrame(Weights, index=AssetPrices.columns)

            elif isinstance(Weights, pd.Series):
                Weights = Weights.to_frame().T.dropna(axis="columns")

            else:
                raise ValueError(
                    "Weights must be a Pandas DataFrame or a numpy array!")

        # Check if the assets of weights are the same as the assets of the asset prices
        if not set(Weights.columns).issubset(set(AssetPrices.columns)):
            raise ValueError(
                "Weights must be a DataFrame with the same columns as the Asset Prices")

        # Nominalize the asset prices
        NominalizeAssetPrices = BacktestBase.NominalizeAssetPrices(
            AssetPrices[Weights.columns])

        # Apply the weights to the asset prices
        portfolio_values = np.multiply(
            NominalizeAssetPrices.values, Weights.values)

        # Create the portfolio DataFrame
        portfolio_dataframe = pd.DataFrame(
            portfolio_values, index=NominalizeAssetPrices.index, columns=NominalizeAssetPrices.columns)

        if ReturnType == "returns":
            return portfolio_dataframe

        elif ReturnType == "total":
            return portfolio_dataframe.sum(axis=1).to_frame().rename(columns={0: "Return"})

        elif ReturnType == "both":
            return portfolio_dataframe, portfolio_dataframe.sum(axis=1).to_frame().rename(columns={0: "Return"})

        else:
            raise ValueError("ReturnType must be 'returns', 'total' or 'both'")

    @staticmethod
    def VectorizedApplyWeights(AssetPrices: pd.DataFrame, Weights: pd.DataFrame, ReturnType: str = "returns", ForceAdjusted=False):
        """
        Apply the weights to the asset prices using vectorized methods (i.e faster way)
        Alternative to :py:func:`ApplyWeights` with Numpy NdArrays.
        Obs: It is recommended for new users to use :py:func:`ApplyWeights` instead of this function.
        Becouse this function its optimized for perfomance and dont check if the parameters are valid.
        Parameters
        ----------
        AssetPrices : :py:class:`pandas.DataFrame` DataFrame with the daily asset prices.
        Weights : :py:class:`pandas.DataFrame` DataFrame with the weights to be applied on the asset prices.
        ReturnType : :py:class:`str` String with the type of return to be returned.
        - "returns" : Returns the asset prices multiplied by the weights.
        - "total" : Returns the total return of the asset prices multiplied by the weights.
        - "both" : Returns 'returns' and 'total'
        ForceAdjusted : :py:class:`bool` Boolean to Adjust the asset prices to Weights Order. It may slow down the process.
        Returns
        -------
        portfolio: :py:class:`pandas.DataFrame` DataFrame with the portfolio values.
        """

        # Check if the ReturnType is valid
        if ReturnType not in ["returns", "total", "both"]:
            raise ValueError("ReturnType must be 'returns', 'total' or 'both'")

        if not isinstance(AssetPrices, pd.DataFrame):
            raise ValueError("Asset Prices must be a Pandas DataFrame!")

        if isinstance(Weights, pd.DataFrame):
            if ForceAdjusted:
                Weights = Weights[AssetPrices.columns]
            Weights = Weights.values

        elif not isinstance(Weights, np.ndarray):
            raise ValueError("Weights must be a numpy array!")

        date_index = AssetPrices.index
        asset_names = AssetPrices.columns

        asset_prices = np.array(AssetPrices)

        weights_array = np.array(Weights)

        asset_returns = asset_prices[0:, :] / asset_prices[0, :]

        if ReturnType == "returns":
            return pd.DataFrame(np.multiply(asset_returns, weights_array), index=asset_names, columns=date_index)

        elif ReturnType == "total":
            return pd.DataFrame(np.dot(asset_returns, weights_array.T), index=date_index).rename(columns={0: "Return"})

        elif ReturnType == "both":
            df_all = pd.DataFrame(np.multiply(
                asset_returns, weights_array), index=asset_names, columns=date_index)
            df_total = df_all.sum(axis=1).to_frame().rename(
                columns={0: "Return"})
            return df_all, df_total

    @staticmethod
    def crop_assetprices(AssetPrices, AssetNames, end_date, remove_nan=True):
        """
        Return the asset prices cropped to the final date and the selected assets.
        Parameters
        ----------
        AssetPrices : :py:class:`pandas.DataFrame` DataFrame with the daily asset prices.
        asset_names : :py:class:`list` List with the names of the assets to be returned.
        end_date : :py:class:`pandas.Timestamp` Date to be used as the final date.
        remove_nan : :py:class:`bool` If True, the nan values will be removed.
        """
        # Check AssetNames
        if not isinstance(AssetNames, list):
            try:
                AssetNames = list(AssetNames)
            except Exception:
                raise ValueError("AssetNames must be listable")

        # Check if the assets of weights are the same as the assets of the asset prices
        if not set(AssetNames).issubset(set(AssetPrices.columns)):
            raise ValueError(
                "Weights must be a DataFrame with the same columns as the Asset Prices")

        # Crop AssetPrices
        AssetPrices = AssetPrices.loc[:end_date, AssetNames]
        return AssetPrices
