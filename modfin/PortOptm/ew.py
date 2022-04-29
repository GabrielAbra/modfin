import numpy as np
import pandas as pd


class EqualWeight():
    """
    Equal Weight Portfolio

    Parameters
    ----------
    AssetPrices : :py:class:`pandas.DataFrame` with the asset prices

    Returns
    -------
    EqualWeight : :py:class:`pandas.DataFrame` with the equal weight portfolio
    """

    def __init__(self):
        pass

    def optimize(self, AssetPrices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate asset weights using equal weight algorithm.

        Parameters
        ----------
        AssetPrices : py:class:`pandas.DataFrame` with the daily asset prices

        Returns
        -------
        Portifolio : :py:class:`pandas.DataFrame` with the weights of the portfolio
        """

        # Check if the asset prices are valid
        if AssetPrices is not None:
            if not isinstance(AssetPrices, pd.DataFrame):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame!")
            if not isinstance(AssetPrices.index, pd.DatetimeIndex):
                raise ValueError(
                    "Asset Prices must be a Pandas DataFrame index by datetime!")

        # Portfolio "construction" LMAO :D
        asset_names = AssetPrices.columns
        value = np.repeat([1 / len(asset_names)],
                          len(asset_names), axis=0)
        Portifolio = pd.Series(value, index=asset_names)
        Portifolio = Portifolio.to_frame().T.dropna(axis="columns")
        return Portifolio
