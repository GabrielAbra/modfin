import numpy as np
import pandas as pd


def get_names(asset_data):
    """
    Generic function to get the asset names.
    """
    if isinstance(asset_data, pd.DataFrame):
        return asset_data.columns.values

    elif isinstance(asset_data, pd.Series):
        return asset_data.index.values

    elif isinstance(asset_data, np.ndarray):
        return np.arange(1, asset_data.shape[0] + 1)

    else:
        raise ValueError(
            "asset_data must be a pandas.DataFrame or a pandas.Series")


def get_index(asset_data):
    """
    Generic function to get the asset dates.
    """
    if isinstance(asset_data, pd.DataFrame):
        return asset_data.index.values

    elif isinstance(asset_data, pd.Series):
        return asset_data.index.values

    elif isinstance(asset_data, np.ndarray):
        return np.arange(1, asset_data.shape[1] + 1)

    else:
        raise ValueError(
            "Asset data must be a pandas.DataFrame or a pandas.Series")