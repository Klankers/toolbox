import pandas as pd
import numpy as np


def safe_median_datetime(x: np.ndarray, axis=None, **kwargs) -> np.datetime64:
    """
    Safely compute the median of datetime64[ns] array using pandas.

    Parameters
    ----------
    x : np.ndarray
        A 1D array of datetime64 values.

    Returns
    -------
    np.datetime64
        Median datetime or NaT if input is empty/all-NaT.
    """
    x = pd.to_datetime(x)

    if isinstance(x, pd.DatetimeIndex):
        x = pd.Series(x)

    if x.empty or x.isna().all():
        return np.datetime64("NaT")

    return x.median()
