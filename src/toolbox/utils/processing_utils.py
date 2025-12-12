import numpy as np

# ----------------------------- NaN Handling ------------------------------
def find_nans(data: np.ndarray):
    """
    Handles generation of masks and location indices of nans.
    Intended for 1D arrays.

    Parameters
    ----------
    data : np.ndarray
        numpy array with nans

    Returns
    -------
    nan_mask : np.ndarray
        mask where incices with nans are True
    nan_indices : np.ndarray
        indices of locations where there are nans
    non_nan_indices : np.ndarray
        indices of locations where there are values
    """
    nan_mask = np.isnan(data)
    nan_indices = np.nonzero(nan_mask)[0]
    non_nan_indices = np.nonzero(~nan_mask)[0]
    return nan_mask, nan_indices, non_nan_indices

def interpolate_nans(data, coords):
    """
    Fills nan values in y using interpolation over x.
    x and y must have the same dimensions.

    Parameters
    ----------
    data : np.ndarray
        1D array of size N to interpolate
    coords : np.ndarray
        1D array of size N which the data will be interpolated over

    Returns
    -------
    filled_data : np.ndarray
        data with nans filled using linear interpolation
    """

    # Convert datetimes to floats
    args = [np.array(data), np.array(coords)]
    for i, array in enumerate(args):
        if np.issubdtype(array.dtype, np.datetime64):
            elapsed_time = (args[i] - args[i][0]) / np.timedelta64(1, "s")
            args[i] = elapsed_time

    data, coords = args
    non_nan_mask = ~np.isnan(data)
    filled_data = np.interp(
        coords,
        coords[non_nan_mask],
        data[non_nan_mask]
    )
    return filled_data

# ----------------------------- Filtering ---------------------------------
def remove_outliers(data):
    """
    Removes outliers (including NaNs) from data. Exclusion is based on
    inter-quartile range.

    Parameters
    ----------
    data : np.ndarray | list
        1D array of size N to interpolate

    Returns
    -------
    filtered_data : np.ndarray
        data with outliers removed

    """
    data = np.array(data)
    data = data[np.isfinite(data)]  # remove NaNs
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered_data = data[(data >= lower) & (data <= upper)]
    return filtered_data