import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from toolbox.utils.time import safe_median_datetime


def plot_time_series(
    data, x_var, y_var, title="Time Series Plot", xlabel=None, ylabel=None, **kwargs
):
    """Generates a time series plot for xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variables exist in the xarray dataset
        if x_var not in data.coords or y_var not in data:
            raise ValueError(
                f"Variables {x_var} and {y_var} must exist in the dataset."
            )
        x_data = data[x_var].values  # Extract x_data (usually time dimension)
        y_data = data[y_var].values  # Extract the y_data (variable to plot)
    else:
        # Assuming custom format such as lists or arrays
        x_data, y_data = data[0], data[1]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, **kwargs)
    plt.xlabel(xlabel or x_var)
    plt.ylabel(ylabel or y_var)
    plt.title(title)
    plt.show()


def plot_histogram(data, var, bins=30, title="Histogram", xlabel=None, **kwargs):
    """Generates a histogram for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins=bins, alpha=0.7, **kwargs)
    plt.xlabel(xlabel or var)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def plot_boxplot(data, var, title="Box Plot", xlabel=None, **kwargs):
    """Generates a box plot for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel or var)
    plt.show()


def plot_correlation_matrix(data, variables=None, title="Correlation Matrix", **kwargs):
    """Generates a heatmap of the correlation matrix for xarray data."""
    if isinstance(data, xr.Dataset):
        if variables is None:
            variables = list(data.data_vars)  # Use all variables by default
        # Extract the variables to calculate the correlation matrix
        corr = data[variables].to_array().T.corr(dim="dim_0")
    else:
        raise TypeError("Data must be a Xarray Dataset to generate correlation matrix.")

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, **kwargs)
    plt.title(title)
    plt.show()


def generate_info(data):
    """Generate info for a given dataset"""
    if isinstance(data, xr.Dataset):
        # For xarray, we'll summarize each data variable
        print("Data Info:")
        print(data.info())
    else:
        print("Data Info only supported for xarray Dataset ")


def check_missing_values(data):
    """Check for missing values in the dataset."""
    if isinstance(data, xr.Dataset):
        missing = data.isnull().sum()
        print("Missing Values in Xarray Dataset:\n", missing)
    else:
        print("Missing value check only supported for xarray Dataset ")


def summarising_profiles(ds: xr.Dataset, source_name: str) -> pd.DataFrame:
    """
    Summarise profiles from an xarray Dataset by computing medians of TIME, LATITUDE, LONGITUDE
    grouped by PROFILE_NUMBER. Handles datetime median safely using pandas.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with PROFILE_NUMBER as a coordinate.

    source_name : str
        Name of the glider/source to include in output.

    Returns
    -------
    pd.DataFrame
        Profile-level summary DataFrame.
    """
    if "PROFILE_NUMBER" not in ds:
        raise ValueError("Dataset must include PROFILE_NUMBER.")

    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")

    summary_vars = [v for v in ["TIME", "LATITUDE", "LONGITUDE"] if v in ds]

    medians = {}
    for var in summary_vars:
        if var not in ds:
            continue

        da = ds[var]
        if "PROFILE_NUMBER" not in da.coords:
            da = da.set_coords("PROFILE_NUMBER")

        grouped = da.groupby("PROFILE_NUMBER")

        if np.issubdtype(da.dtype, np.datetime64):
            # Use pandas to compute median datetime safely
            medians[f"median_{var}"] = grouped.reduce(safe_median_datetime)
        else:
            medians[f"median_{var}"] = grouped.median(skipna=True)

    df = xr.Dataset(medians).to_dataframe().reset_index()
    df["glider_name"] = source_name
    df.rename(columns={"PROFILE_NUMBER": "profile_id"}, inplace=True)
    # sort by time
    df.sort_values(by="median_TIME", inplace=True)
    return df


def find_closest_prof(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    For each profile in df_a, find the closest profile in df_b based on time,
    and calculate spatial distance to it.

    Parameters
    ----------
    df_a : pd.DataFrame
        Summary dataframe for glider A (reference).

    df_b : pd.DataFrame
        Summary dataframe for glider B (comparison).

    Returns
    -------
    pd.DataFrame
        df_a with additional columns:
            - closest_glider_b_profile
            - glider_b_time_diff
            - glider_b_distance_km
    """
    a_times = df_a["median_TIME"].values
    a_lats = df_a["median_LATITUDE"].values
    a_lons = df_a["median_LONGITUDE"].values

    b_times = df_b["median_TIME"].values
    b_lats = df_b["median_LATITUDE"].values
    b_lons = df_b["median_LONGITUDE"].values
    b_ids = df_b["profile_id"].values

    closest_ids = []
    time_diffs = []
    distances = []

    for a_time, a_lat, a_lon in zip(a_times, a_lats, a_lons):
        time_diff = np.abs(b_times - a_time)
        idx = time_diff.argmin()

        closest_ids.append(b_ids[idx])
        time_diffs.append(time_diff[idx])

        if np.all(np.isfinite([a_lat, a_lon, b_lats[idx], b_lons[idx]])):
            dist_km = geodesic((a_lat, a_lon), (b_lats[idx], b_lons[idx])).km
        else:
            dist_km = np.nan
        distances.append(dist_km)

    df_result = df_a.copy()
    df_result["closest_glider_b_profile"] = closest_ids
    df_result["glider_b_time_diff"] = time_diffs
    df_result["glider_b_distance_km"] = distances

    return df_result


def plot_distance_ts(matchup_df: pd.DataFrame, glider_ref: str, glider_comp: str):
    """
    Plot time series of distance between glider_ref and closest glider_comp profiles.

    Parameters
    ----------
    matchup_df : pd.DataFrame
        Output from find_closest_prof().

    glider_ref : str
        Name of the reference glider.

    glider_comp : str
        Name of the comparison glider.
    """
    plt.figure(figsize=(12, 6))
    for name, group in matchup_df.groupby("glider_name"):
        plt.plot(
            group["median_TIME"],
            group["glider_b_distance_km"],
            label=name,
            marker="o",
            linestyle="-",
        )

    plt.xlabel("Datetime")
    plt.ylabel("Distance to Closest Profile (km)")
    plt.ylim(0, 50)
    plt.title(f"Distance between {glider_ref} and {glider_comp} Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
