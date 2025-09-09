import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from typing import Dict, List


def interpolate_DEPTH(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate missing DEPTH values and compute 5m depth bins, all using xarray operations.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with variables 'DEPTH', 'TIME', and 'PROFILE_NUMBER'.

    Returns
    -------
    xr.Dataset
        Same dataset with new data variables:
            - DEPTH_interpolated
            - DEPTH_bin (floored to nearest 5 m)
            - DEPTH_range (string label like "50–55m")
    """
    # Validate required variables
    required_vars = ["DEPTH", "TIME", "PROFILE_NUMBER"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain variable: {var}")

    # Get dimensions
    if "N_MEASUREMENTS" not in ds.dims:
        raise ValueError(
            "Dataset must have a profile-wise measurement dimension (e.g., 'N_MEASUREMENTS')."
        )

    # Interpolate missing values per PROFILE_NUMBER
    # Approach: loop over unique profile numbers using xarray vectorized selection
    if "DEPTH" not in ds:
        raise ValueError("DEPTH not found in dataset.")

    if "PROFILE_NUMBER" not in ds:
        raise ValueError(
            "PROFILE_NUMBER must be present for per-profile interpolation."
        )

    # filter PROFILE_NUMBER > 0
    # (-1 is surfacing behaviour)
    ds = ds.where(ds["PROFILE_NUMBER"] > 0, drop=True)

    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")

    print("[Pipeline Manager] Interpolating missing DEPTH values by PROFILE_NUMBER...")

    ##### DEV ONLY #####
    # For debugging, limit to first 3 profiles
    ds = ds.where(ds["PROFILE_NUMBER"].isin([1, 2, 3]), drop=True)
    ###################

    # Use xarray groupby to interpolate within each profile
    DEPTH_interp = (
        ds["DEPTH"]
        .groupby("PROFILE_NUMBER")
        .map(
            lambda g: g.interpolate_na(
                dim="N_MEASUREMENTS", method="linear", fill_value="extrapolate"
            ).reindex_like(g)
        )
    )

    ds = ds.assign({"DEPTH_interpolated": DEPTH_interp})

    # Compute 5 m bins
    bin_size = 5
    DEPTH_bin = (ds["DEPTH_interpolated"] // bin_size) * bin_size
    ds["DEPTH_bin"] = DEPTH_bin.astype(np.float32)

    # Add range label (as a string)
    bin_start = ds["DEPTH_bin"]
    bin_end = bin_start + bin_size
    ds["DEPTH_range"] = xr.apply_ufunc(
        lambda start, end: np.core.defchararray.add(
            np.char.add(start.astype(int).astype(str), "–"), end.astype(int).astype(str)
        ),
        bin_start,
        bin_end,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str],
    )

    return ds


def aggregate_vars(df, vars_to_aggregate):
    """Aggregate specified variables by PROFILE_NUMBER and DEPTH_bin.

    Args:
        df (pd.DataFrame): Input DataFrame with 'PROFILE_NUMBER', 'DEPTH_bin', and variables to aggregate.
        vars_to_aggregate (list): List of variable names to aggregate.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    if "PROFILE_NUMBER" not in df.columns or "DEPTH_bin" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'PROFILE_NUMBER' and 'DEPTH_bin' columns."
        )

    # Agg over median, with alias of _media{var}
    print("[Pipeline Manager] Aggregating variables...")
    agg_dict = {var: "median" for var in vars_to_aggregate}
    aggregated_df = (
        df.groupby(["PROFILE_NUMBER", "DEPTH_bin"]).agg(agg_dict).reset_index()
    )
    # Rename columns
    aggregated_df.rename(
        columns={var: f"median_{var}" for var in vars_to_aggregate}, inplace=True
    )
    # sort by PROFILE_NUMBER and DEPTH_bin
    aggregated_df = aggregated_df.sort_values(by=["PROFILE_NUMBER", "DEPTH_bin"])
    # drop nulls
    aggregated_df = aggregated_df.dropna(
        subset=[f"median_{var}" for var in vars_to_aggregate]
    )
    return aggregated_df


def merge_depth_binned_profiles(
    target_name: str, binned_dfs: Dict[str, pd.DataFrame], bin_column: str = "DEPTH_bin"
) -> pd.DataFrame:
    """
    Merge depth-binned profile data for a target glider and multiple others.

    Parameters
    ----------
    target_name : str
        The name of the target glider (used as the left/base of the join).

    binned_dfs : dict
        Dictionary of {glider_name: binned_df} where each DataFrame includes
        'PROFILE_NUMBER' and 'DEPTH_bin', and is the result of interpolate_DEPTH + aggregate_vars.

    bin_column : str
        The name of the depth bin column to join on.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with depth-binned values from target and all other gliders,
        using consistent suffixes like _TARGET_{target_name} and _{glider}.
    """
    if target_name not in binned_dfs:
        raise ValueError(f"Target '{target_name}' not found in binned_dfs.")
    print("[Pipeline Manager] Merging depth-binned profiles...")
    # Add glider_PROFILE_NUMBER to each glider's dataframe
    for name, df in binned_dfs.items():
        if f"{name}_PROFILE_NUMBER" not in df.columns:
            binned_dfs[name] = df.copy()
            binned_dfs[name][f"{name}_PROFILE_NUMBER"] = (
                df["PROFILE_NUMBER"].astype(str) + f"_{name}"
            )

    # Use target as base
    target_df = binned_dfs[target_name].copy()
    target_profile_col = f"{target_name}_PROFILE_NUMBER"

    # Rename target columns with _TARGET suffix (except join keys)
    target_df_renamed = target_df.rename(
        columns={
            col: f"{col}_TARGET_{target_name}"
            for col in target_df.columns
            if col not in [bin_column, "PROFILE_NUMBER", target_profile_col]
        }
    )

    # Start with target dataframe as base
    merged_df = target_df_renamed.copy()

    for glider_name, glider_df in binned_dfs.items():
        if glider_name == target_name:
            continue

        profile_col = f"{glider_name}_PROFILE_NUMBER"

        # Rename glider columns (excluding join keys)
        glider_df_renamed = glider_df.rename(
            columns={
                col: f"{col}_{glider_name}"
                for col in glider_df.columns
                if col not in [bin_column, "PROFILE_NUMBER", profile_col]
            }
        )

        merged_df = merged_df.merge(
            glider_df_renamed,
            how="left",
            left_on=[target_profile_col, bin_column],
            right_on=[profile_col, bin_column],
            suffixes=("", f"_{glider_name}"),
        )

    return merged_df


def major_axis_r2(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) for Major Axis (Type II) regression.
    R² is simply the square of the Pearson correlation coefficient.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan  # Not enough points

    r, _ = pearsonr(x_clean, y_clean)
    return r**2


def compute_r2_for_merged_profiles(
    merged_df: pd.DataFrame,
    variables: List[str],
    target_name: str,
    other_names: List[str],
) -> pd.DataFrame:
    """
    Compute R² values for each variable between target and all other gliders.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of merge_depth_binned_profiles()

    variables : list of str
        List of variable names to compare (e.g., ["salinity", "temperature"])

    target_name : str
        Name of the target glider (used in suffixes)

    other_names : list of str
        Other glider names to compare against

    group_columns : list
        Columns to group by (default: PROFILE_NUMBER + DEPTH_bin)

    Returns
    -------
    pd.DataFrame
        One row per target_profile vs comparison_profile pair with R² values for each variable.
    """
    results = []

    grouped = merged_df.groupby(
        [f"{target_name}_PROFILE_NUMBER"]
        + [f"{other}_PROFILE_NUMBER" for other in other_names]
    )

    for keys, group in grouped:
        row = {f"{target_name}_PROFILE_NUMBER": keys[0]}
        for i, other in enumerate(other_names):
            row[f"{other}_PROFILE_NUMBER"] = keys[i + 1]

        for var in variables:
            col_target = f"median_{var}_TARGET_{target_name}"

            for other in other_names:
                col_other = f"median_{var}_{other}"
                if col_target in group.columns and col_other in group.columns:
                    x = group[col_target].to_numpy()
                    y = group[col_other].to_numpy()
                    row[f"r2_{var}_{other}"] = major_axis_r2(x, y)
                else:
                    row[f"r2_{var}_{other}"] = np.nan

        results.append(row)

    return pd.DataFrame(results)
