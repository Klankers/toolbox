"""Class definition for exporting data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
import numpy as np


def find_profiles(
    df,
    gradient_thresholds: list,
    filter_win_sizes=["20s", "10s"],
    time_col="TIME",
    depth_col="DEPTH",
    cust_col=None,
    cust_gradient_thresholds=None 
):
    """
    Identifies vertical profiles in oceanographic or similar data by analyzing depth gradients over time.

    This function processes depth-time data to identify periods where an instrument is performing
    vertical profiling based on gradient thresholds. It handles data interpolation, calculates vertical
    velocities, applies median filtering, and assigns unique profile numbers to identified profiles.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe containing time and depth measurements.
        Time column must be in epoch seconds.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] defining the vertical velocity
        range (in meters/second) that is NOT considered part of a profile. typical values are around [0.02, -0.02]
    filter_win_sizes : list, default= ['20s', '10s']
        Window sizes for the compound filter applied to gradient calculations, in Polars duration format.
        index 0 controls the rolling median window size and index 1 controls the rolling mean window size.
    time_col : str, default='TIME'
        Name of the column containing timestamp data
    depth_col : str, default='DEPTH'
        Name of the column containing depth measurements
    cust_col : str, default=None
        Name of a data column in the input dataframe, with matching time and depth measurements, to be displayed
        alongside profiling plots, e.g. pitch
    cust_filter_win_sizes : list, default=None
        Two-element list [positive_threshold, negative_threshold] defining the
        range of your custom variable that is NOT considered part of a profile.

    Returns
    -------
    polars.DataFrame
        Dataframe with additional columns:
        - 'dt': Time difference between consecutive points (seconds)
        - 'dz': Depth difference between consecutive points (meters)
        - 'grad': Vertical velocity (dz/dt, meters/second)
        - 'smooth_grad': Median-filtered vertical velocity
        - 'is_profile': Boolean indicating if a point belongs to a profile
        - 'profile_num': Unique identifier for each identified profile (0 for non-profile points)
        - NEW : 'phase': Developed for ALR data, indicates the phase of the platform, 0 for downcast, 1 for upcast, 2 for transect

    Notes
    -----
    - The function considers a point part of a profile when its smoothed vertical velocity
      falls outside the range specified by gradient_thresholds.
    - 'depth_col' does not strictly have to be a depth measurement, any variable which follows
      the profile shape (such as pressure) could also be used, though this would change the units
      and interpretation of grad.
    """

    # Get the unedited shape for padding later (to make the input and outputs the same length)
    df_full_len = len(df)

    # Interpolate missing depth values using time as reference
    # Also removes infinite and NaN values before interpolation
    cols = [pl.col(time_col), pl.col(depth_col)]
    if cust_col and cust_col in df.columns:
        cols.append(pl.col(cust_col))

    df = (
        df.select(
            *cols,
            pl.col(depth_col,cust_col)
            .replace([np.inf, -np.inf, np.nan], None)
            .interpolate_by(time_col)
            .name.prefix(f"INTERP_"),
        )
        .with_row_index()
        .drop_nulls(subset=[f"INTERP_{depth_col}",f"INTERP_{cust_col}"])
    )

    # Calculate time differences (dt) and depth differences (dz) between consecutive measurements
    df = df.with_columns(
        (pl.col(time_col).diff().cast(pl.Float64) * 1e-9).alias(
            "dt"
        ),  # Convert nanoseconds to seconds
        pl.col(f"INTERP_{depth_col}").diff().alias("dz"),
        pl.col(f"INTERP_{cust_col}").diff().alias("dC")
    )

    # Calculate vertical velocity (gradient) as depth change divided by time change
    df = df.with_columns(
        (pl.col("dz") / pl.col("dt")).alias("grad"),
    ).drop_nulls(subset="grad")

    df = df.with_columns(
        (pl.col("dC") / pl.col("dt")).alias("dC/dt"),
    ).drop_nulls(subset="dC/dt")

    # Apply a compound filter to smooth the gradient values (rolling median
    # supresses spikes, rolling mean smooths noise)
    # TODO: this window size should be checked against the maximum sample period (dt)
    df = df.with_columns(
        pl.col("grad", f"INTERP_{cust_col}", "dC/dt")
        .rolling_median_by(time_col, window_size=filter_win_sizes[0])
        .rolling_mean_by(time_col, window_size=filter_win_sizes[1])
        .name.prefix("smooth_"),
    )

    # Determine which points are part of profiles based on gradient thresholds
    # A point is considered part of a profile when its gradient is outside the threshold range
    pos_grad, neg_grad = gradient_thresholds

    df = df.with_columns(
        pl.col("smooth_grad").is_between(neg_grad, pos_grad).not_().alias("is_profile")
    )

    # Determine which points are part of profiles based on pitch angle and vertical velocity
    # The two should contrusctively interfere and reduce signal to noise.
    if cust_col == 'pitch': 
        if cust_gradient_thresholds:
            pos_grad, neg_grad = cust_gradient_thresholds

        # Combined metric
        combined_metric = -pl.col(f"smooth_INTERP_{cust_col}") * pl.col("smooth_grad")
        
        df = df.with_columns(
            (combined_metric > pos_grad).alias("is_profile")
        )

    # Assign unique profile numbers to consecutive points identified as profiles
    # This converts the boolean 'is_profile' column into numbered profile segments
    df = df.with_columns(
        (
            pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_profile")
            - 1
        ).alias("profile_num")
    )

    # Reforming the full length dataframe (This executes faster than polars join or merge methods)
    front_pad = np.full((df["index"].min(), len(df.columns)), np.nan)
    end_pad = np.full((df_full_len - df["index"].max() - 1, len(df.columns)), np.nan)

    data = np.vstack((front_pad, df.to_numpy(), end_pad))
    padded_df = pl.DataFrame(data, schema=df.columns).drop("index")

    return padded_df


@register_step
class FindProfilesStep(BaseStep):
    step_name = "Find Profiles"

    def run(self):
        self.log("Attempting to designate profile numbers")

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")
        self.data = self.context["data"]
        self.thresholds = self.parameters["gradient_thresholds"]
        self.win_sizes = self.parameters["filter_window_sizes"]
        self.depth_col = self.parameters["depth_column"]
        self.cust_col = self.parameters.get("custom_column", None)  # <<< NEW

        if self.diagnostics:
            self.log("Generating diagnostics")
            root = self.generate_diagnostics()
            root.mainloop()

        # Convert to polars for processing
        self._df = pl.from_pandas(
            self.data[["TIME", self.depth_col]].to_dataframe(), nan_to_null=False
        )
        self.profile_outputs = find_profiles(
            self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col, cust_col=self.cust_col
        )
        profile_numbers = self.profile_outputs["profile_num"].to_numpy()

        # Add profile numbers to data and update context
        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), profile_numbers)
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. #=-1 indicates no profile, #>=0 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": -1,
            "valid_max": np.inf,
        }

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):

        def generate_plot():
            # Allows interactive plots
            mpl.use("TkAgg")

            # Update data for plot
            self._df = pl.from_pandas(
                self.data[["TIME", self.depth_col]].to_dataframe(), nan_to_null=False
            )
            self.profile_outputs = find_profiles(
                self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col
            )

            # Split data into profile and non-profile points for plotting
            profiles = self.profile_outputs.drop_nans().filter(
                pl.col("is_profile").cast(pl.Boolean)
            )
            not_profiles = self.profile_outputs.drop_nans().filter(
                pl.col("is_profile").cast(pl.Boolean).not_()
            )

            fig, axs = plt.subplots(
                4, 1, figsize=(18, 10), height_ratios=[3, 3, 1, 2], sharex=True  # <<< CHANGED
            )
            axs[0].set(
                xlabel="Time", 
                ylabel="Interpolated Depth",
            )
            axs[1].set(
                xlabel="Time", 
                ylabel="Vertical Velocity",
            )
            axs[2].set(
                xlabel="Time", 
                ylabel="Profile Number",
            )
            axs[3].set(
                xlabel="Time", 
                ylabel=f"{self.cust_col}" if self.cust_col else "Custom",
            )   # <<< NEW

            fig.tight_layout()

            # Plot depth vs time, highlighting profile and non-profile points
            for data, col, label in zip(
                [profiles, not_profiles],
                ["tab:blue", "tab:red"],
                ["Profile", "Not Profile"],
            ):
                axs[0].plot(
                    data["TIME"],
                    -data[f"INTERP_{self.depth_col}"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )
                axs[1].plot(
                    data["TIME"],
                    data["smooth_grad"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )

            # Plot raw and smoothed gradients with threshold lines
            axs[1].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["grad"],
                c="k",
                alpha=0.1,
                label="Raw Velocity",
            )
            for val, label in zip(self.thresholds, ["Gradient Thresholds", None]):
                axs[1].axhline(val, ls="--", color="gray", label=label)

            # Plot profile numbers
            axs[2].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["profile_num"],
                c="gray",
            )

            for ax in axs[:2]:
                ax.legend(loc="upper right")
            plt.show(block=True)

            if self.cust_col and self.cust_col in self.profile_outputs.columns:   # <<< use self.cust_col
                axs[3].plot(
                    self.profile_outputs["TIME"],
                    self.profile_outputs[self.cust_col],   # <<< dynamic
                    c="purple",
                    marker=".",
                    markersize=1,
                    ls="",
                    label=self.cust_col
                )
                axs[3].legend(loc="upper right")

        root = tk.Tk()
        root.title("Parameter Adjustment")
        root.geometry(f"380x{50*len(self.parameters)}")
        entries = {}

        # Gradient thresholds
        row = 0
        values = self.thresholds
        tk.Label(root, text=f"Gradient Thresholds:").grid(row=row, column=0)
        for i, label, value in zip(range(2), ["+ve", "-ve"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Filter window sizes
        row = 2
        values = self.win_sizes
        tk.Label(root, text=f"Filter Window Sizes:").grid(
            row=row, column=0, pady=(20, 0)
        )
        for i, label, value in zip(range(2), ["Median Filter", "Mean Filter"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Depth column name
        row = 4
        value = self.depth_col
        tk.Label(root, text=f"Depth column name:").grid(row=row, column=0, pady=(20, 0))
        entry = tk.Entry(root, textvariable="depth_column")
        entry.insert(0, value)
        entry.grid(row=row, column=1, pady=(20, 0))
        entries["depth_column"] = entry

        # Function to handle Cancel button click
        def on_cancel():
            plt.close()
            root.destroy()

        def on_regenerate():
            # Update parameter attributes
            self.thresholds = [float(entries["+ve"].get()), float(entries["-ve"].get())]
            self.win_sizes = [
                entries["Median Filter"].get(),
                entries["Mean Filter"].get(),
            ]
            self.depth_col = entries["depth_column"].get()

            # Regenerate data and plot it
            plt.close()
            generate_plot()

        def on_save():
            self.log(
                f"continuing with parameters: \n"
                f"  Gradient Thresholds: {self.thresholds}\n"
                f"  Filter Window Sizes: {self.win_sizes}\n"
                f"  Depth column: {self.depth_col}\n"
            )
            plt.close()
            root.destroy()

        tk.Button(root, text="Regenerate", command=on_regenerate).grid(
            row=row + 1, column=0, pady=(20, 0)
        )
        tk.Button(root, text="Save", command=on_save).grid(
            row=row + 1, column=1, pady=(20, 0)
        )
        tk.Button(root, text="Cancel", command=on_cancel).grid(
            row=row + 1, column=2, pady=(20, 0)
        )

        generate_plot()
        return root