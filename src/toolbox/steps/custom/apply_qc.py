"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
import polars as pl
import xarray as xr
from datetime import datetime
from geodatasets import get_path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import shapely as sh
import geopandas

# Defining QC functions from "Argo Quality Control Manual for CTD and Trajectory Data" (https://archimer.ifremer.fr/doc/00228/33951/).
# TODO: Do we need a bin nan rows QC to speed up processing?


class impossible_date_test:
    """
    Target Variable: TIME
    Test Number: 2
    Flag Number: 4 (bad data)
    Checks that the datetime of each point is valid.
    """

    def __init__(self, df):
        self.name = "impossible date test"
        self.required_variables = ["TIME"]
        self.df = df
        self.flags = None

    def return_qc(self):
        # Check if any of the datetime stamps fall outside 1985 and the current datetime
        self.flags = self.df.select(
            (
                (pl.col("TIME") > datetime(1985, 1, 1))
                & (pl.col("TIME") < datetime.now())
                & pl.col("TIME").is_not_null()
            ).alias("TIME_is_valid")
        )

        self.flags = self.flags.select(
            (pl.col("TIME_is_valid").not_().cast(pl.Int64) * 4).alias("TIME_QC"),
        )
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        axs = [ax, ax.twinx()]
        for ax, data, var, col in zip(
            axs, [self.df, self.flags], ["TIME", "TIME_QC"], ["b", "r"]
        ):
            ax.plot(data[var], c=col)
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.yaxis.label.set_color(col)
        fig.tight_layout()
        plt.show(block=True)


class impossible_location_test:
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 3
    Flag Number: 4 (bad data)
    Checks that the latitude and longitude are valid.
    """

    def __init__(self, df):
        self.name = "impossible location test"
        self.required_variables = ["LATITUDE", "LONGITUDE"]
        self.df = df
        self.flags = None

    def return_qc(self):
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            self.df = self.df.with_columns(
                (
                    (pl.col(label) > bounds[0])
                    & (pl.col(label) < bounds[1])
                    & pl.col(label).is_not_null()
                    & pl.col(label).is_finite()
                ).alias(f"{label}_is_valid")
            )

            self.df = self.df.with_columns(
                (pl.col(f"{label}_is_valid").not_().cast(pl.Int64) * 4).alias(
                    f"{label}_QC"
                ),
            )

        self.flags = self.df.select(pl.col("^.*_QC$"))
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, dpi=200)
        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            # Data
            ax.plot(self.df[var], c="b")
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.yaxis.label.set_color("b")
            # Flags
            ax_twin = ax.twinx()
            ax_twin.plot(self.flags[f"{var}_QC"], c="r")
            ax_twin.set(
                xlabel="Index",
                ylabel=f"{var}_QC",
            )
            ax_twin.yaxis.label.set_color("r")
            # Bounds
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")
        fig.tight_layout()
        plt.show(block=True)


class position_on_land_test:
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 4
    Flag Number: 4 (bad data)
    Checks that the measurement location is not on land.
    """
    def __init__(self, df):

        self.name = "position on land test"
        self.required_variables = ["LATITUDE", "LONGITUDE"]
        self.df = df
        self.flags = None

    def return_qc(self):

        # Concat the polygons into a MultiPolygon object
        self.world = geopandas.read_file(
            get_path("naturalearth.land")
        )
        land_polygons = sh.ops.unary_union(
            self.world.geometry
        )

        # Check if lat, long coords fall within the area of the land polygons
        self.df = self.df.with_columns(
            pl.struct("LONGITUDE", "LATITUDE")
            .map_batches(
                lambda x: sh.contains_xy(
                    land_polygons,
                    x.struct.field("LONGITUDE").to_numpy(),
                    x.struct.field("LATITUDE").to_numpy()
                )
                * 4
            )
            .alias("LONGITUDE_QC")
        )
        # Add the flags to LATITUDE as well.
        self.df = self.df.with_columns(pl.col("LONGITUDE_QC").alias("LATITUDE_QC"))

        self.flags = self.df.select(pl.col("^.*_QC$"))
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        # Plot land boundaries
        self.world.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.3)

        # Separate flagged and unflagged points (if LATITUDE has been flagged, so will LONGITUDE)
        unflagged = self.df.filter(pl.col("LATITUDE_QC") == 0)
        flagged = self.df.filter(pl.col("LATITUDE_QC") == 4)

        # Plot points
        for data, col, label in zip([unflagged, flagged], ["b", "r"], ["Good", "Bad"]):
            ax.scatter(
                data["LONGITUDE"],
                data["LATITUDE"],
                c=col,
                s=20,
                alpha=0.6,
                label=label,
            )
        ax.set(
            xlabel="Longitude",
            ylabel="Latitude",
            title=self.name
        )
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)


class impossible_speed_test:
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Test Number: 5
    Flag Number: 4 (bad data)
    Checks that the the glider speed stays below 3m/s
    """

    def __init__(self, df):

        self.name = "impossible speed test"
        self.required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
        self.df = df
        self.flags = None

    def return_qc(self):

        self.df = self.df.with_columns((pl.col("TIME").diff().cast(pl.Float64) * 1e-9).alias("dt"))
        for label in ["LATITUDE", "LONGITUDE"]:
            self.df = self.df.with_columns(
                pl.col(label)
                .replace([np.inf, -np.inf, np.nan], None)
                .interpolate_by("TIME")
                .diff()
                .alias(f"delta_{label}")
            )
            self.df = self.df.with_columns(
                (pl.col(f"delta_{label}") / pl.col("dt")).alias(f"{label}_speed")
            )
        self.df = self.df.with_columns(
            ((pl.col("LATITUDE_speed") ** 2 + pl.col("LONGITUDE_speed") ** 2) ** 0.5).alias(
                "absolute_speed"
            )
        )

        # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
        self.df = self.df.with_columns(
            (
                (pl.col("absolute_speed") < 3)
                & pl.col("absolute_speed").is_not_null()
                & pl.col("absolute_speed").is_finite()
            ).alias("speed_is_valid")
        )

        for label in ["LATITUDE", "LONGITUDE", "TIME"]:
            self.df = self.df.with_columns(
                (pl.col("speed_is_valid").not_().cast(pl.Int64) * 4).alias(f"{label}_QC")
            )

        self.flags = self.df.select(pl.col("^.*_QC$"))
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        # Values
        ax.plot(self.df["TIME"], self.df["absolute_speed"], c="b")
        ax.set(
            xlabel="Time (s)",
            ylabel="Absolute Speed (m/s)",
        )
        ax.yaxis.label.set_color("b")

        # Flags
        ax_twin = ax.twinx()
        ax_twin.plot(self.df["TIME"], self.df[f"LATITUDE_QC"], c="r")
        ax_twin.set(
            xlabel="Time",
            ylabel=f"Flag",
        )
        ax_twin.yaxis.label.set_color("r")

        # Speed threshold
        ax.axhline(3, ls="--", c="k")
        fig.tight_layout()
        plt.show(block=True)


def global_range_test(df):
    """
    Target Variable: PRES, TEMP, PRAC_SALINITY
    Test Number: 6
    Flag Number: 4, 3 (bad data, probably bad data)
    Checks that the pressure, temperature and practically salinity do not lie outside expected
    global extremes.
    """
    # Structured (variable_to_test, [lower_limit, upper_limit], variables_to_flag, flag)
    test_calls = (
        ("PRES", [-np.inf, -5], ["PRES", "TEMP", "PRAC_SALINITY"], 4),
        ("PRES", [-5, -2.4], ["PRES", "TEMP", "PRAC_SALINITY"], 3),
        ("TEMP", [-2.5, 40], ["TEMP"], 4),
        ("PRAC_SALINITY", [2, 41], ["PRAC_SALINITY"], 4),
    )

    # Not the most efficient implementation because of second for loop.
    for var, lims, flag_vars, flag in test_calls:
        for flag_var in flag_vars:
            df = df.with_columns(
                ((pl.col(var) > lims[0]) & (pl.col(var) < lims[1]))
                .not_()
                .cast(pl.Int64)
                * flag.alias(f"{flag_var}_QC")
            )

    return df


def regional_range_test(df):
    """
    Target Variable: TEMP, PRAC_SALINITY
    Test Number: 7
    Flag Number: 4 (bad data)
    Checks that the temperature and practically salinity do not lie outside expected
    regional (Mediterranean and Red Seas) extremes.
    """
    import shapely as sh

    # Define Red and Mediterranean Sea areas
    Red_Sea = sh.geometry.Polygon(
        [(40, 10), (45, 14), (35, 30), (30, 30), (40, 10)]
    )  # (lon, lat)
    Med_Sea = sh.geometry.Polygon(
        [(-6, 30), (40, 30), (35, 40), (20, 42), (15, 50), (-5, 40), (-6, 30)]
    )
    # Check if data falls in those areas
    for poly, name in zip([Red_Sea, Med_Sea], ["in_red_sea", "in_med_sea"]):
        df = df.with_columns(
            pl.shape(["LONGITUDE", "LATITUDE"])
            .map_batches(
                lambda x: sh.contains_xy(
                    poly, x.struct.field("LONGITUDE"), x.struct.field("LATITUDE")
                )
            )
            .alias(f"{name}")
        )

    # define regional temperature and salinity limits
    limits = {
        "red": {"TEMP": (21, 40), "PRAC_SALINITY": (2, 41)},
        "med": {"TEMP": (10, 40), "PRAC_SALINITY": (2, 40)},
    }
    # Check if the data satisfies these limits
    for region, var_lims in limits.items():
        for var, lims in var_lims.items():
            df = df.with_columns(
                ((pl.col(var) > lims[0]) & (pl.col(var) < lims[1])).not_()
            ).alias(f"bad_{region}_{var}")

        # for the flagging, data must fail the regional test AND be within that region.
        for var in ["TEMP", "PRAC_SALINITY"]:
            df = df.with_columns(
                (
                    (pl.col(f"bad_{region}_{var}") & pl.col(f"in_{region}_sea")).cast(
                        pl.Int64
                    )
                    * 4
                ).alias(f"{var}_QC")
            )

    return df


def pressure_increasing_test(df):
    """
    Target Variable: PRES
    Test Number: 8
    Flag Number: 4 (bad data)
    Checks for any egregious spikes in pressure between consecutive points.
    """

    return df


def spike_test(df):
    """
    Target Variable: TEMP, PRAC_SALINITY
    Test Number: 9
    Flag Number: 4 (bad data)
    Checks for spikes in temperature and prctical salinity between nearest neighbour points points.
    """

    return df


@register_step
class ApplyQC(BaseStep):
    
    step_name = "Apply QC"

    def organise_flags(self, new_flags):
        # Method for taking in new flags and cross checking against exiting flags, including upgrading flags when necessary.
        # Update existing flag columns
        flag_columns_to_update = set(new_flags.columns) & set(self.flag_store.columns)
        for column_name in flag_columns_to_update:
            self.flag_store = self.flag_store.with_columns(
                pl.max_horizontal([pl.col(column_name), new_flags[column_name]]).alias(
                    f"{column_name}"
                )
            )
        # Add new QC flag columns if they dont already exist
        flag_columns_to_add = set(new_flags.columns) - set(self.flag_store.columns)
        self.flag_store = self.flag_store.with_columns(
            new_flags[list(flag_columns_to_add)],
        )

    def run(self):
        
        # Register of all QC steps
        self.QC_REGISTER = {
            "impossible date test": impossible_date_test,
            "impossible location test": impossible_location_test,
            "position on land test": position_on_land_test,
            "impossible speed test": impossible_speed_test,
        }
        self.qc_order = []
        
        # Defining the order of operations
        # TODO: Register each QC test and check that all of the config-specified tests exist
        if self.parameters["QC_order"] is None:
            print('Please specify which QC operations to run')
        else:
            for qc_name in self.parameters["QC_order"]:
                if qc_name in self.QC_REGISTER.keys():
                    self.qc_order.append(qc_name)
                else:
                    print(f"The QC test name: {qc_name} was not recognised. Skipping...")

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError(
                "[Argo QC] No data found in context. Please load data first."
            )
        else:
            self.log("Data found in context.")
        data = self.context["data"]
        
        # Convert data to polars for fast processing
        # TODO: Make QC classes have required variables. Collect them and then check they are present in the data
        qc_variables = [
            "TIME",
            "LATITUDE",
            "LONGITUDE",
            "PRES",
            "TEMP",
        ]
        if set(qc_variables).issubset(set(data.keys())):
            df = pl.from_pandas(data[qc_variables].to_dataframe(), nan_to_null=False)
        else:
            raise KeyError(
                f"[Argo QC] The data is missing variables: ({set(qc_variables) - set(data.keys())}) which are required for QC."
                f" Make sure that the variables are present in the data, or use the 'Derive CDT' step to append them."
            )

        # Fetch existing flags from the data and create a place to store them
        existing_flags = [
            flag_column for flag_column in df.columns if "_QC" in flag_column
        ]
        self.flag_store = pl.DataFrame()
        if len(existing_flags) > 0:
            self.flag_store = pl.from_pandas(
                data[existing_flags].to_dataframe(), nan_to_null=False
            )

        # Run through all of the QC steps and add the flags to flag_store
        for qc_test in self.qc_order:
            # Create an instance of this test step
            qc_test_instance = self.QC_REGISTER[qc_test](df)
            returned_flags = qc_test_instance.return_qc()
            self.organise_flags(returned_flags)
            
            # Diagnostic plotting
            if self.parameters["diagnostics"]:
                qc_test_instance.plot_diagnostics()
                
            # Once finished, remove the test instance from memory
            del qc_test_instance

        # Append the flags from self.flag_store to the xarray data and push back into context
        for flag_column in self.flag_store.columns:
            data[flag_column] = (
                ("N_MEASUREMENTS",),
                self.flag_store[flag_column].to_numpy(),
            )
        self.context["data"] = data
        return self.context