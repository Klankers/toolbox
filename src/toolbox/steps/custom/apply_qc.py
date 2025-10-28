"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
import polars as pl
import xarray as xr
from datetime import datetime
from geodatasets import get_path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
import shapely as sh
import geopandas

flag_cols = np.array([
    "black",
    "blue",
    "gray",
    "orange",
    "red",
    "gray",
    "gray",
    "gray",
    "cyan",
    "gray"
])
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[0],
           markersize=10, label='Unchecked (0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[1],
           markersize=10, label='Good (1)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[3],
           markersize=10, label='Probably Bad (3)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[4],
           markersize=10, label='Bad (4)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[8],
           markersize=10, label='Interpolated (8)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=flag_cols[9],
           markersize=10, label='Other'),
]

# Defining QC functions from "Argo Quality Control Manual for CTD and Trajectory Data" (https://archimer.ifremer.fr/doc/00228/33951/).

def range_test(df, variable_name, limits, flag):
    # Not the most efficient implementation because of second for loop.
    df = df.with_columns(
        ((pl.col(variable_name) > limits[0]) & (pl.col(variable_name) < limits[1]))
        .not_()
        .cast(pl.Int64)
        * flag.alias(f"{variable_name}_QC")
    )
    return df


class test_template:
    """
    Target Variable:
    Flag Number:
    Variables Flagged:
    ? description ?
    """

    name = ""
    required_variables = []
    qc_outputs = []

    def __init__(self, df):
        self.df = df
        self.flags = None

    def return_qc(self):
        self.flags = None  # replace with processing of some kind
        return self.flags

    def plot_diagnostics(self):
        # Any relevant diagnostic
        pass

# --------------------------- Universal tests ----------------------------

class impossible_date_test(test_template):
    """
    Target Variable: TIME
    Test Number: 2
    Flag Number: 4 (bad data)
    Checks that the datetime of each point is valid.
    """

    name = "impossible date test"
    required_variables = ["TIME"]
    qc_outputs = ["TIME_QC"]

    def return_qc(self):
        # Check if any of the datetime stamps fall outside 1985 and the current datetime
        self.flags = self.df.select(
            pl.when(
                pl.col("TIME").is_null()
            ).then(9)
            .when(
                ((pl.col("TIME") > datetime(1985, 1, 1))
                & (pl.col("TIME") < datetime.now()))
            ).then(1)
            .otherwise(4)
            .alias("TIME_QC")
        )
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        colors = flag_cols[self.flags["TIME_QC"].to_numpy()]
        ax.scatter(
            range(len(self.df)), self.df["TIME"],
            c=colors, s=1, rasterized=True
        )
        ax.set(
            xlabel="Index",
            ylabel="TIME",
        )
        ax.legend(handles=legend_elements, loc="upper right")
        fig.tight_layout()
        plt.show(block=True)


class impossible_location_test(test_template):
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 3
    Flag Number: 4 (bad data)
    Checks that the latitude and longitude are valid.
    """

    name = "impossible location test"
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Check LAT/LONG exist within expected bounds
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            self.df = self.df.with_columns(
                pl.when(
                    pl.col(label).is_nan()
                ).then(9)
                .when(
                    (pl.col(label) > bounds[0]) & (pl.col(label) < bounds[1])
                ).then(1)
                .otherwise(4).alias(f"{label}_QC")
            )

        self.flags = self.df.select(pl.col("^.*_QC$"))
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, dpi=200)
        colors = flag_cols[self.flags["LATITUDE_QC"].to_numpy()]
        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            # Data
            ax.scatter(
                range(len(self.df)), self.df[var],
                c=colors, s=1, rasterized=True
            )
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.legend(handles=legend_elements, loc="upper right")

            # Bounds
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")

        fig.tight_layout()
        plt.show(block=True)


class position_on_land_test(test_template):
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 4
    Flag Number: 4 (bad data)
    Checks that the measurement location is not on land.
    """

    name = "position on land test"
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

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
            ).replace({0: 1})
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
        unflagged = self.df.filter(pl.col("LATITUDE_QC") == 1)
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


class impossible_speed_test(test_template):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Test Number: 5
    Flag Number: 4 (bad data)
    Checks that the the glider speed stays below 3m/s
    """

    name = "impossible speed test"
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

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
                pl.when(
                    pl.col("speed_is_valid")
                ).then(1)
                .otherwise(4)
                .alias(f"{label}_QC")
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


class pressure_range_test(test_template):
    """
    Target Variable: PRES (pressure)
    Flag Number: 4 (bad data), 3 (Bad, potentially correctable)
    Variables Flagged: PRES (pressure), TEMP (temperature), CNDC (conductivity)
    Checks that the pressure is within a reasonable range.
    """

    name = "pressure range test"
    required_variables = ["PRES"]
    qc_outputs = ["PRES_QC", "TEMP_QC", "CNDC_QC"]

    def return_qc(self):

        # Set flags
        self.flags = self.df.select(
            pl.when(pl.col("PRES").is_between(-5, -2.4)).then(3)
            .when(pl.col("PRES") < -5).then(4)
            .otherwise(1)
            .alias("PRES_QC")
        )

        # Apply to other variables as well
        self.flags = self.flags.with_columns(
            pl.col("PRES_QC").alias("TEMP_QC"),
            pl.col("PRES_QC").alias("CNDC_QC"),
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        # Values
        ax.plot(self.df["PRES"], c="b")
        ax.set(
            xlabel="Index",
            ylabel="Pressure",
        )
        ax.yaxis.label.set_color("b")

        # Flags
        ax_twin = ax.twinx()
        ax_twin.plot(self.flags[f"PRES_QC"], c="r")
        ax_twin.set(
            xlabel="Index",
            ylabel=f"Flag",
        )
        ax_twin.yaxis.label.set_color("r")

        # Speed threshold
        ax.axhline(-2.4, ls="--", c="k")
        ax.axhline(-5, ls="--", c="k")
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



    return df


def regional_range_test(df):
    """
    Target Variable: TEMP, PRAC_SALINITY
    Test Number: 7
    Flag Number: 4 (bad data)
    Checks that the temperature and practically salinity do not lie outside expected
    regional (Mediterranean and Red Seas) extremes.
    """

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
        if len(flag_columns_to_add) > 0:
            self.flag_store = self.flag_store.with_columns(
                new_flags[list(flag_columns_to_add)],
            )

    def run(self):
        
        # Register of all QC steps and required variables
        QC_REGISTER = {}
        for cls in test_template.__subclasses__():
            if hasattr(cls, "name") and cls.name:
                QC_REGISTER[cls.name] = cls


        # Defining the order of operations
        self.qc_order = []
        if self.parameters["QC_order"] is None:
            print("[Apply QC] Please specify which QC operations to run")
        else:
            for qc_name in self.parameters["QC_order"]:
                if qc_name in QC_REGISTER.keys():
                    self.qc_order.append(QC_REGISTER[qc_name])
                else:
                    print(f"[Apply QC] The QC test name: {qc_name} was not recognised. Skipping...")

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError(
                "[Apply QC] No data found in context. Please load data first."
            )
        else:
            self.log("Data found in context.")
        data = self.context["data"]

        # Try and fetch the qc history from context and update it
        qc_history = self.context.setdefault("qc_history", {})
        
        # Collect all of the required varible names and qc outputs
        all_required_variables = set({})
        test_qc_outputs_cols = set({})
        for test in self.qc_order:
            all_required_variables.update(test.required_variables)
            test_qc_outputs_cols.update(test.qc_outputs)

        # Convert data to polars for fast processing
        if set(all_required_variables).issubset(set(data.keys())):
            df = pl.from_pandas(data[all_required_variables].to_dataframe(), nan_to_null=False)
        else:
            raise KeyError(
                f"[Argo QC] The data is missing variables: ({set(all_required_variables) - set(data.keys())}) which are required for QC."
                f" Make sure that the variables are present in the data, or use remove tests from the order."
            )

        # Fetch existing flags from the data and create a place to store them
        existing_flags = [
            flag_col for flag_col in data.data_vars if flag_col in test_qc_outputs_cols
        ]

        self.flag_store = pl.DataFrame()
        if len(existing_flags) > 0:
            self.flag_store = pl.from_pandas(
                data[existing_flags].to_dataframe(), nan_to_null=False
            )

        # Run through all of the QC steps and add the flags to flag_store
        for qc_test in self.qc_order:
            # Create an instance of this test step
            print(f"[Apply QC] Applying: {qc_test.name}")
            qc_test_instance = qc_test(df)
            returned_flags = qc_test_instance.return_qc()
            self.organise_flags(returned_flags)

            # Update QC history
            for flagged_var in returned_flags.columns:
                percent_flagged = (returned_flags[flagged_var].to_numpy() != 0).sum() / len(returned_flags)
                qc_history.setdefault(flagged_var, []).append((qc_test.name, percent_flagged))

            # Diagnostic plotting
            if self.diagnostics:
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
        self.context["qc_history"] = qc_history
        return self.context