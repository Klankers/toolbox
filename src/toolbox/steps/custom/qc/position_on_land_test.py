#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
from geodatasets import get_path
import matplotlib.pyplot as plt
import shapely as sh
import polars as pl
import xarray as xr
import matplotlib
import geopandas

@register_qc
class position_on_land_test(BaseTest):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the measurement location is not on land.
    """

    test_name = "position on land test"
    expected_parameters = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):

        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(),
            nan_to_null=False
        )

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

        # Convert back to xarray
        flags = self.df.select(pl.col("^.*_QC$"))
        self.flags = xr.Dataset(
            data_vars={
                col: ("N_MEASUREMENTS", flags[col].to_numpy())
                for col in flags.columns
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]}
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        # Plot land boundaries
        self.world.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.3)

        for i in range(10):
            # Plot by flag number
            plot_data = self.df.filter(
                pl.col("LATITUDE_QC") == i
            )
            if len(plot_data) == 0:
                continue

            # Plot the data
            ax.plot(
                plot_data["LONGITUDE"],
                plot_data["LATITUDE"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            xlabel="Longitude",
            ylabel="Latitude",
            title="Position On Land Test",
        )
        ax.legend(title="Flags", loc="upper right")
        fig.tight_layout()
        plt.show(block=True)