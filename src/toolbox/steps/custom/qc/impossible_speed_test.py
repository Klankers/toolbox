#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import matplotlib

@register_qc
class impossible_speed_test(BaseTest):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    test_name = "impossible speed test"
    expected_parameters = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):

        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(),
            nan_to_null=False
        )

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

        for i in range(10):
            # Plot by flag number
            plot_data = self.df.filter(
                pl.col("LATITUDE_QC") == i
            )
            if len(plot_data) == 0:
                continue

            # Plot the data
            ax.plot(
                plot_data["TIME"],
                plot_data["absolute_speed"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            title="Impossible Speed Test",
            xlabel="Time (s)",
            ylabel="Absolute Horizontal Speed (m/s)",
            ylim=(0, 4)
        )
        ax.axhline(3, ls="--", c="k")
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)