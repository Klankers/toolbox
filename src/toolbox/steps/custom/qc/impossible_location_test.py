#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
import polars as pl
import matplotlib
import matplotlib.pyplot as plt

@register_qc
class impossible_location_test(BaseTest):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the latitude and longitude are valid.
    """

    test_name = "impossible location test"
    expected_parameters = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):

        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(),
            nan_to_null=False
        )

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

        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            for i in range(10):
                # Plot by flag number
                plot_data = self.df.with_row_index().filter(
                    pl.col(f"{var}_QC") == i
                )
                if len(plot_data) == 0:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["index"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.legend(title="Flags", loc="upper right")
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")

        fig.suptitle("Impossible Location Test")
        fig.tight_layout()
        plt.show(block=True)