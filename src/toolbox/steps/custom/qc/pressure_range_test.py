#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr
import matplotlib

@register_qc
class pressure_range_test(BaseTest):
    """
    Target Variable: PRES
    Flag Number: 4 (bad data), 3 (Bad, potentially correctable)
    Variables Flagged: PRES, TEMP, CNDC
    Checks that the pressure is within a reasonable range.
    """

    test_name = "pressure range test"
    expected_parameters = {}
    required_variables = ["PRES"]
    qc_outputs = ["PRES_QC", "TEMP_QC", "CNDC_QC"]

    def return_qc(self):

        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(),
            nan_to_null=False
        )

        # Set flags
        self.df = self.df.with_columns(
            pl.when(pl.col("PRES").is_between(-5, -2.4)).then(3)
            .when(pl.col("PRES") < -5).then(4)
            .otherwise(1)
            .alias("PRES_QC")
        )

        # Apply to other variables as well
        self.df = self.df.with_columns(
            pl.col("PRES_QC").alias("TEMP_QC"),
            pl.col("PRES_QC").alias("CNDC_QC"),
        )

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
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        for i in range(10):
            # Plot by flag number
            plot_data = self.df.with_row_index().filter(
                pl.col("PRES_QC") == i
            )
            if len(plot_data) == 0:
                continue

            # Plot the data
            ax.plot(
                plot_data["index"],
                plot_data["PRES"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )
        ax.axhline(-2.4, ls="--", c="k")
        ax.axhline(-5, ls="--", c="k")

        ax.set(
            xlabel="Index",
            ylabel="Pressure",
            title="Pressure Range Test",
        )
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)