#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####
import polars as pl
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

@register_qc
class impossible_date_test(BaseTest):
    """
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    test_name = "impossible date test"
    expected_parameters = {}
    required_variables = ["TIME"]
    qc_outputs = ["TIME_QC"]

    def return_qc(self):
        # Check if any of the datetime stamps fall outside 1985 and the current datetime
        self.df = self.df.with_columns(
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
        self.flags = self.df.select(pl.col("^.*_QC$"))
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        for i in range(10):
            # Plot by flag number
            plot_data = self.df.with_row_index().filter(
                pl.col("TIME_QC") == i
            )
            if len(plot_data) == 0:
                continue

            # Plot the data
            ax.plot(
                plot_data["index"],
                plot_data["TIME"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )
        ax.set(
            title="Impossible Date Test",
            xlabel="Index",
            ylabel="TIME",
        )
        ax.legend(title="Flags", loc="upper right")
        fig.tight_layout()
        plt.show(block=True)