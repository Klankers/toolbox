#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt


@register_step
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Find Profile Direction"

    def run(self):

        """
        Config Example
        --------------
        - name: "Find Profile Direction"
            parameters:
            diagnostics: false
        """

        self.filter_qc()

        # Subsetting the data to remove nans and find pressure rate of change
        is_nan = self.data[["PROFILE_NUMBER", "PRES", "TIME"]].isnull()
        nan_mask = is_nan["PROFILE_NUMBER"] | is_nan["PRES"] | is_nan["TIME"]
        data_subset = self.data[["PROFILE_NUMBER", "PRES", "TIME"]].where(~nan_mask, drop=True)

        # Find the gradient of pressure over time
        data_subset = data_subset.set_coords(["TIME"])
        data_subset["direction"] = -1 * np.sign(data_subset["PRES"].differentiate("TIME", datetime_unit="s"))

        # Find the median direction per profile
        direction_mapping = data_subset.groupby("PROFILE_NUMBER").map(lambda x: x["direction"].median())
        data_subset["PROFILE_DIRECTION"] = direction_mapping.sel(
            PROFILE_NUMBER=data_subset["PROFILE_NUMBER"]
        ).drop(["PROFILE_NUMBER", "TIME"])

        # Map the direction back onto self.data
        self.data["PROFILE_DIRECTION"] = xr.DataArray(
            np.full(len(self.data["N_MEASUREMENTS"]), np.nan),
            dims=["N_MEASUREMENTS"]
        )
        self.data["PROFILE_DIRECTION"][~nan_mask] = data_subset["PROFILE_DIRECTION"]


        self.reconstruct_data()
        self.update_qc()
        self.generate_qc({"PROFILE_DIRECTION_QC": ["PROFILE_NUMBER_QC", "PRES_QC", "TIME_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        mpl.use("tkagg")
        fig, ax = plt.subplots()

        for direction, col, label in zip([-1, 1], ['r', 'b'], ['Descending', 'Ascending']):
            plot_data = self.data[["TIME", "PRES", "PROFILE_DIRECTION"]].where(
                self.data["PROFILE_DIRECTION"] == direction
            )
            ax.plot(
                plot_data["TIME"],
                plot_data["PRES"],
                c=col,
                ls='',
                marker="o",
                label=label
            )
        ax.set(
            xlabel="TIME",
            ylabel="PRES",
            title="Profile Directions",
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        plt.show(block=True)

