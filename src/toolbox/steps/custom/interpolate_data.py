"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@register_step
class InterpolateVariables(BaseStep, QCHandlingMixin):
    """
    A processing step class for deriving oceanographic variables from CTD data.

    This class processes Conductivity, Temperature, and Depth (CTD) data to derive
    additional oceanographic variables such as salinity, density, and depth using
    the Gibbs SeaWater (GSW) Oceanographic Toolbox functions.

    Inherits from BaseStep and processes data stored in the context dictionary.

    Attributes:
        step_name (str): Identifier for this processing step ("Derive CTD")
    """

    step_name = "Interpolate Data"

    def run(self):
        """
        """
        self.log(f"Interpolating variables...")

        self.filter_qc()

        # Convert to polars dataframe
        self.df = pl.from_pandas(self.data[list(self.filter_settings.keys() | {"TIME"})].to_dataframe(), nan_to_null=False)
        self.unprocessed_df = self.df.clone()  # Making a copy for plotting change in diagnostics

        # Interpolate
        self.df = self.df.with_columns(
            pl.col(var).replace({np.nan: None}).interpolate_by("TIME").replace({None: np.nan})
            for var in self.filter_settings.keys()
        )

        for var in self.filter_settings.keys():
            self.data[var][:] = self.df[var].to_numpy()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        # Update the context with the enhanced dataset
        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):

        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8), dpi=200)

        plot_var = list(self.filter_settings.keys())[0]
        for ax, data in zip(axs.flatten(), [self.unprocessed_df, self.df]):
            ax.plot(data[plot_var])

        plt.show(block=True)