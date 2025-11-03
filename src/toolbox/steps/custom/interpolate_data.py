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
    A processing step for interpolating data.

    This class processes data to interpolate missing values and fill gaps in
    variables using time-based interpolation. It supports quality control
    handling and optional diagnostic visualization.

    Inherits from BaseStep and processes data stored in the context dictionary.

    Example config usage:
      - name: "Interpolate Data"
        parameters:
          qc_handling_settings: {
            flag_filter_settings: {
              "PRES": [3, 4, 9],
              "LATITUDE": [3, 4, 9],
              "LONGITUDE": [3, 4, 9]
            },
            reconstruction_behaviour: "replace",
            flag_mapping: { 3: 8, 4: 8, 9: 8 }
          }
        diagnostics: false

    Attributes
    ----------
    step_name : str
        Identifier for this processing step. Set to "Interpolate Data".
    """

    step_name = "Interpolate Data"

    def run(self):
        """
        Execute the interpolation workflow.

        This method performs the following steps:
        1. Filters data based on quality control flags
        2. Converts xarray data to a Polars DataFrame
        3. Interpolates missing values using time as the reference dimension
        4. QC and data reconstruction based off of user specification
        6. Updates QC flags for interpolated values
        7. Generates diagnostic plots if enabled

        Returns
        -------
        dict
            The updated context dictionary containing the interpolated dataset
            under the "data" key.
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
        """
        Generate diagnostic plots comparing original and interpolated data.

        Creates a side-by-side comparison visualization showing the first
        variable in filter_settings before and after interpolation.

        This method uses the Tkinter backend for interactive display.
        """

        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8), dpi=200)

        plot_var = list(self.filter_settings.keys())[0]
        for ax, data in zip(axs.flatten(), [self.unprocessed_df, self.df]):
            ax.plot(data[plot_var])

        plt.show(block=True)