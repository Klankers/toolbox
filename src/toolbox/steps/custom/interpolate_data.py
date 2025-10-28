"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@register_step
class InterpolateVariables(BaseStep):
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

        # Validate that data exists in the processing context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")

        data = self.context["data"]

        # Get target flags to interpolate over
        target_flags = self.parameters["target_flags"]

        # Get variable to interpolate from config
        self.vars_to_interp = set(self.parameters["variables_to_interpolate"])
        if "TIME" in self.vars_to_interp:
            print("TIME was listed for interpolation. This is prohibited and TIME will be skipped")
            self.vars_to_interp.remove("TIME")

        # Extract data subset with QC columns
        qc_cols = {col+"_QC" for col in self.vars_to_interp}
        all_cols = self.vars_to_interp.union(qc_cols)
        if not all_cols.issubset(set(data.data_vars)):
            missing_cols = all_cols-set(data.data_vars)
            print(f"The following variables were not found in the data: {missing_cols}. These will be skipped.")
            for col in missing_cols:
                all_cols.remove(col)
        
        # Convert to polars dataframe
        self.df = pl.from_pandas(data[list(all_cols | {"TIME"})].to_dataframe(), nan_to_null=False)
        self.unprocessed_df = self.df.clone()  # Making a copy for plotting change in diagnostics

        # Replacement interpolation strategy
        for var in self.vars_to_interp:
            self.df = self.df.with_columns(
                pl.when(
                    pl.col(f"{var}_QC").is_in(target_flags)
                ).then(np.nan)
                .otherwise(pl.col(var))
                .alias(var)
            )

        # Replace all nans with Nones (polars recognises interpolation gaps by Nones)
        self.df = self.df.with_columns(
            pl.all().exclude("TIME").cast(pl.Float64).replace({np.nan: None})
        )

        # Interpolate
        self.df = self.df.with_columns(
            pl.col(var).interpolate_by("TIME")
            for var in self.vars_to_interp
        )

        # Update flags
        self.df = self.df.with_columns(
            pl.when(
                pl.col(var).is_not_null() & (pl.col(f"{var}_QC").is_in(target_flags))
            ).then(8)
            .otherwise(f"{var}_QC")
            .alias(f"{var}_QC")
            for var in self.vars_to_interp
        )

        # Turn the Nones back into nans
        self.df = self.df.with_columns(
            pl.all().exclude("TIME").replace({None: np.nan})
        )

        if self.diagnostics:
            self.generate_diagnostics()

        # Add derived variables back to the xarray Dataset with proper metadata
        for var in all_cols:
            # Convert Polars column back to numpy array and add to xarray Dataset
            data[var] = (("N_MEASUREMENTS",), self.df[var].to_numpy())

        # Update the context with the enhanced dataset
        self.context["data"] = data
        return self.context

    def generate_diagnostics(self):

        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8), dpi=200)

        plot_var = list(self.vars_to_interp)[0]
        for ax, data in zip(axs.flatten(), [self.unprocessed_df, self.df]):
            ax.plot(data[plot_var])

        plt.show(block=True)