"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np


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
            self.log(f"[Interpolate Data] Data found in context.")

        data = self.context["data"]

        # Get variable to interpolate from config
        variables_to_interpolate = set(self.parameters["variables_to_interpolate"])

        # Extract data subset with QC columns
        cols_to_extract = variables_to_interpolate | {"TIME"}  #TODO: Unfinished - get the qc columns as well
        df = pl.from_pandas(data[list(variables_to_interpolate | {"TIME"})].to_dataframe(), nan_to_null=False)

        df = df.with_columns(
            pl.col(col).replace({np.nan: None}).interpolate_by("TIME").name.prefix("INTERP_")
            for col in variables_to_interpolate
        )

        # Add derived variables back to the xarray Dataset with proper metadata
        for var_name in variables_to_interpolate.items():
            # Convert Polars column back to numpy array and add to xarray Dataset
            data[var_name] = (("N_MEASUREMENTS",), df[var_name].to_numpy())
            # Attach CF-compliant metadata attributes
            data[var_name].attrs = meta

        # Update the context with the enhanced dataset
        self.context["data"] = data
        return self.context
