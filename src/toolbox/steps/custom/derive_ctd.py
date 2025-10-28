"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import gsw


@register_step
class DeriveCTDVariables(BaseStep):
    """
    A processing step class for deriving oceanographic variables from CTD data.

    This class processes Conductivity, Temperature, and Depth (CTD) data to derive
    additional oceanographic variables such as salinity, density, and depth using
    the Gibbs SeaWater (GSW) Oceanographic Toolbox functions.

    Inherits from BaseStep and processes data stored in the context dictionary.

    Attributes:
        step_name (str): Identifier for this processing step ("Derive CTD")
    """

    step_name = "Derive CTD"

    def run(self):
        """
        Execute the CTD variable derivation process. The following varibales are
        required: ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]

        This method performs the following operations:
        1. Validates that data exists in the context
        2. Applies unit conversions to raw measurements
        3. Optionally interpolates missing position data
        4. Derives oceanographic variables using GSW functions
        5. Adds metadata to derived variables
        6. Updates the context with processed data

        Returns:
            dict: Updated context dictionary containing original and derived variables

        Raises:
            ValueError: If no data is found in the context
        """
        self.log(f"Processing CTD...")

        # Validate that data exists in the processing context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")

        data = self.context["data"]
        to_derive = self.parameters["to_derive"]

        # Convert xarray Dataset to Polars DataFrame for efficient numerical processing
        # Extract only the variables needed for GSW calculations
        df = pl.from_pandas(
            data[
                ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
            ].to_dataframe(),
            nan_to_null=False,
        )

        # Define GSW (Gibbs SeaWater) function calls for deriving oceanographic variables
        # Each tuple contains: (output_variable_name, gsw_function, [required_input_variables])
        gsw_function_calls = (
            ("DEPTH", gsw.z_from_p, ["PRES", "LATITUDE"]),
            ("PRAC_SALINITY", gsw.SP_from_C, ["CNDC", "TEMP", "PRES"]),
            (
                "ABS_SALINITY",
                gsw.SA_from_SP,
                ["PRAC_SALINITY", "PRES", "LONGITUDE", "LATITUDE"],
            ),
            ("CONS_TEMP", gsw.CT_from_t, ["ABS_SALINITY", "TEMP", "PRES"]),
            ("DENSITY", gsw.rho, ["ABS_SALINITY", "CONS_TEMP", "PRES"]),
        )

        # Process each GSW function call to derive new variables
        for var_name, func, args in gsw_function_calls:
            if var_name not in to_derive:
                continue

            self.log(f"Deriving {var_name}...")

            # Use Polars struct operations to efficiently apply GSW functions
            # This approach handles vectorized operations across the entire dataset
            df = df.with_columns(
                pl.struct(args)
                .map_batches(lambda x: func(*(x.struct.field(arg) for arg in args)))
                .alias(var_name)
            )

        # self.log diagnostic information if diagnostics are enabled
        if self.diagnostics:
            self.log(df.describe(percentiles=[]))

        # Define metadata for each derived variable following CF conventions
        variable_metadata = {
            "DEPTH": {
                "long_name": "Depth from surface (negative down as defined by TEOS-10)",
                "units": "meters [m]",
                "standard_name": "DEPTH",
                "valid_min": -10925,  # Mariana Trench depth
                "valid_max": 1,  # Above sea level
            },
            "PRAC_SALINITY": {
                "long_name": "Practical salinity",
                "units": "unitless",
                "standard_name": "PRAC_SALINITY",
                "valid_min": 2,  # Extremely fresh water
                "valid_max": 42,  # Hypersaline conditions
            },
            "ABS_SALINITY": {
                "long_name": "Absolute salinity",
                "units": "g/kg",
                "standard_name": "ABS_SALINITY",
                "valid_min": 0,  # Pure water
                "valid_max": 1000,  # Pure salt (theoretical maximum)
            },
            "CONS_TEMP": {
                "long_name": "Conservative temperature",
                "units": "°C",
                "standard_name": "CONS_TEMP",
                "valid_min": -2,  # Freezing point of seawater
                "valid_max": 102,  # Boiling point of seawater
            },
            "DENSITY": {
                "long_name": "Density",
                "units": "kg/m3",
                "standard_name": "DENSITY",
                "valid_min": 900,  # Warm, low salinity surface water
                "valid_max": 1100,  # Cold, high salinity bottom water
            },
        }

        # Add derived variables back to the xarray Dataset with proper metadata
        for var_name, meta in variable_metadata.items():
            if var_name not in to_derive:
                continue
            # Convert Polars column back to numpy array and add to xarray Dataset
            data[var_name] = (("N_MEASUREMENTS",), df[var_name].to_numpy())
            # Attach CF-compliant metadata attributes
            data[var_name].attrs = meta

        # Update the context with the enhanced dataset
        self.context["data"] = data
        return self.context
