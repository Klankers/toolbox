"""Class definition for deriving CDT variables."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag
import polars as pl
import numpy as np
import gsw


class DeriveCDTVariables(BaseStep):
    step_name = "Derive CDT"

    def run(self):
        print(
            f"[Derive CTD Variables] Processing CTD..."
        )

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("[Derive CTD Variables] No data found in context. Please load data first.")
        else:
            print(f"[Derive CTD Variables] Data found in context.")
        data = self.context["data"]

        # Converting units
        # TODO: This following two lines should become redundant when BODC releases OG1-like datasets
        conversion_dict = {
            'CNDC': self.parameters["conductivity_unit_conversion_factor"],
            'PRES': self.parameters["pressure_unit_conversion_factor"],
            'LATITUDE': self.parameters["latitude_longitude_conversion_factor"],
            'LONGITUDE': self.parameters["latitude_longitude_conversion_factor"]
        }
        for var_name, multiplier in conversion_dict.items():
            data[var_name][:] *= float(multiplier)

        # Convert data to polars for fast processing
        df = pl.from_pandas(
            data[['TIME', 'LATITUDE', 'LONGITUDE', 'CNDC', 'PRES', 'TEMP']]
            .to_dataframe(), nan_to_null=False
        )

        # Apply interpolation on position variables if required
        if self.parameters["interpolate_latitude_longitude"]:
            df = df.with_columns(
                *(pl.col(var_name).replace([np.inf, -np.inf, np.nan], None).interpolate_by('TIME')
                  for var_name in ['LATITUDE', 'LONGITUDE'])
            )

        # Collect calls for gsw functions (variable_name, gsw function, function args)
        gsw_function_calls = (
            ('DEPTH', gsw.z_from_p, ['PRES', 'LATITUDE']),
            ('PRAC_SALINITY', gsw.SP_from_C, ['CNDC', 'TEMP', 'PRES']),
            ('ABS_SALINITY', gsw.SA_from_SP, ['PRAC_SALINITY', 'PRES', 'LONGITUDE', 'LATITUDE']),
            ('CONS_TEMP', gsw.CT_from_t, ['ABS_SALINITY', 'TEMP', 'PRES']),
            ('DENSITY', gsw.rho, ['ABS_SALINITY', 'CONS_TEMP', 'PRES'])
        )
        for var_name, func, args in gsw_function_calls:
            print(f'[Derive CTD Variables] Deriving {var_name}...')
            df = df.with_columns(
                pl.struct(args).map_batches(
                    lambda x: func(*(x.struct.field(arg) for arg in args))
                ).alias(var_name)
            )

        if self.diagnostics:
            print(df.describe(percentiles=[]))

        variable_metadata = {
            'DEPTH': {
                "long_name": "Depth from surface (negative down as defined by TEOS-10)",
                "units": "meters [m]",
                "standard_name": "DEPTH",
                "valid_min": -2550000, # Mariana Trench
                "valid_max": 1,
            },
            'PRAC_SALINITY': {
                "long_name": "Practical salinity",
                "units": "unitless",
                "standard_name": "PRAC_SALINITY",
                "valid_min": 2,
                "valid_max": 42, # Valid min and max are considered limits beyond which errors become impractical
            },
            'ABS_SALINITY': {
                "long_name": "Absolute salinity",
                "units": "g/kg",
                "standard_name": "ABS_SALINITY",
                "valid_min": 0,
                "valid_max": 1000, # Pure salt
            },
            'CONS_TEMP': {
                "long_name": "Conservative temperature",
                "units": "°C",
                "standard_name": "CONS_TEMP",
                "valid_min": -2, # Freezing point of seawater
                "valid_max": 102, # Boiling point of seawater
            },
            'DENSITY': {
                "long_name": "Density",
                "units": "kg/m3",
                "standard_name": "DENSITY",
                "valid_min": 900, # Warm, low salinity surface water
                "valid_max": 1100, # Cold, high salinity bottom water
            },
        }
        # Append to xarray and update context
        for var_name, meta in variable_metadata.items():
            data[var_name] = (("N_MEASUREMENTS",), df[var_name].to_numpy())
            data[var_name].attrs = meta
        self.context["data"] = data
        return self.context