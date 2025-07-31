"""Class definition for loading data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import pandas as pd
import numpy as np
import gsw


@register_step
class LoadOG1(BaseStep):
    """
    Step for loading OG1 data.

    Derived from Phyto-Phys Repo by Obsidian500

    Available Parameters:
        - file_path: Path to the OG1 data file.
        - add_meta: Boolean flag to indicate whether to add metadata.
        - add_elapsed_time: Boolean flag to indicate whether to add elapsed time.
        - diagnostics: Boolean flag to indicate whether to generate diagnostics.
    """

    step_name = "Load OG1"

    def run(self):

        source = self.parameters["file_path"]
        self.log(f"Params: {self.parameters}")
        self.log(f"Loading {source} OG1")
        # load data from xarray
        self.data = xr.open_dataset(source)

        if self.add_meta:
            self.f_addMeta()

        if self.add_elapsed_time:
            self.f_addElapsedTime()

        # Generate diagnostics if enabled
        self.log(f"Diagnostics: {self.diagnostics}")
        if self.diagnostics:
            self.generate_diagnostics()

        # add data to context
        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        self.log(f"Generating diagnostics...")
        # self.log summary stats
        diag.generate_info(self.data)

    def f_addMeta(self):
        self.log(f"Adding metadata...")
        # Add Length
        self.data.attrs["N_MEASUREMENTS_Length"] = int(
            (len(self.data.N_MEASUREMENTS.values))
        )
        # Add Variable Meta
        self.data.attrs["Variable Information"] = pd.DataFrame(
            index=self.data.keys(),
            columns=["Is Numeric", "Length", "# NaNs", "NaNless Length"],
        )
        for k, v in self.data.items():
            # Check if the variable is numeric
            b_is_numeric = (
                np.issubdtype(v.dtype, np.number)
                | np.issubdtype(v.dtype, np.datetime64)
            ) & (v.ndim != 0)
            self.data.attrs["Variable Information"].loc[k] = [
                b_is_numeric,
                len(v.values) if b_is_numeric else v.values,
                np.sum(np.isnan(v.values)) if b_is_numeric else None,
                len(v.values) - np.sum(np.isnan(v.values)) if b_is_numeric else None,
            ]

    def f_addElapsedTime(self):
        """
        Appends epoch and elapsed time to the dataset
        """
        try:
            epoch_time = self.data.TIME.values.astype("float")
            elapsed_time = (epoch_time - epoch_time[0]) * 1e-9
            self.data["EPOCH_TIME"] = (("N_MEASUREMENTS",), epoch_time)
            self.data.EPOCH_TIME.attrs = {
                "long_name": "Time in nanoseconds since 01/01/1970",
                "units": "ns",
                "standard_name": "Epoch time",
                "valid_min": -np.inf,
                "valid_max": np.inf,
            }

            self.data["ELAPSED_TIME"] = (("N_MEASUREMENTS",), elapsed_time)
            self.data.ELAPSED_TIME.attrs = {
                "long_name": "Elapsed time in seconds since deployment",
                "units": "s",
                "standard_name": "Elapsed time",
                "valid_min": 0,
                "valid_max": np.inf,
            }
        except Exception as e:
            if type(e) == AttributeError:
                self.log(
                    "ERROR: The TIME variable does not appear in the netCDF file. These functions are only intended"
                    " for use with OG1 format netCDF files."
                )
            else:
                self.log(f"{type(e)}: Something unexpected happened: \n {e}")
