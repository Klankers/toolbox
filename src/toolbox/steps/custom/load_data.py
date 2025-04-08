"""Class definition for loading data steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import pandas as pd
import numpy as np
import gsw


class LoadOG1(BaseStep):
    """
    Step for loading OG1 data.

    Derived from Phyto-Phys Repo by Obsidian500

    Available Parameters:
        - file_path: Path to the OG1 data file.
        - add_meta: Boolean flag to indicate whether to add metadata.
        - add_elapsed_time: Boolean flag to indicate whether to add elapsed time.
        - add_depth: Boolean flag to indicate whether to add depth.
        - lat_label: Label for latitude variable in the dataset. (default: "LATITUDE")
        - diagnostics: Boolean flag to indicate whether to generate diagnostics.
    """

    step_name = "Load OG1"

    def run(self):

        source = self.parameters["file_path"]
        print(f"[LoadData] Loading {source} OG1")
        # load data from xarray
        self.data = xr.open_dataset(source)

        if "add_meta" in self.parameters:
            self.add_meta()

        if "add_elapsed_time" in self.parameters:
            self.add_elapsed_time()

        if "add_depth" in self.parameters:
            self.add_depth(lat_label=self.parameters["lat_label"])

        # Generate diagnostics if enabled
        if self.diagnostics:
            self.generate_diagnostics()

        # Continue with the rest of the step logic...

    def generate_diagnostics(self):
        print(f"[LoadData] Generating diagnostics...")
        # Print summary stats
        diag.generate_summary_stats(self.data)

    def add_meta(self):
        print(f"[LoadData] Adding metadata...")
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
                len(v.values),
                np.sum(np.isnan(v.values)) if b_is_numeric else None,
                len(v.values) - np.sum(np.isnan(v.values)) if b_is_numeric else None,
            ]

    def add_elapsed_time(self):
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
                print(
                    "ERROR: The TIME variable does not appear in the netCDF file. These functions are only intended"
                    " for use with OG1 format netCDF files."
                )
            else:
                print(f"{type(e)}: Something unexpected happened: \n {e}")

    def add_depth(self, lat_label="LATITUDE"):
        """
        Converts pressure to depth and appends it to the dataset
        """
        p, lat = self.data["PRES"].values, self.data[lat_label].values
        # use GSW to convert pressure to depth
        depth = -1 * gsw.conversions.z_from_p(p, lat)
        self.data["DEPTH"] = (("N_MEASUREMENTS",), depth)
        self.data.DEPTH.attrs = {
            "long_name": "Depth calculated following TEOS-10, implementation by GSW, see"
            "https://github.com/TEOS-10/GSW-python. Dynamic height anomoly and"
            "sea surface geopotential are assimed to be 0",
            "units": "m",
            "standard_name": "Depth (z)",
            "valid_min": -10,
            "valid_max": 10935,
        }
