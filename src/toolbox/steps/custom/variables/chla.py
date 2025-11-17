#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


@register_step
class chla_deep_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Deep Correction"

    def run(self):
        """
        Example
        -------

        - name: "Chla Deep Correction"
          parameters:
            dark_value: null
            depth_threshold: 200
        diagnostics: true

        Returns
        -------

        """
        self.filter_qc()

        self.compute_dark_value()
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"CHLA_FLUORESCENCE_QC": ["CHLA_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def compute_dark_value(self):
        """
        Compute dark value for chlorophyll-a correction.

        The dark value represents the sensor's baseline reading in the absence of
        chlorophyll fluorescence. Computed as the median of minimum CHLA values from
        deep profiles (>= depth_threshold).

        Parameters
        ----------
        ds : xarray.Dataset
            Glider dataset with variables: CHLA, DEPTH (or PRES), PROFILE_NUMBER
        depth_threshold : float, optional
            Minimum depth [m] for dark value calculation (default: 200)
        n_profiles : int, optional
            Number of deep profiles to use (default: 5)
        config_path : str or Path, optional
            Path to config file to check for existing dark value

        Returns
        -------
        dark_value : float
            Computed dark value
        profile_data : dict
            Dictionary containing profile information used in calculation
            Keys are profile numbers, values are dicts with 'depth', 'chla', 'min_value', 'min_depth'
        """

        # Check config file for existing dark value
        if self.dark_value:
            self.log(f"Using dark value from config: {self.dark_value}")
            return self.dark_value

        print(f"Computing dark value from profiles reaching >= {self.depth_threshold}m")

        # Get DEPTH and CHLA data
        missing_vars = {"TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"} - set(self.data.data_vars)
        if missing_vars:
            raise KeyError(f"[Chla Deep Correction] {missing_vars} could not be found in the data.")

        # Convert to pandas dataframe and interpolate the DEPTH data
        interp_data = self.data[["TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"]].to_pandas()
        interp_data["DEPTH"] = interp_data.set_index("TIME")["DEPTH"].interpolate().reset_index(drop=True)
        interp_data = interp_data.dropna(subset=["CHLA", "PROFILE_NUMBER"])

        # Subset the data to only deep measurements
        interp_data = interp_data[
            interp_data["DEPTH"] < self.depth_threshold
        ]

        # Remove profiles that do not have CHLA data below the threshold depth
        deep_profiles = interp_data.groupby("PROFILE_NUMBER").agg({"CHLA": "count"}).reset_index()
        deep_profiles = deep_profiles[deep_profiles["CHLA"] > 0]["PROFILE_NUMBER"].to_numpy()
        if len(deep_profiles) == 0:
            raise ValueError(
                "[Chla Deep Correction] No deep profiles could be identified. "
                "Try adjusting the 'depth_threshold' parameter."
            )
        interp_data = interp_data[interp_data["PROFILE_NUMBER"].isin(deep_profiles)]

        # Extract the profile number, depth and chla data for all chla minima per profile
        self.chla_deep_minima = interp_data.loc[
            interp_data.groupby("PROFILE_NUMBER")["CHLA"].idxmin(),
            ["TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"]
        ]

        # Compute median of minimum values
        self.dark_value = np.nanmedian(self.chla_deep_minima["CHLA"])
        self.log(
            f"\nComputed dark value: {self.dark_value:.6f} "
            f"(median of {len(self.chla_deep_minima)} profile minimums)\n"
            f"Min values range: {np.min(self.chla_deep_minima["CHLA"]):.6f} "
            f"to {np.max(self.chla_deep_minima["CHLA"]):.6f}"
        )

    def apply_dark_correction(self):
        """
        Apply dark value correction to CHLA data.
        """

        # Create adjusted chlorophyll variable
        self.data["CHLA_FLUORESCENCE"] = xr.DataArray(
            self.data["CHLA"] - self.dark_value,
            dims=self.data["CHLA"].dims,
            coords=self.data["CHLA"].coords,
        )

        # Copy and update attributes
        if hasattr(self.data["CHLA"], 'attrs'):
            self.data["CHLA_FLUORESCENCE"].attrs = self.data["CHLA"].attrs.copy()
        self.data["CHLA_FLUORESCENCE"].attrs["comment"] = (
            f"CHLA fluorescence with dark value correction (dark_value={self.dark_value:.6f})"
        )
        self.data["CHLA_FLUORESCENCE"].attrs["dark_value"] = self.dark_value

        self.log(f"Applied dark value correction: CHLA_FLUORESCENCE = CHLA_FLUORESCENCE - {self.dark_value:.6f}")

    def generate_diagnostics(self):

        mpl.use("tkagg")

        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        ax.plot(
            self.chla_deep_minima["CHLA"],
            self.chla_deep_minima["DEPTH"],
            ls="",
            marker="o",
            c="b"
        )

        ax.axhline(self.depth_threshold, ls="--", c="k", label="Depth Threshold")
        ax.axvline(self.dark_value, ls="--", c="r", label="Dark Value")
        ax.legend(loc="upper right")

        ax.set(
            xlabel="CHLA",
            ylabel="DEPTH",
            title="Deep Minima Values",
        )

        fig.tight_layout()
        plt.show(block=True)

