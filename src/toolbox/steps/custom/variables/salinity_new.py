#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import matplotlib.pyplot as plt
from scipy import interpolate
import xarray as xr
import numpy as np
import gsw


def running_average_nan(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Estimate running average mean
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for symmetry.")

    pad_size = window_size // 2  # Symmetric padding
    padded = np.pad(arr, pad_size, mode="reflect")  # Edge handling

    kernel = np.ones(window_size)
    # Compute weighted sums while ignoring NaNs
    sum_vals = np.convolve(np.nan_to_num(padded), kernel, mode="valid")
    count_vals = np.convolve(~np.isnan(padded), kernel, mode="valid")

    # Compute the moving average, handling NaNs properly
    avg = np.divide(sum_vals, count_vals, where=(count_vals != 0))
    avg[count_vals == 0] = np.nan  # Set to NaN where all values were NaN

    return avg


def compute_optimal_lag(profile_data, filter_window_size):
    """
    Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
    Mimimize the standard deviation of the difference between a lagged CNDC and a high-pass filtered CNDC.
    The optimal lag is returned. The lag is chosen from -2 to 2s every 0.1s.
    This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.
    This correction is described in Woo (2019) but the mimimization is done between salinity and high-pass filtered salinity (as done by RBR, https://bitbucket.org/rbr/pyrsktools/src/master/pyrsktools) instead of comparing downcast vs upcast.

    Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/ 10.26198/5c997b5fdc9bd).


    Parameters
    ----------
    self.tsr: xarray.Dataset with raw CTD dataset for one single profile, which should contain:
        - TIME_CTD, sci_ctd41cp_timestamp, [numpy.datetime64]
        - PRES: pressure [dbar]
        - CNDC: conductivity [S/m]
        - TEMP: in-situ temperature [de C]

    windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.

    Returns
    -------
    self.tsr: with lags.

    """

    # remove any rows where conductivity is bad (nan)
    profile_data = profile_data[
        ["TIME_CTD",
         "CNDC",
         "PRES",
         "TEMP"]
    ].dropna(dim="N_MEASUREMENTS", subset=["CNDC"])

    # Find the elapsed time in seconds from the start of the profile
    t0 = profile_data["TIME_CTD"].values[0]
    profile_data["ELAPSED_TIME[s]"] = (profile_data["TIME_CTD"] - t0).dt.total_seconds()

    # Creates a callable function that predicts what CNDC would be at any given time
    conductivity_from_time = interpolate.interp1d(
        profile_data["ELAPSED_TIME[s]"].values,
        profile_data["CNDC"].values,
        bounds_error=False
    )

    # Specify the range time lags that the optimum will be found from. Column indexes are: (lag value, lag score)
    time_lags = np.array(
        [np.linspace(-2, 2, 41),
         np.full(41, np.nan)]
    ).T

    # For each lag find its score and add it to the time_lags array
    for i, lag in enumerate(lags[:, 0]):
        # Apply the time shift
        time_shifted_conductivity = conductivity_from_time(
            profile_data["ELAPSED_TIME[s]"] + lag
        )
        # Derive salinity with the time shifted CNDC (spiking will be minimized when CNDC and TEMP are aligned)
        PSAL = gsw.conversions.SP_from_C(
            time_shifted_conductivity,
            profile_data["TEMP"],
            profile_data["PRES"]
        )

        # Smooth the salinity profile (to remove spiking)
        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)

        # Subtracting the raw and smoothed salinity gives an idication of "spikiness".
        PSAL_Diff = PSAL - PSAL_Smooth

        # More spiky data will have higher standard deviation - which is used to score the effectiveness of the applied lag
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

    # return the time lag which has the lowerst score (standard deviation)
    best_score_index = np.argmin(time_lags[:, 1])
    return time_lags[best_score_index, 0]


@register_step
class AdjustSalinity(BaseStep, QCHandlingMixin):
    step_name = "Salinity Adjustment"

    def run(self):
        """
        Apply the thermal-lag correction for Salinity presented in Morrison et al 1994.
        The temperature is estimated inside the conductivity cell to estimate Salinity.
        This is based on eq.5 of Morrison et al. (1994), which doesn't require to know the sensitivity of temperature to conductivity (eq.2 of Morrison et al. 1994).
        No attempt is done yet to minimize the coefficients alpha/tau in T/S space, as in Morrison et al. (1994) or Garau et al. (2011).
        The fixed coefficients (alpha and tau) presented in Morrison et al. (1994) are used.
        These coefficients should be valid for pumped SeaBird CTsail as described in Woo (2019) by using their flow rate in the conductivity cell.
        This function should further be adapted to unpumped CTD by taking into account the glider velocity through the water based on the pitch angle or a hydrodynamic flight model.

        Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/10.26198/5c997b5fdc9bd).

        Config Example
        --------------
          - name: "ADJ: Salinity"
            parameters:
              CTLag: true
              thermal_lag_correction: true
              filter_window_size: 21
            diagnostics: false

        Parameters
        -----------

        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - time, sci_m_present_time, [numpy.datetime64]
            - PRES: pressure [dbar]
            - CNDC: conductivity [S/c]
            - TEMP: in-situ temperature [deg C]
            - LON: longitude
            - LAT: latitude

        Returns
        -------
            Nil - serves on self in-place
                MUST APPLY self.data to self.context["data"] to save the changes

        """


        self.log(f"Running adjustment...")
        # TODO: TIME_CTD checking

        # Filter user-specified flags
        self.filter_qc()

        # Correct conductivity-temperature response time misalignment (C-T Lag)
        self.correct_ct_lag()

        # Correct thermal mass error
        self.correct_thermal_lag()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        plt.ion()
        self.display_CTLag()
        self.display_adj_profiles()
        self.display_tsr_raw()
        self.display_tsr_adj()
        plt.ioff()

    def call_CTlag(self):
        """
        For the full deployment, calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
        If more than 300 profiles are present, the optimal lag is estimated every 10 profiles.
        Display the optimal conductivity time lag calculated for each profile, estimate the median of this lag, and apply this median lag to corrected variables (CNDC_ADJ/PSAL_ADJ).
        This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.


        Parameters
        ----------
        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - PROFILE_NUMBER
            - TIME_CTD, sci_ctd41cp_timestamp, [numpy.datetime64]
            - PRES: pressure [dbar]
            - CNDC: conductivity [mS/cm]
            - TEMP: in-situ temperature [de C]

        windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.

        Returns
        -------
        self.tsr: with tau and prof_i.

        """

        # Estimate the CT lag every profile or 10 profiles for more than 300 profiles.
        # Note that profile_numbers is not a list of consecutive integers as some profiles may have been filtered out.
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].values).astype("int32")
        if len(profile_numbers) > 300:
            profile_numbers = profile_numbers[::10]

        # Making a place to store intermediate products. The two column dimentions are (profile number, time lag)
        per_profile_optimal_lag = np.full((len(profile_numbers), 2), np.nan)

        # TODO: The following could be optimized using xarray groupby() applying a user defined CTLag function
        # Loop through all good profiles and store the optimal C-T lag for each.
        for i, profile_number in enumerate(profile_numbers):
            profile = self.data.where((self.data["PROFILE_NUMBER"] == profile_number), drop=True)
            if len(profile["TIME_CTD"]) > 3 * self.filter_window_size:
                optimal_lag = compute_optimal_lag(profile, self.filter_window_size)
                per_profile_optimal_lag[i, :] = [profile_number, optimal_lag]

        # TODO: Apply the optimal time lag and then stitch self.data back up
        # Find median time lag
        # Apply to all conductivity

    def run_thermal_lag_correction(self):

        nan_mask = self.data["TEMP"].isnull()
        data_subset = self.data[["TIME_CTD", "TEMP", "PRES"]][~nan_mask]

        # Interpolate TEMP onto a 1 Hz sample rate
        t0 = data_subset["TIME_CTD"].values[0]
        data_subset["ELAPSED_TIME[s]"] = (data_subset["TIME_CTD"] - t0).dt.total_seconds()

        t_second = interpolate.interp1d(TIME_CTD_s, TEMP, bounds_error=False)
        TIME_CTD_1s = np.arange(TIME_CTD_s[0], TIME_CTD_s[-1], 1)
        TEMP_1s = t_second(TIME_CTD_1s)

        # Tau and alpha are the fixed coefficients of Morison94 for unpumped cell.
        # alpha: initial amplitude of the temperature error for a unit step change in ambient temperature [without unit].
        # tau = beta^-1: time constant of the error, the e-folding time of the temperature error [s].
        # The flow rate in the conductivity cell (spd) comes from Woo (2019).
        self.alpha_offset = 0.0135
        self.alpha_slope = 0.0264
        self.tau_offset = 7.1499
        self.tau_slope = 2.7858
        spd = 0.4867

        self.tau = self.tau_offset + self.tau_slope / np.sqrt(spd)
        alpha = self.alpha_offset + self.alpha_slope / spd

        fn = 1/2  # Nyquist frequency for 1 Hz sampling (= sample frequency / 2)
        a = 4 * fn * alpha * self.tau / (1 + 4 * fn * self.tau)
        b = 1 - (2 * a / alpha)

        dTemp = np.full(TEMP_1s.shape[0], 0.0)
        TEMP_corrected_1s = np.full(TEMP_1s.shape[0], np.nan)
        TEMP_corrected = np.full(TEMP.shape[0], 0.0)
        for i in range(2, TEMP_1s.shape[0]):
            dTemp[i] = -b * dTemp[i - 1] + a * (TEMP_1s[i] - TEMP_1s[i - 1])

        TEMP_corrected_1s = TEMP_1s - dTemp

        t_sampled = interpolate.interp1d(
            TIME_CTD_1s, TEMP_corrected_1s, bounds_error=False
        )
        TEMP_corrected = t_sampled(TIME_CTD_s)

        # adj variables do not exist if call_CTlag was not run before
        varsi = ["PSAL_ADJ", "DENSITY_ADJ", "SIGMA0_ADJ"]
        for vari in varsi:
            if 1 in self.tsr.keys():  # Means call_CTlag was called before
                self.tsr[2][vari] = xr.DataArray(
                    np.full(self.tsr[0].TIME.shape[0], np.nan),
                    dims=["time"],
                    coords=dict(time=self.tsr[0].TIME),
                )
                self.tsr[2][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
                self.tsr[2][vari].attrs["comment"] = (
                    vari
                    + " with CT lag alignment and cell thermal mass correction (coefficients of Morrison et al. 1994)"
                )
            else:
                self.tsr[2][vari].values = np.full(self.tsr[0].TIME.shape[0], np.nan)
                self.tsr[2][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
                self.tsr[2][vari].attrs["comment"] = (
                    vari
                    + " with cell thermal mass correction (coefficients of Morrison et al. 1994)"
                )

        # Check that call_CTlag was run first
        if 1 in self.tsr.keys():
            CNDC = self.tsr[1].CNDC_ADJ.values[nonan]
        else:
            CNDC = self.tsr[0].CNDC.values[nonan]
        self.tsr[2]["PSAL_ADJ"].values[nonan] = gsw.SP_from_C(
            10 * CNDC, TEMP_corrected, PRES
        )  # OG1 is in [S/m] but should be in [mS/cm] for gsw.
        SA = gsw.SA_from_SP(
            self.tsr[2]["PSAL_ADJ"].values[nonan],
            PRES,
            self.tsr[2].LON.values[nonan],
            self.tsr[2].LAT.values[nonan],
        )
        CT = gsw.CT_from_t(SA, TEMP, PRES)
        self.tsr[2]["DENSITY_ADJ"].values[nonan] = gsw.rho(SA, CT, PRES)
        self.tsr[2]["SIGMA0_ADJ"].values[nonan] = gsw.sigma0(SA, CT)

    def display_CTLag(self):
        # Display optimal CTlag for each profile
        prof_min, prof_max = np.nanmin(self.prof_i), np.nanmax(self.prof_i)
        tau_median = np.nanmedian(self.tau)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            [prof_min, prof_max],
            [tau_median, tau_median],
            c="indianred",
            linestyle="--",
            linewidth=2,
            label=f"Median CTlag: {tau_median:.2f}s",
        )
        ax.plot([prof_min, prof_max], [0, 0], "k")
        ax.scatter(self.prof_i, self.tau, c="k")
        ax.legend(prop={"weight": "bold"}, labelcolor="indianred")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis([prof_min, prof_max, -1, 1])
        ax.set_ylabel(
            "CTlag [s]\n < 0: delay Cond by CTlag\n > 0: advance Cond by CTlag",
            fontweight="bold",
        )
        ax.set_xlabel("Profile Index", fontweight="bold")

        # Save corrected salinity/density after applying the optimal CTlag on conductivity (this conductivity is not saved, only used to reduce salinity noise).
        nonan = ~np.isnan(self.tsr[0].CNDC.values)
        TIME_CTD_s = (
            self.tsr[0].TIME_CTD.values[nonan] - self.tsr[0].TIME_CTD.values[nonan][0]
        ) / np.timedelta64(1, "s")
        CNDC = self.tsr[0].CNDC.values[nonan]
        shft = interpolate.interp1d(TIME_CTD_s, CNDC, bounds_error=False)

        varsi = ["CNDC_ADJ", "PSAL_ADJ", "DENSITY_ADJ", "SIGMA0_ADJ"]
        for vari in varsi:
            dim_name = self.tsr[1].TIME.dims[0]  # typically 'N_MEASUREMENTS'
            self.tsr[1][vari] = xr.DataArray(
                np.full(self.tsr[1].dims[dim_name], np.nan),
                dims=[dim_name],
                coords={dim_name: self.tsr[1].TIME},
            )
            self.tsr[1][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
            self.tsr[1][vari].attrs["comment"] = vari + " with CT lag alignment"
        self.tsr[1]["CNDC_ADJ"].values[nonan] = shft(TIME_CTD_s + tau_median)
        self.tsr[1]["PSAL_ADJ"].values[nonan] = gsw.conversions.SP_from_C(
            10 * self.tsr[1]["CNDC_ADJ"].values[nonan],
            self.tsr[1].TEMP.values[nonan],
            self.tsr[1].PRES.values[nonan],
        )
        SA = gsw.SA_from_SP(
            self.tsr[1]["PSAL_ADJ"].values[nonan],
            self.tsr[1].PRES.values[nonan],
            self.tsr[1].LON.values[nonan],
            self.tsr[1].LAT.values[nonan],
        )
        CT = gsw.CT_from_t(
            SA,
            self.tsr[1].TEMP.values[nonan],
            self.tsr[1].PRES.values[nonan],
        )
        self.tsr[1]["DENSITY_ADJ"].values[nonan] = gsw.rho(
            SA, CT, self.tsr[1].PRES.values[nonan]
        )
        self.tsr[1]["SIGMA0_ADJ"].values[nonan] = gsw.sigma0(SA, CT)

    def display_adj_profiles(self):
        """
        Display profiles for ~20 mid profiles of:
            (1) PSAL: raw salinity
            (2) PSAL_ADJ: salinity corrected from CTlag
            (3) PSAL_ADJ: salinity with the thermal lag correction
            (4) difference between raw and ADJ (CTlag + thermal lag correction) salinity and temperature

        """

        # Define a subset of ~20 profiles in the mid-mission
        prof_idx = self.tsr[0].PROFILE_NUMBER
        unique_prof = np.unique(prof_idx)
        prof1 = int(np.nanmedian(unique_prof))
        prof2 = prof1 + int(np.nanmin([unique_prof.shape[0] / 2, 20]))
        self.tsrs = {}
        for i in range(3):
            self.tsrs[i] = self.tsr[i].where(
                (prof_idx > prof1) & (prof_idx < prof2), drop=True
            )

        nonna = np.isfinite(self.tsrs[0].TEMP)
        idx_u = np.where(self.tsrs[0].profile_direction.values[nonna] == -1)  # Ups
        idx_d = np.where(self.tsrs[0].profile_direction.values[nonna] == 1)  # Downs

        fig, ax = plt.subplots(ncols=4, figsize=(15, 8), sharey=True)

        # Function to plot data for each category (downs and ups)
        def plot_profiles(self, index, color, labels):
            for ct, i in enumerate(
                np.unique(self.tsrs[0].PROFILE_NUMBER[index].values)
            ):
                prof_id = np.where(self.tsrs[0].PROFILE_NUMBER[index].values == i)

                ax[0].plot(
                    self.tsrs[0].PSAL[index][prof_id],
                    -self.tsrs[0].DEPTH[index][prof_id],
                    color,
                    label=labels[0] if ct == 0 else None,
                )
                ax[1].plot(
                    self.tsrs[1].PSAL_ADJ[index][prof_id],
                    -self.tsrs[1].DEPTH[index][prof_id],
                    color,
                    label=labels[1] if ct == 0 else None,
                )
                ax[2].plot(
                    self.tsrs[2].PSAL_ADJ[index][prof_id],
                    -self.tsrs[2].DEPTH[index][prof_id],
                    color,
                    label=labels[2] if ct == 0 else None,
                )

                ax[3].plot(
                    self.tsrs[2].PSAL_ADJ[index][prof_id]
                    - self.tsrs[0].PSAL[index][prof_id],
                    -self.tsrs[2].DEPTH[index][prof_id],
                    color,
                    label=labels[3] if ct == 0 else None,
                )
            if color == "grey":
                ax3 = ax[3].twiny()
                ax3.plot(
                    self.tsrs[2].TEMP[index][prof_id],
                    -self.tsrs[2].DEPTH[index][prof_id],
                    c="k",
                    label="Temperature" if ct == 0 else None,
                )
                ax3.set_xlabel("Temperature [$^{\circ}$C]", fontweight="bold")

        plot_profiles(
            self.tsrs,
            idx_d,
            "grey",
            ["downs", "downs (CTlag)", "downs (thermal mass)", "downs (ADJ - raw)"],
        )
        plot_profiles(
            self.tsrs,
            idx_u,
            "blue",
            ["ups", "ups (CTlag)", "ups (thermal mass)", "ups (ADJ - raw)"],
        )

        for i in range(4):
            ax[i].grid()
            ax[i].legend(prop={"weight": "bold", "size": 12})
            ax[i].set_xlabel(
                r"$\Delta$ PSAL [psu]" if i == 3 else "PSAL [psu]",
                fontweight="bold",
            )
        ax[2].set_xlim(
            [
                np.nanmin(self.tsrs[0].PSAL) - 0.02,
                np.nanmax(self.tsrs[0].PSAL) + 0.02,
            ]
        )
        minmax = np.nanmax(
            np.concatenate(
                (
                    np.abs(
                        self.tsrs[2].PSAL_ADJ.values[idx_u]
                        - self.tsrs[0].PSAL.values[idx_u]
                    ),
                    np.abs(
                        self.tsrs[2].PSAL_ADJ.values[idx_d]
                        - self.tsrs[0].PSAL.values[idx_d]
                    ),
                )
            )
        )
        ax[3].set_xlim([-minmax, minmax])
        ax[0].set_ylabel("Depth [m]", fontweight="bold")

        plt.tight_layout()

        if save == 1:
            plt.savefig(
                "plots/3_adj_profiles",
                bbox_inches="tight",
                pad_inches=0.02,
                dpi=160,
                facecolor="white",
            )

    def display_tsr_raw(self):
        """
        Display for ~20 mid profiles of:
            (1) TEMP: in situ temperature
            (2) PSAL: raw salinity
            (3) DENSITY: density

        """

        variables = ["TEMP", "PSAL", "DENSITY"]
        units = [r" [$^{\circ}$C]", " [psu]", r" [kg/m$^3$]"]
        colormaps = ["RdYlBu_r", "viridis", "Blues"]
        cs = {}

        fig, ax = plt.subplots(nrows=3, figsize=(7, 10), sharex=True, sharey=False)
        # Loop through variables to create scatter plots
        for i, var in enumerate(variables):
            data = self.tsrs[var] - (1000 if var == "DENSITY" else 0)  # Adjust density
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            vmin += 0.1 * (vmax - vmin)
            vmax -= 0.1 * (vmax - vmin)

            cs[i] = ax[i].scatter(
                self.tsrs.TIME,
                -self.tsrs.DEPTH,
                c=data,
                cmap=colormaps[i],
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(cs[i], ax=ax[i])
            cbar.set_label(var + units[i], fontsize=12, fontweight="bold")
            ax[i].set_ylabel("Depth [m]", fontweight="bold")

        ax[2].set_xlabel("TIME", fontweight="bold")
        plt.tight_layout()

    def display_tsr_adj(self):
        """
        Display for ~20 mid profiles (all=0) or the entire section (all=1) of:
            (1) PSAL: raw salinity
            (2) PSAL_ADJ: ADJ salinity
            (3) raw - ADJ salinity

        """

        prof_idx = self.tsr.PROFILE_NUMBER
        unique_prof = np.unique(prof_idx)
        prof1 = int(np.nanmedian(unique_prof))
        prof2 = prof1 + int(np.nanmin([unique_prof.shape[0] / 2, 20]))
        self.tsrs = self.tsr.where((prof_idx > prof1) & (prof_idx < prof2), drop=True)

        if all == 1:  # Display the entire deployment
            self.tsrs = self.tsr

        variables = ["PSAL", "PSAL_ADJ", "diff"]
        colormaps = ["viridis", "viridis", "RdBu_r"]
        cs = {}

        fig, ax = plt.subplots(
            nrows=3, figsize=(7 if all == 0 else 14, 10), sharex=True, sharey=False
        )
        # Loop through variables to create scatter plots
        for i, var in enumerate(variables):
            data = (
                self.tsrs["PSAL"] - self.tsrs["PSAL_ADJ"]
                if var == "diff"
                else self.tsrs[var]
            )
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            vmin += 0.1 * (vmax - vmin)
            vmax -= 0.1 * (vmax - vmin)

            if i == 2:
                cs[i] = ax[i].scatter(
                    self.tsrs.TIME,
                    -self.tsrs.DEPTH,
                    c=data,
                    cmap=colormaps[i],
                    vmin=-0.8 * np.nanmin([np.abs(vmin), np.abs(vmax)]),
                    vmax=0.8 * np.nanmin([np.abs(vmin), np.abs(vmax)]),
                )
            else:
                cs[i] = ax[i].scatter(
                    self.tsrs.TIME,
                    -self.tsrs.DEPTH,
                    c=data,
                    cmap=colormaps[i],
                    vmin=vmin,
                    vmax=vmax,
                )
            cbar = plt.colorbar(cs[i], ax=ax[i])
            if i == 2:
                cbar.set_label("PSAL - PSAL_ADJ [psu]", fontsize=12, fontweight="bold")
            else:
                cbar.set_label(var + " [psu]", fontsize=12, fontweight="bold")
            ax[i].set_ylabel("Depth [m]", fontweight="bold")

        ax[2].set_xlabel("TIME", fontweight="bold")
        plt.tight_layout()