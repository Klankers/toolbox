"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import interpolate
import gsw

def running_average_nan( arr: np.ndarray, window_size: int ) -> np.ndarray:
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

class SalinityQC(BaseStep):
    step_name = "SalinityQC"
    def run(self):
        """
        Apply the thermal-lag correction for salinity presented in Morrison et al 1994.
        The temperature is estimated inside the conductivity cell to estimate salinity.
        This is based on eq.5 of Morrison et al. (1994), which doesn't require to know the sensitivity of temperature to conductivity (eq.2 of Morrison et al. 1994).
        No attempt is done yet to minimize the coefficients alpha/tau in T/S space, as in Morrison et al. (1994) or Garau et al. (2011).
        The fixed coefficients (alpha and tau) presented in Morrison et al. (1994) are used.
        These coefficients should be valid for pumped SeaBird CTsail as described in Woo (2019) by using their flow rate in the conductivity cell.
        This function should further be adapted to unpumped CTD by taking into account the glider velocity through the water based on the pitch angle or a hydrodynamic flight model.

        Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/ 10.26198/5c997b5fdc9bd).

        Parameters
        -----------

        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - time, sci_m_present_time, [numpy.datetime64]
            - pressure [dbar]
            - conductivity [mS/cm]
            - in-situ temperature [de C]
            - longitude
            - latitude

        Returns
        -------
            Nil - serves on self in-place
                MUST APPLY self.data to self.context["data"] to save the changes

        """
        print(f"[SalinityQC] Running QC")
        print(f"[SalinityQC] Run with diagnostics to determine if CTlag and thermal lag correction should be applied. Diagnostics: {self.diagnostics}")

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            # Prepare the data for Salinity QC
            self.tsr = {}
            self.tsrs = {}
            self.tsr[0] = self.context["data"]
            prof_idx = self.tsr[0].profile_index
            unique_prof = np.unique(prof_idx)
            prof1 = int(np.nanmedian(unique_prof))
            prof2 = prof1 + int(np.nanmin([unique_prof.shape[0] / 2, 20]))
            self.tsrs[0] = self.tsr[0].where((prof_idx > prof1) & (prof_idx < prof2), drop=True)

            # Run functions based on params
            if self.diagnostics:
                self.generate_diagnostics()
            else:
                if self.parameters["call_CTlag"]:
                    self.tsr = self.call_CTlag()
                if self.parameters["thermal_lag_correction"]:
                    self.thermal_lag_correction()


            self.context["data"] = self.tsr[0]
            return self.context

    def generate_diagnostics(self):
        plt.ion()
        print(f"[SalinityQC] Generating diagnostics...")

        if self.parameters["call_CTlag"]:
            self.tsr = self.call_CTlag()
            self.display_CTLag()
        if self.parameters["thermal_lag_correction"]:
            self.thermal_lag_correction()
            self.display_qc_profiles()
        self.display_tsr_raw()
        self.display_tsr_qc()
        plt.ioff()
        print(f"[SalinityQC] Diagnostics generated.")

    def thermal_lag_correction(self):
        

        # Check that call_CTlag was run first
        if 1 in self.tsr.keys():
            self.tsr[2] = self.tsr[1].copy()
        else:
            self.tsr[2] = self.tsr[0].copy()
        nonan = np.where(np.isfinite(self.tsr[0].temperature))
        temperature = self.tsr[0].temperature.values[nonan]
        time = self.tsr[0].time_ctd.values[nonan]
        pressure = self.tsr[0].pressure.values[nonan]

        # Apply a 1-second interpolation to have a constant sampling rate (1 Hz)
        freq = 1
        fn = 1.0 / (2 * freq)  # Nyquist frequency

        time_s = (time - time[0]) / np.timedelta64(
            1, "s"
        )  # Array of time difference in seconds from 1st measurement.
        t_second = interpolate.interp1d(time_s, temperature, bounds_error=False)
        time_1s = np.arange(time_s[0], time_s[-1], 1)
        temperature_1s = t_second(time_1s)

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

        a = 4 * fn * alpha * self.tau / (1 + 4 * fn * self.tau)
        b = 1 - (2 * a / alpha)

        dTemp = np.full(temperature_1s.shape[0], 0.0)
        temperature_corrected_1s = np.full(temperature_1s.shape[0], np.nan)
        temperature_corrected = np.full(temperature.shape[0], 0.0)
        for i in range(2, temperature_1s.shape[0]):
            dTemp[i] = -b * dTemp[i - 1] + a * (
                temperature_1s[i] - temperature_1s[i - 1]
            )

        temperature_corrected_1s = temperature_1s - dTemp

        t_sampled = interpolate.interp1d(
            time_1s, temperature_corrected_1s, bounds_error=False
        )
        temperature_corrected = t_sampled(time_s)

        # qc variables do not exist if call_CTlag was not run before
        varsi = ["salinity_qc", "density_qc", "potential_density_qc"]
        for vari in varsi:
            if 1 in self.tsr.keys():
                self.tsr[2][vari] = xr.DataArray(
                    np.full(self.tsr[0].time.shape[0], np.nan),
                    dims=["time"],
                    coords=dict(time=self.tsr[0].time),
                )
                self.tsr[2][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
                self.tsr[2][vari].attrs["comment"] = (
                    vari
                    + " with CT lag alignment and cell thermal mass correction (coefficients of Morrison et al. 1994)"
                )
            else:
                self.tsr[2][vari].values = np.full(self.tsr[0].time.shape[0], np.nan)
                self.tsr[2][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
                self.tsr[2][vari].attrs["comment"] = (
                    vari
                    + " with cell thermal mass correction (coefficients of Morrison et al. 1994)"
                )

        # Check that call_CTlag was run first
        if 1 in self.tsr.keys():
            conductivity = self.tsr[1].conductivity_qc.values[nonan]
        else:
            conductivity = self.tsr[0].conductivity.values[nonan]
        self.tsr[2]["salinity_qc"].values[nonan] = gsw.SP_from_C(
            10 * conductivity, temperature_corrected, pressure
        )  # C*10 because needs to be in [ mS/cm ] instead of [ S/m ].
        SA = gsw.SA_from_SP(
            self.tsr[2]["salinity_qc"].values[nonan],
            pressure,
            self.tsr[2].longitude.values[nonan],
            self.tsr[2].latitude.values[nonan],
        )
        CT = gsw.CT_from_t(SA, temperature, pressure)
        self.tsr[2]["density_qc"].values[nonan] = gsw.rho(SA, CT, pressure)
        self.tsr[2]["potential_density_qc"].values[nonan] = gsw.sigma0(SA, CT)

    def call_CTlag(self):
        """
        For the full deployment, calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
        If more than 300 profiles are present, the optimal lag is estimated every 10 profiles.
        Display the optimal conductivity time lag calculated for each profile, estimate the median of this lag, and apply this median lag to corrected variables (conductivity_qc/salinity_qc).
        This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.


        Parameters
        ----------
        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - profile_index
            - time, sci_m_present_time, [numpy.datetime64]
            - pressure [dbar]
            - conductivity [mS/cm]
            - in-situ temperature [de C]
            - longitude
            - latitude
            
        windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.
                    
        Returns
        -------
        self.tsr: with conductivity_qc and salinity_qc.

        """
        print(f"[SalinityQC] Running CTlag")
        self.tsr[1] = self.tsr[0].copy()
        prof_idx = np.unique(self.tsr[0].profile_index.values).astype('int32')
        # Estimate the CT lag every profile or 10 profiles for more than 300 profiles.
        d_prof = 10 if prof_idx.shape[0] > 300 else 1
        prof_size = prof_idx[::d_prof].shape[0]
        self.tau = np.full(prof_size, np.nan)
        self.prof_i = self.tau.copy()
        # Loop through every profile, keep the optimal CTlag for each profile, with the min standard dev of raw vs smoothed salinity.
        for count, i in enumerate(prof_idx[::d_prof]):
            self.tsr_i = self.tsr[0].where((self.tsr[0].profile_index == i), drop=True)
            if self.tsr_i.time.shape[0] > 3* self.parameters['windowLength']:
                self.prof_i[count] = i
                self.tau[count] = self.compute_optimal_lag( self.tsr_i ) 

        
        return self.tsr

    def compute_optimal_lag( self ):
        """
        Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
        Mimimize the standard deviation of the difference between a lagged salinity and a high-pass filtered salinity.
        The optimal lag is returned. The lag is chosen from -2 to 2s every 0.1s.
        This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.
        This correction is described in Woo (2019) but the mimimization is done between salinity and high-pass filtered salinity (as done by RBR, https://bitbucket.org/rbr/pyrsktools/src/master/pyrsktools) instead of comparing downcast vs upcast. 

        Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/ 10.26198/5c997b5fdc9bd).


        Parameters
        ----------
        self.tsr: xarray.Dataset with raw CTD dataset for one single profile, which should contain:
            - time_ctd, sci_ctd41cp_timestamp, [numpy.datetime64]
            - pressure [dbar]
            - conductivity [mS/cm]
            - in-situ temperature [de C]
            
        windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.
                    
        Returns
        -------
        self.tsr: with conductivity_qc and salinity_qc.

        """
        

        idx = ~np.isnan(self.tsr_i['conductivity'].values)
        time_i = self.tsr_i.time_ctd.values[idx]
        conductivity = self.tsr_i.conductivity.values[idx]
        temperature = self.tsr_i.temperature.values[idx]
        pressure = self.tsr_i.pressure.values[idx]

        time_s = (time_i - time_i[0]) / np.timedelta64(1,'s') # Array of time difference in seconds from 1st measurement.
        
        shft = interpolate.interp1d(time_s, conductivity, bounds_error=False)

        lags = np.linspace(-2,2,41)
        runningStdDiff = np.empty_like(lags, dtype=np.float64)

        for i, l in enumerate(lags):
            cShift = shft(time_s + l)
            salinity = gsw.conversions.SP_from_C(10 * cShift, temperature, pressure)
            salinitySmooth = running_average_nan(salinity, self.parameters['windowLength'])
            salinityDiff = salinity - salinitySmooth
            runningStdDiff[i] = np.nanstd(salinityDiff)
        return lags[np.argmin(runningStdDiff)]

    def display_CTLag(self):
        # Display optimat CTlag for each profile
        prof_min, prof_max = np.nanmin(self.prof_i), np.nanmax(self.prof_i)
        tau_median = np.nanmedian(self.tau)
        
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot([prof_min, prof_max], [tau_median, tau_median], c='indianred', linestyle='--', linewidth=2, label=f'Median CTlag: {tau_median:.2f}s')
        ax.plot([prof_min, prof_max], [0,0], 'k')
        ax.scatter(self.prof_i, self.tau, c='k')
        ax.legend(prop={'weight': 'bold'}, labelcolor='indianred')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis([prof_min, prof_max, -1, 1])
        ax.set_ylabel('CTlag [s]\n < 0: delay Cond by CTlag\n > 0: advance Cond by CTlag', fontweight='bold')
        ax.set_xlabel('Profile Index', fontweight='bold')

        # Save corrected salinity/density after applying the optimal CTlag on conductivity (this conductivity is not saved, only used to reduce salinity noise).
        nonan = ~np.isnan(self.tsr[0].conductivity.values)
        time_second = ((self.tsr[0].time.values[nonan] - self.tsr[0].time.values[nonan][0]) / np.timedelta64(1,'s'))
        conductivity = self.tsr[0].conductivity.values[nonan]
        shft = interpolate.interp1d(time_second, conductivity, bounds_error=False)

        varsi = ['conductivity_qc','salinity_qc', 'density_qc', 'potential_density_qc']
        for vari in varsi:
            self.tsr[1][vari] = xr.DataArray( np.full(self.tsr[1].time.shape[0],np.nan), dims=['time'], coords=dict(time=self.tsr[1].time))
            self.tsr[1][vari].attrs = self.tsr[0][vari[:-3]].attrs.copy()
            self.tsr[1][vari].attrs['comment'] = vari + ' with CT lag alignment'
        self.tsr[1]['conductivity_qc'].values[nonan] = shft(time_second + tau_median)
        self.tsr[1]['salinity_qc'].values[nonan] = gsw.conversions.SP_from_C(10*self.tsr[1]['conductivity_qc'].values[nonan], self.tsr[1].temperature.values[nonan], self.tsr[1].pressure.values[nonan])
        SA = gsw.SA_from_SP( self.tsr[1]['salinity_qc'].values[nonan], self.tsr[1].pressure.values[nonan], self.tsr[1].longitude.values[nonan], self.tsr[1].latitude.values[nonan])
        CT = gsw.CT_from_t(SA, self.tsr[1].temperature.values[nonan], self.tsr[1].pressure.values[nonan])
        self.tsr[1]['density_qc'].values[nonan] = gsw.rho(SA, CT, self.tsr[1].pressure.values[nonan])
        self.tsr[1]['potential_density_qc'].values[nonan] = gsw.sigma0(SA, CT)

        if save==1:
            plt.savefig( 'plots/2_CTlag', bbox_inches='tight', pad_inches=0.02, dpi=160, facecolor='white')
 
    def display_qc_profiles(self):
        """
        Display profiles for ~20 mid profiles of:
            (1) raw salinity
            (2) salinity corrected from CTlag
            (3) salinity with the thermal lag correction
            (4) difference between raw and QC (CTlag + thermal lag correction) salinity and temperature

        """

        # Define a subset of ~20 profiles in the mid-mission
        prof_idx = self.tsr[0].profile_index
        unique_prof = np.unique(prof_idx)
        prof1 = int(np.nanmedian(unique_prof))
        prof2 = prof1 + int(np.nanmin([unique_prof.shape[0] / 2, 20]))
        self.tsrs={}
        for i in range(3):
            self.tsrs[i] = self.tsr[i].where((prof_idx > prof1) & (prof_idx < prof2), drop=True)

        nonna = np.isfinite( self.tsrs[0].temperature )
        idx_u = np.where( self.tsrs[0].profile_direction.values[nonna]==-1 ) # Ups
        idx_d = np.where( self.tsrs[0].profile_direction.values[nonna]==1 ) # Downs

        fig, ax = plt.subplots(ncols=4, figsize=(15, 8), sharey=True)
        # Function to plot data for each category (downs and ups)
        def plot_profiles(self.tsrs, index, color, labels):
            for ct, i in enumerate(np.unique(self.tsrs[0].profile_index[index].values)):
                prof_id = np.where(self.tsrs[0].profile_index[index].values == i)

                ax[0].plot(self.tsrs[0].salinity[index][prof_id], -self.tsrs[0].depth[index][prof_id], color, label=labels[0] if ct == 0 else None)
                ax[1].plot(self.tsrs[1].salinity_qc[index][prof_id], -self.tsrs[1].depth[index][prof_id], color, label=labels[1] if ct == 0 else None)
                ax[2].plot(self.tsrs[2].salinity_qc[index][prof_id], -self.tsrs[2].depth[index][prof_id], color, label=labels[2] if ct == 0 else None)
                
                ax[3].plot(self.tsrs[2].salinity_qc[index][prof_id] - self.tsrs[0].salinity[index][prof_id], -self.tsrs[2].depth[index][prof_id], color, label=labels[3] if ct == 0 else None)
            if color=='grey':
            ax3 = ax[3].twiny()
            ax3.plot(self.tsrs[2].temperature[index][prof_id], -self.tsrs[2].depth[index][prof_id], c='k', label='Temperature' if ct == 0 else None)
            ax3.set_xlabel('Temperature [$^{\circ}$C]', fontweight='bold')

        plot_profiles(self.tsrs, idx_d, 'grey', ['downs', 'downs (CTlag)', 'downs (thermal mass)', 'downs (qc - raw)'])
        plot_profiles(self.tsrs, idx_u, 'blue', ['ups', 'ups (CTlag)', 'ups (thermal mass)', 'ups (qc - raw)'])

        for i in range(4):
            ax[i].grid()
            ax[i].legend(prop={'weight': 'bold', 'size': 12})
            ax[i].set_xlabel(r'$\Delta$ Salinity [psu]' if i==3 else 'Salinity [psu]', fontweight='bold')
        ax[2].set_xlim([np.nanmin(self.tsrs[0].salinity) - 0.02, np.nanmax(self.tsrs[0].salinity) + 0.02])
        minmax = np.nanmax( np.concatenate((np.abs(self.tsrs[2].salinity_qc.values[idx_u] - self.tsrs[0].salinity.values[idx_u]), np.abs(self.tsrs[2].salinity_qc.values[idx_d] - self.tsrs[0].salinity.values[idx_d]))) )
        ax[3].set_xlim([-minmax, minmax])
        ax[0].set_ylabel('Depth [m]', fontweight='bold')

        plt.tight_layout()

        if save==1:
            plt.savefig( 'plots/3_qc_profiles', bbox_inches='tight', pad_inches=0.02, dpi=160, facecolor='white')

    def display_tsr_raw( self):
        """
        Display for ~20 mid profiles of:
            (1) in situ temperature
            (2) raw salinity
            (3) density

        """

        variables = ['temperature', 'salinity', 'density']
        units = [r' [$^{\circ}$C]', ' [psu]', r' [kg/m$^3$]']
        colormaps = ['RdYlBu_r', 'viridis', 'Blues']
        cs = {}

        fig, ax = plt.subplots(nrows=3, figsize=(7, 10), sharex=True, sharey=False)
        # Loop through variables to create scatter plots
        for i, var in enumerate(variables):
            data = self.tsrs[var] - (1000 if var == 'density' else 0)  # Adjust density
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            vmin += 0.1 * (vmax - vmin)
            vmax -= 0.1 * (vmax - vmin)

            cs[i] = ax[i].scatter(self.tsrs.time, -self.tsrs.depth, c=data, cmap=colormaps[i], vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(cs[i], ax=ax[i])
            cbar.set_label(var + units[i], fontsize=12, fontweight="bold")
            ax[i].set_ylabel('Depth [m]', fontweight='bold')

        ax[2].set_xlabel('Time', fontweight='bold')
        plt.tight_layout()

        if save==1:
            plt.savefig( 'plots/1_raw_profiles', bbox_inches='tight', pad_inches=0.02, dpi=160, facecolor='white')

    def display_tsr_qc( self ):
        """
        Display for ~20 mid profiles (all=0) or the entire section (all=1) of:
            (1) raw salinity
            (2) QC salinity
            (3) raw - QC salinity

        """

        prof_idx = self.tsr.profile_index
        unique_prof = np.unique(prof_idx)
        prof1 = int(np.nanmedian(unique_prof))
        prof2 = prof1 + int(np.nanmin([unique_prof.shape[0] / 2, 20]))
        self.tsrs = self.tsr.where((prof_idx > prof1) & (prof_idx < prof2), drop=True)

        if all==1: # Display the entire deployment
            self.tsrs = self.tsr

        variables = ['salinity', 'salinity_qc', 'diff']
        colormaps = ['viridis', 'viridis', 'RdBu_r']
        cs = {}

        fig, ax = plt.subplots(nrows=3, figsize=(7 if all==0 else 14, 10), sharex=True, sharey=False)
        # Loop through variables to create scatter plots
        for i, var in enumerate(variables):
            data =  (self.tsrs['salinity'] - self.tsrs['salinity_qc'] if var == 'diff' else self.tsrs[var])
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            vmin += 0.1 * (vmax - vmin)
            vmax -= 0.1 * (vmax - vmin)

            if i==2:
                cs[i] = ax[i].scatter(self.tsrs.time, -self.tsrs.depth, c=data, cmap=colormaps[i], vmin=-0.8*np.nanmin([np.abs(vmin),np.abs(vmax)]), vmax=0.8*np.nanmin([np.abs(vmin),np.abs(vmax)]))
            else:
                cs[i] = ax[i].scatter(self.tsrs.time, -self.tsrs.depth, c=data, cmap=colormaps[i], vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(cs[i], ax=ax[i])
            if i==2:
                cbar.set_label('salinity - salinity_qc [psu]', fontsize=12, fontweight="bold")
            else:
                cbar.set_label(var + ' [psu]', fontsize=12, fontweight="bold")
            ax[i].set_ylabel('Depth [m]', fontweight='bold')

        ax[2].set_xlabel('Time', fontweight='bold')
        plt.tight_layout()

        if save==1:
            plt.savefig( 'plots/4_qc_section', bbox_inches='tight', pad_inches=0.02, dpi=160, facecolor='white')

class TemperatureQC(BaseStep):
    def run(self):
        print(
            f"[TemperatureQC] Running QC with threshold {self.parameters['threshold']}"
        )
