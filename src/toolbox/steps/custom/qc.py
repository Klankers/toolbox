"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag
import polars as pl
import xarray as xr
from datetime import datetime


# Defining QC functions from "Argo Quality Control Manual for CTD and Trajectory Data" (https://archimer.ifremer.fr/doc/00228/33951/).
# TODO: Do we need a bin nan rows QC to speed up processing?

def platform_identification_test(df):
    """
    Target Variable: ?
    Test Number: 1
    TODO: Unclear if this is redundant for gliders (Argo QC is designed for floats)
    """
    return df

def impossible_date_test(df):
    """
    Target Variable: TIME
    Test Number: 2
    Flag Number: 4 (bad data)
    Checks that the datetime of each point is valid.
    """
    df = df.with_columns(
        (
                (pl.col('TIME') > datetime(1985, 1, 1)) &
                (pl.col('TIME') < datetime.now()) &
                pl.col('TIME').is_not_null()
        ).alias('TIME_is_valid')
    )

    df = df.with_columns(
        (pl.not_(pl.col('TIME_is_valid')).cast(pl.Int64) * 4).alias('TIME_QC'),
    )
    return df

def impossible_location_test(df):
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 3
    Flag Number: 4 (bad data)
    Checks that the latitude and longitude are valid.
    """
    for label, bounds in zip(['LATITUDE', 'LONGITUDE'], [(-90, 90), (-180, 180)]):
        df = df.with_columns(
            (
                (pl.col(label) > bounds[0]) &
                (pl.col(label) < bounds[1]) &
                pl.col(label).is_not_null() &
                pl.col(label).is_finite()
            ).alias(f'{label}_is_valid')
        )

        df = df.with_columns(
            (pl.not_(pl.col(f'{label}_is_valid')).cast(pl.Int64) * 4).alias(f'{label}_QC'),
        )
    return df

def position_on_land_test(df):
    """
    Target Variable: LATITUDE, LONGITUDE
    Test Number: 4
    Flag Number: 4 (bad data)
    Checks that the measurement location is not on land.
    """
    #TODO: Find a lookup table to perform this check with

def impossible_speed_test(df):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Test Number: 5
    Flag Number: 4 (bad data)
    Checks that the the glider speed stays below 3m/s
    """
    df = df.with_columns((pl.col(time_col).diff().cast(pl.Float64) * 1e-9).alias('dt'))
    for label in ['LATITUDE', 'LONGITUDE']:
        df = df.with_columns(
            pl.col(label)
            .replace([np.inf, -np.inf, np.nan], None)
            .interpolate_by('TIME')
            .diff()
            .alias(f'delta_{label}')
        )
        df = df.with_columns(
            (pl.col(f'delta_{label}')/pl.col('dt')).alias(f'{label}_speed')
        )
    df = df.with_columns(
        ((pl.col('LATITUDE_speed')**2 + pl.col('LONGITUDE_speed')**2)**0.5)
        .alias('absolute_speed')
    )

    df = df.with_columns(
        (
            (pl.col(label) < 3) &
            pl.col(label).is_not_null() &
            pl.col(label).is_finite()
        ).alias('speed_is_valid')
    )

    for label in ['LATITUDE', 'LONGITUDE', 'TIME']:
        df = df.with_columns((pl.col('speed_is_valid').cast(pl.Int64) * 4).alias(f'{label}_QC'))

    return df

def global_range_test(df):
    """
    Target Variable: PRES, TEMP, PRAC_SALINITY
    Test Number: 6
    Flag Number: 4, 3 (bad data, probably bad data)
    Checks that the pressure, temperature and practically salinity do not lie outside expected
    global extremes.
    """
    # Structured (variable_to_test, [lower_limit, upper_limit], variables_to_flag, flag)
    test_calls = (
        ('PRES', [-np.inf, -5], ['PRES', 'TEMP', 'PRAC_SALINITY'], 4),
        ('PRES', [-5, -2.4], ['PRES', 'TEMP', 'PRAC_SALINITY'], 3),
        ('TEMP', [-2.5, 40], ['TEMP'], 4),
        ('PRAC_SALINITY', [2, 41], ['PRAC_SALINITY'], 4)
    )

    # Not the most efficient implementation because of second for loop.
    for var, lims, flag_vars, flag in test_calls:
        for flag_var in flag_vars:
            df = df.with_columns(
                ((pl.col(var) > lims[0]) & (pl.col(var) < lims[1])).not_().cast(pl.Int64) * flag
                .alias(f'{flag_var}_QC')
            )

    return df

def regional_range_test(df):
    """
    Target Variable: TEMP, PRAC_SALINITY
    Test Number: 7
    Flag Number: 4 (bad data)
    Checks that the temperature and practically salinity do not lie outside expected
    regional (Mediterranean and Red Seas) extremes.
    """
    import shapely as sh

    # Define Red and Mediterranean Sea areas
    Red_Sea = sh.geometry.Polygon([(40, 10), (45, 14), (35, 30), (30, 30), (40, 10)])  # (lon, lat)
    Med_Sea = sh.geometry.Polygon([(-6, 30), (40, 30), (35, 40), (20, 42), (15, 50), (-5, 40), (-6, 30)])
    # Check if data falls in those areas
    for poly, name in zip([Red_Sea, Med_Sea], ['in_red_sea', 'in_med_sea']):
        df = df.with_columns(
            pl.shape(['LONGITUDE', 'LATITUDE']).map_batches(
                lambda x: sh.contains_xy(poly, x.struct.field('LONGITUDE'), x.struct.field('LATITUDE'))
            ).alias(f'{name}')
        )

    # define regional temperature and salinity limits
    limits = {
        'red': {
            'TEMP': (21, 40),
            'PRAC_SALINITY': (2, 41)
        },
        'med': {
            'TEMP': (10, 40),
            'PRAC_SALINITY': (2, 40)
        }
    }
    # Check if the data satisfies these limits
    for region, var_lims in limits.items():
        for var, lims in var_lims.items():
            df = df.with_columns(
                ((pl.col(var) > lims[0]) & (pl.col(var) < lims[1])).not_()
            ).alias(f'bad_{region}_{var}')

        # for the flagging, data must fail the regional test AND be within that region.
        for var in ['TEMP', 'PRAC_SALINITY']:
            df = df.with_columns(
                ((pl.col(f'bad_{region}_{var}') & pl.col(f'in_{region}_sea')).cast(pl.Int64) * 4)
                .alias(f'{var}_QC')
            )

    return df

def pressure_increasing_test(df):
    """
    Target Variable: PRES
    Test Number: 8
    Flag Number: 4 (bad data)
    Checks for any egregious spikes in pressure between consecutive points.
    """
    # TODO: Check with others that this test is relevant alongside test 9

    return df

class SalinityQCStep(BaseStep):
    def run(self):
        print(f"[SalinityQC] Running QC with threshold {self.parameters['threshold']}")


class TemperatureQCStep(BaseStep):
    def run(self):
        print(
            f"[TemperatureQC] Running QC with threshold {self.parameters['threshold']}"
        )
