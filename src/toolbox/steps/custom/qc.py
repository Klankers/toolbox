"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag
import polars as pl
import xarray as xr
from datetime import datetime


# Defining QC functions from "Argo Quality Control Manual for CTD and Trajectory Data" (https://archimer.ifremer.fr/doc/00228/33951/).

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


class SalinityQCStep(BaseStep):
    def run(self):
        print(f"[SalinityQC] Running QC with threshold {self.parameters['threshold']}")


class TemperatureQCStep(BaseStep):
    def run(self):
        print(
            f"[TemperatureQC] Running QC with threshold {self.parameters['threshold']}"
        )
