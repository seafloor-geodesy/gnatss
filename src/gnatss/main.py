from typing import Any, Dict

import numpy as np
import pandas as pd
import typer

from .configs.main import Configuration
from .configs.solver import ArrayCenter
from .constants import (
    GPS_COV_DIAG,
    GPS_GEOCENTRIC,
    GPS_GEODETIC,
    GPS_LOCAL_TANGENT,
    GPS_TIME,
)
from .harmonic_mean import sv_harmonic_mean
from .loaders import load_sound_speed
from .utilities.geo import geocentric2enu, geocentric2geodetic
from .utilities.io import _get_filesystem


def gather_files(config: Configuration) -> Dict[str, Any]:
    all_files_dict = {}
    for k, v in config.solver.input_files.dict().items():
        path = v.get("path", "")
        typer.echo(f"Gathering {k} at {path}")
        storage_options = v.get("storage_options", {})

        fs = _get_filesystem(path, storage_options)
        if "**" in path:
            all_files = fs.glob(path)
        else:
            all_files = path

        all_files_dict.setdefault(k, all_files)
    return all_files_dict


def _find_gps_record(
    gps_solutions: pd.DataFrame, travel_time: pd.Timestamp
) -> pd.Series:
    match = gps_solutions.iloc[
        gps_solutions[GPS_TIME]
        .apply(lambda row: (row - travel_time))
        .abs()
        .argsort()[0],
        :,
    ]

    return match


def _compute_std_and_verify(
    gps_series: pd.Series, std_dev: bool = True, sigma_limit: float = 0.05
) -> float:
    # Compute the 3d std (sum variances of GPS components and take sqrt)
    sig_3d = np.sqrt(np.sum(gps_series[GPS_COV_DIAG] ** (2 if std_dev else 1)))

    # Verify sigma value, throw error if greater than gps sigma limit
    if sig_3d > sigma_limit:
        raise ValueError(
            f"3D Standard Deviation of {sig_3d} exceeds GPS Sigma Limit of {sigma_limit}!"
        )

    return sig_3d


def _compute_enu_series(
    input_series: pd.Series, array_center: ArrayCenter
) -> pd.Series:
    array_center_coords = [array_center.lon, array_center.lat, array_center.alt]

    location_series = input_series.copy()

    geodetic_coords = geocentric2geodetic(*location_series[GPS_GEOCENTRIC].values)
    enu_coords = geocentric2enu(
        *location_series[GPS_GEOCENTRIC].values, *array_center_coords
    ).flatten()

    # Set geodetic lon,lat,alt to the series
    for idx, v in enumerate(geodetic_coords):
        location_series[GPS_GEODETIC[idx]] = v

    # Set local tangent e,n,u to the series
    for idx, v in enumerate(enu_coords):
        location_series[GPS_LOCAL_TANGENT[idx]] = v

    return location_series


def calc_uv(input_vector: np.ndarray) -> np.ndarray:
    """Calculate unit vector"""

    if len(input_vector.shape) > 1:
        raise ValueError("Unit vector calculation must be 1-D array!")

    vector_norm = np.linalg.norm(input_vector)

    if vector_norm == 0:
        return np.array([2.0, 0.0, 0.0])

    return input_vector / vector_norm


def main(config: Configuration, all_files_dict: Dict[str, Any]):
    # Read sound speed
    svdf = load_sound_speed(all_files_dict["sound_speed"])
    transponders = config.solver.transponders
    start_depth = config.solver.harmonic_mean_start_depth

    # Compute harmonic mean of each transponder
    typer.echo("Computing harmonic mean...")
    for transponder in transponders:
        # Compute the harmonic mean and round to 3 decimal places
        harmonic_mean = round(
            sv_harmonic_mean(svdf, start_depth, transponder.height), 3
        )
        transponder.sv_mean = harmonic_mean
        typer.echo(transponder)
    typer.echo("Finished computing harmonic mean")
