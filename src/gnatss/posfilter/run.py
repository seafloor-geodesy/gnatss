from __future__ import annotations

import typer

from .. import constants
from ..ops.data import ensure_monotonic_increasing, standardize_data
from .posfilter import kalman_filtering, rotation, spline_interpolate
from .utilities import export_gps_solution, filter_columns


def run_posfilter(config, data_dict):
    typer.echo("Performing Kalman filtering ...")
    # These are antenna positions and covariances
    pos_twtt = kalman_filtering(
        inspvaa_df=data_dict.get("novatel"),
        insstdeva_df=data_dict.get("novatel_std"),
        gps_df=data_dict.get("gps_positions"),
        twtt_df=data_dict.get("travel_times"),
    )
    typer.echo("Performing Spline Interpolation ...")
    cov_rph_twtt = spline_interpolate(
        inspvaa_df=data_dict.get("novatel"),
        insstdeva_df=data_dict.get("novatel_std"),
        twtt_df=data_dict.get("travel_times"),
    )
    typer.echo("Performing Rotation ...")
    pos_freed_trans_twtt = rotation(
        pos_twtt, cov_rph_twtt, config.posfilter.atd_offsets, config.array_center
    )
    typer.echo(f"Standardizing data to specification version {constants.DATA_SPEC.version} ...")
    all_observations = standardize_data(pos_freed_trans_twtt)

    # Ensure the data is sorted properly
    all_observations = ensure_monotonic_increasing(all_observations)

    # This will overwrite the gps_solution file
    # if also set on the solver input
    data_dict.update({"gps_solution": all_observations})

    export = config.posfilter.export
    if export and not export.full:
        data_dict = filter_columns(data_dict)
        export_gps_solution(config, data_dict)
    return data_dict
