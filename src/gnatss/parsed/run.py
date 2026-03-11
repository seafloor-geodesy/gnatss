from __future__ import annotations

import typer

from .. import constants
from ..ops.data import ensure_monotonic_increasing, standardize_data
from ..posfilter import rotation
from ..posfilter.utilities import export_gps_solution, filter_columns
from .parsed import get_parsed_ant_positions, get_parsed_rph


def run_parsed(config, data_dict):
    typer.echo("Calculating antenna positions ...")
    # These are antenna positions and covariances
    pos_twtt = get_parsed_ant_positions(
        raw_df=data_dict.get("raw_data"),
        gps_df=data_dict.get("gps_positions"),
    )
    typer.echo("Extracting platform roll/pitch/heading ...")
    cov_rph_twtt = get_parsed_rph(
        raw_df=data_dict.get("raw_data"),
    )
    typer.echo("Performing Rotation ...")
    pos_freed_trans_twtt = rotation(
        pos_twtt, cov_rph_twtt, config.parsed.atd_offsets, config.array_center
    )
    typer.echo(f"Standardizing data to specification version {constants.DATA_SPEC.version} ...")
    all_observations = standardize_data(pos_freed_trans_twtt)

    # Ensure the data is sorted properly
    all_observations = ensure_monotonic_increasing(all_observations)

    # This will overwrite the gps_solution file
    # if also set on the solver input
    data_dict.update({"gps_solution": all_observations})

    export = config.parsed.export
    if export and not export.full:
        data_dict = filter_columns(data_dict)
        export_gps_solution(config, data_dict)
    return data_dict
