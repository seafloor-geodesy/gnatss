from __future__ import annotations

import typer

from .. import constants
from ..configs.io import CSVOutput
from ..ops.io import to_file


def filter_columns(data_dict):
    all_observations = data_dict.get("gps_solution")
    if all_observations is not None:
        typer.echo("Filtering observations columns for necessary columns only...")
        partial_obs = all_observations.drop(
            [
                *list(constants.DATA_SPEC.platform_rx_fields.keys()),
                *list(constants.DATA_SPEC.platform_tx_fields.keys()),
            ],
            axis="columns",
        )
        partial_obs = partial_obs.drop(
            [
                name
                for name in constants.DATA_SPEC.gnss_rx_fields
                if name not in constants.DATA_SPEC.gnss_rx_cov_fields
            ],
            axis="columns",
        )
        partial_obs = partial_obs.drop(
            [
                name
                for name in constants.DATA_SPEC.gnss_tx_fields
                if name not in constants.DATA_SPEC.gnss_tx_cov_fields
            ],
            axis="columns",
        )
        data_dict.update({"gps_solution": partial_obs})
    return data_dict


def export_gps_solution(config, data_dict):
    to_file(config, data_dict, "gps_solution", CSVOutput.gps_solution)
