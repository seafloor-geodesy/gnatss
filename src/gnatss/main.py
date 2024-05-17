from typing import Optional

import typer

from .configs.io import CSVOutput
from .ops.data import data_loading, preprocess_data
from .ops.io import to_file
from .ops.qc import export_qc_plots
from .posfilter.run import run_posfilter
from .solver.run import run_solver


def run_gnatss(
    config_yaml: str,
    distance_limit: Optional[float] = None,
    residual_limit: Optional[float] = None,
    outlier_threshold: Optional[float] = None,
    from_cache: bool = False,
    return_raw: bool = False,
    remove_outliers: bool = False,
    extract_process_dataset: bool = True,
    extract_dist_center: bool = True,
    qc: bool = True,
    skip_posfilter: bool = False,
    skip_solver: bool = False,
):
    typer.echo("Starting GNATSS ...")
    if from_cache:
        typer.echo(
            "Flag `from_cache` is set. Skipping data loading and processing for posfilter step."
        )
    config, data_dict = data_loading(
        config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        outlier_threshold=outlier_threshold,
        from_cache=from_cache,
        remove_outliers=remove_outliers,
        skip_posfilter=skip_posfilter,
        skip_solver=skip_solver,
    )
    config, data_dict = preprocess_data(config, data_dict)

    if config.posfilter and not from_cache and not skip_posfilter:
        data_dict = run_posfilter(config, data_dict)

    if config.solver and not skip_solver:
        data_dict = run_solver(config, data_dict, return_raw=return_raw)
        # Write out to residuals.csv file
        to_file(config, data_dict, "residuals", CSVOutput.residuals)

        # Write out to outliers.csv file
        to_file(config, data_dict, "outliers", CSVOutput.outliers)

        # Write out distance from center to dist_center.csv file
        if extract_dist_center:
            to_file(config, data_dict, "distance_from_center", CSVOutput.dist_center)

        # Write out to process_dataset.nc file
        if extract_process_dataset:
            to_file(
                config,
                data_dict,
                "process_dataset",
                "process_dataset.nc",
                file_format="netcdf",
            )
        # Export QC plots
        if qc:
            export_qc_plots(config, data_dict)

    typer.echo("Finished GNATSS.")
    return config, data_dict
