from __future__ import annotations

import typer

from .configs.io import CSVOutput
from .configs.main import Configuration
from .ops.data import data_loading, preprocess_data
from .ops.io import to_file
from .ops.qc import export_qc_plots
from .posfilter.run import run_posfilter
from .solver.run import run_solver


def run_gnatss(
    config_yaml: str,
    distance_limit: float | None = None,
    residual_limit: float | None = None,
    residual_range_limit: float | None = None,
    outlier_threshold: float | None = None,
    from_cache: bool = False,
    return_raw: bool = False,
    remove_outliers: bool = False,
    extract_process_dataset: bool = True,
    extract_dist_center: bool = True,
    qc: bool = True,
    skip_posfilter: bool = False,
    skip_solver: bool = False,
) -> tuple[Configuration, dict[str, any]]:
    """
    The main function to run GNATSS from end-to-end.

    Parameters
    ----------
    config_yaml : str
        Path to the configuration yaml file

    distance_limit : float | None, optional
        Distance in meters from center beyond
        which points will be excluded from solution

        *Setting this argument will override the value set in the configuration.*

    residual_limit : float | None, optional
        Maximum residual in centimeters beyond
        which data points will be excluded from solution

        *Setting this argument will override the value set in the configuration.*

    residual_range_limit : float | None, optional
        Maximum residual range (maximum - minimum) in centimeters for
        a given epoch, beyond which data points will be excluded from solution

        *Setting this argument will override the value set in the configuration.*

    outlier_threshold : float | None, optional
        Residual outliers threshold acceptable before throwing an error in percent

        *Setting this argument will override the value set in the configuration.*

    from_cache : bool, optional
        Flag to load the GNSS-A Level-2 Data from cache.

        *Setting this to ``True`` will load the data from the cache output directory
        set in the configuration*

    return_raw : bool, optional
        Flag to return raw processing data as
        part of the result dictionary, by default False

    remove_outliers : bool, optional
        Flag to execute removing outliers from the GNSS-A Level-2 Data
        before running the solver process, by default False

    extract_process_dataset : bool, optional
        Flag to extract the process dataset as a netCDF file, by default True

    extract_dist_center : bool, optional
        Flag to extract the distance from center as a CSV file, by default True

    qc : bool, optional
        Flag to plot residuals from run and store in output folder, by default True

    skip_posfilter : bool, optional
        Flag to skip the posfilter step, by default False

    skip_solver : bool, optional
        Flag to skip the solver step, by default False

    Returns
    -------
    config : Configuration
        Configuration object containing all the configuration values
    data_dict : dict
        Dictionary containing the results of the GNATSS run
    """
    typer.echo("Starting GNATSS ...")
    if from_cache:
        typer.echo(
            "Flag `from_cache` is set. Skipping data loading and processing for posfilter step."
        )
    config, data_dict = data_loading(
        config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        residual_range_limit=residual_range_limit,
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
