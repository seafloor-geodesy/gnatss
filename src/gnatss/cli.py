"""The main command line interface for gnatss"""
from typing import Optional

import typer

from . import constants, package_name
from .configs.io import CSVOutput
from .configs.solver import Solver
from .main import run_gnatss
from .ops.io import to_file
from .ops.qc import export_qc_plots

# Global variables
OVERRIDE_MESSAGE = "Note that this will override the value set as configuration."

app = typer.Typer(name=package_name, pretty_exceptions_show_locals=False)


@app.callback()
def callback():
    """
    GNSS-A Processing in Python
    """


@app.command()
def run(
    config_yaml: str = typer.Argument(
        ...,
        help="Custom path to configuration yaml file. **Currently only support local files!**",
    ),
    extract_dist_center: Optional[bool] = typer.Option(
        True, help="Flag to extract distance from center from run."
    ),
    extract_process_dataset: Optional[bool] = typer.Option(
        True, help="Flag to extract process results."
    ),
    outlier_threshold: Optional[float] = typer.Option(
        constants.DATA_OUTLIER_THRESHOLD,
        help=(
            "Threshold for allowable percentage of outliers "
            "before raising a runtime error."
        ),
    ),
    distance_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('distance_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    residual_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('residual_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    qc: Optional[bool] = typer.Option(
        True, help="Flag to plot residuals from run and store in output folder."
    ),
    from_cache: Optional[bool] = typer.Option(
        False, help="Flag to load the GNSS-A L2 Data from cache."
    ),
) -> None:
    """Runs the full pre-processing routine for GNSS-A

    Note: Currently only supports 3 transponders
    """
    config, result_dict = run_gnatss(
        config_yaml=config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        outlier_threshold=outlier_threshold,
        from_cache=from_cache,
    )

    # Write out distance from center to dist_center.csv file
    if extract_dist_center:
        to_file(config, result_dict, "distance_from_center", CSVOutput.dist_center)

    # Write out to residuals.csv file
    to_file(config, result_dict, "residuals", CSVOutput.residuals)

    # Write out to outliers.csv file
    to_file(config, result_dict, "outliers", CSVOutput.outliers)

    # Write out to process_dataset.nc file
    if extract_process_dataset:
        to_file(
            config,
            result_dict,
            "process_dataset",
            "process_dataset.nc",
            file_format="netcdf",
        )

    # Export QC plots
    if qc:
        export_qc_plots(config, result_dict)
