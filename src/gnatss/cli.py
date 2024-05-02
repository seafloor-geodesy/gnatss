"""The main command line interface for gnatss"""

from typing import Optional

import typer

from . import __version__, package_name
from .configs.solver import Solver
from .main import run_gnatss

# Global variables
OVERRIDE_MESSAGE = "Note that this will override the value set as configuration."

app = typer.Typer(name=package_name, pretty_exceptions_show_locals=False)


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    )
):
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
        None,
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
    remove_outliers: Optional[bool] = typer.Option(
        False,
        help=(
            "Flag to execute removing outliers from the GNSS-A L2 Data "
            "before running the solver process."
        ),
    ),
    run_all: Optional[bool] = typer.Option(
        True, help="Flag to run the full end-to-end GNSS-A processing routine."
    ),
    solver: Optional[bool] = typer.Option(
        False, help="Flag to run the solver process only. Requires GNSS-A L2 Data."
    ),
    posfilter: Optional[bool] = typer.Option(
        False,
        help="Flag to run the posfilter process only. Requires GNSS-A L1 Data Inputs.",
    ),
) -> None:
    """Runs the full pre-processing routine for GNSS-A

    Note: Currently only supports 3 transponders
    """
    if all([run_all, solver, posfilter]):
        raise ValueError("Cannot run all and solver or posfilter at the same time.")
    elif all([not x for x in [run_all, solver, posfilter]]):
        raise ValueError("Must specify either all, solver, or posfilter.")

    skip_posfilter = False
    skip_solver = False
    if solver or posfilter:
        skip_posfilter = not posfilter
        skip_solver = not solver

    typer.echo(f"skip_posfilter: {skip_posfilter}")
    typer.echo(f"skip_solver: {skip_solver}")

    run_gnatss(
        config_yaml=config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        outlier_threshold=outlier_threshold,
        from_cache=from_cache,
        remove_outliers=remove_outliers,
        extract_dist_center=extract_dist_center,
        extract_process_dataset=extract_process_dataset,
        qc=qc,
        skip_posfilter=skip_posfilter,
        skip_solver=skip_solver,
    )
