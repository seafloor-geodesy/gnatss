"""The main command line interface for gnatss"""
from pathlib import Path
from typing import Optional

import typer

from . import constants, package_name
from .configs.io import CSVOutput
from .configs.solver import Solver
from .loaders import load_configuration
from .main import gather_files, main

# Global variables
OVERRIDE_MESSAGE = "Note that this will override the value set as configuration."

app = typer.Typer(name=package_name)


@app.callback()
def callback():
    """
    GNSS-A Processing in Python
    """


@app.command()
def run(
    config_yaml: Optional[str] = typer.Option(
        None,
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
) -> None:
    """Runs the full pre-processing routine for GNSS-A

    Note: Currently only supports 3 transponders
    """
    typer.echo("Loading configuration ...")
    config = load_configuration(config_yaml)

    # Override the distance and residual limits if provided
    # this short-circuits pydantic model
    if distance_limit is not None:
        config.solver.distance_limit = distance_limit

    if residual_limit is not None:
        config.solver.residual_limit = residual_limit

    typer.echo("Configuration loaded.")
    all_files_dict = gather_files(config)

    # Run the main function
    # TODO: Clean up so that we aren't throwing data away
    _, _, resdf, dist_center_df, process_ds, outliers_df = main(
        config,
        all_files_dict,
        extract_process_dataset=extract_process_dataset,
        outlier_threshold=outlier_threshold,
    )

    # TODO: Switch to fsspec so we can save anywhere
    output_path = Path(config.output.path)
    if extract_dist_center:
        dist_center_csv = output_path / CSVOutput.dist_center.value
        typer.echo(
            f"Saving the distance from center file to {str(dist_center_csv.absolute())}"
        )
        dist_center_df.to_csv(dist_center_csv, index=False)

    # Write out to residuals.csv file
    res_csv = output_path / CSVOutput.residuals.value
    typer.echo(f"Saving the latest residuals to {str(res_csv.absolute())}")
    resdf.to_csv(res_csv, index=False)

    # Write out to outliers.csv file
    if len(outliers_df) > 0:
        outliers_csv = output_path / CSVOutput.outliers.value
        typer.echo(
            f"Saving the latest residual outliers to {str(outliers_csv.absolute())}"
        )
        outliers_df.to_csv(outliers_csv, index=False)

    if extract_process_dataset:
        # Write out to process_dataset.nc file
        process_dataset_nc = output_path / "process_dataset.nc"
        typer.echo(
            "Saving the process results "
            f"dataset to {str(process_dataset_nc.absolute())}"
        )
        process_ds.to_netcdf(process_dataset_nc)

    if qc:
        from .ops.qc import plot_enu_comps, plot_residuals

        res_png = output_path / "residuals.png"
        enu_comp_png = output_path / "residuals_enu_components.png"

        # Plot the figures
        res_figure = plot_residuals(resdf, outliers_df)
        enu_figure = plot_enu_comps(resdf, config)

        # Save the figures
        res_figure.savefig(res_png)
        enu_figure.savefig(enu_comp_png)
