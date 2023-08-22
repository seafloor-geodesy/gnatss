"""The main command line interface for gnatss"""
from pathlib import Path
from typing import Optional

import typer

from . import package_name
from .loaders import load_configuration
from .main import gather_files, main

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
    extract_res: Optional[bool] = typer.Option(
        False, help="Flag to extract residual files from run."
    ),
    extract_dist_center: Optional[bool] = typer.Option(
        False, help="Flag to extract distance from center from run."
    ),
    extract_process_dataset: Optional[bool] = typer.Option(
        False, help="Flag to extract process results."
    ),
) -> None:
    """Runs the full pre-processing routine for GNSS-A

    Note: Currently only supports 3 transponders
    """
    typer.echo("Loading configuration ...")
    config = load_configuration(config_yaml)
    typer.echo("Configuration loaded.")
    all_files_dict = gather_files(config)

    # Run the main function
    # TODO: Clean up so that we aren't throwing data away
    _, _, resdf, dist_center_df, process_ds = main(
        config,
        all_files_dict,
        extract_res=extract_res,
        extract_dist_center=extract_dist_center,
        extract_process_dataset=extract_process_dataset,
    )

    # TODO: Switch to fsspec so we can save anywhere
    output_path = Path(config.output.path)
    if extract_dist_center:
        dist_center_csv = output_path / "dist_center.csv"
        typer.echo(
            f"Saving the distance from center file to {str(dist_center_csv.absolute())}"
        )
        dist_center_df.to_csv(dist_center_csv, index=False)

    if extract_res:
        # Write out to residuals.csv file
        res_csv = output_path / "residuals.csv"
        typer.echo(f"Saving the latest residuals to {str(res_csv.absolute())}")
        resdf.to_csv(res_csv, index=False)

    if extract_process_dataset:
        # Write out to process_dataset.nc file
        process_dataset_nc = output_path / "process_dataset.nc"
        typer.echo(
            f"Saving the process results dataset to {str(process_dataset_nc.absolute())}"
        )
        process_ds.to_netcdf(process_dataset_nc)
