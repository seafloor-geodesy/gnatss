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
    _, _, resdf = main(config, all_files_dict, extract_res=extract_res)

    if extract_res:
        # Write out to residuals.csv file
        # TODO: Switch to fsspec so we can save anywhere
        output_path = Path(config.output.path)
        csv_path = output_path / "residuals.csv"
        typer.echo(f"Saving the latest residuals to {str(csv_path.absolute())}")
        resdf.to_csv(csv_path, index=False)
