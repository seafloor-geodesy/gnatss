"""The main command line interface for gnatss"""
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
    )
):
    """
    Run the full pre-processing routine for GNSS-A
    """
    typer.echo("Loading configuration ...")
    config = load_configuration(config_yaml)
    typer.echo("Configuration loaded.")
    all_files_dict = gather_files(config)

    main(config, all_files_dict)
