"""The main command line interface for seagap"""
from typing import Any, Dict, Optional

import typer

from . import package_name
from .configs.main import Configuration, load_configuration
from .harmonic_mean import sv_harmonic_mean
from .utilities.io import _get_filesystem

app = typer.Typer(name=package_name)


def load_files(config: Configuration) -> Dict[str, Any]:
    all_files_dict = {}
    for k, v in config.solver.input_files.dict().items():
        path = v.get("path", "")
        typer.echo(f"Loading {k} at {path}")
        storage_options = v.get("storage_options", {})

        fs = _get_filesystem(path, storage_options)
        if "**" in path:
            all_files = fs.glob(path)
        else:
            all_files = path

        all_files_dict.setdefault(k, all_files)
    return all_files_dict


def main(config: Configuration, all_files_dict: Dict[str, Any]):
    import pandas as pd

    # Read sound speed
    svdf = pd.read_csv(
        all_files_dict["sound_speed"],
        delim_whitespace=True,
        header=None,
        names=["dd", "sv"],
    )
    transponders = config.solver.transponders
    start_depth = config.solver.harmonic_mean_start_depth

    # Compute harmonic mean of each transponder
    typer.echo("Computing harmonic mean...")
    for transponder in transponders:
        # Compute the harmonic mean and round to 3 decimal places
        harmonic_mean = round(
            sv_harmonic_mean(svdf, start_depth, transponder.height), 3
        )
        transponder.sv_mean = harmonic_mean
        typer.echo(transponder)
    typer.echo("Finished computing harmonic mean")


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
    all_files_dict = load_files(config)

    main(config, all_files_dict)
