from typing import Any, Dict

import typer

from .configs.main import Configuration
from .harmonic_mean import sv_harmonic_mean
from .loaders import load_sound_speed
from .utilities.io import _get_filesystem


def gather_files(config: Configuration) -> Dict[str, Any]:
    all_files_dict = {}
    for k, v in config.solver.input_files.dict().items():
        path = v.get("path", "")
        typer.echo(f"Gathering {k} at {path}")
        storage_options = v.get("storage_options", {})

        fs = _get_filesystem(path, storage_options)
        if "**" in path:
            all_files = fs.glob(path)
        else:
            all_files = path

        all_files_dict.setdefault(k, all_files)
    return all_files_dict


def main(config: Configuration, all_files_dict: Dict[str, Any]):
    # Read sound speed
    svdf = load_sound_speed(all_files_dict["sound_speed"])
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
