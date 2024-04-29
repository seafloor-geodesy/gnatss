import enum
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import typer
import xarray as xr

from .. import constants
from ..configs.io import CSVOutput
from ..configs.main import Configuration
from ..loaders import (
    load_configuration,
    load_deletions,
    load_gps_positions,
    load_gps_solutions,
    load_novatel,
    load_novatel_std,
    load_quality_control,
    load_sound_speed,
    load_travel_times,
)


def _to_file_fs(fs, file_path: str, export_func, **kwargs):
    with fs.open(file_path, "wb") as f:
        export_func(f, **kwargs)
        typer.echo(f"Successfully exported data to {file_path}")


def to_file(
    config: Configuration,
    data_dict: Dict[str, Any],
    key: str,
    file_name: Union[str, enum.Enum],
    file_format: Literal["csv", "netcdf", "zarr"] = "csv",
):
    to_format = f"to_{file_format}"

    if config.output is None:
        raise ValueError("Output configuration is not set")

    if isinstance(file_name, enum.Enum):
        file_name = file_name.value

    typer.echo(f"Exporting {key} to {file_name} ...")
    data = data_dict.get(key)
    no_data_message = f"No data found for {key}, skipping export!"
    if data is None:
        typer.echo(no_data_message)
        return

    if file_format == "csv" and not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Data must be a pandas DataFrame for CSV export, got {type(data)}"
        )
    elif file_format in ["netcdf", "zarr"] and not isinstance(data, xr.Dataset):
        raise ValueError(
            f"Data must be an xarray Dataset for {file_format} export, got {type(data)}"
        )

    if isinstance(data, pd.DataFrame):
        if len(data) == 0:
            typer.echo(no_data_message)
            return
        export_kwargs = {"index": False}
    elif isinstance(data, xr.Dataset):
        if not data:
            typer.echo(no_data_message)
            return
        export_kwargs = {
            "mode": "w",
        }

    # Perform export
    file_path = config.output.path + file_name
    export_func = getattr(data, to_format)
    _to_file_fs(config.output._fsmap.fs, file_path, export_func, **export_kwargs)


def set_limits(
    config,
    distance_limit: Optional[float] = None,
    residual_limit: Optional[float] = None,
    residual_outliers_threshold: Optional[float] = None,
):
    # Override the distance and residual limits if provided
    # this short-circuits pydantic model
    if distance_limit is not None:
        config.solver.distance_limit = distance_limit

    if residual_limit is not None:
        config.solver.residual_limit = residual_limit

    if residual_outliers_threshold is not None:
        config.solver.residual_outliers_threshold = residual_outliers_threshold

    return config


def gather_files(
    config: Configuration,
    proc: Literal["main", "solver", "posfilter"] = "solver",
    mode: Literal["files", "object"] = "files",
) -> Dict[str, List[str]]:
    """Gather file paths for the various dataset files defined in proc config.

    Parameters
    ----------
    config : Configuration
        A configuration object
    proc: Literal["solver", "posfilter"]
        Process name as defined in config

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the various datasets file paths
    """
    all_files_dict = {}
    if proc == "main":
        proc_config = config
    else:
        # Check for process type first
        if not hasattr(config, proc):
            raise AttributeError(f"Unknown process type: {proc}")
        proc_config = getattr(config, proc)

    if proc_config:
        input_files = proc_config.input_files
        for field in input_files.model_fields_set:
            input_data = getattr(input_files, field)
            typer.echo(f"Gathering {field} at {input_data.path}")
            if mode == "files":
                all_files = input_data.files
            elif mode == "object":
                all_files = input_data
            else:
                raise ValueError(f"Unknown mode: {mode}")
            all_files_dict.setdefault(field, all_files)
    return all_files_dict


def gather_files_all_procs(
    config: Configuration,
    mode: Literal["files", "object"] = "files",
    from_cache: bool = False,
) -> Dict[str, List[str]]:
    """Gather file paths for the various dataset files from all procs in config.

    Parameters
    ----------
    config : Configuration
        A configuration object

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the various datasets file paths
    """
    all_files_dict = dict()
    for proc in constants.DEFAULT_CONFIG_PROCS:
        if from_cache and proc == "posfilter":
            # Skip performing posfilter operations
            # if we are loading solution from cache
            continue

        if proc == "main" or hasattr(config, proc):
            all_files_dict.update(gather_files(config, proc, mode))
    return all_files_dict


def load_config(
    config_yaml: str,
    distance_limit: Optional[float] = None,
    residual_limit: Optional[float] = None,
    outlier_threshold: Optional[float] = None,
):
    config = load_configuration(config_yaml)
    config = set_limits(
        config,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        residual_outliers_threshold=outlier_threshold,
    )
    return config


def load_datasets(config: Configuration, from_cache: bool = False):
    all_files_dict = gather_files_all_procs(
        config, mode="object", from_cache=from_cache
    )
    data_dict = {}
    gps_solution_key = "gps_solution"
    if from_cache:
        # Special case for gps_solution
        typer.echo("Loading gps solution from cache ...")
        file_path = config.output.path + CSVOutput.gps_solution
        gps_solution = _load_csv_data(config, file_path)
        data_dict[gps_solution_key] = gps_solution

        # Cache gps_solution will override the gps solution given
        # from configuration
        if gps_solution_key in all_files_dict:
            all_files_dict.pop(gps_solution_key)

    deletions_key = "deletions"
    if deletions_key not in all_files_dict:
        # Add deletions key if not found
        # this will grab the deletions file from the output path
        # if program has been run before
        all_files_dict.setdefault(deletions_key, None)

    for key, input_data in all_files_dict.items():
        data_dict[key] = load_files_to_dataframe(key, input_data, config)

    return data_dict


def load_files_to_dataframe(key, input_data, config: Configuration):
    if input_data is None:
        typer.echo(f"Loading {key} from {config.output.path}")
        file_paths = input_data
        loader_kwargs = {}
    else:
        typer.echo(f"Loading {key} from {input_data.path}")
        file_paths = input_data.files
        loader_kwargs = input_data.loader_kwargs

    loaders_map = {
        "sound_speed": load_sound_speed,
        "quality_controls": load_quality_control,
        "novatel": load_novatel,
        "novatel_std": load_novatel_std,
    }
    if key == "travel_times":
        return load_travel_times(
            file_paths,
            transponder_ids=[tp.pxp_id for tp in config.transponders],
            **loader_kwargs,
        )
    elif key == "deletions":
        return load_deletions(config=config, file_paths=file_paths)
    elif key == "gps_positions":
        # Posfilter input
        return load_gps_positions(file_paths)
    elif key == "gps_solution":
        # Solver input
        return load_gps_solutions(file_paths, from_legacy=False)
    else:
        loader = loaders_map[key]
        return loader(file_paths)


def _load_csv_data(config, file_path):
    if not config.output.fs.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with config.output.fs.open(file_path, "rb") as f:
        return pd.read_csv(f)
