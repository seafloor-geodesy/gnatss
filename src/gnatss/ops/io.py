from __future__ import annotations

import enum
from typing import Any, Literal

import pandas as pd
import typer
import xarray as xr

from .. import constants
from ..configs.io import CSVOutput, InputData, OutputPath
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


def _check_and_delete_outliers_file(output: OutputPath):
    file_path = output.path + CSVOutput.outliers
    if output.fs.exists(file_path):
        # Delete previous outliers file if it exists
        # this is mostly for outliers
        output.fs.rm(file_path)


def to_file(
    config: Configuration,
    data_dict: dict[str, Any],
    key: str,
    file_name: str | enum.Enum,
    file_format: Literal["csv", "netcdf", "zarr"] = "csv",
):
    to_format = f"to_{file_format}"

    if config.output is None:
        msg: str = "Output configuration is not set"
        raise ValueError(msg)

    if isinstance(file_name, enum.Enum):
        file_name = file_name.value

    typer.echo(f"Exporting {key} to {file_name} ...")
    data = data_dict.get(key)
    no_data_message = f"No data found for {key}, skipping export!"
    if data is None:
        typer.echo(no_data_message)
        return

    if file_format == "csv" and not isinstance(data, pd.DataFrame):
        msg: str = f"Data must be a pandas DataFrame for CSV export, got {type(data)}"
        raise ValueError(msg)

    if file_format in ["netcdf", "zarr"] and not isinstance(data, xr.Dataset):
        msg = f"Data must be an xarray Dataset for {file_format} export, got {type(data)}"
        raise ValueError(msg)

    if isinstance(data, pd.DataFrame):
        if data.empty:
            typer.echo(no_data_message)
            # For outliers, we need to delete the file if it exists
            # when there is no data
            if key == "outliers":
                _check_and_delete_outliers_file(config.output)
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
    distance_limit: float | None = None,
    residual_limit: float | None = None,
    residual_range_limit: float | None = None,
    residual_outliers_threshold: float | None = None,
):
    # Override the distance and residual limits if provided
    # this short-circuits pydantic model
    if distance_limit is not None:
        config.solver.distance_limit = distance_limit

    if residual_limit is not None:
        config.solver.residual_limit = residual_limit

    if residual_outliers_threshold is not None:
        config.solver.residual_outliers_threshold = residual_outliers_threshold

    if residual_range_limit is not None:
        config.solver.residual_range_limit = residual_range_limit

    return config


def gather_files(
    config: Configuration,
    proc: Literal["main", "solver", "posfilter"] = "solver",
    mode: Literal["files", "object"] = "files",
) -> dict[str, list[str | InputData]]:
    """Gather file paths for the various dataset files defined in proc config.

    Parameters
    ----------
    config : Configuration
        A configuration object
    proc: Literal["solver", "posfilter"]
        Process name as defined in config

    Returns
    -------
    dict[str, Any]
        A dictionary containing the various datasets file paths
    """
    all_files_dict = {}
    if proc == "main":
        proc_config = config
    else:
        # Check for process type first
        if not hasattr(config, proc):
            msg: str = f"Unknown process type: {proc}"
            raise AttributeError(msg)
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
                msg: str = f"Unknown mode: {mode}"
                raise ValueError(msg)
            all_files_dict.setdefault(field, all_files)
    return all_files_dict


def gather_files_all_procs(
    config: Configuration,
    mode: Literal["files", "object"] = "files",
    from_cache: bool = False,
) -> dict[str, list[str]]:
    """Gather file paths for the various dataset files from all procs in config.

    Parameters
    ----------
    config : Configuration
        A configuration object

    Returns
    -------
    dict[str, Any]
        A dictionary containing the various datasets file paths
    """
    all_files_dict = {}
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
    distance_limit: float | None = None,
    residual_limit: float | None = None,
    residual_range_limit: float | None = None,
    outlier_threshold: float | None = None,
):
    config = load_configuration(config_yaml)
    return set_limits(
        config,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        residual_range_limit=residual_range_limit,
        residual_outliers_threshold=outlier_threshold,
    )


def load_datasets(
    config: Configuration,
    from_cache: bool = False,
    remove_outliers: bool = False,
    skip_posfilter: bool = False,
    skip_solver: bool = False,
):
    all_files_dict = {}
    mode = "object"

    # Gather main
    all_files_dict.update(gather_files(config, proc="main", mode=mode))

    # Gather posfilter (Skip if from_cache set)
    if not skip_posfilter and not from_cache:
        all_files_dict.update(gather_files(config, proc="posfilter", mode=mode))

    # Gather solver
    if not skip_solver:
        all_files_dict.update(gather_files(config, proc="solver", mode=mode))

    data_dict = {}
    if from_cache:
        # Special case for gps_solution
        typer.echo("Loading gps solution from cache ...")
        gps_solution_key = "gps_solution"
        file_path = config.output.path + CSVOutput.gps_solution
        gps_solution = _load_csv_data(config, file_path)
        data_dict[gps_solution_key] = gps_solution

        # Cache gps_solution will override the gps solution given
        # from configuration
        if gps_solution_key in all_files_dict:
            all_files_dict.pop(gps_solution_key)

        # TODO: Skip travel times if gps_solution is loaded from cache

    deletions_key = "deletions"
    if deletions_key not in all_files_dict:
        # Add deletions key if not found
        # this will grab the deletions file from the output path
        # if program has been run before
        all_files_dict.setdefault(deletions_key, None)

    for key, input_data in all_files_dict.items():
        data_dict[key] = load_files_to_dataframe(
            key, input_data, config, remove_outliers=remove_outliers
        )

    return data_dict


def load_files_to_dataframe(key, input_data, config: Configuration, remove_outliers: bool = False):
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

    if key == "deletions":
        return load_deletions(config=config, file_paths=file_paths, remove_outliers=remove_outliers)

    if key == "gps_positions":
        # Posfilter input
        return load_gps_positions(file_paths)

    if key == "gps_solution":
        # Solver input
        return load_gps_solutions(file_paths, from_legacy=False)

    loader = loaders_map[key]
    return loader(file_paths)


def _load_csv_data(config, file_path):
    if not config.output.fs.exists(file_path):
        msg: str = f"File {file_path} not found"
        raise FileNotFoundError(msg)

    with config.output.fs.open(file_path, "rb") as f:
        return pd.read_csv(f)
