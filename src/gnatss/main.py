from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import typer
import xarray as xr
from nptyping import Float, NDArray, Shape

from . import constants
from .configs.io import CSVOutput
from .configs.main import Configuration
from .configs.solver import ArrayCenter, SolverTransponder
from .loaders import (
    get_atd_offsets,
    load_deletions,
    load_gps_solutions,
    load_quality_control,
    load_roll_pitch_heading,
    load_sound_speed,
    load_travel_times,
)
from .ops.data import clean_tt, data_loading, filter_tt, preprocess_data
from .ops.harmonic_mean import sv_harmonic_mean
from .ops.io import to_file
from .ops.qc import export_qc_plots
from .ops.validate import check_sig3d
from .posfilter.posfilter import rotation
from .posfilter.run import run_posfilter
from .solver.run import run_solver
from .solver.utilities import (
    _create_process_dataset,
    _get_latest_process,
    extract_distance_from_center,
    extract_latest_residuals,
    prepare_and_solve,
)
from .utilities.time import AstroTime


def get_transmit_times(
    cleaned_travel_times: pd.DataFrame,
    all_gps_solutions: pd.DataFrame,
    rph_data: pd.DataFrame,
    gps_sigma_limit: float,
    atd_offsets: Optional[NDArray[Shape["3"], Float]] = None,
    array_center: Optional[ArrayCenter] = None,
) -> pd.DataFrame:
    """
    Merges cleaned transmit times with gps solutions and roll-pitch-heading solutions into
    one dataframe. Then calculates GNSS antenna positions, and checks for 3d std deviation.

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    all_gps_solutions : pd.DataFrame
        The full gps solutions data
    rph_data : pd.DataFrame
        The full roll-pitch-heading data. If non-empty Dataframe, calculate GNSS antenna positions.
    gps_sigma_limit : float
        Maximum positional sigma allowed to use GPS positions
    atd_offsets : Optional[NDArray[Shape["3"], Float]]
        (Optional argument) Numpy array containing forward, rightward, and downward atd offset
        values. Required argument if GNSS antenna position calculation being performed.
    array_center : Optional[ArrayCenter]
        (Optional argument) An object containing the center of the array.
        Required argument if GNSS antenna position calculation being performed.

    Returns
    -------
    pd.DataFrame
        The transmit times data with gps solutions included
    """
    # Merge with gps solutions
    transmit_times = pd.merge(
        cleaned_travel_times[[constants.TT_TIME]],
        all_gps_solutions,
        left_on=constants.TT_TIME,
        right_on=constants.GPS_TIME,
    )

    # Merge with rph data if rph_data df is non-empty
    if not rph_data.empty:
        transmit_times = pd.merge(
            transmit_times,
            rph_data,
            left_on=constants.TT_TIME,
            right_on=constants.RPH_TIME,
        )
        # Remove RPH_TIME column from transmit_times df after merge
        if constants.TT_TIME != constants.RPH_TIME:
            transmit_times.drop(constants.RPH_TIME, axis="columns", inplace=True)

        # Calculate GNSS antenna positions
        transmit_times = rotation(
            transmit_times,
            atd_offsets,
            array_center,
            constants.RPH_LOCAL_TANGENTS,
            constants.GPS_GEOCENTRIC,
            constants.ANTENNA_DIRECTIONS,
        )

    # Compute and check 3d standard deviation
    transmit_times = check_sig3d(data=transmit_times, gps_sigma_limit=gps_sigma_limit)

    # Adds a 0 to column names for transmit values
    transmit_times.columns = [
        f"{col}0" if col != constants.TT_TIME else constants.DATA_SPEC.tx_time
        for col in transmit_times.columns
    ]

    return transmit_times


def get_reply_times(
    cleaned_travel_times: pd.DataFrame,
    all_gps_solutions: pd.DataFrame,
    rph_data: pd.DataFrame,
    gps_sigma_limit: float,
    transponder_ids: List[str],
    atd_offsets: Optional[NDArray[Shape["3"], Float]] = None,
    array_center: Optional[ArrayCenter] = None,
):
    """
    Merges cleaned reply times with gps solutions and roll-pitch-heading solutions into one
    dataframe. Then calculates GNSS antenna positions, and checks for 3d std deviation.

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    all_gps_solutions : pd.DataFrame
        The full gps solutions data
    rph_data : pd.DataFrame
        The full roll-pitch-heading data. If non-empty Dataframe, calculate GNSS antenna positions.
    gps_sigma_limit : float
        Maximum positional sigma allowed to use GPS positions
    transponder_ids : List[str]
        A list of the transponder ids that matches the order
        with ``cleaned_travel_times`` data
    atd_offsets : Optional[NDArray[Shape["3"], Float]]
        (Optional argument) Numpy array containing forward, rightward, and downward atd
        offset values. Required argument if GNSS antenna position calculation being performed.
    array_center : Optional[ArrayCenter]
        (Optional argument) An object containing the center of the array.
        Required argument if GNSS antenna position calculation being performed.

    Returns
    -------
    pd.DataFrame
        The reply times data with gps solutions included
    """
    reply_times = cleaned_travel_times[transponder_ids]
    reply_times[constants.DATA_SPEC.tx_time] = cleaned_travel_times[constants.TT_TIME]

    # Pivot the table by stacking
    reply_times = reply_times.set_index(constants.DATA_SPEC.tx_time).stack()
    reply_times = reply_times.rename(constants.DATA_SPEC.travel_time)
    reply_times.index = reply_times.index.rename(
        [constants.DATA_SPEC.tx_time, constants.DATA_SPEC.transponder_id]
    )
    reply_times = reply_times.to_frame().reset_index()
    # Set RT
    reply_times[constants.DATA_SPEC.rx_time] = reply_times.apply(
        lambda row: row[constants.DATA_SPEC.tx_time]
        + row[constants.DATA_SPEC.travel_time],
        axis=1,
    )

    # Merge with gps solutions
    reply_times = pd.merge(
        reply_times,
        all_gps_solutions,
        left_on=constants.DATA_SPEC.rx_time,
        right_on=constants.GPS_TIME,
    )
    reply_times = reply_times.drop(constants.GPS_TIME, axis="columns")

    # Merge with rph data if rph_data df is non-empty
    if not rph_data.empty:
        # Merge with rph data
        reply_times = pd.merge(
            reply_times,
            rph_data,
            left_on=constants.DATA_SPEC.rx_time,
            right_on=constants.RPH_TIME,
        )
        # Remove RPH_TIME column from reply_times df after merge
        if constants.DATA_SPEC.rx_time != constants.RPH_TIME:
            reply_times.drop(constants.RPH_TIME, axis="columns", inplace=True)

        # Calculate GNSS antenna positions
        reply_times = rotation(
            reply_times,
            atd_offsets,
            array_center,
            constants.RPH_LOCAL_TANGENTS,
            constants.GPS_GEOCENTRIC,
            constants.ANTENNA_DIRECTIONS,
        )

    # Compute and check 3d standard deviation
    reply_times = check_sig3d(data=reply_times, gps_sigma_limit=gps_sigma_limit)

    # Currently looks for even value counts... check fortran code what to do here?
    time_counts = reply_times[constants.DATA_SPEC.tx_time].value_counts()
    reply_times = reply_times[
        reply_times[constants.DATA_SPEC.tx_time].isin(
            time_counts[time_counts == len(transponder_ids)].index
        )
    ]

    # Adds a 1 to column names for reply values
    reply_times.columns = [
        (
            f"{col}1"
            if col
            not in [
                constants.DATA_SPEC.tx_time,
                constants.DATA_SPEC.transponder_id,
                constants.DATA_SPEC.rx_time,
                constants.DATA_SPEC.travel_time,
            ]
            else col
        )
        for col in reply_times.columns
    ]

    return reply_times


def _print_final_stats(
    transponders: List[SolverTransponder], process_data: Dict[str, Any]
):
    """Print out final solution statistics and results"""
    num_transponders = len(transponders)
    # Get the latest process data
    process_info = _get_latest_process(process_data)
    typer.echo("---- FINAL SOLUTION ----")
    data = process_info["data"]
    lat_lon = process_info["transponders_lla"]
    enu_arr = process_info["enu"]
    sig_enu = process_info["sig_enu"]
    transponders_xyz = process_info["transponders_xyz"]
    for idx, tp in enumerate(transponders):
        pxp_id = tp.pxp_id
        typer.echo(pxp_id)
        x, y, z = transponders_xyz[idx]
        lat, lon, alt = lat_lon[idx]

        SIGPX = np.array_split(data["sigpx"], num_transponders)
        sigX, sigY, sigZ = SIGPX[idx]

        # Compute enu
        e, n, u = enu_arr[idx]

        # Get sig enu
        sigE, sigN, sigU = sig_enu[idx]

        typer.echo(
            (
                f"x = {np.round(x, 4)} +/- {np.format_float_scientific(sigX, 6)} m "
                f"del_e = {np.round(e, 4)} +/- {np.format_float_scientific(sigE, 6)} m"
            )
        )
        typer.echo(
            (
                f"y = {np.round(y, 4)} +/- {np.format_float_scientific(sigY, 6)} m "
                f"del_n = {np.round(n, 4)} +/- {np.format_float_scientific(sigN, 6)} m"
            )
        )
        typer.echo(
            (
                f"z = {np.round(z, 4)} +/- {np.format_float_scientific(sigZ, 6)} m "
                f"del_u = {np.round(u, 4)} +/- {np.format_float_scientific(sigU, 6)} m"
            )
        )
        typer.echo(f"Lat. = {lat} deg, Long. = {lon}, Hgt.msl = {alt} m")
    typer.echo("------------------------")
    typer.echo()


def load_data(
    all_files_dict: Dict[str, Any],
    config: Configuration,
    all_observations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Loads all of the necessary datasets for processing into a singular
    pandas dataframe object.

    Parameters
    ----------
    all_files_dict : Dict[str, Any]
        A dictionary containing the various datasets file paths
    config : Configuration
        The configuration file object

    Returns
    -------
    pd.DataFrame
        All observations dataframe
    """
    # Read sound speed
    typer.echo("Load sound speed profile data...")
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

    # Read deletion file
    # Set default to empty string
    all_files_dict.setdefault("deletions", "")
    typer.echo("Load deletions data...")
    cut_df = load_deletions(all_files_dict["deletions"], config=config)

    # Read quality control file
    # Set default to empty string
    all_files_dict.setdefault("quality_controls", "")
    typer.echo("Load quality controls data...")
    qc_df = load_quality_control(all_files_dict["quality_controls"])
    # Concatenate quality_controls data onto deletions data
    if not qc_df.empty:
        cut_df = pd.concat([cut_df, qc_df]).reset_index(drop=True)

    if all_observations is None:
        # Load travel times data
        typer.echo("Load travel times...")
        transponder_ids = [t.pxp_id for t in transponders]
        all_travel_times = load_travel_times(
            files=all_files_dict["travel_times"], transponder_ids=transponder_ids
        )

        if all_files_dict.get("roll_pitch_heading"):
            # Load roll-pitch-heading data
            typer.echo("Load roll-pitch-heading data...")
            rph_data = load_roll_pitch_heading(
                files=all_files_dict["roll_pitch_heading"]
            )

        # Cleaning travel times
        typer.echo("Cleaning travel times data...")
        typer.echo(f"{all_travel_times=}")
        filtered_travel_times = filter_tt(all_travel_times, cut_df)
        cleaned_travel_times = clean_tt(
            filtered_travel_times,
            transponder_ids,
            config.solver.travel_times_correction,
            config.solver.transducer_delay_time,
        )
        typer.echo(
            f"clean_tt:\n{filtered_travel_times}\n{all_files_dict['travel_times']}\n"
            f"{transponder_ids}\n{config.solver.travel_times_correction}\n"
            f"{config.solver.transducer_delay_time}"
        )

        # Load gps solutions data
        typer.echo("Load GPS data...")
        all_gps_solutions = load_gps_solutions(all_files_dict["gps_solution"])

        typer.echo("Cross referencing transmit, reply, and gps solutions...")

        atd_offsets = get_atd_offsets(config)

        # Parse transmit times
        transmit_times = get_transmit_times(
            cleaned_travel_times,
            all_gps_solutions,
            rph_data,
            config.solver.gps_sigma_limit,
            atd_offsets,
            config.solver.array_center,
        )

        # Parse reply times
        reply_times = get_reply_times(
            cleaned_travel_times,
            all_gps_solutions,
            rph_data,
            config.solver.gps_sigma_limit,
            transponder_ids,
            atd_offsets,
            config.solver.array_center,
        )

        # Merge times
        all_observations = pd.merge(
            transmit_times, reply_times, on=constants.DATA_SPEC.tx_time
        ).reset_index(
            drop=True
        )  # Reset index ensures that it is sequential
    else:
        all_observations = filter_tt(
            all_observations, cut_df, constants.DATA_SPEC.tx_time
        )
    return all_observations


def main(
    config: Configuration,
    all_files_dict: Dict[str, Any],
    all_observations: Optional[pd.DataFrame] = None,
    extract_process_dataset: bool = False,
    outlier_threshold: float = constants.DATA_OUTLIER_THRESHOLD,
) -> Tuple[
    List[float],
    Dict[str, Any],
    Union[pd.DataFrame, None],
    Union[pd.DataFrame, None],
    Union[xr.Dataset, None],
    Union[pd.DataFrame, None],
]:
    """
    The main function that performs the full pre-processing

    Parameters
    ----------
    config : Configuration
        The configuration object
    all_files_dict : Dict[str, Any]
        A dictionary of file paths for the input data
    extract_process_dataset : bool, default False
        A flag to extract the process data as a netCDF file

    Returns
    -------
    all_epochs : List[float]
        A list of all the epoch values
    process_data : Dict[str, Any]
        The full processing data results
    resdf : Union[pd.DataFrame, None]]
        Extracted latest residuals as dataframe, by default None
    dist_center_df : Union[pd.DataFrame, None]
        Extracted distance from center as dataframe, by default None
    process_ds : Union[xr.Dataset, None]
        Extracted process results as xarray dataset, by default None
    outliers_df : Union[pd.DataFrame, None]
        Extracted residual outliers as dataframe, by default None
    """
    all_observations = load_data(all_files_dict, config, all_observations)

    # Extracts distance from center
    dist_center_df = extract_distance_from_center(all_observations, config)
    typer.echo("Filtering out data outside of distance limit...")
    # Extract distance limit
    distance_limit = config.solver.distance_limit

    # Extract the rows of observations with distances beyond the limit
    filtered_rows = dist_center_df[
        dist_center_df[constants.GPS_DISTANCE] > distance_limit
    ][constants.DATA_SPEC.tx_time]

    # Filter out data based on the filtered rows and reset index
    all_observations = all_observations[
        ~all_observations[constants.DATA_SPEC.tx_time].isin(filtered_rows)
    ].reset_index(drop=True)

    all_epochs = all_observations[constants.DATA_SPEC.tx_time].unique()
    process_data, is_converged = prepare_and_solve(all_observations, config)

    if is_converged:
        _print_final_stats(config.solver.transponders, process_data)

    # Extracts latest residuals when specified
    resdf = extract_latest_residuals(config, all_epochs, process_data)

    # Get data outside of the residual limit
    truthy_df = (
        resdf[[t.pxp_id for t in config.solver.transponders]].apply(np.abs)
        > config.solver.residual_limit
    )
    truthy_series = truthy_df.apply(np.any, axis=1)
    outliers_df = resdf[truthy_series]

    # Print out the number of outliers detected
    n_outliers = len(outliers_df)
    percent_outliers = np.round((n_outliers / all_epochs.size) * 100.0, 2)
    message = f"There are {n_outliers} outliers found during this run.\n"
    if n_outliers > 0:
        message += f"This is {percent_outliers}% of the total number of data points.\n"
        message += "Please re-run the program again to remove these outliers.\n"
        if percent_outliers > outlier_threshold:
            raise RuntimeError(
                f"The number of outliers ({percent_outliers}%) is greater than the threshold of "
                f"{outlier_threshold}%. Please check your residual limit"
            )

    typer.echo(message)

    # Extracts process dataset when specified
    process_ds = None
    if extract_process_dataset:
        process_ds = xr.concat(
            [_create_process_dataset(v, k, config) for k, v in process_data.items()],
            dim="iteration",
        )

        # Get the median time of residuals
        median_time = AstroTime(
            np.median(resdf[constants.TIME_J2000].values), format="unix_j2000"
        )
        median_time_str = median_time.strftime("%Y-%m-%dT%H:00:00")

        # Set the median time to the process dataset
        process_ds.attrs["session_time"] = median_time_str
    return all_epochs, process_data, resdf, dist_center_df, process_ds, outliers_df


def run_gnatss(
    config_yaml: str,
    distance_limit: Optional[float] = None,
    residual_limit: Optional[float] = None,
    outlier_threshold: Optional[float] = None,
    from_cache: bool = False,
    return_raw: bool = False,
    remove_outliers: bool = False,
    extract_process_dataset: bool = True,
    extract_dist_center: bool = True,
    qc: bool = True,
    skip_posfilter: bool = False,
    skip_solver: bool = False,
):
    typer.echo("Starting GNATSS ...")
    if from_cache:
        typer.echo(
            "Flag `from_cache` is set. Skipping data loading and processing for posfilter step."
        )
    config, data_dict = data_loading(
        config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        outlier_threshold=outlier_threshold,
        from_cache=from_cache,
        remove_outliers=remove_outliers,
        skip_posfilter=skip_posfilter,
        skip_solver=skip_solver,
    )
    config, data_dict = preprocess_data(config, data_dict)

    if config.posfilter and not from_cache and not skip_posfilter:
        data_dict = run_posfilter(config, data_dict)

    if config.solver and not skip_solver:
        data_dict = run_solver(config, data_dict, return_raw=return_raw)
        # Write out to residuals.csv file
        to_file(config, data_dict, "residuals", CSVOutput.residuals)

        # Write out to outliers.csv file
        to_file(config, data_dict, "outliers", CSVOutput.outliers)

        # Write out distance from center to dist_center.csv file
        if extract_dist_center:
            to_file(config, data_dict, "distance_from_center", CSVOutput.dist_center)

        # Write out to process_dataset.nc file
        if extract_process_dataset:
            to_file(
                config,
                data_dict,
                "process_dataset",
                "process_dataset.nc",
                file_format="netcdf",
            )
        # Export QC plots
        if qc:
            export_qc_plots(config, data_dict)

    typer.echo("Finished GNATSS.")
    return config, data_dict
