import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import typer
import xarray as xr
from pymap3d import ecef2enu, ecef2geodetic, geodetic2ecef

from . import constants
from .configs.main import Configuration
from .configs.solver import SolverTransponder
from .harmonic_mean import sv_harmonic_mean
from .loaders import (
    load_deletions,
    load_gps_solutions,
    load_quality_control,
    load_sound_speed,
    load_travel_times,
)
from .ops.data import get_data_inputs
from .ops.solve import perform_solve
from .ops.utils import _prep_col_names
from .ops.validate import check_sig3d, check_solutions
from .utilities.geo import _get_rotation_matrix
from .utilities.io import _get_filesystem
from .utilities.time import AstroTime


def gather_files(
    config: Configuration, proc: Literal["solver", "posfilter"] = "solver"
) -> Dict[str, List[str]]:
    """Gather file paths for the various dataset files

    Parameters
    ----------
    config : Configuration
        A configuration object

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the various datasets file paths
    """
    all_files_dict = {}
    # Check for process type first
    if not hasattr(config, proc):
        raise AttributeError(f"Unknown process type: {proc}")

    proc_config = getattr(config, proc)
    for k, v in proc_config.input_files.model_dump().items():
        if v:
            path = v.get("path", "")
            typer.echo(f"Gathering {k} at {path}")
            storage_options = v.get("storage_options", {})

            fs = _get_filesystem(path, storage_options)
            if "**" in path:
                all_files = fs.glob(path)
            else:
                all_files = [path]

            all_files_dict.setdefault(k, all_files)
    return all_files_dict


def clean_tt(
    all_travel_times: pd.DataFrame,
    cut_df: pd.DataFrame,
    transponder_ids: List[str],
    travel_times_correction: float,
    transducer_delay_time: float,
) -> pd.DataFrame:
    """
    Clean travel times using deletions data

    Parameters
    ----------
    all_travel_times : pd.DataFrame
        The original travel times data
    cut_df : pd.DataFrame
        The deletions data to be removed
    transponder_ids : List[str]
        A list of the transponder ids that matches the order
        with all_travel_times data
    travel_times_correction : float
        Correction to times in travel times (secs.)
    transducer_delay_time : float
        Transducer Delay Time - delay at surface transducer (secs).

    Returns
    -------
    pd.DataFrame
        The cleaned travel times data

    Notes
    -----
    Original implementation by @SquirrelKnight
    """

    if len(cut_df.index) > 0:
        # Only cut the data with deletions file if there are data
        cut_ids_all = []
        for _, cut in cut_df.iterrows():
            cut_ids = all_travel_times[
                (all_travel_times[constants.TT_TIME] >= cut.starttime)
                & (all_travel_times[constants.TT_TIME] <= cut.endtime)
            ].index.values
            cut_ids_all = cut_ids_all + cut_ids.tolist()
        cut_ids_all = list(set(cut_ids_all))
        all_travel_times = all_travel_times.loc[
            ~all_travel_times.index.isin(cut_ids_all)
        ]

    # TODO: Store junk travel times? These are travel times with 0 values
    # _ = all_travel_times.loc[
    #     all_travel_times.where(all_travel_times[transponder_ids] == 0)
    #     .dropna(how="all")
    #     .index
    # ]

    # Get cleaned travel times
    # This is anything that has 0 reply time
    cleaned_travel_times = all_travel_times.loc[
        all_travel_times[transponder_ids]
        .where(all_travel_times[transponder_ids] != 0)
        .dropna()
        .index
    ]

    # Apply travel time correction
    cleaned_travel_times.loc[:, constants.TT_TIME] = (
        cleaned_travel_times[constants.TT_TIME]
        + travel_times_correction
        + transducer_delay_time
    )

    return cleaned_travel_times


def get_transmit_times(
    cleaned_travel_times: pd.DataFrame,
    input_df: pd.DataFrame,
    input_time_col: str,
    is_gps: bool = False,
    gps_sigma_limit: Optional[float] = None,
) -> pd.DataFrame:
    """
    Merges cleaned transmit times with gps solutions into one
    dataframe and check for 3d std deviation

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    input_df : pd.DataFrame
        The full data to extract transmit times from
    input_time_col : str
        The column name for the time column in ``input_df``
    is_gps : bool, default False
        Flag to indicate if the input data is gps data
    gps_sigma_limit : float, optional
        Maximum positional sigma allowed to use GPS positions.
        This will be used to check the 3d standard deviation,
        and only used when ``is_gps`` is True

    Returns
    -------
    pd.DataFrame
        The transmit times data with gps solutions included
    """
    # Merge with gps solutions
    transmit_times = pd.merge(
        cleaned_travel_times[[constants.TT_TIME]],
        input_df,
        left_on=constants.TT_TIME,
        right_on=input_time_col,
    )

    if is_gps:
        # Compute and check 3d standard deviation
        transmit_times = check_sig3d(data=transmit_times, gps_sigma_limit=gps_sigma_limit)

    # Adds a 0 to column names for transmit values
    transmit_times.columns = [
        f"{col}0" if col != constants.TT_TIME else constants.garpos.ST
        for col in transmit_times.columns
    ]

    return transmit_times


def get_reply_times(
    cleaned_travel_times: pd.DataFrame,
    input_df: pd.DataFrame,
    input_time_col: str,
    transponder_ids: List[str],
    is_gps: bool = False,
    gps_sigma_limit: Optional[float] = None,
):
    """
    Merges cleaned reply times with gps solutions into one
    dataframe and check for 3d std deviation

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    input_df : pd.DataFrame
        The full data to extract reply times from
    input_time_col : str
        The column name for the time column in ``input_df``
    transponder_ids : List[str]
        A list of the transponder ids that matches the order
        with ``cleaned_travel_times`` data
    is_gps : bool, default False
        The flag to indicate if the input data is gps data
    gps_sigma_limit : float, optional
        Maximum positional sigma allowed to use GPS positions.
        This will be used to check the 3d standard deviation,
        and only used when ``is_gps`` is True

    Returns
    -------
    pd.DataFrame
        The reply times data with gps solutions included
    """
    reply_times = cleaned_travel_times[transponder_ids]
    reply_times[constants.garpos.ST] = cleaned_travel_times[constants.TT_TIME]

    # Pivot the table by stacking
    reply_times = reply_times.set_index(constants.garpos.ST).stack()
    reply_times = reply_times.rename(constants.garpos.TT)
    reply_times.index = reply_times.index.rename(
        [constants.garpos.ST, constants.garpos.MT]
    )
    reply_times = reply_times.to_frame().reset_index()
    # Set RT
    reply_times[constants.garpos.RT] = reply_times.apply(
        lambda row: row[constants.garpos.ST] + row[constants.garpos.TT], axis=1
    )
    # Merge with gps solutions
    reply_times = pd.merge(
        reply_times,
        input_df,
        left_on=constants.garpos.RT,
        right_on=input_time_col,
    )
    reply_times = reply_times.drop(input_time_col, axis="columns")

    if is_gps:
        # Compute and check 3d standard deviation
        reply_times = check_sig3d(data=reply_times, gps_sigma_limit=gps_sigma_limit)

    # Currently looks for even value counts... check fortran code what to do here?
    time_counts = reply_times[constants.garpos.ST].value_counts()
    reply_times = reply_times[
        reply_times[constants.garpos.ST].isin(
            time_counts[time_counts == len(transponder_ids)].index
        )
    ]

    # Adds a 1 to column names for reply values
    reply_times.columns = [
        f"{col}1"
        if col
        not in [
            constants.garpos.ST,
            constants.garpos.MT,
            constants.garpos.RT,
            constants.garpos.TT,
        ]
        else col
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


def prepare_and_solve(
    all_observations: pd.DataFrame, config: Configuration, max_iter: int = 6
) -> Tuple[Dict[int, Any], bool]:
    """
    Prepare data inputs and perform solving algorithm

    Parameters
    ----------
    all_observations : pd.DataFrame
        The whole dataset that includes,
        transmit, reply, and gps solutions data
    config : Configuration
        The configuration object

    Returns
    -------
    Dict[int, Any]
        The process dictionary that contains stats and data results,
        for all of the iterations
    """
    transponders = config.solver.transponders
    # convert orthonomal heights of PXPs into ellipsoidal heights and convert to x,y,z
    transponders_xyz = None
    if transponders_xyz is None:
        transponders_xyz = np.array(
            [
                geodetic2ecef(t.lat, t.lon, t.height + config.solver.geoid_undulation)
                for t in transponders
            ]
        )
    transponders_mean_sv = np.array([t.sv_mean for t in transponders])
    transponders_delay = np.array([t.internal_delay for t in transponders])

    # Get travel times variance
    travel_times_variance = config.solver.travel_times_variance

    # Store original xyz
    original_positions = transponders_xyz.copy()

    typer.echo("Preparing data inputs...")
    data_inputs = get_data_inputs(all_observations)

    typer.echo("Perform solve...")
    is_converged = False
    n_iter = 0
    num_transponders = len(transponders)
    process_dict = {}
    num_data = len(all_observations)
    typer.echo(f"--- {len(data_inputs)} epochs, {num_data} measurements ---")
    while not is_converged:
