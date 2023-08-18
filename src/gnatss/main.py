from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import typer
from pymap3d import ecef2enu, ecef2geodetic, geodetic2ecef

from . import constants
from .configs.main import Configuration
from .harmonic_mean import sv_harmonic_mean
from .loaders import (
    load_deletions,
    load_gps_solutions,
    load_sound_speed,
    load_travel_times,
)
from .ops.data import get_data_inputs
from .ops.solve import perform_solve
from .ops.utils import _prep_col_names
from .ops.validate import check_sig3d, check_solutions
from .utilities.geo import _get_rotation_matrix
from .utilities.io import _get_filesystem


def gather_files(config: Configuration) -> Dict[str, Any]:
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
    all_gps_solutions: pd.DataFrame,
    gps_sigma_limit: float,
) -> pd.DataFrame:
    """
    Merges cleaned transmit times with gps solutions into one
    dataframe and check for 3d std deviation

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    all_gps_solutions : pd.DataFrame
        The full gps solutions data
    gps_sigma_limit : float
        Maximum positional sigma allowed to use GPS positions

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
    all_gps_solutions: pd.DataFrame,
    gps_sigma_limit: float,
    transponder_ids: List[str],
):
    """
    Merges cleaned reply times with gps solutions into one
    dataframe and check for 3d std deviation

    Parameters
    ----------
    cleaned_travel_times : pd.DataFrame
        The full cleaned travel times data
    all_gps_solutions : pd.DataFrame
        The full gps solutions data
    gps_sigma_limit : float
        Maximum positional sigma allowed to use GPS positions
    transponder_ids : List[str]
        A list of the transponder ids that matches the order
        with ``cleaned_travel_times`` data

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
        all_gps_solutions,
        left_on=constants.garpos.RT,
        right_on=constants.GPS_TIME,
    )
    reply_times = reply_times.drop(constants.GPS_TIME, axis="columns")

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


def prepare_and_solve(
    all_observations: pd.DataFrame, config: Configuration
) -> Dict[int, Any]:
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
        # TODO: Add max converge attempt failure
        # if n_iter > max_iter:
        #     raise RuntimeError("Exceeds the allowed number of attempt, please adjust your data.")

        # Increase iter num
        n_iter += 1

        # Keep track of process
        process_dict[n_iter] = {"transponders_xyz": transponders_xyz}

        # Perform solving
        all_results = perform_solve(
            data_inputs,
            transponders_mean_sv,
            transponders_xyz,
            transponders_delay,
            travel_times_variance,
        )

        is_converged, transponders_xyz, data = check_solutions(
            all_results, transponders_xyz
        )

        process_dict[n_iter]["data"] = data

        # Compute one way travel time residual in centimeter
        # This uses a constant assume sound speed of 1500 m/s
        # since this is only used for quality control.
        process_dict[n_iter]["rescm"] = (100 * 1500 * np.array(data["address"])) / 2

        # Print out some stats below

        # This assumes that all data is ADSIG > 0
        RMSRES = np.sum(np.array(data["address"]) ** 2)
        RMSRESCM = np.sum(
            ((100 * transponders_mean_sv) * np.array(data["address"])) ** 2
        )
        ERRFAC = np.sum((np.array(data["address"]) / np.array(data["adsig"])) ** 2)

        RMSRES = np.sqrt(RMSRES / num_data)
        RMSRESCM = np.sqrt(RMSRESCM / num_data)
        ERRFAC = np.sqrt(ERRFAC / (num_data - (3 * num_transponders)))

        typer.echo(
            (
                f"After iteration: {n_iter}, "
                f"rms residual = {np.round(RMSRESCM, 2)} cm, "
                f"error factor = {np.round(ERRFAC, 3)}"
            )
        )

        for idx, tp in enumerate(transponders):
            pxp_id = tp.pxp_id
            SIGPX = np.array_split(data["sigpx"], num_transponders)
            DELP = np.array_split(data["delp"], num_transponders)
            dX, dY, dZ = DELP[idx]
            sigX, sigY, sigZ = SIGPX[idx]
            typer.echo(pxp_id)
            typer.echo(
                (
                    f"D_x = {np.format_float_scientific(dX, 6)} m, "
                    f"Sigma(x) = {np.format_float_scientific(sigX, 6)} m"
                )
            )
            typer.echo(
                (
                    f"D_y = {np.format_float_scientific(dY, 6)} m, "
                    f"Sigma(y) = {np.format_float_scientific(sigY, 6)} m"
                )
            )
            typer.echo(
                (
                    f"D_z = {np.format_float_scientific(dZ, 6)} m, "
                    f"Sigma(z) = {np.format_float_scientific(sigZ, 6)} m"
                )
            )

        if is_converged:
            typer.echo()
            typer.echo("---- FINAL SOLUTION ----")
            for idx, tp in enumerate(transponders):
                typer.echo(pxp_id)
                x, y, z = transponders_xyz[idx]
                original_xyz = original_positions[idx]
                original_lla = ecef2geodetic(*original_xyz)

                SIGPX = np.array_split(data["sigpx"], num_transponders)
                sigX, sigY, sigZ = SIGPX[idx]
                lat, lon, alt = ecef2geodetic(x, y, z)

                # Compute enu
                e, n, u = ecef2enu(x, y, z, *original_lla)

                # Find enu covariance
                latr, lonr = np.radians([lat, lon])
                R = _get_rotation_matrix(latr, lonr, False)
                covpx = np.array(
                    [arr[:3] for arr in data["covpx"][idx * 3 : 3 * (idx + 1)]]  # noqa
                )
                covpe = R.T @ covpx @ R
                # Retrieve diagonal and change negative values to 0
                diag = covpe.diagonal().copy()
                diag[diag < 0] = 0

                sigE, sigN, sigU = np.sqrt(diag)

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
            return process_dict


def load_data(all_files_dict: Dict[str, Any], config: Configuration) -> pd.DataFrame:
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
    typer.echo("Load deletions data...")
    cut_df = load_deletions(all_files_dict["deletions"])

    # Load travel times data
    typer.echo("Load travel times...")
    transponder_ids = [t.pxp_id for t in transponders]
    all_travel_times = load_travel_times(
        files=all_files_dict["travel_times"], transponder_ids=transponder_ids
    )

    # Cleaning travel times
    typer.echo("Cleaning travel times data...")
    cleaned_travel_times = clean_tt(
        all_travel_times,
        cut_df,
        transponder_ids,
        config.solver.travel_times_correction,
        config.solver.transducer_delay_time,
    )

    # Load gps solutions data
    typer.echo("Load GPS data...")
    all_gps_solutions = load_gps_solutions(all_files_dict["gps_solution"])

    typer.echo("Cross referencing transmit, reply, and gps solutions...")
    # Parse transmit times
    transmit_times = get_transmit_times(
        cleaned_travel_times, all_gps_solutions, config.solver.gps_sigma_limit
    )
    # Parse reply times
    reply_times = get_reply_times(
        cleaned_travel_times,
        all_gps_solutions,
        config.solver.gps_sigma_limit,
        transponder_ids,
    )

    # Merge times
    all_observations = pd.merge(
        transmit_times, reply_times, on=constants.garpos.ST
    ).reset_index(
        drop=True
    )  # Reset index ensures that it is sequential

    return all_observations


def extract_distance_from_center(
    all_observations: pd.DataFrame, config: Configuration
) -> pd.DataFrame:
    """Extracts and calculates the distance from the array center

    Parameters
    ----------
    all_observations : pd.DataFrame
        The full dataset for computation
    config : Configuration
        The configuration object

    Returns
    -------
    pd.DataFrame
        The final dataframe for distance from center
    """

    def _compute_enu(coords, array_center):
        return ecef2enu(
            *coords, array_center.lat, array_center.lon, array_center.alt, deg=True
        )

    # Set up transmit columns
    transmit_cols = _prep_col_names(constants.GPS_GEOCENTRIC, True)

    # Since we're only working with transmit,
    # we can just group by transmit time to avoid repetition.
    # This extracts transmit data coords only
    transmit_obs = (
        all_observations[[constants.garpos.ST] + transmit_cols]
        .groupby(constants.garpos.ST)
        .first()
        .reset_index()
    )

    # Get geocentric x,y,z for array center
    array_center = config.solver.array_center

    # Extract coordinates only
    transmit_coords = transmit_obs[transmit_cols]
    enu_arrays = np.apply_along_axis(
        _compute_enu, axis=1, arr=transmit_coords, array_center=array_center
    )
    enu_df = pd.DataFrame.from_records(enu_arrays, columns=constants.GPS_LOCAL_TANGENT)
    # Compute azimuth from north to east
    enu_df.loc[:, constants.GPS_AZ] = enu_df.apply(
        lambda row: np.degrees(
            np.arctan2(row[constants.GPS_EAST], row[constants.GPS_NORTH])
        ),
        axis=1,
    )
    # Compute distance from center
    enu_df.loc[:, constants.GPS_DISTANCE] = enu_df.apply(
        lambda row: np.sqrt(
            row[constants.GPS_NORTH] ** 2 + row[constants.GPS_EAST] ** 2
        ),
        axis=1,
    )

    # Merge with equivalent index
    return pd.merge(
        transmit_obs[constants.garpos.ST], enu_df, left_index=True, right_index=True
    )


def extract_latest_residuals(
    config: Configuration, all_epochs: List[float], process_data: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extracts the latest residuals from process data,
    and convert them into a pandas dataframe.

    Parameters
    ----------
    config : Configuration
        The configuration object
    all_epochs : List[float]
        A list of all the epoch values
    process_data : Dict[str, Any]
        The full processing data results

    Returns
    -------
    pd.DataFrame
        The final dataframe for residuals
    """
    from .utilities.time import AstroTime  # noqa

    def to_iso(astro_time):
        return [t.strftime("%Y-%m-%dT%H:%M:%S.%f") for t in astro_time]

    # Convert j2000 seconds time to astro time and then convert to iso
    astro_epochs = np.apply_along_axis(AstroTime, 0, all_epochs, format="unix_j2000")
    iso_epochs = np.apply_along_axis(to_iso, 0, astro_epochs)

    # Get the latest process data
    process_info = process_data[max(process_data.keys())]

    # Retrieve residuals data
    all_residuals_data = []
    for ep, iso, address in zip(all_epochs, iso_epochs, process_info["rescm"]):
        all_residuals_data.append([ep, iso] + list(address))

    return pd.DataFrame(
        all_residuals_data,
        columns=[constants.TIME_J2000, constants.TIME_ISO]
        + [t.pxp_id for t in config.solver.transponders],
    )


def main(
    config: Configuration,
    all_files_dict: Dict[str, Any],
    extract_res: bool = False,
    extract_dist_center: bool = False,
) -> Tuple[
    List[float], Dict[str, Any], Union[pd.DataFrame, None], Union[pd.DataFrame, None]
]:
    """
    The main function that performs the full pre-processing

    Parameters
    ----------
    config : Configuration
        The configuration object
    all_files_dict : Dict[str, Any]
        A dictionary of file paths for the input data
    extract_res : bool, optional
        A flag to extract latest residual data as dataframe,
        by default False
    extract_dist_center : bool, optional
        A flag to extract distance from center data as dataframe,
        by default False

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
    """
    all_observations = load_data(all_files_dict, config)

    # Extracts distance from center when specified
    dist_center_df = None
    if extract_dist_center:
        dist_center_df = extract_distance_from_center(all_observations, config)

    all_epochs = all_observations[constants.garpos.ST].unique()
    process_data = prepare_and_solve(all_observations, config)

    # Extracts latest residuals when specified
    resdf = None
    if extract_res:
        resdf = extract_latest_residuals(config, all_epochs, process_data)
    return all_epochs, process_data, resdf, dist_center_df
