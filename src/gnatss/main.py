from typing import Any, Dict

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
from .ops.validate import check_sig3d, check_solutions
from .utilities.geo import _get_rotation_matrix
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


def clean_tt(
    all_travel_times,
    cut_df,
    transponder_ids,
    travel_times_correction,
    transducer_delay_time,
):
    """
    Clean travel times

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


def get_transmit_times(cleaned_travel_times, all_gps_solutions, gps_sigma_limit):
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
    cleaned_travel_times, all_gps_solutions, gps_sigma_limit, transponder_ids
):
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


def prepare_and_solve(all_observations, config):
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
            return process_dict
        typer.echo()


def load_data(all_files_dict, config):
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
    all_observations = pd.merge(transmit_times, reply_times, on=constants.garpos.ST)

    # TODO: Get lat lon alt and enu, and azimuth
    # Calculate transmit azimuth angle in degrees
    # transmit_azimuth = np.degrees(
    #   np.arctan2(transmit_location[GPS_EAST], transmit_location[GPS_NORTH])
    # )

    # Get geocentric x,y,z for array center
    # array_center = config.solver.array_center
    # array_center_xyz = np.array(
    #     geodetic2ecef(array_center.lat, array_center.lon, array_center.alt)
    # )
    return all_observations


def main(config: Configuration, all_files_dict: Dict[str, Any]):
    all_observations = load_data(all_files_dict, config)
    all_epochs = all_observations[constants.garpos.ST].unique()
    return all_epochs, prepare_and_solve(all_observations, config)
