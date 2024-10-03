from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import typer
import xarray as xr
from pymap3d import ecef2enu, ecef2geodetic, geodetic2ecef

from .. import constants
from ..configs.main import Configuration
from ..configs.solver import SolverTransponder
from ..ops.data import filter_tt, get_data_inputs, prefilter_replies
from ..ops.validate import check_solutions
from ..utilities.geo import _get_rotation_matrix
from ..utilities.time import AstroTime
from .solve import perform_solve


def _print_detected_outliers(outliers_df, outlier_threshold, all_epochs, residual_limit) -> None:
    # Print out the number of outliers detected
    n_outliers = len(outliers_df)
    percent_outliers = np.round((n_outliers / all_epochs.size) * 100.0, 2)
    message = f"There are {n_outliers} outliers found during this run.\n"
    if n_outliers > 0:
        message += f"This is {percent_outliers}% of the total number of data points.\n"
        message += "Please re-run the program again with `--remove-outliers` flag to remove these outliers.\n"
        if percent_outliers > outlier_threshold:
            msg = (
                f"The number of outliers ({percent_outliers}%) is greater than the threshold of "
                f"{outlier_threshold}%. Please modify your current residual limit of {residual_limit}."
            )
            raise RuntimeError(msg)
    typer.echo(message)


def _print_final_stats(transponders: list[SolverTransponder], process_data: dict[str, Any]):
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
            f"x = {np.round(x, 4)} +/- {np.format_float_scientific(sigX, 6)} m "
            f"del_e = {np.round(e, 4)} +/- {np.format_float_scientific(sigE, 6)} m"
        )
        typer.echo(
            f"y = {np.round(y, 4)} +/- {np.format_float_scientific(sigY, 6)} m "
            f"del_n = {np.round(n, 4)} +/- {np.format_float_scientific(sigN, 6)} m"
        )
        typer.echo(
            f"z = {np.round(z, 4)} +/- {np.format_float_scientific(sigZ, 6)} m "
            f"del_u = {np.round(u, 4)} +/- {np.format_float_scientific(sigU, 6)} m"
        )
        typer.echo(f"Lat. = {lat} deg, Long. = {lon}, Hgt.msl = {alt} m")
    typer.echo("------------------------")
    typer.echo()


def _get_latest_process(process_data: dict[str, Any]) -> dict[str, Any]:
    """Get the latest process data"""
    return process_data[max(process_data.keys())]


def _create_process_dataset(
    proc_d: dict[str, Any], n_iter: int, config: Configuration
) -> xr.Dataset:
    """Creates a process dataset from the process dictionary

    Parameters
    ----------
    proc_d : dict[str, Any]
        Process dictionary
    n_iter : int
        Iteration number
    config : Configuration
        The configuration object

    Returns
    -------
    xr.Dataset
        The resulting process dataset
    """
    transponders = config.solver.transponders
    num_transponders = len(transponders)
    transponders_ids = [tp.pxp_id for tp in transponders]

    return xr.Dataset(
        data_vars={
            "transponders_xyz": (
                ("transponder", "coords"),
                proc_d["transponders_xyz"],
                {"units": "meters", "long_name": "Transponder ECEF location"},
            ),
            "delta_xyz": (
                ("transponder", "coords"),
                np.array(np.array_split(proc_d["data"]["delp"], num_transponders)),
                {
                    "units": "meters",
                    "long_name": "Transponder location differences from apriori",
                },
            ),
            "sigma_xyz": (
                ("transponder", "coords"),
                np.array(np.array_split(proc_d["data"]["sigpx"], num_transponders)),
                {
                    "units": "meters",
                    "long_name": "Transponder location differences standard deviation",
                },
            ),
            "rms_residual": (
                ("iteration"),
                [proc_d["rmsrescm"]],
                {
                    "units": "centimeters",
                    "long_name": "Root mean square (RMS) of residuals",
                },
            ),
            "error_factor": (
                ("iteration"),
                [proc_d["errfac"]],
                {"units": "unitless", "long_name": "Error factor value"},
            ),
            "delta_enu": (
                ("transponder", "coords"),
                proc_d["enu"],
                {
                    "units": "meters",
                    "long_name": "Transponder ENU differences from apriori",
                },
            ),
            "sigma_enu": (
                ("transponder", "coords"),
                proc_d["sig_enu"],
                {
                    "units": "meters",
                    "long_name": "Transponder ENU differences standard deviation",
                },
            ),
            "transponders_lla": (
                ("transponder", "coords"),
                proc_d["transponders_lla"],
                {"units": "degrees", "long_name": "Transponder Geodetic location"},
            ),
        },
        coords={
            "transponder": (
                ("transponder"),
                transponders_ids,
                {"long_name": "Transponder id"},
            ),
            "coords": (("coords"), ["x", "y", "z"], {"long_name": "Coordinate label"}),
            "iteration": (("iteration"), [n_iter], {"long_name": "Iteration number"}),
        },
    )


def generate_process_xr_dataset(process_data, resdf, config) -> xr.Dataset:
    # Extracts process dataset when specified
    process_ds = None
    process_ds = xr.concat(
        [_create_process_dataset(v, k, config) for k, v in process_data.items()],
        dim="iteration",
    )

    # Get the median time of residuals
    median_time = AstroTime(np.median(resdf[constants.TIME_J2000].values), format="unix_j2000")
    median_time_str = median_time.strftime("%Y-%m-%dT%H:00:00")

    # Set the median time to the process dataset
    process_ds.attrs["session_time"] = median_time_str
    return process_ds


def get_residual_outliers(config, resdf):
    # Get data that fall outside various quality metrics

    # Get data that fall outside the residual limit
    truthy_limit = get_residuals_by_limit(config, resdf)

    # Get data outside of the residual range limit
    # (Epoch-wise Max - Min)
    truthy_range = get_residuals_by_range_limit(config, resdf)

    return resdf[truthy_limit | truthy_range]


def get_residuals_by_limit(config, resdf):
    # Get data outside of the residual limit
    truthy_limit = (
        resdf[[t.pxp_id for t in config.solver.transponders]].apply(np.abs)
        > config.solver.residual_limit
    )
    return truthy_limit.apply(np.any, axis=1)


def get_residuals_by_range_limit(config, resdf):
    # Get data outside of the residual range limit
    # (Epoch-wise Max - Min)
    return (
        resdf[[t.pxp_id for t in config.solver.transponders]].max(axis=1)
        - resdf[[t.pxp_id for t in config.solver.transponders]].min(axis=1)
        > config.solver.residual_range_limit
    )


def filter_by_distance_limit(all_observations, config):
    # Extracts distance from center
    dist_center_df = extract_distance_from_center(all_observations, config)
    typer.echo("Filtering out data outside of distance limit...")
    # Extract distance limit
    distance_limit = config.solver.distance_limit

    # Extract the rows of observations with distances beyond the limit
    filtered_rows = dist_center_df[dist_center_df[constants.GPS_DISTANCE] > distance_limit][
        constants.DATA_SPEC.tx_time
    ]

    # Filter out data based on the filtered rows and reset index
    all_observations = all_observations[
        ~all_observations[constants.DATA_SPEC.tx_time].isin(filtered_rows)
    ].reset_index(drop=True)
    return all_observations, dist_center_df


def filter_deletions_and_qc(all_observations, data_dict):
    cut_df = data_dict.get("deletions")
    qc_df = data_dict.get("quality_controls")
    if cut_df is not None:
        if qc_df is not None and not qc_df.empty:
            # Concatenate quality_controls data onto deletions data
            cut_df = pd.concat([cut_df, qc_df]).reset_index(drop=True)
        return filter_tt(all_observations, cut_df, constants.DATA_SPEC.tx_time)
    return all_observations


def get_all_epochs(all_observations):
    return all_observations[constants.DATA_SPEC.tx_time].unique()


def extract_latest_residuals(
    config: Configuration, all_epochs: list[float], process_data: dict[str, Any]
) -> pd.DataFrame:
    """
    Extracts the latest residuals from process data,
    and convert them into a pandas dataframe.

    Parameters
    ----------
    config : Configuration
        The configuration object
    all_epochs : list[float]
        A list of all the epoch values
    process_data : dict[str, Any]
        The full processing data results

    Returns
    -------
    pd.DataFrame
        The final dataframe for residuals
    """

    def to_iso(astro_time):
        return [t.strftime("%Y-%m-%dT%H:%M:%S.%f") for t in astro_time]

    # Convert j2000 seconds time to astro time and then convert to iso
    astro_epochs = np.apply_along_axis(AstroTime, 0, all_epochs, format="unix_j2000")
    iso_epochs = np.apply_along_axis(to_iso, 0, astro_epochs)

    # Get the latest process data
    process_info = _get_latest_process(process_data)

    # Retrieve residuals data
    all_residuals_data = []
    for ep, iso, address in zip(all_epochs, iso_epochs, process_info["rescm"], strict=False):
        all_residuals_data.append([ep, iso, *list(address)])

    return pd.DataFrame(
        all_residuals_data,
        columns=[constants.TIME_J2000, constants.TIME_ISO]
        + [t.pxp_id for t in config.solver.transponders],
    )


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
        return ecef2enu(*coords, array_center.lat, array_center.lon, array_center.alt, deg=True)

    # Set up transmit columns
    transmit_cols = constants.DATA_SPEC.transducer_tx_fields.keys()

    # Since we're only working with transmit,
    # we can just group by transmit time to avoid repetition.
    # This extracts transmit data coords only
    transmit_obs = (
        all_observations[[constants.DATA_SPEC.tx_time, *transmit_cols]]
        .groupby(constants.DATA_SPEC.tx_time)
        .first()
        .reset_index()
    )

    # Get geocentric x,y,z for array center
    array_center = config.array_center

    # Extract coordinates only
    transmit_coords = transmit_obs[transmit_cols]
    enu_arrays = np.apply_along_axis(
        _compute_enu, axis=1, arr=transmit_coords, array_center=array_center
    )
    enu_df = pd.DataFrame.from_records(enu_arrays, columns=constants.GPS_LOCAL_TANGENT)
    # Compute azimuth from north to east
    enu_df.loc[:, constants.GPS_AZ] = enu_df.apply(
        lambda row: np.degrees(np.arctan2(row[constants.GPS_EAST], row[constants.GPS_NORTH])),
        axis=1,
    )
    # Compute distance from center
    enu_df.loc[:, constants.GPS_DISTANCE] = enu_df.apply(
        lambda row: np.sqrt(row[constants.GPS_NORTH] ** 2 + row[constants.GPS_EAST] ** 2),
        axis=1,
    )

    # Merge with equivalent index
    return pd.merge(  # noqa: PD015
        transmit_obs[constants.DATA_SPEC.tx_time],
        enu_df,
        left_index=True,
        right_index=True,
    )


def prepare_and_solve(
    all_observations: pd.DataFrame,
    config: Configuration,
    max_iter: int = 6,
    twtt_model: Literal["simple_twtt"] = "simple_twtt",
) -> tuple[dict[int, Any], bool]:
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
    dict[int, Any]
        The process dictionary that contains stats and data results,
        for all of the iterations
    """
    transponders = config.transponders
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
    travel_times_variance = config.travel_times_variance

    # Store original xyz
    original_positions = transponders_xyz.copy()

    # Store number of transponders
    num_transponders = len(transponders)

    typer.echo("Preparing data inputs...")
    typer.echo(f"Pre-filtering data with fewer than {num_transponders} replies...")
    reduced_observations = prefilter_replies(all_observations, num_transponders)
    data_inputs = get_data_inputs(reduced_observations)

    typer.echo("Perform solve...")
    is_converged = False
    n_iter = 0
    process_dict = {}
    num_data = len(reduced_observations)
    typer.echo(f"--- {len(data_inputs)} epochs, {num_data} measurements ---")
    while not is_converged:
        # Max converge attempt failure
        if n_iter > max_iter:
            msg = "Exceeds the allowed number of attempt, please adjust your data."
            warnings.warn(msg, stacklevel=1)
            break

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
            twtt_model,
        )

        is_converged, transponders_xyz, data = check_solutions(all_results, transponders_xyz)

        process_dict[n_iter]["data"] = data

        # Compute one way travel time residual in centimeter
        # This uses a constant assume sound speed of 1500 m/s
        # since this is only used for quality control.
        process_dict[n_iter]["rescm"] = (100 * 1500 * np.array(data["address"])) / 2

        # Print out some stats below

        # This assumes that all data is ADSIG > 0
        RMSRES = np.sum(np.array(data["address"]) ** 2)
        RMSRESCM = np.sum(((100 * transponders_mean_sv) * np.array(data["address"])) ** 2)
        ERRFAC = np.sum((np.array(data["address"]) / np.array(data["adsig"])) ** 2)

        RMSRES = np.sqrt(RMSRES / num_data)
        RMSRESCM = np.sqrt(RMSRESCM / num_data)
        ERRFAC = np.sqrt(ERRFAC / (num_data - (3 * num_transponders)))

        process_dict[n_iter]["rmsrescm"] = RMSRESCM
        process_dict[n_iter]["errfac"] = ERRFAC
        typer.echo(
            f"After iteration: {n_iter}, "
            f"rms residual = {np.round(RMSRESCM, 2)} cm, "
            f"error factor = {np.round(ERRFAC, 3)}"
        )

        enu_arr = []
        sig_enu = []
        lat_lon = []
        for idx, tp in enumerate(transponders):
            pxp_id = tp.pxp_id
            # Get xyz for a transponder
            x, y, z = transponders_xyz[idx]

            # Get lat lon alt
            lat, lon, alt = ecef2geodetic(x, y, z)
            lat_lon.append([lat, lon, alt - config.solver.geoid_undulation])

            # Retrieve apriori xyz and lat lon alt
            original_xyz = original_positions[idx]
            original_lla = ecef2geodetic(*original_xyz)

            # Compute enu w.r.t apriori lat lon alt
            e, n, u = ecef2enu(x, y, z, *original_lla)
            enu_arr.append([e, n, u])

            # Find enu covariance
            latr, lonr = np.radians([lat, lon])
            R = _get_rotation_matrix(latr, lonr, False)
            covpx = np.array([arr[:3] for arr in data["covpx"][idx * 3 : 3 * (idx + 1)]])
            covpe = R.T @ covpx @ R
            # Retrieve diagonal and change negative values to 0
            diag = covpe.diagonal().copy()
            diag[diag < 0] = 0
            sig_enu.append(np.sqrt(diag))

            # Find location differences and its std dev
            SIGPX = np.array_split(data["sigpx"], num_transponders)
            DELP = np.array_split(data["delp"], num_transponders)
            dX, dY, dZ = DELP[idx]
            sigX, sigY, sigZ = SIGPX[idx]
            typer.echo(pxp_id)
            typer.echo(
                f"D_x = {np.format_float_scientific(dX, 6)} m, "
                f"Sigma(x) = {np.format_float_scientific(sigX, 6)} m"
            )
            typer.echo(
                f"D_y = {np.format_float_scientific(dY, 6)} m, "
                f"Sigma(y) = {np.format_float_scientific(sigY, 6)} m"
            )
            typer.echo(
                f"D_z = {np.format_float_scientific(dZ, 6)} m, "
                f"Sigma(z) = {np.format_float_scientific(sigZ, 6)} m"
            )
        process_dict[n_iter]["enu"] = np.array(enu_arr)
        process_dict[n_iter]["sig_enu"] = np.array(sig_enu)
        process_dict[n_iter]["transponders_lla"] = np.array(lat_lon)
        typer.echo()
    return process_dict, is_converged
