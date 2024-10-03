from __future__ import annotations

from typing import Literal

import numba
import numpy as np
import pandas as pd
import typer
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList
from pymap3d import ecef2enu, ecef2geodetic

from .. import constants
from ..configs.io import CSVOutput
from ..configs.main import Configuration
from ..configs.solver import ArrayCenter
from .harmonic_mean import sv_harmonic_mean
from .io import load_config, load_datasets
from .utils import _prep_col_names

META_COLUMNS = [
    constants.DATA_SPEC.transponder_id,
    constants.DATA_SPEC.tx_time,
    constants.DATA_SPEC.travel_time,
]

TRANSMIT_LOC_COLS = list(constants.DATA_SPEC.transducer_tx_fields.keys())
REPLY_LOC_COLS = list(constants.DATA_SPEC.transducer_rx_fields.keys())
TRANSMIT_COV_LOC_COLS = list(constants.DATA_SPEC.gnss_tx_cov_fields.keys())


@numba.njit(cache=True)
def _split_cov(
    cov_values: NDArray[Shape[9], Float64],
) -> NDArray[Shape[3, 3], Float64]:
    """
    Splits an array of covariance values of shape (9,)
    to a matrix array of shape (3, 3)

    Parameters
    ----------
    cov_values : (9,) ndarray
        Covariance array of shape (9,)
        in the order [xx, xy, xz,yx, yy, yx, yz, zx, zy, zz]

    Returns
    -------
    (3,3) ndarray
        The final covariance matrix of shape (3,3)
    """

    n = 3
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i] = cov_values[i * n : i * n + n]
    return cov


def _get_standard_columns(
    columns: list[str], data_type: Literal["transmit", "receive"] = "receive"
) -> list[str]:
    """
    Get the standard columns based on the data type

    Parameters
    ----------
    columns : list of str
        The columns to be standardized
    data_type : {'transmit', 'receive'}, optional
        The data type which can either be 'transmit' or 'receive', by default "receive"

    Returns
    -------
    list of str
        The standard columns

    Notes
    -----
    This function is used to standardize the columns of the data
    based on the data type. It is used to ensure that the columns
    are in the correct order and format for the solver algorithm.
    However, this function is very specific to the v1 data spec
    and will need to be updated for future versions.
    """
    # Very specific to v1 atm... will need to be updated for v2 and beyond
    new_columns = []
    fields = {
        "transmit": constants.DATA_SPEC.tx_fields,
        "receive": constants.DATA_SPEC.rx_fields,
    }
    standard_cols = {k.lower(): k for k, v in fields[data_type].items()}
    suffix = f"_{data_type}"

    for col in columns:
        if col in META_COLUMNS:
            new_columns.append(col)
        elif col == constants.TIME_J2000:
            new_columns.append(col[0].upper() + suffix)
        elif col in constants.GPS_GEOCENTRIC:
            new_columns.append(col.upper() + suffix)
        else:
            num_suffix = (
                constants.DATA_SPEC.tx_code
                if data_type == "transmit"
                else constants.DATA_SPEC.rx_code
            )
            new_column_name = col + str(num_suffix)
            match_col = standard_cols.get(new_column_name.lower())
            if match_col:
                new_columns.append(match_col)
    return new_columns


def ecef_to_enu(
    df: pd.DataFrame,
    input_ecef_columns: list[str],
    output_enu_columns: list[str],
    array_center: ArrayCenter,
) -> pd.DataFrame:
    """
    Calculate ENU coordinates from input ECEF coordinates

    Parameters
    ----------
    df: pd.DataFrame
        The full dataset for computation
    input_ecef_columns: list[str]
        Columns in the df that contain ENU coordinates
    output_enu_columns: list[str]
        Columns that should be created in the df for ENU coordinates
    array_center : ArrayCenter
        An object containing the center of the array

    Returns
    -------
    pd.DataFrame
        Modified dataset with ECEF and ENU coordinates
    """
    enu = df[input_ecef_columns].apply(
        lambda row: ecef2enu(
            *row.to_numpy(),
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    return df.assign(**dict(zip(output_enu_columns, zip(*enu, strict=False), strict=False)))


def calc_lla_and_enu(all_observations: pd.DataFrame, array_center: ArrayCenter) -> pd.DataFrame:
    """
    Calculates the LLA and ENU coordinates for all observations

    Parameters
    ----------
    all_observations : pd.DataFrame
        The full dataset for computation
    array_center : ArrayCenter
        An object containing the center of the array

    Returns
    -------
    pd.DataFrame
        Modified dataset with LLA and ENU coordinates
    """
    lla = all_observations[TRANSMIT_LOC_COLS].apply(
        lambda row: ecef2geodetic(*row.to_numpy()), axis=1
    )
    enu = all_observations[TRANSMIT_LOC_COLS].apply(
        lambda row: ecef2enu(
            *row.to_numpy(),
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    all_observations = all_observations.assign(
        **dict(zip(_prep_col_names(constants.GPS_GEODETIC), zip(*lla, strict=False), strict=False))
    )
    return all_observations.assign(
        **dict(
            zip(_prep_col_names(constants.GPS_LOCAL_TANGENT), zip(*enu, strict=False), strict=False)
        )
    )


def get_data_inputs(all_observations: pd.DataFrame) -> NumbaList:
    """Extracts data inputs to perform solving algorithm

    Parameters
    ----------
    all_observations : pd.DataFrame
        The full dataset for computation

    Returns
    -------
    NumbaList
        A list of data inputs
    """
    # Set up special numba list so it can be passed
    # into numba functions for just in time compilation
    data_inputs = NumbaList()

    # Group obs by the transmit time
    grouped_obs = all_observations.groupby(constants.DATA_SPEC.tx_time)

    # Get transmit xyz
    transmit_xyz = grouped_obs[TRANSMIT_LOC_COLS].first().to_numpy()

    # Get reply xyz
    reply_xyz_list = []
    grouped_obs[REPLY_LOC_COLS].apply(lambda group: reply_xyz_list.append(group.to_numpy()))

    # Get observed delays
    observed_delay_list = []
    grouped_obs[constants.DATA_SPEC.travel_time].apply(
        lambda group: observed_delay_list.append(group.to_numpy())
    )

    # Get transmit cov matrices
    cov_vals_df = grouped_obs[TRANSMIT_COV_LOC_COLS].first()
    gps_covariance_matrices = [_split_cov(row.to_numpy()) for _, row in cov_vals_df.iterrows()]

    # Merge all inputs
    for data in zip(
        transmit_xyz, reply_xyz_list, gps_covariance_matrices, observed_delay_list, strict=False
    ):
        data_inputs.append(data)
    return data_inputs


def prefilter_replies(
    all_observations: pd.DataFrame,
    num_transponders: int,
) -> pd.DataFrame:
    """
    Remove pings that do receive replies from each
    transponder in the array.

    Parameters
    ----------
    all_observations : pd.DataFrame
        The original observations that include every ping and reply
    num_transponders : int
        The number of transponders in the array

    Returns
    -------
    pd.DataFrame
        The observations where the number of replies equal the
        number of transponders
    """
    # Get value counts for transmit times
    time_counts = all_observations[constants.DATA_SPEC.tx_time].value_counts()

    return all_observations[
        all_observations[constants.DATA_SPEC.tx_time].isin(
            time_counts[time_counts == num_transponders].index
        )
    ]


def clean_tt(
    travel_times: pd.DataFrame,
    transponder_ids: list[str],
    travel_times_correction: float,
    transducer_delay_time: float,
) -> pd.DataFrame:
    """
    Clean travel times by doing the following steps:
    1. remove any travel times that have 0 reply time
    2. apply travel time correction and transducer delay time.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The original travel times data
    transponder_ids : list[str]
        A list of the transponder ids that matches the order
        with travel_times data
    travel_times_correction : float
        Correction to times in travel times (secs.)
    transducer_delay_time : float
        Transducer Delay Time - delay at surface transducer (secs).

    Returns
    -------
    pd.DataFrame
        The cleaned travel times data
    """
    # Get cleaned travel times
    # This is anything that has 0 reply time
    cleaned_travel_times = travel_times.loc[
        travel_times[transponder_ids].where(travel_times[transponder_ids] != 0).dropna().index
    ]

    # Apply travel time correction
    cleaned_travel_times.loc[:, constants.TT_TIME] = (
        cleaned_travel_times[constants.TT_TIME] + travel_times_correction + transducer_delay_time
    )

    # TODO: Store junk travel times? These are travel times with 0 values
    # _ = all_travel_times.loc[
    #     all_travel_times.where(all_travel_times[transponder_ids] == 0)
    #     .dropna(how="all")
    #     .index
    # ]

    return cleaned_travel_times


def filter_tt(
    travel_times: pd.DataFrame,
    cut_df: pd.DataFrame,
    time_column: str = constants.TT_TIME,
) -> pd.DataFrame:
    """
    Filter travel times data by removing the data that falls within
    the time range specified in the deletions file.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The original travel times data
    cut_df : pd.DataFrame
        The deletions data to be removed

    Returns
    -------
    pd.DataFrame
        The filtered travel times data
    """

    if len(cut_df.index) > 0:
        # Only cut the data with deletions file if there are data
        cut_ids_all = []
        for _, cut in cut_df.iterrows():
            cut_ids = travel_times[
                (travel_times[time_column] >= cut.starttime)
                & (travel_times[time_column] <= cut.endtime)
            ].index.to_numpy()
            cut_ids_all = cut_ids_all + cut_ids.tolist()
        cut_ids_all = list(set(cut_ids_all))
        return travel_times.loc[~travel_times.index.isin(cut_ids_all)]
    return travel_times


def preprocess_tt(travel_times: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess travel times data by creating a dataframe that
    contains the travel times and the reply times contiguously.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The travel times data

    Returns
    -------
    pd.DataFrame
        The preprocessed travel times data
    """
    data = []
    for tx, tds in travel_times.set_index(constants.TT_TIME).iterrows():
        data.append((tx, constants.DATA_SPEC.tx_time, 0, np.nan))
        for k, td in tds.to_dict().items():
            data.append((tx + td, k, td, tx))

    return pd.DataFrame(
        data,
        columns=[
            constants.TT_TIME,
            constants.DATA_SPEC.transponder_id,
            constants.DATA_SPEC.travel_time,
            constants.DATA_SPEC.tx_time,
        ],
    )


def standardize_data(pos_freed_trans_twtt: pd.DataFrame, data_precision: int = 8) -> pd.DataFrame:
    is_transmit = pos_freed_trans_twtt[constants.DATA_SPEC.tx_time].isna()

    # Standardize receive data
    receive_df = pos_freed_trans_twtt[~is_transmit]
    receive_df.columns = _get_standard_columns(receive_df.columns, "receive")

    # Standardize transmit data
    transmit_df = pos_freed_trans_twtt[is_transmit]
    transmit_df = transmit_df.drop(
        [
            constants.DATA_SPEC.tx_time,
            constants.DATA_SPEC.travel_time,
            constants.DATA_SPEC.transponder_id,
        ],
        axis=1,
    )
    transmit_df.columns = _get_standard_columns(transmit_df.columns, "transmit")

    return receive_df.merge(transmit_df, on=constants.DATA_SPEC.tx_time).round(data_precision)


def data_loading(
    config_yaml: str,
    distance_limit: float | None = None,
    residual_limit: float | None = None,
    residual_range_limit: float | None = None,
    outlier_threshold: float | None = None,
    from_cache: bool = False,
    remove_outliers: bool = False,
    skip_posfilter: bool = False,
    skip_solver: bool = False,
):
    config = load_config(
        config_yaml,
        distance_limit=distance_limit,
        residual_limit=residual_limit,
        residual_range_limit=residual_range_limit,
        outlier_threshold=outlier_threshold,
    )
    # Switch off cache if gps solution does not exist
    if from_cache and not gps_solution_exists(config):
        from_cache = False

    return config, load_datasets(config, from_cache, remove_outliers, skip_posfilter, skip_solver)


def preprocess_data(config, data_dict):
    twtt_df = preprocess_travel_times(data_dict.get("travel_times"), config)
    config = compute_harmonic_mean(config, svdf=data_dict.get("sound_speed", None))
    data_dict.update({"travel_times": twtt_df})
    return config, data_dict


def gps_solution_exists(config) -> bool:
    if config.output is None:
        return False

    file_path = config.output.path + CSVOutput.gps_solution
    return config.output._fsmap.fs.exists(file_path)


def preprocess_travel_times(pxp_df, config: Configuration):
    typer.echo("Preprocessing Travel Times Data")
    transponder_ids = [tp.pxp_id for tp in config.transponders]
    pxp_df = clean_tt(
        pxp_df,
        transponder_ids=transponder_ids,
        travel_times_correction=config.travel_times_correction,
        transducer_delay_time=config.transducer_delay_time,
    )
    twtt_df = preprocess_tt(pxp_df)
    typer.echo("Finished Preprocessing Travel Times Data")
    return twtt_df


def compute_harmonic_mean(
    config: Configuration,
    svdf: pd.DataFrame | None = None,
):
    # Compute harmonic mean of each transponder
    if svdf is not None and config.solver:
        typer.echo("Computing harmonic mean...")
        start_depth = config.solver.harmonic_mean_start_depth
        for transponder in config.transponders:
            # Compute the harmonic mean and round to 3 decimal places
            harmonic_mean = round(sv_harmonic_mean(svdf, start_depth, transponder.height), 3)
            transponder.sv_mean = harmonic_mean
            typer.echo(transponder)
        typer.echo("Finished computing harmonic mean")
    return config


def ensure_monotonic_increasing(all_observations: pd.DataFrame) -> pd.DataFrame:
    # In case things are not sorted, let's sort on the fly
    # This is important for the solver to work properly
    # as it assumes the data is sorted by receive time and that
    # the data is monotonic increasing
    if not all_observations[constants.DATA_SPEC.rx_time].is_monotonic_increasing:
        all_observations = all_observations.sort_values(by=constants.DATA_SPEC.rx_time).reset_index(
            drop=True
        )
    return all_observations
