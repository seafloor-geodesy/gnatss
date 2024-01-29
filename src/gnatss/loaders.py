import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from pandas.api.types import is_integer_dtype, is_string_dtype
from pydantic import ValidationError

from . import constants
from .configs.io import CSVOutput
from .configs.main import Configuration


def load_configuration(config_yaml: Optional[str] = None) -> Configuration:
    """
    Loads configuration yaml file into a Config object
    to be used throughout the pre-processing
    """
    try:
        config = Configuration()
    except ValidationError:
        warnings.warn(
            "Loading attempt failed, trying to load configuration from file path given."
        )
        if config_yaml is None or not Path(config_yaml).exists():
            raise FileNotFoundError(
                "Configuration file not found. Unable to create configuration"
            )

        yaml_dict = yaml.safe_load(Path(config_yaml).read_text("utf-8"))
        config = Configuration(**yaml_dict)
    return config


def load_sound_speed(sv_files: List[str]) -> pd.DataFrame:
    """
    Loads sound speed file data into pandas dataframe

    Parameters
    ----------
    sv_files : List[str]
        The list of path string to the sound speed files to be loaded

    Returns
    -------
    pd.DataFrame
        Sound speed profile pandas dataframe
    """
    columns = [constants.SP_DEPTH, constants.SP_SOUND_SPEED]

    sv_dfs = [
        pd.read_csv(sv_file, delim_whitespace=True, header=None, names=columns)
        .drop_duplicates(constants.SP_DEPTH)
        .reset_index(drop=True)
        for sv_file in sv_files
    ]

    return pd.concat(sv_dfs).reset_index(drop=True)


def load_travel_times(
    files: List[str],
    transponder_ids: List[str],
    is_j2k: bool = False,
    time_scale: str = "tt",
) -> pd.DataFrame:
    """
    Loads travel times data into a pandas dataframe from a list of files.

    Time conversions from date and time string will occur under the hood
    using `astropy` package.
    The final time with epoch J2000 will be returned in the pandas dataframe.

    Parameters
    ----------
    files : list
        The list of path string to files to load
    transponder_ids : list
        A list of transponder ids corresponding to the transponder data
        within the travel times. Note that ids order must match to travel times
        column order for transponders
    is_j2k : bool
        A flag to signify to expect j2000 time column only rather
        than date and time. No conversion will be performed if this
        is the case.
    time_scale : str
        The time scale of the datetime string input.
        Default is 'tt' for Terrestrial Time,
        Must be one of the following:
        (`tai`, `tcb`, `tcg`, `tdb`,
        `tt`, `ut1`, `utc`)

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing all of the travel times data.
        Expected columns: 'time', 'transponder-x1', ..., 'transponder-xid'

    Notes
    -----
    The input travel time data is assumed to follow the structure below.

    Structure 1:

    Date, Time, TT Transponder 1, TT Transponder 2, ..., TT Transponder N

    - Date is a date string. e.g. '27-JUL-22'
    - Time is a time string. e.g. '17:37:45.00'
    - TT is the 2 way travel times in microseconds (µs). e.g. 2263261

    Structure 2:

    Time, TT Transponder 1, TT Transponder 2, ..., TT Transponder N

    - Time is a J2000 epoch time float (seconds since Jan 1, 2000 12:00 UTC).
      e.g. '712215465.000'
    - TT is the 2 way travel times in microseconds (µs). e.g. 2263261

    These files are often called `pxp_tt` or `pxp_tt_j2k`.

    Additionally, there's an assumption that wave glider delays
    have been removed from the data.
    """
    DATETIME_FORMAT = "%d-%b-%y %H:%M:%S.%f"

    columns = [constants.TT_DATE, constants.TT_TIME, *transponder_ids]
    if is_j2k:
        # If it's already j2k then pop off date column, idx 0
        columns.pop(0)
    # Read all travel times
    travel_times = [pd.read_csv(i, delim_whitespace=True, header=None) for i in files]
    all_travel_times = pd.concat(travel_times).reset_index(drop=True)

    # Remove any columns that are not being used
    column_num_diff = len(columns) - len(all_travel_times.columns)
    if column_num_diff < 0:
        all_travel_times = all_travel_times.iloc[:, :column_num_diff]

    # Set standard column name
    all_travel_times.columns = columns

    # Convert microseconds to seconds for delay times
    all_travel_times[transponder_ids] = all_travel_times[transponder_ids] * 1e-6

    # Skip time conversion if it's already j2000 time
    if not is_j2k:
        from .utilities.time import AstroTime

        # Determine j2000 time from date and time string,
        # assuming that they're in Terrestrial Time (TT) scale
        all_travel_times[constants.TIME_ASTRO] = all_travel_times.apply(
            lambda row: AstroTime.strptime(
                f"{row[constants.TT_DATE].lower()} {row[constants.TT_TIME]}",
                DATETIME_FORMAT,
                scale=time_scale,
            ),
            axis=1,
        )
        # Replace time to j2000 rather than time string
        all_travel_times[constants.TT_TIME] = all_travel_times.apply(
            lambda row: row[constants.TIME_ASTRO].unix_j2000, axis=1
        )

        # Drop unused columns for downstream computation
        all_travel_times = all_travel_times.drop(
            [constants.TT_DATE, constants.TIME_ASTRO], axis=1
        )

    return all_travel_times


def load_roll_pitch_heading(files: List[str]) -> pd.DataFrame:
    """
    Loads roll pitch heading data into a pandas dataframe from a list of files.

    Parameters
    ----------
    files : List[str]
        The list of path string to files to load

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing all of
        the roll pitch heading data.
        Expected columns will have 'time' and
        the 'roll', 'pitch', 'heading' values
    """
    columns = [
        constants.RPH_TIME,
        constants.RPH_ROLL,
        constants.RPH_PITCH,
        constants.RPH_HEADING,
    ]
    # Read all rph files
    rph_dfs = [
        pd.read_csv(i, delim_whitespace=True, header=None, names=columns)
        .drop_duplicates(constants.RPH_TIME)
        .reset_index(drop=True)
        for i in files
    ]
    all_rph = pd.concat(rph_dfs).reset_index(drop=True)
    return all_rph


def load_gps_solutions(
    files: List[str], time_round: int = constants.DELAY_TIME_PRECISION
) -> pd.DataFrame:
    """
    Loads gps solutions into a pandas dataframe from a list of files.

    Parameters
    ----------
    files : list
        The list of path string to files to load
    time_round : int
        The precision value to round the time values

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing all of the gps solutions data.
        Expected columns will have 'time',
        the geocentric 'x,y,z',
        and covariance matrix 'xx' to 'zz'

    Notes
    -----
    The input gps solutions data is assumed to follow the structure below:

    The numbers are column number.

    1: Time unit seconds in J2000 epoch
    2: Geocentric x in meters
    3: Geocentric y in meters
    4: Geocentric z in meters
    5 - 13: Covariance matrix (3x3) xx, xy, xz, yx, yy, yz, zx, zy, zz

    These files are often called `POS_FREED_TRANS_TWTT`.
    """
    columns = [constants.GPS_TIME, *constants.GPS_GEOCENTRIC, *constants.GPS_COV]
    # Real all gps solutions
    gps_solutions = [
        pd.read_csv(i, delim_whitespace=True, header=None, names=columns) for i in files
    ]
    all_gps_solutions = pd.concat(gps_solutions).reset_index(drop=True)

    # Round to match the delays precision
    # TODO: Find a way to determine this precision dynamically?
    if isinstance(time_round, int) and time_round > 0:
        all_gps_solutions.loc[:, constants.GPS_TIME] = all_gps_solutions[
            constants.GPS_TIME
        ].round(time_round)

    return all_gps_solutions


def load_deletions(
    file_paths: List[str], config: Configuration, time_scale="tt"
) -> pd.DataFrame:
    """
    Loads the raw deletion text file into a pandas dataframe

    Parameters
    ----------
    file_paths : List[str]
        Path to the deletion file to be loaded
    config : Configuration
        The configuration object
    time_scale : str
        The time scale of the datetime string input.
        Default is 'tt' for Terrestrial Time,
        Must be one of the following:
        (`tai`, `tcb`, `tcg`, `tdb`,
        `tt`, `ut1`, `utc`)

    Returns
    -------
    pd.DataFrame
        Deletion ranges data pandas dataframe
    """
    output_path = Path(config.output.path)
    # TODO: Add additional files to be used for deletions
    default_deletions = output_path / CSVOutput.deletions.value
    if file_paths:
        from .utilities.time import AstroTime

        cut_dfs = [pd.read_fwf(file_path, header=None) for file_path in file_paths]
        cut_df = pd.concat(cut_dfs).reset_index(drop=True)

        # Date example: 28-JUL-22 12:30:00
        cut_df[constants.DEL_STARTTIME] = pd.to_datetime(
            cut_df[0] + "T" + cut_df[1], format="%d-%b-%yT%H:%M:%S"
        )
        cut_df[constants.DEL_ENDTIME] = pd.to_datetime(
            cut_df[2] + "T" + cut_df[3], format="%d-%b-%yT%H:%M:%S"
        )
        # Got rid of the other columns
        # TODO: Parse the other columns
        cut_columns = cut_df.columns[0:-2]
        cut_df.drop(columns=cut_columns, inplace=True)

        # Convert time string to j2000,
        # assuming that they're in Terrestrial Time (TT) scale
        cut_df[constants.DEL_STARTTIME] = cut_df[constants.DEL_STARTTIME].apply(
            lambda row: AstroTime(row, scale=time_scale).unix_j2000
        )
        cut_df[constants.DEL_ENDTIME] = cut_df[constants.DEL_ENDTIME].apply(
            lambda row: AstroTime(row, scale=time_scale).unix_j2000
        )
    elif default_deletions.exists():
        cut_df = pd.read_csv(default_deletions)
    else:
        cut_df = pd.DataFrame()

    # Try to find outliers.csv file if there is one run already
    # this currently assumes that the file is in the output directory
    outliers_csv = output_path / CSVOutput.outliers.value
    if outliers_csv.exists():
        import typer

        typer.echo(f"Found {str(outliers_csv.absolute())} file. Including into cuts...")
        outliers_df = pd.read_csv(outliers_csv)
        outlier_cut = pd.DataFrame.from_records(
            outliers_df[constants.TT_TIME].apply(lambda row: (row, row)).to_numpy(),
            columns=[constants.DEL_STARTTIME, constants.DEL_ENDTIME],
        )
        # Include the outlier cut into here
        cut_df = pd.concat([cut_df, outlier_cut])
        outliers_csv.unlink()

    # Export to a deletions csv
    if not cut_df.empty:
        cut_df.to_csv(output_path / CSVOutput.deletions.value, index=False)

    return cut_df


def load_quality_control(qc_files: List[str], time_scale="tt") -> pd.DataFrame:
    """
    Loads the quality controls csv files into a pandas dataframe

    Parameters
    ----------
    qc_files : List[str]
        Path to the quality control files to be loaded
    time_scale : str
        The time scale of the datetime string input.
        Default is 'tt' for Terrestrial Time,
        Must be one of the following:
        (`tai`, `tcb`, `tcg`, `tdb`,
        `tt`, `ut1`, `utc`)

    Returns
    -------
    pd.DataFrame
        Deletion ranges data pandas dataframe
    """
    csv_columns = [constants.QC_STARTTIME, constants.QC_ENDTIME, constants.QC_NOTES]

    if qc_files:
        qc_dfs = [
            pd.read_csv(qc_file, header=0, names=csv_columns).reset_index(drop=True)
            for qc_file in qc_files
        ]
        qc_df = (
            pd.concat(qc_dfs).reset_index(drop=True).drop(columns=[constants.QC_NOTES])
        )

        # If QC_STARTTIME & QC_ENDTIME are of string type, convert to unix_j2000 float
        if is_string_dtype(qc_df[constants.QC_STARTTIME]) and is_string_dtype(
            qc_df[constants.QC_ENDTIME]
        ):
            from .utilities.time import AstroTime

            qc_df[constants.QC_STARTTIME] = qc_df[constants.QC_STARTTIME].apply(
                lambda row: AstroTime(row, scale=time_scale, format="isot").unix_j2000
            )
            qc_df[constants.QC_ENDTIME] = qc_df[constants.QC_ENDTIME].apply(
                lambda row: AstroTime(row, scale=time_scale, format="isot").unix_j2000
            )

        # If QC_STARTTIME & QC_ENDTIME are of int type,
        # convert to float for consistency with unix_j2000 float
        elif is_integer_dtype(qc_df[constants.QC_STARTTIME]) and is_integer_dtype(
            qc_df[constants.QC_ENDTIME]
        ):
            qc_df[constants.QC_ENDTIME] = qc_df[constants.QC_ENDTIME].apply(
                lambda row: float(row)
            )

        else:
            msg = (
                f"Unsupported data type found in quality_controls "
                f"{constants.QC_STARTTIME} or {constants.QC_ENDTIME} columns."
            )
            raise ValueError(msg)

    # If quality control files are not configured, return empty dataframe
    else:
        qc_df = pd.DataFrame(columns=[constants.QC_STARTTIME, constants.QC_ENDTIME])

    return qc_df
