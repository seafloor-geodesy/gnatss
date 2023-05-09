import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from pydantic.error_wrappers import ValidationError

from .configs.main import Configuration
from .constants import TIME_ASTRO, TT_DATE, TT_TIME, TT_TRANSPONDER
from .utilities.time import AstroTime


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


def load_travel_times(
    files: List[str],
    transponder_ids: List[str],
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

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing all of the travel times data.
        Expected columns: 'time', 'transponder-x1', ..., 'transponder-xid'

    Notes
    -----
    The input travel time data is assumed to follow the structure below:

    Date, Time, TT Transponder 1, TT Transponder 2, ..., TT Transponder N

    - Date is a date string. e.g. '27-JUL-22'
    - Time is a time string. e.g. '17:37:45.00'
    - TT is the 2 way travel times in microseconds (µs). e.g. 2263261

    These files are often called `pxp_tt`.
    Additionally, there's an assumption that wave glider delays
    have been removed from the data.
    """
    PARSED_FILE = "parsed"
    DATETIME_FORMAT = "%d-%b-%y %H:%M:%S.%f"

    columns = [TT_DATE, TT_TIME] + [TT_TRANSPONDER(tid) for tid in transponder_ids]
    # Read all travel times
    travel_times = [
        pd.read_csv(i, delim_whitespace=True, header=None)
        for i in files
        if PARSED_FILE not in i
    ]
    all_travel_times = pd.concat(travel_times)

    # Remove any columns that are not being used
    column_num_diff = len(columns) - len(all_travel_times.columns)
    if column_num_diff < 0:
        all_travel_times = all_travel_times.iloc[:, :column_num_diff]

    # Set standard column name
    all_travel_times.columns = columns

    # Determine j2000 time from date and time string
    all_travel_times.loc[:, TIME_ASTRO] = all_travel_times.apply(
        lambda row: AstroTime.strptime(
            f"{row[TT_DATE].lower()} {row[TT_TIME]}", DATETIME_FORMAT
        ),
        axis=1,
    )
    # Replace time to j2000 rather than time string
    all_travel_times[TT_TIME] = all_travel_times.apply(
        lambda row: row[TIME_ASTRO].unix_j2000, axis=1
    )

    # Drop unused columns for downstream computation
    all_travel_times = all_travel_times.drop([TT_DATE, TIME_ASTRO], axis=1)

    return all_travel_times


def load_gps_solutions(files: List[str]) -> pd.DataFrame:
    ...
