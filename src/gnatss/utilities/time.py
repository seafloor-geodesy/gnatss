"""time.py

Time utilities module utilizing astropy
"""
from typing import Union

from astropy import units as u
from astropy.time import Time as AstroTime  # noqa
from astropy.time import TimeDelta
from astropy.time.formats import TimeFromEpoch, erfa

__all__ = ["AstroTime", "erfa"]


class TimeUnixJ2000(TimeFromEpoch):
    """
    Seconds from 2000-01-01 12:00:00 TT (equivalent to 2000-01-01 11:58:55.816 UTC),
    ignoring leap seconds.
    """

    name = "unix_j2000"
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = "2000-01-01 12:00:00"
    epoch_val2 = None
    epoch_scale = "tt"  # Scale for epoch_val class attribute
    epoch_format = "iso"  # Format for epoch_val class attribute


def gpsws_to_time(week: int, seconds: Union[int, float]) -> AstroTime:
    """Converts GPS week and seconds to Astropy Time

    Parameters
    ----------
    week : int
        The GPS reference week number
    seconds : Union[int, float]
        Seconds from midnight terrestrial time (TT) of
        GPS reference week day;
        accurate to the millisecond level
    Returns
    -------
    AstroTime
        The time in Astropy Time format
    """
    # Cast week to int if it's a string
    if isinstance(week, str):
        week = int(week)

    # Cast seconds to float if it's a string
    if isinstance(seconds, str):
        seconds = float(seconds)

    # Get the origin of GPS time
    gps_epoch = AstroTime(0, format="gps", scale="tai")
    # Convert week to days
    num_weeks = week * u.wk
    # Add days to time 0, includes leap seconds
    week_time = gps_epoch + TimeDelta(num_weeks.to(u.d), format="jd", scale="tai")
    # Get the week start time exactly at midnight, doesn't include leap seconds, TT scale
    week_start = AstroTime(week_time.strftime("%Y-%m-%d"), format="iso", scale="tt")
    # Add seconds to beginning of week
    final_time = week_start + TimeDelta(seconds, format="sec")
    return final_time
