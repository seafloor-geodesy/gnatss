"""time.py

Time utilities module utilizing astropy
"""
import numpy as np
from astropy import units as u
from astropy.time import Time as AstroTime  # noqa
from astropy.time import TimeDelta
from astropy.time.formats import TimeFromEpoch, erfa
from nptyping import Float, Int, NDArray, Shape

__all__ = ["AstroTime", "erfa", "gps_ws_time_to_j2000_time"]


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


def gps_ws_time_to_j2000_time(
    weeks: NDArray[Shape["*"], Int], seconds: NDArray[Shape["*"], Float]
) -> NDArray[Shape["*"], Float]:
    """Converts GPS weeks and seconds to TimeUnixJ2000 seconds
    Parameters
    ----------
    weeks : NDArray[(Any,), Int]
        The GPS reference week number
    seconds : NDArray[(Any,), Float]
        Seconds from midnight terrestrial time (TT) of GPS reference week day;
        accurate to the millisecond level
    Returns
    -------
    NDArray[(Any,), Float]
        Float Numpy Array representing seconds from J2000 epoch
    """
    # Find unique week values, and their indexes
    unique, unique_index = np.unique(weeks, return_inverse=True)

    gps_epoch = AstroTime(0, format="gps", scale="tt")

    # Convert week to days
    num_days = (unique * u.wk).to(u.d)

    # Add days to time 0, includes leap seconds
    week_time = gps_epoch + TimeDelta(num_days, format="jd", scale="tai")

    # Get the week start time exactly at midnight, doesn't include leap seconds, TT scale
    week_start = AstroTime(week_time.strftime("%Y-%m-%d"), format="iso", scale="tt")

    # Add seconds to beginning of week
    final_time = week_start[unique_index] + TimeDelta(seconds, format="sec")

    final_time_j2000 = final_time.unix_j2000
    return final_time_j2000
