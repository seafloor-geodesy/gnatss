"""time.py

Time utilities module utilizing astropy
"""
from astropy.time import Time as AstroTime  # noqa
from astropy.time.formats import TimeFromEpoch, erfa


class TimeUnixJ2000(TimeFromEpoch):

    """

    Seconds from 2000-01-01 12:00:00 UTC

    """

    name = "unix_j2000"

    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)

    epoch_val = "2000-01-01 12:00:00"

    epoch_val2 = None

    epoch_scale = "utc"  # Scale for epoch_val class attribute
    epoch_format = "iso"  # Format for epoch_val class attribute
