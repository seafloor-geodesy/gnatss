"""constants.py

Module for storing constants used in package
"""

# Data columns for sound profile
SP_DEPTH = "dd"
SP_SOUND_SPEED = "sv"

# Data columns for time
TIME_ASTRO = "astro_time"  # astropy time obj
TIME_TAI = "tai_time"  # unix tai (includes leap seconds)
TIME_ISO = "iso_string"  # ISO 8601 string (YYYY-MM-DD HH:mm:ss.f)
TIME_J2000 = "time"  # Default J2000 time (sec since 2000-01-01 12:00:00)

# Travel times columns
TT_TIME = TIME_J2000  # Time string that will become J2000
TT_DATE = "date"  # Date string
TT_TRANSPONDER = "transponder-{}".format  # Transponder id. ex. TT_TRANSPONDER('01')

# GPS solutions columns
GPS_TIME = TIME_J2000  # Default J2000 time (sec since 2000-01-01 12:00:00)
GPS_X = "x"  # Geocentric x
GPS_Y = "y"  # Geocentric y
GPS_Z = "z"  # Geocentric z
GPS_LON = "lon"  # Geodetic longitude
GPS_LAT = "lat"  # Geodetic latitude
GPS_ALT = "alt"  # Geodetic altitude
GPS_EAST = "east"  # Local tangent East
GPS_NORTH = "north"  # Local tangent North
GPS_UP = "up"  # Local tangent Up
GPS_COV_XX = "xx"
GPS_COV_XY = "xy"
GPS_COV_XZ = "xz"
GPS_COV_YX = "yx"
GPS_COV_YY = "yy"
GPS_COV_YZ = "yz"
GPS_COV_ZX = "zx"
GPS_COV_ZY = "zy"
GPS_COV_ZZ = "zz"
GPS_COV = [
    GPS_COV_XX,
    GPS_COV_XY,
    GPS_COV_XZ,
    GPS_COV_YX,
    GPS_COV_YY,
    GPS_COV_YZ,
    GPS_COV_ZX,
    GPS_COV_ZY,
    GPS_COV_ZZ,
]  # Covariance matrix columns
