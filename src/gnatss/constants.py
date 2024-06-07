"""constants.py

Module for storing constants used in package
"""

from __future__ import annotations

from .dataspec import DataSpec

# Config constants
DEFAULT_CONFIG_PROCS = ("main", "solver", "posfilter")

DATA_SPEC = DataSpec(version="v1")

# General constants
SIG_3D = "sig_3d"
DELAY_TIME_PRECISION = 6
DATA_OUTLIER_THRESHOLD = 25.0

# Data columns for sound profile
SP_DEPTH = "depth"
SP_SOUND_SPEED = "speed"

# Data columns for deletion file
DEL_STARTTIME = "starttime"
DEL_ENDTIME = "endtime"

# Data columns for quality control file
QC_STARTTIME = "starttime"
QC_ENDTIME = "endtime"
QC_NOTES = "notes"

# Data columns for time
TIME = "time"
TIME_ASTRO = f"astro_{TIME}"  # astropy time obj
TIME_TAI = f"tai_{TIME}"  # unix tai (includes leap seconds)
TIME_ISO = "iso_string"  # ISO 8601 string (YYYY-MM-DD HH:mm:ss.f)
TIME_J2000 = TIME  # Default J2000 time (sec since 2000-01-01 12:00:00)

# Travel times columns
TT_TIME = TIME_J2000  # Time string that will become J2000
TT_DATE = "date"  # Date string

# GPS solutions columns
GPS_TIME = TIME_J2000  # Default J2000 time (sec since 2000-01-01 12:00:00)
GPS_AZ = "azimuth"  # Azimuth
GPS_DISTANCE = "distance"  # Distance
GPS_X = "x"  # Geocentric x
GPS_Y = "y"  # Geocentric y
GPS_Z = "z"  # Geocentric z
GPS_GEOCENTRIC = [GPS_X, GPS_Y, GPS_Z]  # Geocentric x,y,z
ANT_GPS_GEOCENTRIC = [f"ant_{d}" for d in GPS_GEOCENTRIC]  # Geocentric ant x,y,z
ANT_GPS_GEOCENTRIC_STD = [f"ant_sig{d}" for d in GPS_GEOCENTRIC]  # Geocentric ant x,y,z std

GPS_LON = "lon"  # Geodetic longitude
GPS_LAT = "lat"  # Geodetic latitude
GPS_ALT = "alt"  # Geodetic altitude
GPS_GEODETIC = [GPS_LON, GPS_LAT, GPS_ALT]  # Geodetic lon,lat,alt
ANT_GPS_GEODETIC = [f"ant_{d}" for d in GPS_GEODETIC]  # Geodetic ant lon,lat,alt
GPS_EAST = "east"  # Local tangent East
GPS_NORTH = "north"  # Local tangent North
GPS_UP = "up"  # Local tangent Up
GPS_LOCAL_TANGENT = [GPS_EAST, GPS_NORTH, GPS_UP]  # Local tangent e,n,u
ANT_GPS_LOCAL_TANGENT = [f"ant_{d}" for d in GPS_LOCAL_TANGENT]  # Local tangent ant e,n,u
GPS_COV_XX = "xx"
GPS_COV_XY = "xy"
GPS_COV_XZ = "xz"
GPS_COV_YX = "yx"
GPS_COV_YY = "yy"
GPS_COV_YZ = "yz"
GPS_COV_ZX = "zx"
GPS_COV_ZY = "zy"
GPS_COV_ZZ = "zz"
GPS_COV_DIAG = [GPS_COV_XX, GPS_COV_YY, GPS_COV_ZZ]  # Covariance matrix diagonal values
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
ANT_GPS_COV_DIAG = [f"ant_cov_{c}" for c in GPS_COV_DIAG]  # Antenna covariance diagonal
ANT_GPS_COV = [f"ant_cov_{c}" for c in GPS_COV]  # Antenna covariance columns

# Roll Pitch Heading columns
RPH_TIME = TIME_J2000
RPH_ROLL = "roll"
RPH_PITCH = "pitch"
RPH_HEADING = "heading"
RPH_LOCAL_TANGENTS = [RPH_ROLL, RPH_PITCH, RPH_HEADING]
RPH_COV_RR = "rr"
RPH_COV_RP = "rp"
RPH_COV_RH = "rh"
RPH_COV_PR = "pr"
RPH_COV_PP = "pp"
RPH_COV_PH = "ph"
RPH_COV_HR = "hr"
RPH_COV_HP = "hp"
RPH_COV_HH = "hh"
RPH_COV_DIAG = [RPH_COV_RR, RPH_COV_PP, RPH_COV_HH]  # Covariance matrix diagonal values
RPH_COV = [
    RPH_COV_RR,
    RPH_COV_RP,
    RPH_COV_RH,
    RPH_COV_PR,
    RPH_COV_PP,
    RPH_COV_PH,
    RPH_COV_HR,
    RPH_COV_HP,
    RPH_COV_HH,
]  # Covariance matrix columns
PLATFORM_COV_RPH_DIAG = [f"cov_{c}" for c in RPH_COV_DIAG]  # Platform covariance diagonal
PLATFORM_COV_RPH = [f"cov_{c}" for c in RPH_COV]  # Platform covariance columns

# Antenna Position Direction columns
ANTENNA_EASTWARD = "ant_e"
ANTENNA_NORTHWARD = "ant_n"
ANTENNA_UPWARD = "ant_u"
ANTENNA_DIRECTIONS = [ANTENNA_EASTWARD, ANTENNA_NORTHWARD, ANTENNA_UPWARD]

L1_DATA_FORMAT = {
    "INSPVAA": {
        "regex_pattern": r"#INSPVAA,.*?,.*?,.*?,FINESTEERING,(.*?),(.*?),"
        r".*?,.*?,.*?;.*?,.*?,(.*?),"
        r"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),"
        r"(.*?),(.*?),INS_SOLUTION_GOOD\*+.*\n",
        "data_fields": (
            "week",
            "seconds",
            GPS_LAT,
            GPS_LON,
            GPS_ALT,
            GPS_NORTH,
            GPS_EAST,
            GPS_UP,
            RPH_ROLL,
            RPH_PITCH,
            RPH_HEADING,
        ),
        "data_fields_dtypes": (
            "int",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
        ),
    },
    "INSSTDEVA": {
        "regex_pattern": r"#INSSTDEVA,.*?,.*?,.*?,FINESTEERING,(.*?),(.*?),.*?,.*?,.*?;(.*?),(.*?),"
        r"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),"
        r"(.*?),(.*?),(.*?),.*?,.*?,.*?\*+.*?\n",
        "data_fields": (
            "week",
            "seconds",
            f"{GPS_LAT}_sig",
            f"{GPS_LON}_sig",
            f"{GPS_ALT}_sig",
            f"{GPS_NORTH}_sig",
            f"{GPS_EAST}_sig",
            f"{GPS_UP}_sig",
            *PLATFORM_COV_RPH_DIAG,
            "ext_sol_stat",
            "time_since_update",
        ),
        "data_fields_dtypes": (
            "int",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "object",
            "object",
        ),
    },
}


# Kalman filter configs. # TODO move to config file
gnss_pos_psd = 3.125e-5
vel_psd = 0.0025
cov_err = 0.25
start_dt = 5e-2
