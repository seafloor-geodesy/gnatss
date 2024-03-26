"""constants.py

Module for storing constants used in package
"""
from . import garpos

__all__ = ["garpos"]

# Config constants
DEFAULT_CONFIG_PROCS = ("solver", "posfilter")

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
TIME_ASTRO = "astro_time"  # astropy time obj
TIME_TAI = "tai_time"  # unix tai (includes leap seconds)
TIME_ISO = "iso_string"  # ISO 8601 string (YYYY-MM-DD HH:mm:ss.f)
TIME_J2000 = "time"  # Default J2000 time (sec since 2000-01-01 12:00:00)

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
GPS_LON = "lon"  # Geodetic longitude
GPS_LAT = "lat"  # Geodetic latitude
GPS_ALT = "alt"  # Geodetic altitude
GPS_GEODETIC = [GPS_LON, GPS_LAT, GPS_ALT]  # Geodetic lon,lat,alt
GPS_EAST = "east"  # Local tangent East
GPS_NORTH = "north"  # Local tangent North
GPS_UP = "up"  # Local tangent Up
GPS_LOCAL_TANGENT = [GPS_EAST, GPS_NORTH, GPS_UP]  # Local tangent e,n,u
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

# Roll Pitch Heading columns
RPH_TIME = TIME_J2000
RPH_ROLL = "roll"
RPH_PITCH = "pitch"
RPH_HEADING = "heading"
RPH_LOCAL_TANGENTS = [RPH_ROLL, RPH_PITCH, RPH_HEADING]
RPH_COV_RR = "roll_roll"
RPH_COV_RP = "roll_pitch"
RPH_COV_RH = "roll_heading"
RPH_COV_PR = "pitch_roll"
RPH_COV_PP = "pitch_pitch"
RPH_COV_PH = "pitch_heading"
RPH_COV_HR = "heading_roll"
RPH_COV_HP = "heading_pitch"
RPH_COV_HH = "heading_heading"
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

# Antenna Position Direction columns
ANTENNA_EASTWARD = "ant_e"
ANTENNA_NORTHWARD = "ant_n"
ANTENNA_UPWARD = "ant_u"
ANTENNA_DIRECTIONS = [ANTENNA_EASTWARD, ANTENNA_NORTHWARD, ANTENNA_UPWARD]

# L1 rph data
L1_DATA_HEADER = (
    "Message",
    "Port",
    "Sequence #",
    "% Idle Time",
    "Time Status",
    "Week",
    "Seconds",
    "Receiver Status",
    "Reserved",
    "Receiver S/W Version",
)
L1_DATA_CONFIG = {
    "INSPVAA": {
        "regex_pattern": r"#INSPVAA,.*?,.*?,.*?,.*?,.*?,.*?,.*?;(.*?),(.*?),(.*?),"
        r"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),"
        r"(.*?),(.*?),(.*?)\*+.*\n",
        "data_header": L1_DATA_HEADER,
        "data_fields": (
            "Week",
            "Seconds",
            "Latitude",
            "Longitude #",
            "Height",
            "North Velocity",
            "East Velocity",
            "Up Velocity",
            RPH_ROLL,
            RPH_PITCH,
            RPH_HEADING,
            "Status",
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
        ),
    },
    "INSSTDEVA": {
        "regex_pattern": r"#INSSTDEVA,.*?,.*?,.*?,.*?,.*?,.*?,.*?;(.*?),(.*?),"
        r"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),"
        r"(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)(\*+)(.*?)\n",
        "data_header": L1_DATA_HEADER,
        "data_fields": (
            "Latitude std",
            "Longitude std",
            "Height std",
            "North Velocity std",
            "East Velocity std",
            "Up Velocity std",
            f"{RPH_ROLL} std",
            f"{RPH_PITCH} std",
            f"{RPH_HEADING} std",
            "Ext sol stat",
            "Time Since Update",
            "Reserved1",
            "Reserved2",
            "Reserved3",
            "crc",
            "Sentence terminator",
        ),
        "data_fields_dtypes": (
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
            "object",
            "object",
            "object",
            "object",
            "object",
        ),
    },
}
