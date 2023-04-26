"""geo.py

Geospatial utilities module
"""
from typing import Tuple

import numpy as np
import pyproj

ECEF_PROJ = pyproj.Proj(proj="geocent", ellps="WGS84")  # Earth Centered Fixed (x, y, z)
LLA_PROJ = pyproj.Proj(
    proj="longlat", ellps="WGS84"
)  # Longitude Latitude (lon, lat, alt)
GEODETIC_PRECISION = 9  # 9 decimal points
GEOCENTRIC_PRECISION = 3  # 3 decimal points


def geocentric2geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Converts Geocentric coordinate (x,y,z) to Geodetic coordinate (lon,lat,alt)
    based on Ellipsoid `WGS84`

    Parameters
    ----------
    x : float
        Geocentric x in meters
    y : float
        Geocentric y in meters
    z : float
        Geocentric z in meters

    Returns
    -------
    longitude, latitude, altitude
        The lon, lat, alt coordinates in degrees
    """
    transformer = pyproj.Transformer.from_proj(ECEF_PROJ, LLA_PROJ)
    coordinates = transformer.transform(x, y, z, radians=False)
    return np.round(coordinates, GEODETIC_PRECISION)


def geodetic2geocentric(
    lon: float, lat: float, alt: float
) -> Tuple[float, float, float]:
    """
    Converts Geodetic coordinate (lon,lat,alt) to Geocentric coordinate (x,y,z)
    based on Ellipsoid `WGS84`

    Parameters
    ----------
    lon : float
        Longitude in degrees
    lat : float
        Latitude in degrees
    alt : float
        Altitude in degrees

    Returns
    -------
    x, y, z
        The x, y, z coordinates in meters
    """
    transformer = pyproj.Transformer.from_proj(LLA_PROJ, ECEF_PROJ)
    coordinates = transformer.transform(lon, lat, alt, radians=False)
    return np.round(coordinates, GEOCENTRIC_PRECISION)


def __get_rotation_matrix(
    lat_org: float, lon_org: float, to_enu: bool = True
) -> np.ndarray:
    """Helper function for ECEF to ENU and vice versa"""
    # Setup
    cos_lat = np.cos(lat_org)
    sin_lat = np.sin(lat_org)
    cos_lon = np.cos(lon_org)
    sin_lon = np.sin(lon_org)

    if to_enu:
        return np.array(
            [
                [-sin_lon, cos_lon, 0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )
    return np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0, cos_lat, sin_lat],
        ]
    )


def geocentric2enu(x, y, z, lon_org, lat_org, alt_org):
    """
    Transform Geocentric coordinate (x,y,z) to a local ENU coordinate(east, north, up)
    based on a reference point in Geodetic coordinate (lon, lat, alt)

    See https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

    Parameters
    ----------
    x : float
        Geocentric x in meters
    y : float
        Geocentric y in meters
    z : float
        Geocentric z in meters
    lon_org : float
        Reference origin longitude in degrees
    lat_org : float
        Reference origin latitude in degrees
    alt_org : float
        Reference origin altitude in degrees

    Returns
    -------
    east, north, up
        The east, north, up coordinates in meters
    """
    origin_xyz = geodetic2geocentric(lon_org, lat_org, alt_org)
    delta_xyz = np.column_stack([np.array([x, y, z]) - origin_xyz])  # D(3, 1)

    # Rotation matrix R(3, 3)
    R = __get_rotation_matrix(lat_org, lon_org)
    return np.dot(R, delta_xyz)  # E(3, 1)


def enu2geocentric(e, n, u, lat_org, lon_org, alt_org):
    """
    Transform local ENU coordinate(east, north, up) to Geocentric coordinate (x,y,z)
    based on a reference point in Geodetic coordinate (lon, lat, alt)

    See https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF

    Parameters
    ----------
    east : float
        East in meters
    north : float
        North in meters
    up : float
        Up in meters
    lon_org : float
        Reference origin longitude in degrees
    lat_org : float
        Reference origin latitude in degrees
    alt_org : float
        Reference origin altitude in degrees

    Returns
    -------
    x, y, z
        The x, y, z coordinates in meters
    """
    origin_xyz = np.column_stack(
        [geodetic2geocentric(lon_org, lat_org, alt_org)]
    )  # O(3, 1)
    enu = np.column_stack([np.array([e, n, u])])  # E(3, 1)
    # Rotation matrix R(3, 3)
    R = __get_rotation_matrix(lat_org, lon_org, to_enu=False)
    return np.dot(R, enu) + origin_xyz  # X(3, 1)


# 04/25/2023 Don Setiawan
# Code below uses Proj... but it provides different result
# need to investigate as to what's going on here and its approach compared to the fortran
# The Proj method does take into account the ellipsoid
#
# def _get_topo_proj(lon, lat, alt) -> pyproj.Proj:
#     """Get topocentric Proj object"""
#     return pyproj.Proj(proj="topocentric", ellps="WGS84", lat_0=lat, lon_0=lon, h_0=alt)

# def geodetic2enu(
#     lon: float, lat: float, alt: float, lon_org: float, lat_org: float, alt_org: float
# ) -> Tuple[float, float, float]:
#     """
#     Convert Geodetic coordinates (lon,lat,alt) to Topocentric coordinates (east,north,up)
#     based on a specific topocentric origin described as geographic coordinates.

#     See https://proj.org/operations/conversions/topocentric.html for more details.

#     Parameters
#     ----------
#     lon : float
#         Longitude in degrees
#     lat : float
#         Latitude in degrees
#     alt : float
#         Altitude in degrees
#     lon_org : float
#         Topocentric origin (x) as longitude in degrees
#     lat_org : float
#         Topocentric origin (y) as latitude in degrees
#     alt_org : float
#         Topocentric origin (z) as altitude in degrees

#     Returns
#     -------
#     east, north, up
#         The east, north, up coordinates in meters
#     """

#     # Create topocentric proj obj based on specified origin
#     topo_proj = _get_topo_proj(lon_org, lat_org, alt_org)

#     # Create the projection transform pipeline string
#     # This pipeline is for lonlatalt -> xyz -> enu
#     projs = ["+proj=pipeline", ECEF_PROJ, topo_proj]
#     pyproj_pipe = [proj.srs if not isinstance(proj, str) else proj for proj in projs]
#     pyproj_pipeline = " +step ".join(pyproj_pipe)

#     # Get transformer from pipeline string
#     transformer = pyproj.Transformer.from_pipeline(pyproj_pipeline)

#     return transformer.transform(lon, lat, alt, radians=False)


# def enu2geodetic(
#     east: float, north: float, up: float, lon_org: float, lat_org: float, alt_org: float
# ) -> Tuple[float, float, float]:
#     """
#     Convert Topocentric coordinates (east,north,up) to Geodetic coordinates (lon,lat,alt)
#     based on a specific topocentric origin described as geographic coordinates.

#     See https://proj.org/operations/conversions/topocentric.html for more details.

#     Parameters
#     ----------
#     east : float
#         East in meters
#     north : float
#         North in meters
#     up : float
#         Up in meters
#     lon_org : float
#         Topocentric origin (x) as longitude in degrees
#     lat_org : float
#         Topocentric origin (y) as latitude in degrees
#     alt_org : float
#         Topocentric origin (z) as altitude in degrees

#     Returns
#     -------
#     longitude, latitude, altitude
#         The lon, lat, alt coordinates in degrees
#     """

#     # Create topocentric proj obj based on specified origin
#     topo_proj = _get_topo_proj(lon_org, lat_org, alt_org)

#     # Create projection transform pipeline string
#     # This pipeline is for enu -> xyz
#     pyproj_pipeline = f"+proj=pipeline +inv +step {topo_proj.srs}"

#     # Get transformer from pipeline string
#     transformer1 = pyproj.Transformer.from_pipeline(pyproj_pipeline)

#     x, y, z = transformer1.transform(east, north, up, radians=False)

#     # Return lonlatalt by converting xyz -> lonlatalt
#     return geocentric2geodetic(x, y, z)
