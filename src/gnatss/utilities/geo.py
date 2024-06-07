"""geo.py

Geospatial utilities module
"""

from __future__ import annotations

import numpy as np
from nptyping import Float64, NDArray, Shape
from pymap3d import Ellipsoid, ecef2enu


def _get_rotation_matrix(
    lat_org: float, lon_org: float, to_enu: bool = True
) -> NDArray[Shape[3, 3], Float64]:
    """
    Get rotation matrix for converting between ECEF and ENU

    Parameters
    ----------
    lat_org : float
        Origin latitude
    lon_org : float
        Origin longitude
    to_enu : bool, default=True
        If True, return matrix for converting from ECEF to ENU

    Returns
    -------
    (3, 3) ndarray
        Rotation matrix
    """
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


def ecef2ae(
    x: float,
    y: float,
    z: float,
    lat0: float,
    lon0: float,
    h0: float,
    ell: Ellipsoid = None,
    deg: bool = True,
) -> tuple[float, float]:
    """
    Compute azimuth and elevation from ECEF coordinates
    w.r.t. a reference point

    Parameters
    ----------
    x: float
        target x ECEF coordinate (meters)
    y: float
        target y ECEF coordinate (meters)
    z: float
        target z ECEF coordinate (meters)
    lat0: float
        observer geodetic latitude
    lon0: float
        observer geodetic longitude
    h0: float
        observer altitude above geodetic ellipsoid (meters)
    ell : Ellipsoid, optional
        reference ellipsoid
    deg : bool, optional
        degrees input/output  (False: radians in/out)

    Returns
    -------
    az: float
        azimuth (degrees)
    elev: float
        elevation (degrees)
    """
    if ell is None:
        ell = Ellipsoid.from_name("wgs84")

    e, n, u = ecef2enu(x, y, z, lat0, lon0, h0, ell, deg=deg)

    # 1 millimeter precision for singularity stability
    try:
        e[abs(e) < 1e-3] = 0.0
        n[abs(n) < 1e-3] = 0.0
        u[abs(u) < 1e-3] = 0.0
    except TypeError:
        if abs(e) < 1e-3:
            e = 0.0
        if abs(n) < 1e-3:
            n = 0.0
        if abs(u) < 1e-3:
            u = 0.0

    # Calculate hypotenuse
    r = np.hypot(n, e)
    elev = np.arctan(r / u)
    az = np.arctan2(n, e)

    if deg:
        az = np.degrees(az)
        elev = np.degrees(elev)
    return az, elev


def calc_enu_comp(
    residuals: NDArray[Shape["*"], Float64],
    az: NDArray[Shape["*"], Float64],
    el: NDArray[Shape["*"], Float64],
) -> NDArray[Shape["*"], Float64]:
    """
    Calculates the mean east, north, up components of the residuals
    from N number of transponders.

    Returns
    -------
    residuals : (N,) ndarray
        Residuals values in centimeters (cm)
    az : (N,) ndarray
        Azimuth values in degrees (deg)
    el : (N,) ndarray
        Elevation values in degrees (deg)
    """
    # Perform checks
    assert residuals.ndim == 1, "Residuals must be 1D"
    assert az.ndim == 1, "Azimuth must be 1D"
    assert el.ndim == 1, "Elevation must be 1D"
    assert (
        az.shape == el.shape == residuals.shape
    ), "Azimuth, elevation, and residuals must have same shape"

    # Compute north, east, and vertical components
    res_north = np.mean(residuals * np.sin(np.radians(az)) * np.sin(np.radians(el)))
    res_east = np.mean(residuals * np.cos(np.radians(az)) * np.cos(np.radians(el)))
    res_vert = np.mean(residuals * np.cos(np.radians(el)))

    return np.array([res_east, res_north, res_vert])
