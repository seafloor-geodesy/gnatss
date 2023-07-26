"""geo.py

Geospatial utilities module
"""
import numpy as np
from nptyping import Float64, NDArray, Shape


def _get_rotation_matrix(
    lat_org: float, lon_org: float, to_enu: bool = True
) -> NDArray[Shape["3, 3"], Float64]:
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
