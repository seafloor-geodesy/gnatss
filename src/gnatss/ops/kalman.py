from math import pi

import numba
import numpy as np
from nptyping import NDArray, Shape
from numpy import arctan as atan
from numpy import arctan2 as atan2
from numpy import asarray, cos, degrees, empty_like, finfo, hypot, sin, sqrt, tan, where

# Process noise matrix
gnss_pos_psd = 3.125e-5
vel_psd = 0.0025

WGS84_ELL = {"name": "WGS-84 (1984)", "a": 6378137.0, "b": 6356752.31424518}
semimajor_axis = WGS84_ELL["a"]
semiminor_axis = WGS84_ELL["b"]


@numba.njit
def ecef2geodetic(
    x: float,
    y: float,
    z: float,
    semimajor_axis: float = semimajor_axis,
    semiminor_axis: float = semiminor_axis,
    eps=finfo(np.float32).eps,
) -> tuple:
    """
    convert ECEF (meters) to geodetic coordinates

    rewrite of pymap3d.ecef2geodetic to use numba for speedup...
    not as robust
    """

    x = asarray(x)
    y = asarray(y)
    z = asarray(z)

    r = sqrt(x**2 + y**2 + z**2)

    E = sqrt(semimajor_axis**2 - semiminor_axis**2)

    # eqn. 4a
    u = sqrt(0.5 * (r**2 - E**2) + 0.5 * hypot(r**2 - E**2, 2 * E * z))

    hxy = hypot(x, y)

    huE = hypot(u, E)

    # eqn. 4b
    Beta = empty_like(r)
    Beta = atan(huE / u * z / hxy)
    # eqn. 13
    Beta += ((semiminor_axis * u - semimajor_axis * huE + E**2) * sin(Beta)) / (
        semimajor_axis * huE * 1 / cos(Beta) - E**2 * cos(Beta)
    )

    # eqn. 4c
    # %% final output
    lat = atan(semimajor_axis / semiminor_axis * tan(Beta))

    # # patch latitude for float32 precision loss
    lim_pi2 = pi / 2 - eps
    lat = where(Beta >= lim_pi2, pi / 2, lat)
    lat = where(Beta <= -lim_pi2, -pi / 2, lat)

    lon = atan2(y, x)

    # eqn. 7
    cosBeta = cos(Beta)

    # patch altitude for float32 precision loss
    cosBeta = where(Beta >= lim_pi2, 0, cosBeta)
    cosBeta = where(Beta <= -lim_pi2, 0, cosBeta)

    alt = hypot(z - semiminor_axis * sin(Beta), hxy - semimajor_axis * cosBeta)

    lat = degrees(lat)
    lon = degrees(lon)

    return lat, lon, alt


@numba.njit
def predict(dt, X, P, Q, F):
    F[0:3, 3:6] = np.identity(3) * dt
    Q = updateQ(dt, Q)
    X = F @ X
    P = F @ P @ F.T + Q

    return X, P, Q, F


@numba.njit
def updateQ(dt, Q):
    # Position estimation noise
    # Initial Q values from Chadwell code 3.125d-5 3.125d-5 3.125d-5 0.0025 0.0025 0.0025, assumes white noise of 2.5 cm over a second
    Q[0:3, 0:3] = np.identity(3) * gnss_pos_psd * dt

    # Velocity estimation noise (acc psd)
    Q[3:6, 3:6] = np.identity(3) * vel_psd

    return Q


@numba.njit
def rot_vel(row, lat, lon):
    """
    ------------------- Rotate ENU velocity into ECEF velocity --------------------------------
                dX = |  -sg   -sa*cg  ca*cg | | de |        de = |  -sg       cg      0 | | dX |
                dY = |   cg   -sa*sg  ca*sg | | dn |  and   dn = |-sa*cg  -sa*sg     ca | | dY |
                dZ = |    0    ca     sa    | | du |        du = | ca*cg   ca*sg     sa | | dZ |
    -------------------------------------------------------------------------------------------
    """
    v_enu = row[1:4].copy().reshape((3, 1))
    lat = np.deg2rad(lat)
    lam = np.deg2rad(lon)

    ca = np.cos(lat)
    sa = np.sin(lat)

    cg = np.cos(lam)
    sg = np.sin(lam)

    rot = np.zeros((3, 3))
    rot[0, 0] = -sg
    rot[0, 1] = -sa * cg
    rot[0, 2] = ca * cg
    rot[1, 0] = cg
    rot[1, 1] = -sa * sg
    rot[1, 2] = ca * sg
    rot[2, 0] = 0
    rot[2, 1] = ca
    rot[2, 2] = sa
    v_xyz = rot @ v_enu

    return rot, v_xyz


@numba.njit
def update_vel_cov(row, R_velocity, rot):
    # Velocity measurement noise
    # Vel STD
    R_velocity[0, 0] = row[13] ** 2
    R_velocity[1, 1] = row[14] ** 2
    R_velocity[2, 2] = row[15] ** 2

    R_velocity[0, 1] = row[16]
    R_velocity[0, 2] = row[17]
    R_velocity[1, 2] = row[18]

    R_velocity[1, 0] = R_velocity[0, 1]
    R_velocity[2, 0] = R_velocity[0, 2]
    R_velocity[2, 1] = R_velocity[1, 2]

    R_velocity = rot @ R_velocity @ rot.T
    return R_velocity


@numba.njit
def update_rpos(row, R_position):
    R_position[0, 0] = row[7] ** 2
    R_position[1, 1] = row[8] ** 2
    R_position[2, 2] = row[9] ** 2

    R_position[0, 1] = np.sign(row[10]) * row[10] ** 2
    R_position[0, 2] = np.sign(row[11]) * row[11] ** 2
    R_position[1, 2] = np.sign(row[12]) * row[12] ** 2

    R_position[1, 0] = R_position[0, 1]
    R_position[2, 0] = R_position[0, 2]
    R_position[2, 1] = R_position[1, 2]
    return R_position


@numba.njit
def update_position(row, Nx, X, P, R_position):
    pos_xyz = row[4:7].copy().reshape((3, 1))

    H = np.zeros((3, Nx))
    H[0:3, 0:3] = np.identity(3)
    R_position = update_rpos(row, R_position)

    y = pos_xyz - H @ X
    S = H @ P @ H.T + R_position
    K = (P @ H.T) @ np.linalg.inv(S)
    X = X + K @ y

    I = np.identity(Nx)  # noqa
    P = (I - K @ H) @ P
    return X, P, R_position


@numba.njit
def update_velocity(Nx, X, P, R_velocity, v_xyz):
    H = np.zeros((3, Nx))
    H[0:3, 3:6] = np.identity(3)

    y = v_xyz - H @ X
    S = H @ P @ H.T + R_velocity
    K = (P @ H.T) @ np.linalg.inv(S)
    X = X + K @ y

    IDENTITY = np.identity(Nx)
    P = (IDENTITY - K @ H) @ P
    return X, P, R_velocity


@numba.njit()
def rts_smoother(Xs, Ps, F, Q):
    # Rauch, Tongue, and Striebel smoother

    n, dim_x, _ = Xs.shape

    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()

    i = 0
    for k in range(n - 2, -1, -1):
        Pp[k] = F @ P[k] @ F.T + Q  # predicted covariance

        K[k] = P[k] @ F.T @ np.linalg.inv(Pp[k])
        x[k] += K[k] @ (x[k + 1] - (F @ x[k]))
        P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T
        i += 1

    return x, P, K, Pp


@numba.njit
def kalman_init(row):
    dt = 5e-2
    Nx = 6
    Nu = 3  # noqa

    # error states: pos_xyz, v_ned, eul, bias_acc, bias_gyro
    X = np.zeros((Nx, 1))
    # Initial antenna location
    pos_xyz = row[4:7].copy().reshape((3, 1))
    X[0:3] = pos_xyz

    lat, lon, _ = ecef2geodetic(X[0], X[1], X[2])
    # Initial antenna velocity
    rot, v_xyz = rot_vel(row, lat[0], lon[0])
    X[3:6] = v_xyz

    # Process model

    # State transition matrix
    F = np.identity(Nx)
    F[0:3, 3:6] = np.identity(3) * dt

    # Set initial values to 0.25?
    P = np.identity(Nx)

    # Process noise matrix
    Q = np.zeros((Nx, Nx))
    Q = updateQ(dt, Q)

    # Position measurement noise
    R_position = np.identity(3)
    R_position = update_rpos(row, R_position)

    # Velocity measurement noise
    R_velocity = np.identity(3) * 4e-8
    R_velocity[0, 1] = 1.5e-9
    R_velocity[0, 2] = 1.5e-9
    R_velocity[1, 2] = 1.5e-9
    R_velocity[1, 0] = R_velocity[0, 1]
    R_velocity[2, 0] = R_velocity[0, 2]
    R_velocity[2, 1] = R_velocity[1, 2]

    R_velocity = update_vel_cov(row, R_velocity, rot)

    return Nx, X, P, Q, F, R_position, R_velocity


@numba.njit()
def run_filter_simulation(records: NDArray) -> NDArray:
    """
    Performs Kalman filtering of the GPS_GEOCENTRIC and GPS_COV_DIAG fields

    Parameters
    ----------
    records : Numpy Array
        Numpy Array containing the fields: # TODO -> Fill field names after verification of algorithm

    Returns
    -------
    DataFrame
        Pandas Dataframe containing Time and Kalman filtered GPS_GEOCENTRIC and GPS_COV_DIAG columns
    """
    last_time = np.nan
    records_len = records.shape[0]
    Xs = np.zeros((records_len, 6, 1))
    Ps = np.zeros((records_len, 6, 6))
    n_records = len(records)
    for i in range(n_records):
        row = records[i]
        dts = row[0]
        if i == 0:
            Nx, X, P, Q, F, R_position, R_velocity = kalman_init(row)
            last_time = dts
        else:
            dt = np.abs(
                dts - last_time
            )  # This helps to stabilize the solution, abs ensures reverse filtering works.
            X, P, Q, F = predict(dt, X, P, Q, F)

            last_time = dts

            lat, lon, _ = ecef2geodetic(X[0], X[1], X[2])
            rot, v_xyz = rot_vel(row, lat[0], lon[0])

            # New velocity standard deviations
            if ~np.isnan(row[16]):
                R_velocity = update_vel_cov(row, R_velocity, rot)

            # new GNSS measurement
            if ~np.isnan(row[4]):
                X, P, R_position = update_position(row, Nx, X, P, R_position)

            # new velocity measurement
            if ~np.isnan(row[1]):
                X, P, R_velocity = update_velocity(Nx, X, P, R_velocity, v_xyz)

        Xs[i] = X
        Ps[i] = P

    x, P, K, Pp = rts_smoother(Xs, Ps, F, Q)

    return x, P, K, Pp
