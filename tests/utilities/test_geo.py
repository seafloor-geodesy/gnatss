import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from nptyping import Float64, NDArray, Shape
from pymap3d import ecef2enu, geodetic2ecef

from gnatss.utilities.geo import _get_rotation_matrix

from ..fortran import flib

AE = 6378137
RF = 298.257222101
HERE = Path(__file__).parent.absolute()


@pytest.fixture(
    params=[
        [
            (45.302840001, -124.965792528, -27.447300516),
            (-2575288.225, -3682570.227, 4511064.31),
            (30.65833784516843, -48.5432379637279, 35.870278707225),
        ],
        [
            (45.302842926, -124.965768018, -26.772966283),
            (-2575286.789, -3682571.528, 4511065.018),
            (28.738677758968027, -48.62310043527105, 36.62160296509438),
        ],
        [
            (45.30284281, -124.965767059, -26.767161545),
            (-2575286.735, -3682571.582, 4511065.013),
            (28.66260099824154, -48.63079082770161, 36.618411673613465),
        ],
        [
            (45.302842778, -124.965766917, -26.767907543),
            (-2575286.727, -3682571.59, 4511065.01),
            (28.65133036714429, -48.63249083166475, 36.61575030903799),
        ],
        [
            (45.302849701, -124.965658148, -26.727371295),
            (-2575279.439, -3682576.054, 4511065.58),
            (20.54714122627723, -51.119870518143685, 37.84139327034435),
        ],
    ]
)
def coordinates(request):
    """Coordinates for `lat lon lat` and their `x y z` equivalent"""
    return request.param


@pytest.fixture
def array_center():
    return (45.3023, -124.9656, 0.0)


def test_plh2xyz(coordinates):
    """Test for geodetic to geocentric conversion, comparing to fortran code"""
    (lat, lon, alt), _, _ = coordinates
    input_data = f"{AE} {RF}\n{lat} {lon} {alt}".encode("utf-8")
    fortran_program = f"plh2xyz-{platform.machine().lower()}"
    # Calls on fortran code
    result = subprocess.run(
        [str((HERE.parent / "fortran" / fortran_program).absolute())],
        input=input_data,
        capture_output=True,
    )
    # Parse the string exported from the fortran code
    fx, fy, fz = np.round(
        np.array(re.findall(r"(-?\d+.\d+)+", result.stdout.decode("utf-8"))).astype(
            float
        ),
        3,
    )  # round to 3 decimal places

    # Calls on the python code
    px, py, pz = np.round(geodetic2ecef(lat, lon, alt), 3)

    # Compare fortran and python values
    assert (fx, fy, fz) == (px, py, pz)


def test_xyz2enu(coordinates, array_center):
    _, (x, y, z), _ = coordinates
    olat, olon, oalt = array_center
    # Python enu
    penu = ecef2enu(x, y, z, olat, olon, oalt)

    # Find delta array first
    origin_array = geodetic2ecef(olat, olon, oalt)
    delta_array = np.column_stack([np.array([x, y, z]) - origin_array])

    # Calculate enu with fortran lib
    fenu = flib.xyz2enu(olat, olon, delta_array)
    assert np.array_equal(np.round(fenu.flatten(), 9), np.round(penu, 9))


@pytest.mark.parametrize(
    "to_enu, expected",
    [
        (True, np.array([[-0.0, 1.0, 0.0], [-0.0, -0.0, 1.0], [1.0, 0.0, 0.0]])),
        (False, np.array([[-0.0, -0.0, 1.0], [1.0, -0.0, 0.0], [0.0, 1.0, 0.0]])),
    ],
)
def test__get_rotation_matrix(
    to_enu: bool, expected: NDArray[Shape["3, 3"], Float64]
) -> None:
    lat, lon = 0.0, 0.0
    res_array = _get_rotation_matrix(lat, lon, to_enu)
    assert np.array_equal(res_array, expected)
