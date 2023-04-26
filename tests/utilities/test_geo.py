import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from seagap.utilities.geo import (
    enu2geocentric,
    geocentric2enu,
    geocentric2geodetic,
    geodetic2geocentric,
)

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


def test_geodetic2geocentric(coordinates):
    """Test for geodetic to geocentric conversion, comparing to fortran code"""
    (lat, lon, alt), (x, y, z), _ = coordinates
    input_data = f"{AE} {RF}\n{lat} {lon} {alt}".encode("utf-8")
    # Calls on fortran code
    result = subprocess.run(
        [str((HERE.parent / "fortran" / "plh2xyz").absolute())],
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
    px, py, pz = geodetic2geocentric(lon, lat, alt)

    assert (fx, fy, fz) == (x, y, z)
    assert (px, py, pz) == (x, y, z)

    # Compare fortran and python values
    assert (fx, fy, fz) == (px, py, pz)


def test_geocentric2geodetic(coordinates):
    """Test for geocentric to geodetic conversion"""
    (lat, lon, alt), (x, y, z), _ = coordinates

    plon, plat, palt = geocentric2geodetic(x, y, z)

    assert (plat, plon, palt) == (lat, lon, alt)


def test_geocentric2enu(coordinates, array_center):
    _, (x, y, z), _ = coordinates
    olat, olon, oalt = array_center
    # Python enu
    penu = geocentric2enu(x, y, z, olon, olat, oalt)

    # Find delta array first
    origin_array = geodetic2geocentric(olon, olat, oalt)
    delta_array = np.column_stack([np.array([x, y, z]) - origin_array])

    # Calculate enu with fortran lib
    fenu = flib.xyz2enu(olat, olon, delta_array)
    assert np.array_equal(np.round(fenu, 9), np.round(penu, 9))


def test_enu2geocentric(coordinates, array_center):
    _, (x, y, z), (e, n, u) = coordinates
    olat, olon, oalt = array_center

    pxyz = enu2geocentric(e, n, u, olat, olon, oalt)
    expected = np.column_stack([np.array([x, y, z])])

    assert np.array_equal(pxyz, expected)
