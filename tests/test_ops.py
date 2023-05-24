import pandas as pd
import pytest

from gnatss.ops import calc_std_and_verify


@pytest.fixture(
    params=[
        {
            "time": 712215465.0,
            "x": -2575288.225,
            "y": -3682570.227,
            "z": 4511064.31,
            "xx": 0.0007642676487,
            "xy": 1.464868147e-07,
            "xz": -2.28909888e-06,
            "yx": 1.464868147e-07,
            "yy": 0.001003773991,
            "yz": -3.469954175e-06,
            "zx": -2.28909888e-06,
            "zy": -3.469954175e-06,
            "zz": 0.0008874766165,
        },
        {
            "time": 712215480.0,
            "x": -2575279.439,
            "y": -3682576.054,
            "z": 4511065.58,
            "xx": 0.0007632827522,
            "xy": 4.247534112e-07,
            "xz": -2.52535522e-06,
            "yx": 4.247534112e-07,
            "yy": 0.001002904626,
            "yz": -3.809993913e-06,
            "zx": -2.52535522e-06,
            "zy": -3.809993913e-06,
            "zz": 0.0008868617746,
        },
        {
            "time": 712215495.0,
            "x": -2575269.558,
            "y": -3682579.137,
            "z": 4511068.196,
            "xx": 0.0007636394742,
            "xy": 2.351279563e-07,
            "xz": -2.343107339e-06,
            "yx": 2.351279563e-07,
            "yy": 0.001003126674,
            "yz": -3.537116345e-06,
            "zx": -2.343107339e-06,
            "zy": -3.537116345e-06,
            "zz": 0.0008869383932,
        },
        {
            "time": 712215510.0,
            "x": -2575262.122,
            "y": -3682585.47,
            "z": 4511066.817,
            "xx": 0.0007622707048,
            "xy": -2.964120039e-08,
            "xz": -2.149297811e-06,
            "yx": -2.964120039e-08,
            "yy": 0.001003263125,
            "yz": -3.258009041e-06,
            "zx": -2.149297811e-06,
            "zy": -3.258009041e-06,
            "zz": 0.0008847786993,
        },
        {
            "time": 712215525.0,
            "x": -2575253.07,
            "y": -3682595.498,
            "z": 4511065.769,
            "xx": 0.0007639192564,
            "xy": -2.477204992e-07,
            "xz": -2.233920204e-06,
            "yx": -2.477204992e-07,
            "yz": -3.374849755e-06,
            "zx": -2.233920204e-06,
            "zy": -3.374849755e-06,
            "zz": 0.0008861840659,
        },
    ]
)
def gps_dataseries(request) -> pd.Series:
    return pd.Series(request.param)


@pytest.fixture(params=[0.05, 0.001])
def gps_sigma_limit(request):
    return request.param


def test_calc_std_and_verify(gps_dataseries, gps_sigma_limit):
    """Tests the ``calc_std_and_verify`` function"""
    try:
        sig_3d = calc_std_and_verify(
            gps_series=gps_dataseries, sigma_limit=gps_sigma_limit
        )

        assert isinstance(sig_3d, float)
    except Exception as e:
        if "yy" not in gps_dataseries:
            # One of the dataseries doesn't have yy
            # so Key error is expected
            assert isinstance(e, KeyError)
        elif gps_sigma_limit == 0.001:
            # This is a really small sigma limit
            # it will fail
            assert isinstance(e, ValueError)
