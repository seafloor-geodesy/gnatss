import numpy as np
import pandas as pd
import pytest

from gnatss.configs.solver import ArrayCenter
from gnatss.constants import GPS_GEOCENTRIC, GPS_GEODETIC, GPS_LOCAL_TANGENT, GPS_TIME
from gnatss.ops import (
    calc_std_and_verify,
    calc_tt_residual,
    calc_twtt_model,
    calc_uv,
    clean_zeros,
    compute_enu_series,
    find_gps_record,
)

# Datasets
GPS_DATASET = [
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
    # This should be missing yy and z
    {
        "time": 712215525.0,
        "x": -2575253.07,
        "y": -3682595.498,
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

TRAVEL_TIMES_DATASET = [
    7.12215465e08,
    7.12215480e08,
    7.12215495e08,
    7.12215510e08,
    7.12215525e08,
    7.12215540e08,
]

TT_DELAY_SECONDS = [
    np.array([2.281219, 2.377755, 2.388229]),
    np.array([2.288577, 2.371308, 2.383921]),
    np.array([2.301908, 2.363096, 2.382359]),
]

TWTT_MODEL = [
    np.array([2.08140459, 2.05803764, 1.94834694]),
    np.array([2.08875182, 2.0515515, 1.94400265]),
    np.array([2.10208738, 2.04335558, 1.9424679]),
]

TT_RESIDUAL = [
    np.array([-0.00018559, -0.00028264, -0.00011794]),
    np.array([-1.74815363e-04, -2.43499730e-04, -8.16499168e-05]),
    np.array([-0.00017938, -0.00025958, -0.0001089]),
]

TRANSMIT_VECTORS = [
    np.array(
        [
            [-375.921, 1193.236, -900.247],
            [550.164, -141.056, -1414.852],
            [1160.014, 812.795, -277.465],
        ]
    ),
    np.array(
        [
            [-383.357, 1199.569, -898.868],
            [542.728, -134.723, -1413.473],
            [1152.578, 819.128, -276.086],
        ]
    ),
    np.array(
        [
            [-392.409, 1209.597, -897.82],
            [533.676, -124.695, -1412.425],
            [1143.526, 829.156, -275.038],
        ]
    ),
]

REPLY_VECTORS = [
    np.array(
        [
            [-376.828, 1194.152, -900.553],
            [549.157, -140.103, -1415.148],
            [1158.996, 813.753, -277.761],
        ]
    ),
    np.array(
        [
            [-384.879, 1199.434, -898.625],
            [541.175, -134.871, -1413.261],
            [1151.021, 818.979, -275.88],
        ]
    ),
    np.array(
        [
            [-393.932, 1211.999, -895.664],
            [532.113, -122.293, -1410.229],
            [1141.949, 831.558, -272.829],
        ]
    ),
]


@pytest.fixture(
    params=[
        ArrayCenter(lat=45.3023, lon=-124.9656, alt=0.0),
        dict(lat=45.3023, lon=-124.9656, alt=0.0),
    ]
)
def array_center(request):
    return request.param


@pytest.fixture(params=GPS_DATASET)
def gps_dataseries(request) -> pd.Series:
    return pd.Series(request.param)


@pytest.fixture(params=TRAVEL_TIMES_DATASET)
def travel_time(request) -> float:
    return request.param


@pytest.fixture(params=[0.05, 0.001])
def gps_sigma_limit(request):
    return request.param


@pytest.fixture
def mean_sv():
    return [1481.542, 1481.513, 1481.5]


@pytest.fixture
def transponders_delay():
    return [0.2, 0.32, 0.44]


def test_find_gps_record(travel_time):
    gps_df = pd.DataFrame.from_records(GPS_DATASET)

    gps_ds = find_gps_record(gps_solutions=gps_df, travel_time=travel_time)

    assert isinstance(gps_ds, pd.Series)


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


def test_compute_enu_series(gps_dataseries, array_center):
    """Tests the ``compute_enu_series`` function"""
    location_columns = [GPS_TIME, *GPS_GEOCENTRIC]
    try:
        enu_series = compute_enu_series(
            input_series=gps_dataseries, array_center=array_center
        )
        expected_columns = [*location_columns, *GPS_GEODETIC, *GPS_LOCAL_TANGENT]

        assert all(col in enu_series for col in expected_columns)
    except Exception as e:
        if "z" not in gps_dataseries:
            assert isinstance(e, KeyError)
        elif isinstance(array_center, dict):
            assert isinstance(e, ValueError)


@pytest.mark.parametrize(
    "input_vector,expected",
    [
        (np.array([0, 0, 0]), np.array([2.0, 0.0, 0.0])),
        (np.array([1, 1, 1]), np.array([0.57735027, 0.57735027, 0.57735027])),
        (np.array([-1, -2, 2]), np.array([-0.33333333, -0.66666667, 0.66666667])),
        (np.array([[-1, 1, 3], [1, 1, 4]]), ValueError),
        (np.array([-1, 2]), ValueError),
    ],
)
def test_calc_uv(input_vector, expected):
    """Test calculating unit vector"""
    try:
        unit_vector = calc_uv(input_vector=input_vector)
        assert np.allclose(unit_vector, expected)
    except Exception as e:
        assert isinstance(e, expected)


@pytest.mark.parametrize(
    "transmit_vectors,reply_vectors,expected",
    list(zip(TRANSMIT_VECTORS, REPLY_VECTORS, TWTT_MODEL)),
)
def test_calc_twtt_model(transmit_vectors, reply_vectors, expected, mean_sv):
    """Test calculating two way travel time model"""
    result = calc_twtt_model(
        transmit_vectors=transmit_vectors,
        reply_vectors=reply_vectors,
        transponders_mean_sv=mean_sv,
    )
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "tt_delay_seconds,twtt_model,expected_residual",
    list(zip(TT_DELAY_SECONDS, TWTT_MODEL, TT_RESIDUAL)),
)
def test_calc_tt_residual(
    tt_delay_seconds, twtt_model, expected_residual, transponders_delay
):
    """Test calculating travel time residual"""
    res = calc_tt_residual(
        delays=tt_delay_seconds,
        transponder_delays=transponders_delay,
        twtt_model=twtt_model,
    )
    assert np.allclose(res, expected_residual)


def test_calc_partials():
    ...


def test_calc_weight_matrix():
    ...


@pytest.mark.parametrize(
    "input_array,expected",
    [
        # 2-D case
        (
            np.array([[1, 1, 1, 0, 0, 0], [2, 2, 2, 0, 0, 0], [3, 4, 5, 0, 0, 0]]),
            np.array([[1, 1, 1], [2, 2, 2], [3, 4, 5]]),
        ),
        # 1-D case
        (np.array([1, 3, 4, 0, 7, 8, 0, 0, 0]), np.array([1, 3, 4, 0, 7, 8])),
        # 3-D case, fails
        (np.ones(shape=(2, 2, 3)), ValueError),
    ],
)
def test_clean_zeros(input_array, expected):
    try:
        result = clean_zeros(input_array=input_array)
        assert np.array_equal(result, expected)
    except Exception as e:
        assert isinstance(e, expected)


def test_calc_lsq_contrained():
    ...
