import numpy as np
import pandas as pd
import pytest

from gnatss.configs.solver import ArrayCenter
from gnatss.ops import (
    _check_cols_in_series,
    calc_std_and_verify,
    calc_uv,
    clean_zeros,
    find_gps_record,
)
from gnatss.ops.solve import calc_tt_residual, calc_twtt_model
from gnatss.ops.utils import DEFAULT_VECTOR_NORM
from gnatss.ops.validate import calc_lsq_constrained

from .ops_data import (
    ATWA_SAMPLE,
    ATWF_SAMPLE,
    GPS_DATASET,
    LSQ_RESULT,
    REPLY_VECTORS,
    TRANSMIT_VECTORS,
    TRAVEL_TIMES_DATASET,
    TT_DELAY_SECONDS,
    TT_RESIDUAL,
    TWTT_MODEL,
)

# TODO: Figure out how to test with multiple number of transponders data


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
    return np.array([1481.542, 1481.513, 1481.5])


@pytest.fixture
def transponders_delay():
    return np.array([0.2, 0.32, 0.44])


@pytest.fixture
def b_cov():
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def num_transponders():
    return 3


@pytest.fixture
def travel_times_variance():
    return 1e-10


@pytest.mark.parametrize(
    "input_data",
    {
        "c": 1,
        "d": 2,
        "e": 3,
        "f": 4,
        "g": 5,
        "a": 6,
        "b": 7,
    },
)
@pytest.mark.parametrize("check_columns", [["a", "b", "c"], ["e", "z"], ["x"]])
def test__check_cols_in_series(input_data, check_columns):
    try:
        input_series = pd.Series(input_data)
        _check_cols_in_series(input_series=input_series, columns=check_columns)
    except Exception as e:
        if ("z" in check_columns) or ("x" in check_columns):
            assert isinstance(e, KeyError)


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


@pytest.mark.parametrize(
    "input_vector,expected",
    [
        (np.array([0.0, 0.0, 0.0]), DEFAULT_VECTOR_NORM),
        (np.array([1.0, 1.0, 1.0]), np.array([0.57735027, 0.57735027, 0.57735027])),
        (np.array([-1.0, -2.0, 2.0]), np.array([-0.33333333, -0.66666667, 0.66666667])),
        (np.array([[-1.0, 1.0, 3.0], [1.0, 1.0, 4.0]]), ValueError),
        (np.array([-1.0, 2.0]), ValueError),
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


# @pytest.mark.parametrize(
#     "transmit_vectors,reply_vectors,tt_delays,expected_a",
#     list(zip(TRANSMIT_VECTORS, REPLY_VECTORS, TT_DELAY_SECONDS, A_PARTIALS)),
# )
# def test_calc_partials(
#     transmit_vectors,
#     reply_vectors,
#     tt_delays,
#     expected_a,
#     num_transponders,
#     mean_sv,
#     b_cov,
# ):
#     A_partials, B_cov, _, _ = calc_partials(
#         transmit_vectors=transmit_vectors,
#         reply_vectors=reply_vectors,
#         transponders_mean_sv=mean_sv,
#         delays=tt_delays,
#         num_transponders=num_transponders,
#     )
#     assert np.allclose(A_partials, expected_a)
#     assert np.array_equal(B_cov, b_cov)


# @pytest.mark.parametrize(
#     "gps_dict,transmit_vectors,expected_wm",
#     list(zip(GPS_DATASET[2:], TRANSMIT_VECTORS, WEIGHT_MATRIX)),
# )
# def test_calc_weight_matrix(
#     gps_dict, transmit_vectors, expected_wm, mean_sv, b_cov, travel_times_variance
# ):
#     gps_dataseries = pd.Series(gps_dict)
#     # Add missing data
#     if "z" not in gps_dataseries:
#         gps_dataseries["z"] = 4511065.769
#     if "yy" not in gps_dataseries:
#         gps_dataseries["yy"] = 0.001007144325

#     gps_covariance_matrix = np.array(np.array_split(gps_dataseries[GPS_COV], 3)).astype(
#         "float64"
#     )
#     transmit_uv = np.array([calc_uv(v) for v in transmit_vectors])
#     weight_matrix = calc_weight_matrix(
#         transmit_uv=transmit_uv,
#         gps_covariance_matrix=gps_covariance_matrix,
#         transponders_mean_sv=mean_sv,
#         b_cov=b_cov,
#         travel_times_variance=travel_times_variance,
#     )
#     assert np.allclose(weight_matrix, expected_wm)


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
        (np.ones(shape=(2, 2, 3)), NotImplementedError),
    ],
)
def test_clean_zeros(input_array, expected):
    try:
        result = clean_zeros(input_array=input_array)
        assert np.array_equal(result, expected)
    except Exception as e:
        assert isinstance(e, expected)


def test_calc_lsq_constrained():
    num_transponders = 3
    X, XP, MX, MXP = calc_lsq_constrained(ATWA_SAMPLE, ATWF_SAMPLE, num_transponders)

    # The results are in meters and we'd like to ensure cm
    # precision, so 1e-4 should be very precise,
    # might even be overkill
    assert np.allclose(X, LSQ_RESULT[0], atol=1e-4)
    assert np.allclose(XP, LSQ_RESULT[1], atol=1e-4)
    assert np.allclose(MX, LSQ_RESULT[2], atol=1e-4)
    assert np.allclose(MXP, LSQ_RESULT[3], atol=1e-4)
