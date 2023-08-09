import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList
from numpy.testing import assert_allclose

from gnatss.ops.solve import (
    __get_diagonal,
    _calc_cov,
    _calc_partial_derivatives,
    _calc_tr_vectors,
    _calc_unit_vectors,
    _calc_weight_fac,
    _calc_weight_mat,
    _setup_ab,
    calc_tt_residual,
    calc_twtt_model,
    perform_solve,
    solve_transponder_locations,
)

# TODO: Use real data examples for testing

N_TRANSPONDERS = 3


@given(
    transponders_xyz=st_arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=6), st.integers(min_value=3, max_value=3)
        ),
        elements=st.floats(min_value=0.0, max_value=100.0),
    ),
    transmit_xyz=st_arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=1.0, max_value=10.0),
    ),
    reply_xyz=st_arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=6), st.integers(min_value=3, max_value=3)
        ),
        elements=st.floats(min_value=0.0, max_value=10.0),
    ),
)
@settings(deadline=None)
def test__calc_tr_vectors(
    transponders_xyz: NDArray[Shape["*,3"], Float64],  # noqa
    transmit_xyz: NDArray[Shape["3"], Float64],
    reply_xyz: NDArray[Shape["*,3"], Float64],  # noqa
) -> None:
    """Test calculate transmit and reply vectors"""
    try:
        transmit_vectors, reply_vectors = _calc_tr_vectors(
            transponders_xyz=transponders_xyz,
            transmit_xyz=transmit_xyz,
            reply_xyz=reply_xyz,
        )
        ext_tv = transponders_xyz - transmit_xyz
        ext_rv = transponders_xyz - reply_xyz

        assert np.array_equal(transmit_vectors, ext_tv)
        assert np.array_equal(reply_vectors, ext_rv)
    except AssertionError:
        # The shape should be different
        assert transponders_xyz.shape[0] != reply_xyz.shape[0]


@given(
    vectors=st_arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=6), st.integers(min_value=3, max_value=3)
        ),
        elements=st.floats(min_value=0.0, max_value=100.0),
    )
)
@settings(deadline=None)
def test__calc_unit_vectors(vectors) -> None:
    """Test calculate unit vectors"""
    result_array = _calc_unit_vectors(vectors)

    assert result_array.shape == vectors.shape


@given(
    n_transponders=st.integers(min_value=N_TRANSPONDERS, max_value=N_TRANSPONDERS),
)
@settings(deadline=None)
def test__calc_partial_derivatives(n_transponders) -> None:
    """Test calculate partial derivatives"""
    transmit_uv = np.random.rand(n_transponders, 3)
    reply_uv = np.random.rand(n_transponders, 3)
    transponders_mean_sv = np.random.rand(n_transponders) + 1052.0

    expected_res = (transmit_uv + reply_uv) / transponders_mean_sv

    d_arr = _calc_partial_derivatives(transmit_uv, reply_uv, transponders_mean_sv)

    assert d_arr.shape == (transponders_mean_sv.shape[0], 3)
    assert np.allclose(d_arr, expected_res)


@given(
    # Currently hard-coded to 3 transponders
    n_transponders=st.integers(min_value=3, max_value=3),
)
@settings(deadline=None)
def test__setup_ab(n_transponders) -> None:
    delays = np.random.rand(n_transponders)
    num_transponders = n_transponders
    partial_derivatives = np.random.rand(n_transponders, 3)

    A_partials, B_cov = _setup_ab(delays, num_transponders, partial_derivatives)

    assert A_partials.shape == (num_transponders, num_transponders * 3)
    assert B_cov.shape == (num_transponders, num_transponders * 3)


@given(
    # Currently hard-coded to 3 transponders
    n_transponders=st.integers(min_value=3, max_value=3),
    travel_times_variance=st.floats(min_value=1e-10, max_value=0.1),
)
@settings(deadline=None)
def test__calc_cov(n_transponders, travel_times_variance) -> None:
    """Test calculate covariance"""
    transmit_uv = np.random.rand(n_transponders, 3)
    gps_covariance_matrix = np.random.rand(3, 3)
    transponders_mean_sv = np.random.rand(n_transponders) + 1052.0
    covariance_matrix = _calc_cov(
        transmit_uv, gps_covariance_matrix, travel_times_variance, transponders_mean_sv
    )

    assert covariance_matrix.shape == (n_transponders, n_transponders)


@given(
    transponders_mean_sv=st_arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(min_value=1, max_value=6)),
        elements=st.floats(min_value=1052.0, max_value=1053.0),
    )
)
@settings(deadline=None)
def test__calc_weight_fac(transponders_mean_sv):
    """Testing calculate weight factor"""
    expected_output = 2.0 / (transponders_mean_sv**2)
    assert_allclose(_calc_weight_fac(transponders_mean_sv), expected_output)


@given(
    covariance_std=st_arrays(
        dtype=np.float64, shape=(2, 2), elements=st.integers(min_value=1, max_value=6)
    )
)
@settings(deadline=None)
def test__calc_weight_mat(covariance_std):
    if np.linalg.det(covariance_std) == 0:
        return
    expected_result = np.linalg.inv(covariance_std)
    assert_allclose(_calc_weight_mat(covariance_std), expected_result)


@given(
    array=st_arrays(
        dtype=np.float64, shape=(3, 3), elements=st.integers(min_value=1, max_value=6)
    )
)
@settings(deadline=None)
def test___get_diagonal(array):
    expected_result = array.diagonal()
    assert_allclose(__get_diagonal(array), expected_result)


@pytest.mark.skip(reason="Need to figure out a good way to test this")
def test_calc_twtt_model():
    # Test case 1
    transmit_vectors = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    reply_vectors = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=np.float64)
    transponders_mean_sv = np.array([1500, 1500, 1500], dtype=np.float64)
    expected_twtt_model = np.array([0.006, 0.008, 0.01])
    assert_allclose(
        calc_twtt_model(transmit_vectors, reply_vectors, transponders_mean_sv),
        expected_twtt_model,
        rtol=1e-3,
    )

    # Test case 2
    transmit_vectors = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    reply_vectors = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=np.float64)
    transponders_mean_sv = np.array([1600, 1600, 1600], dtype=np.float64)
    expected_twtt_model = np.array([0.005625, 0.0075, 0.009375])
    assert_allclose(
        calc_twtt_model(transmit_vectors, reply_vectors, transponders_mean_sv),
        expected_twtt_model,
        rtol=1e-3,
    )


@pytest.mark.skip(reason="Need to figure out a good way to test this")
def test_calc_tt_residual():
    # Test case 1
    delays = np.array([0.007, 0.009, 0.011])
    transponder_delays = np.array([0.001, 0.001, 0.001])
    twtt_model = np.array([0.006, 0.008, 0.01])
    expected_residual = np.array([0.001, 0.001, 0.001])
    assert_allclose(
        calc_tt_residual(delays, transponder_delays, twtt_model),
        expected_residual,
        rtol=1e-3,
    )

    # Test case 2
    delays = np.array([0.007, 0.009, 0.011])
    transponder_delays = np.array([0.002, 0.002, 0.002])
    twtt_model = np.array([0.006, 0.008, 0.01])
    expected_residual = np.array([-0.001, -0.001, -0.001])
    assert_allclose(
        calc_tt_residual(delays, transponder_delays, twtt_model),
        expected_residual,
        rtol=1e-3,
    )


@pytest.mark.skip(reason="Need to figure out a good way to test this")
def test_solve_transponder_locations():
    # Define test inputs
    transmit_xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    reply_xyz = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    gps_covariance_matrix = np.eye(3)
    observed_delays = np.array([1.0, 2.0, 3.0])
    transponders_xyz = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    transponders_delay = np.array([0.1, 0.2, 0.3])
    transponders_mean_sv = np.array([1500.0, 1500.0, 1500.0])
    travel_times_variance = 0.1

    # Define expected outputs
    expected_atwa = np.array(
        [
            [1.5e-06, -1.5e-06, 0.0],
            [-1.5e-06, 3.0e-06, -1.5e-06],
            [0.0, -1.5e-06, 1.5e-06],
        ]
    )
    expected_atwf = np.array([0.0, 0.0, 0.0])
    expected_tt_residual = np.array([-0.1, -0.2, -0.3])
    expected_sigma_delay = np.array([0.00070711, 0.00070711, 0.00070711])

    # Call the function
    atwa, atwf, tt_residual, sigma_delay = solve_transponder_locations(
        transmit_xyz,
        reply_xyz,
        gps_covariance_matrix,
        observed_delays,
        transponders_xyz,
        transponders_delay,
        transponders_mean_sv,
        travel_times_variance,
    )

    # Check the outputs
    assert_allclose(atwa, expected_atwa)
    assert_allclose(atwf, expected_atwf)
    assert_allclose(tt_residual, expected_tt_residual)
    assert_allclose(sigma_delay, expected_sigma_delay)


@pytest.mark.skip(reason="Need to figure out a good way to test this")
def test_perform_solve():
    # Define test data
    transponders_mean_sv = np.array([1500.0, 1500.0, 1500.0])
    transponders_xyz = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1000.0], [1000.0, 0.0, 0.0]]
    )
    transponders_delay = np.array([0.0, 0.0, 0.0])
    travel_times_variance = 1.0

    data_inputs = NumbaList()
    for i in range(10):
        transmit_xyz = np.random.rand(3)
        reply_xyz = np.random.rand(3)
        gps_covariance_matrix = np.eye(3)
        observed_delays = np.random.rand(3)
        data_inputs.append(
            (transmit_xyz, reply_xyz, gps_covariance_matrix, observed_delays)
        )

    # Call the function
    results = perform_solve(
        data_inputs,
        transponders_mean_sv,
        transponders_xyz,
        transponders_delay,
        travel_times_variance,
    )

    # Check the output
    assert len(results) == len(data_inputs)
    for result in results:
        assert len(result) == 4
        assert result[0].shape == (3, 3)
        assert result[1].shape == (3,)
        assert result[2].shape == (3,)
        assert isinstance(result[3], float)
