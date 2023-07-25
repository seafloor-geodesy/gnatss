import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from nptyping import Float64, NDArray, Shape

from gnatss.ops.solve import _calc_tr_vectors


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
