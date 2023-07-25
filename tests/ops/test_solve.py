import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from nptyping import Float64, NDArray, Shape

from gnatss.ops.solve import _calc_tr_vectors


@given(
    n_instruments=st.integers(min_value=1, max_value=6),
    transmit_xyz=st_arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=1.0, max_value=10.0),
    ),
)
@settings(deadline=None)
def test__calc_tr_vectors(
    n_instruments: int, transmit_xyz: NDArray[Shape["3"], Float64]
) -> None:
    """Test calculate transmit and reply vectors"""
    transponders_xyz = np.random.rand(n_instruments, 3)
    reply_xyz = np.random.rand(n_instruments, 3)

    transmit_vectors, reply_vectors = _calc_tr_vectors(
        transponders_xyz=transponders_xyz,
        transmit_xyz=transmit_xyz,
        reply_xyz=reply_xyz,
    )
    ext_tv = transponders_xyz - transmit_xyz
    ext_rv = transponders_xyz - reply_xyz

    assert transmit_vectors.shape == (n_instruments, 3)
    assert reply_vectors.shape == (n_instruments, 3)
    assert np.array_equal(transmit_vectors, ext_tv)
    assert np.array_equal(reply_vectors, ext_rv)
