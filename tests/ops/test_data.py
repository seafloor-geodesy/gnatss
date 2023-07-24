import numpy as np
import pandas as pd
import pytest
from numba.typed import List as NumbaList

from gnatss import constants
from gnatss.configs.solver import ArrayCenter
from gnatss.ops.data import (
    _prep_col_names,
    _split_cov,
    calc_lla_and_enu,
    get_data_inputs,
)

from .. import TEST_DATA_FOLDER


@pytest.fixture()
def all_observations() -> pd.DataFrame:
    obs_csv = TEST_DATA_FOLDER / "test_obs.csv"
    return pd.read_csv(obs_csv)


def test_calc_lla_and_enu(all_observations: pd.DataFrame) -> None:
    array_center = ArrayCenter(lat=0, lon=0, alt=0)
    final_obs = calc_lla_and_enu(all_observations, array_center=array_center)
    assert isinstance(final_obs, pd.DataFrame)
    print(final_obs.columns)
    assert all(
        [
            col in final_obs.columns
            for col in _prep_col_names(constants.GPS_GEODETIC)
            + _prep_col_names(constants.GPS_LOCAL_TANGENT)
        ]
    )


def test__split_cov() -> None:
    cov_values = np.ones(9)

    cov_matrix = _split_cov(cov_values)

    assert cov_matrix.shape == (3, 3)


def test_get_data_inputs(all_observations: pd.DataFrame) -> None:
    data_inputs = get_data_inputs(all_observations)

    assert isinstance(data_inputs, NumbaList)
    assert len(data_inputs) == 4
