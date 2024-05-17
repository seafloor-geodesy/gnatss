import numpy as np
import pytest
from astropy.units import isclose

from gnatss.utilities.time import AstroTime, erfa, gps_ws_time_to_astrotime

DATETIME_FORMAT = "%Y-%b-%d %H:%M:%S.%f"


@pytest.fixture(
    params=[
        (
            np.array([0, 1042, 2307]),
            np.array([0.0, 561600.0, 288327.0]),
            "expected_result",
            np.array([[-630763200.0, 0.0, 764798727.0]]),
        ),
        (
            np.array([0.0, 1042.0, 2307.0]),
            np.array([0, 561600, 288327]),
            "expected_result",
            np.array([[-630763200.0, 0.0, 764798727.0]]),
        ),
        (
            np.array([0, float(0 + float(1 / 7))]),  # Adding 1 day = 0.14285... weeks
            np.array([0.0, 0.0]),
            "expected_result",
            np.array(
                [[-630763200.0, float(-630763200.0 + 86400)]]
            ),  # Expect to see additional 86400 seconds
        ),
        (
            np.array(["definitely not an int", 1042, 2307]),
            np.array([0.0, 561600.0, 288327.0]),
            "TypeError",
            np.array([[-630763200.0, 0.0, 764798727.0]]),
        ),
        (
            np.array([0, 1042, 2307]),
            np.array(["definitely not a float", 561600.0, 288327.0]),
            "ValueError",
            np.array([[-630763200.0, 0.0, 764798727.0]]),
        ),
    ]
)
def gps_ws_time_to_astrotime_unittests(request):
    return request.param


@pytest.fixture(
    params=[
        ((2220, 322665), 712215465.0, "2022-07-27 17:37:45.000"),
        (("2220", "322665"), 712215465, "2022-07-27 17:37:45.000"),
        (("2220", 322665.0), "712215465.0", "2022-07-27 17:37:45.000"),
        ((2220.0, 322665.0), 712215465, "2022-07-27 17:37:45.000"),
        (("sometext", 322665.0), "2022-07-27 17:37:45.000", None),
    ]
)
def test_unix_j2000_convert_to_iso_unittests(request):
    return request.param


def test_unix_j2000_time_format():
    """
    Tests custom time format unix_j2000.
    Calculated times are verified with millisecond level tolerance.
    """

    first_jan_2000_noon_unix_j2000 = AstroTime(0.0, format="unix_j2000")
    initial_unix_j2000_tt = AstroTime.strptime(
        "2000-Jan-01 12:00:00.000", DATETIME_FORMAT, scale="tt"
    )
    initial_unix_j2000_utc = AstroTime.strptime(
        "2000-Jan-01 11:58:55.816", DATETIME_FORMAT, scale="utc"
    )
    assert isclose(first_jan_2000_noon_unix_j2000.unix_j2000, 0.0, atol=1e-3)
    assert isclose(initial_unix_j2000_tt.unix_j2000, 0.0, atol=1e-3)
    assert isclose(initial_unix_j2000_utc.unix_j2000, 0.0, atol=1e-3)

    # Number of seconds elapsed from 01-Jan-2000-noon-tt to 02_Feb-2004-noon-tt
    secs_elapsed_till_second_feb_2004_noon = float(((366 + 365 + 365 + 365) + 31 + 1) * erfa.DAYSEC)

    second_feb_2004_noon_unix_j2000 = AstroTime(
        0.0 + secs_elapsed_till_second_feb_2004_noon, format="unix_j2000"
    )
    elapsed_unix_j2000_tt = AstroTime.strptime(
        "2004-Feb-02 12:00:00.000", DATETIME_FORMAT, scale="tt"
    )
    elapsed_unix_j2000_utc = AstroTime.strptime(
        "2004-Feb-02 11:58:55.816", DATETIME_FORMAT, scale="utc"
    )
    assert isclose(
        second_feb_2004_noon_unix_j2000.unix_j2000,
        secs_elapsed_till_second_feb_2004_noon,
        atol=1e-3,
    )
    assert isclose(
        elapsed_unix_j2000_tt.unix_j2000,
        secs_elapsed_till_second_feb_2004_noon,
        atol=1e-3,
    )
    assert isclose(
        elapsed_unix_j2000_utc.unix_j2000,
        secs_elapsed_till_second_feb_2004_noon,
        atol=1e-3,
    )


def test_unix_j2000_convert_to_iso(test_unix_j2000_convert_to_iso_unittests):
    _, unix_j2000, expected_iso = test_unix_j2000_convert_to_iso_unittests
    if unix_j2000 == "2022-07-27 17:37:45.000":
        with pytest.raises(ValueError):
            AstroTime(unix_j2000, format="unix_j2000").iso
    else:
        assert AstroTime(unix_j2000, format="unix_j2000").iso == expected_iso


def test_gps_ws_time_to_astrotime(gps_ws_time_to_astrotime_unittests):
    (
        week_array,
        sec_array,
        result_type,
        expected_j2000_array,
    ) = gps_ws_time_to_astrotime_unittests
    if result_type == "expected_result":
        returned_astrotime = gps_ws_time_to_astrotime(week_array, sec_array)
        assert isinstance(returned_astrotime, AstroTime)
        returned_j2000 = returned_astrotime.unix_j2000
        assert np.allclose(expected_j2000_array, returned_j2000)
    elif result_type == "TypeError":
        with pytest.raises(TypeError):
            returned_astrotime = gps_ws_time_to_astrotime(week_array, sec_array)

    elif result_type == "ValueError":
        with pytest.raises(ValueError):
            returned_astrotime = gps_ws_time_to_astrotime(week_array, sec_array)
    else:
        assert False
