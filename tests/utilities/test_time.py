import pytest

from gnatss.utilities.time import AstroTime, erfa, isclose

DATETIME_FORMAT = "%Y-%b-%d %H:%M:%S.%f"


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
    secs_elapsed_till_second_feb_2004_noon = float(
        ((366 + 365 + 365 + 365) + 31 + 1) * erfa.DAYSEC
    )

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


@pytest.mark.parametrize(
    "unix_j2000, expected_iso",
    [
        (712215465.0, "2022-07-27 17:37:45.000"),
        (712215465, "2022-07-27 17:37:45.000"),
        ("712215465.0", "2022-07-27 17:37:45.000"),
        (712215465, "2022-07-27 17:37:45.000"),
        ("2022-07-27 17:37:45.000", None),
    ],
)
def test_unix_j2000_convert_to_iso(unix_j2000, expected_iso):
    if unix_j2000 == "2022-07-27 17:37:45.000":
        with pytest.raises(ValueError):
            AstroTime(unix_j2000, format="unix_j2000").iso
    else:
        assert AstroTime(unix_j2000, format="unix_j2000").iso == expected_iso
