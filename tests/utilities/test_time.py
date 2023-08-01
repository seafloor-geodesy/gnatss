from gnatss.utilities.time import AstroTime, erfa

DATETIME_FORMAT = "%Y-%b-%dT%H:%M:%S.%f"


def test_unix_j2000_time_format():
    """Tests custom time format unix_j2000"""

    first_jan_2000_noon_unix_j2000 = AstroTime(0.0, format="unix_j2000")
    first_jan_2000_noon_utc = AstroTime.strptime(
        "2000-Jan-01T12:00:00.000", DATETIME_FORMAT, scale="utc"
    )
    assert (
        first_jan_2000_noon_unix_j2000.unix_j2000
        == first_jan_2000_noon_utc.unix_j2000
        == 0.0
    )

    # Number of seconds elapsed from 01-Jan-2000-noon-utc to 02_Feb-2004-noon-utc
    secs_elapsed_till_second_feb_2004_noon = float(
        ((365 + 365 + 365 + 366) + 31 + 1) * erfa.DAYSEC
    )

    second_feb_2004_noon_unix_j2000 = AstroTime(
        0.0 + secs_elapsed_till_second_feb_2004_noon, format="unix_j2000"
    )
    second_feb_2004_noon_utc = AstroTime.strptime(
        "2004-Feb-02T12:00:00.000", DATETIME_FORMAT, scale="utc"
    )
    assert (
        second_feb_2004_noon_unix_j2000.unix_j2000
        == second_feb_2004_noon_utc.unix_j2000
        == secs_elapsed_till_second_feb_2004_noon
    )
