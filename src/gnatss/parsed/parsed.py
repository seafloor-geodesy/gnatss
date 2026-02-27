from __future__ import annotations

import pandas as pd

from .. import constants


def get_parsed_rph(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reformats the INS RPH data from 2-minute parsed QC data.
    These data have only rough estimates of RPH at ping transmit
    and reply, so we cannot interpolate them. There is also no
    covariance data, so we insert dummy values to approximate
    "normal" uncertainties.

    Parameters
    ----------
    raw_df :  DataFrame
        Pandas Dataframe containing parsed data at ping transmit/reply

    Returns
    -------
    cov_rph_twtt : pd.DataFrame
        RPH data approximated at ping transmit/reply
    """
    # Grab desired columns from raw data.
    rph_twtt = raw_df[[
        constants.DATA_SPEC.transponder_id,
        constants.RPH_TIME,
        constants.DATA_SPEC.tx_time,
        constants.DATA_SPEC.travel_time,
        *constants.RPH_LOCAL_TANGENTS
    ]]
    
    # Create columns of dummy covariance
    cov = pd.DataFrame(index=rph_twtt.index,columns=constants.PLATFORM_COV_RPH)
    cov[constants.PLATFORM_COV_RPH] = [0.03,0.00,0.00,0.00,0.03,0.00,0.00,0.00,0.15]

    return pd.concat([rph_twtt,cov],axis=1)


def get_parsed_ant_positions(
    raw_df: pd.DataFrame,
    gps_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculates the approximate antenna positions at ping transmit
    and reply using a less robust method than the standard Kalman
    Filter in order to accomodate sparsely sampled data.

    This routine is meant to process 2-minute parsed QC data
    telemetered to shore in real time and operates in two possible
    modes. If processed GNSS antenna positions are provided this
    routine will use them. Otherwise it defaults to the positions
    logged by the INS.

    If no processed GNSS antenna positions are provided, we will
    also insert dummy values for position uncertainty.

    Parameters
    ----------
    raw_df :  pd.DataFrame
        Pandas Dataframe containing parsed data at ping transmit/reply
    gps_df :  pd.DataFrame
        Pandas Dataframe containing GPS solutions Novatel Level-1 data

    Returns
    -------
    pos_twtt : pd.DataFrame
        Pandas Dataframe containing the resulting
        antenna positions, covariance, and standard deviation
    """
    
    # Check to see if GPS positions were provided
    if gps_df is None:
        # No GPS positions
        # Assume that the real-time antenna positions are our best guess
        pos_twtt_df = raw_df[[
            constants.DATA_SPEC.transponder_id,
            constants.GPS_TIME,
            constants.DATA_SPEC.tx_time,
            constants.DATA_SPEC.travel_time,
            "ant_x1",
            "ant_y1",
            "ant_z1",
            "ant_x0",
            "ant_y0",
            "ant_z0",
        ]]
        pos_twtt_df = pos_twtt_df.rename(columns={"ant_x1":"ant_x", "ant_y1":"ant_y", "ant_z1":"ant_z"})
        
        # Define dummy values for the uncertainties
        pos_twtt_df[constants.ANT_GPS_GEOCENTRIC_STD] = [0.01,0.01,0.01]
    else:
        # GPS positions provided
        # Merge provided GPS positions into the dataframe.
        pos_twtt_df = pd.concat(
            [raw_df.merge(gps_df,on=constants.TIME_J2000),
             raw_df.merge(gps_df.rename(columns={constants.TIME_J2000:constants.DATA_SPEC.tx_time}),on=constants.DATA_SPEC.tx_time)],
             ignore_index=True
        )

        # The GPS positions will not have values at the specific epochs for ping reply.
        # To accomodate this, we assume that the INS position offsets between
        # transmit and reply are accurate and add these offsets to the GPS position
        # at transmit.
        pos_twtt_df["ant_x"] = pos_twtt_df["ant_x"]+pos_twtt_df["ant_x1"]-pos_twtt_df["ant_x0"]
        pos_twtt_df["ant_y"] = pos_twtt_df["ant_y"]+pos_twtt_df["ant_y1"]-pos_twtt_df["ant_y0"]
        pos_twtt_df["ant_z"] = pos_twtt_df["ant_z"]+pos_twtt_df["ant_z1"]-pos_twtt_df["ant_z0"]

    # Calculate the covariance matrix
    # We assume no off-diagonal terms
    pos_twtt_df["ant_cov_xx"] = pos_twtt_df["ant_sigx"].pow(2)
    pos_twtt_df["ant_cov_yy"] = pos_twtt_df["ant_sigy"].pow(2)
    pos_twtt_df["ant_cov_zz"] = pos_twtt_df["ant_sigz"].pow(2)

    pos_twtt_df["ant_cov_xy"] = 0.0
    pos_twtt_df["ant_cov_xz"] = 0.0
    pos_twtt_df["ant_cov_yx"] = 0.0
    pos_twtt_df["ant_cov_yz"] = 0.0
    pos_twtt_df["ant_cov_zx"] = 0.0
    pos_twtt_df["ant_cov_zy"] = 0.0

    columns_out = [
        constants.DATA_SPEC.transponder_id,
        constants.GPS_TIME,
        constants.DATA_SPEC.tx_time,
        constants.DATA_SPEC.travel_time,
        *constants.ANT_GPS_GEOCENTRIC,
        *constants.ANT_GPS_GEOCENTRIC_STD,
        *constants.ANT_GPS_COV
    ]

    return pos_twtt_df[columns_out]
