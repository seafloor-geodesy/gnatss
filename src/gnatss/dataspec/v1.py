"""DataSpec v1

Based on https://hal.science/hal-04319233/document
"""

from typing import Optional

from pydantic import Field

from .base import BaseData


class DataV1(BaseData):
    # Private attributes for internal use
    # This refers to the field names in the model
    _travel_time: str = "TravelTime"
    _tx_time: str = "T_transmit"
    _rx_time: str = "T_receive"
    _transponder_id: str = "MT_ID"

    # Private attributes that indicate
    # what is the vocabulary for tx, rx, and gnss
    _tx: str = "transmit"
    _tx_code: int = 0
    _rx: str = "receive"
    _rx_code: int = 1
    _gnss: str = "ant"
    _covariance: str = "cov"
    _standard_dev: str = "sig"

    # --- Required Fields ---
    MT_ID: str = Field(..., description="ID of mirror transponder")
    TravelTime: float = Field(..., description="Observed travel time [sec.]")
    # Transducer Transmit
    T_transmit: float = Field(
        ..., description="Transmission time of acoustic signal [sec. from origin]"
    )
    X_transmit: float = Field(..., description="Transducer position at T_transmit in ECEF [m]")
    Y_transmit: float = Field(..., description="Transducer position at T_transmit in ECEF [m]")
    Z_transmit: float = Field(..., description="Transducer position at T_transmit in ECEF [m]")
    # Transducer Receive
    T_receive: float = Field(
        ..., description="Reception time of acoustic signal [sec. from origin]"
    )
    X_receive: float = Field(..., description="Transducer position at T_receive in ECEF [m]")
    Y_receive: float = Field(..., description="Transducer position at T_receive in ECEF [m]")
    Z_receive: float = Field(..., description="Transducer position at T_receive in ECEF [m]")
    # --- Optional Fields ---
    # Antenna (GNSS) Transmit
    ant_X0: Optional[float] = Field(None, description="GNSS position at T_transmit in ECEF [m]")
    ant_Y0: Optional[float] = Field(None, description="GNSS position at T_transmit in ECEF [m]")
    ant_Z0: Optional[float] = Field(None, description="GNSS position at T_transmit in ECEF [m]")
    # Standard Deviation Antenna (GNSS) Transmit
    ant_sigX0: Optional[float] = Field(
        None,
        description="Standard deviation of GNSS position at T_receive in ECEF",
    )
    ant_sigY0: Optional[float] = Field(
        None, description="Standard deviation of GNSS position at T_receive in ECEF"
    )
    ant_sigZ0: Optional[float] = Field(
        None, description="Standard deviation of GNSS position at T_receive in ECEF"
    )
    # Covariance Matrix Antenna (GNSS) Transmit
    ant_cov_XX0: Optional[float] = Field(
        None, description="XX covariance value of GNSS position at T_receive"
    )
    ant_cov_XY0: Optional[float] = Field(
        None, description="XY covariance value of GNSS position at T_receive"
    )
    ant_cov_XZ0: Optional[float] = Field(
        None, description="XZ covariance value of GNSS position at T_receive"
    )
    ant_cov_YX0: Optional[float] = Field(
        None, description="YX covariance value of GNSS position at T_receive"
    )
    ant_cov_YY0: Optional[float] = Field(
        None, description="YY covariance value of GNSS position at T_receive"
    )
    ant_cov_YZ0: Optional[float] = Field(
        None, description="YZ covariance value of GNSS position at T_receive"
    )
    ant_cov_ZX0: Optional[float] = Field(
        None, description="ZX covariance value of GNSS position at T_receive"
    )
    ant_cov_ZY0: Optional[float] = Field(
        None, description="ZY covariance value of GNSS position at T_receive"
    )
    ant_cov_ZZ0: Optional[float] = Field(
        None, description="ZZ covariance value of GNSS position at T_receive"
    )
    # Platform Transmit
    roll0: Optional[float] = Field(
        None,
        description="Roll at T_transmit [deg.]; *rotation around 'forward' axis in ATD offset",
    )
    pitch0: Optional[float] = Field(
        None,
        description="Pitch at T_transmit [deg.]; *rotation around 'rightward' axis in ATD offset",
    )
    heading0: Optional[float] = Field(
        None,
        description="Heading at T_transmit [deg.]; *rotation around 'downward' axis in ATD offset",
    )
    # Covariance Platform Transmit
    cov_RR0: Optional[float] = Field(None, description="RR covariance value at T_transmit")
    cov_PP0: Optional[float] = Field(None, description="PP covariance value at T_transmit")
    cov_HH0: Optional[float] = Field(None, description="HH covariance value at T_transmit")
    cov_RP0: Optional[float] = Field(None, description="RP covariance value at T_transmit")
    cov_RH0: Optional[float] = Field(None, description="RH covariance value at T_transmit")
    cov_PR0: Optional[float] = Field(None, description="PR covariance value at T_transmit")
    cov_PH0: Optional[float] = Field(None, description="PH covariance value at T_transmit")
    cov_HR0: Optional[float] = Field(None, description="HR covariance value at T_transmit")
    cov_HP0: Optional[float] = Field(None, description="HP covariance value at T_transmit")
    # Antenna (GNSS) Receive
    ant_X1: Optional[float] = Field(None, description="GNSS position at T_receive in ECEF [m]")
    ant_Y1: Optional[float] = Field(None, description="GNSS position at T_receive in ECEF [m]")
    ant_Z1: Optional[float] = Field(None, description="GNSS position at T_receive in ECEF [m]")
    # Standard Deviation Antenna (GNSS) Receive
    ant_sigX1: Optional[float] = Field(
        None,
        description="Standard deviation of GNSS position at T_receive in ECEF",
    )
    ant_sigY1: Optional[float] = Field(
        None, description="Standard deviation of GNSS position at T_receive in ECEF"
    )
    ant_sigZ1: Optional[float] = Field(
        None, description="Standard deviation of GNSS position at T_receive in ECEF"
    )
    # Covariance Matrix Antenna (GNSS) Receive
    ant_cov_XX1: Optional[float] = Field(
        None, description="XX covariance value of GNSS position at T_receive"
    )
    ant_cov_XY1: Optional[float] = Field(
        None, description="XY covariance value of GNSS position at T_receive"
    )
    ant_cov_XZ1: Optional[float] = Field(
        None, description="XZ covariance value of GNSS position at T_receive"
    )
    ant_cov_YX1: Optional[float] = Field(
        None, description="YX covariance value of GNSS position at T_receive"
    )
    ant_cov_YY1: Optional[float] = Field(
        None, description="YY covariance value of GNSS position at T_receive"
    )
    ant_cov_YZ1: Optional[float] = Field(
        None, description="YZ covariance value of GNSS position at T_receive"
    )
    ant_cov_ZX1: Optional[float] = Field(
        None, description="ZX covariance value of GNSS position at T_receive"
    )
    ant_cov_ZY1: Optional[float] = Field(
        None, description="ZY covariance value of GNSS position at T_receive"
    )
    ant_cov_ZZ1: Optional[float] = Field(
        None, description="ZZ covariance value of GNSS position at T_receive"
    )
    # Platform Receive
    roll1: Optional[float] = Field(
        None,
        description="Roll at T_receive [deg.]; *rotation around 'forward' axis in ATD offset",
    )
    pitch1: Optional[float] = Field(
        None,
        description="Pitch at T_receive [deg.]; *rotation around 'rightward' axis in ATD offset",
    )
    heading1: Optional[float] = Field(
        None,
        description="Heading at T_receive [deg.]; *rotation around 'downward' axis in ATD offset",
    )
    # Covariance Platform Receive
    cov_RR1: Optional[float] = Field(None, description="RR covariance value at T_receive")
    cov_PP1: Optional[float] = Field(None, description="PP covariance value at T_receive")
    cov_HH1: Optional[float] = Field(None, description="HH covariance value at T_receive")
    cov_RP1: Optional[float] = Field(None, description="RP covariance value at T_receive")
    cov_RH1: Optional[float] = Field(None, description="RH covariance value at T_receive")
    cov_PR1: Optional[float] = Field(None, description="PR covariance value at T_receive")
    cov_PH1: Optional[float] = Field(None, description="PH covariance value at T_receive")
    cov_HR1: Optional[float] = Field(None, description="HR covariance value at T_receive")
    cov_HP1: Optional[float] = Field(None, description="HP covariance value at T_receive")
