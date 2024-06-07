from __future__ import annotations

from pydantic import BaseModel


class BaseData(BaseModel):
    # Private attributes for internal use
    # This refers to the field names in the model
    _travel_time: str
    _tx_time: str
    _rx_time: str
    _transponder_id: str

    # Private attributes that indicate
    # what is the vocabulary for tx, rx, and gnss
    _tx: str
    _tx_code: int
    _rx: str
    _rx_code: int
    _gnss: str
    _covariance: str
    _standard_dev: str
