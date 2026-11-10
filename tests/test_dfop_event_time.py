"""DFOP event clocks and exact transmit-key joins must survive stale GNSS snapshots."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gnatss import constants
from gnatss.loaders import _round_time_precision, load_dfo
from gnatss.ops.data import standardize_data
from gnatss.utilities.time import AstroTime


def ping(event_time, lag):
    return [
        {
            "event": "interrogation",
            "time": {"common": event_time},
            "observations": {"GNSS": {"time": {"common": event_time - lag}}},
        },
        {
            "event": "range",
            "time": {"common": event_time + 2.125},
            "range": {"range": 2.255, "cn": "IR5209"},
        },
    ]


@pytest.mark.parametrize("fraction", [0, 0.1, 0.2, 0.3, 0.8, 0.9])
@pytest.mark.parametrize("precision", [3, 6])
def test_event_time_and_fractional_join_are_consistent(tmp_path, fraction, precision):
    event_time = 1788147712 + fraction
    rows = ping(event_time, 0.2)
    path = tmp_path / "DFOP00.raw"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    loaded = load_dfo(SimpleNamespace(site_id="NCC1"), [str(path)], time_round=precision)
    event = loaded[loaded[constants.DATA_SPEC.transponder_id] == "T_transmit"].iloc[0]
    reply = loaded[loaded[constants.DATA_SPEC.transponder_id] == "NCC1-1"].iloc[0]
    expected = round(
        float(AstroTime(event_time, format="unix", scale="tt").rangea_j2000), precision
    )
    assert event[constants.GPS_TIME] == expected
    assert reply[constants.DATA_SPEC.tx_time] == event[constants.GPS_TIME]
    assert reply[constants.DATA_SPEC.travel_time] == pytest.approx(2.125)
    assert reply[constants.GPS_TIME] - reply[constants.DATA_SPEC.tx_time] == pytest.approx(
        2.125, abs=0.0011
    )
    assert len(standardize_data(loaded.copy())) == 1


@pytest.mark.parametrize(
    "clock", [None, {}, {"common": None}, {"common": float("nan")}, {"common": True}]
)
def test_no_silent_fallback_to_snapshot_for_missing_event_clock(tmp_path, clock):
    rows = ping(1788147712.0, 0.2)
    rows[0]["time"] = clock
    path = tmp_path / "DFOP00.raw"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="event time.common"):
        load_dfo(SimpleNamespace(site_id="NCC1"), [str(path)])


def test_precision_normalizes_both_join_columns_and_preserves_nan():
    frame = pd.DataFrame(
        {
            constants.GPS_TIME: [841419729.8999999, 841419732.025],
            constants.DATA_SPEC.tx_time: [np.nan, 841419729.8999999],
        }
    )
    rounded = _round_time_precision(frame, 6)
    assert rounded[constants.GPS_TIME].iloc[0] == rounded[constants.DATA_SPEC.tx_time].iloc[1]
    assert np.isnan(rounded[constants.DATA_SPEC.tx_time].iloc[0])


@pytest.mark.parametrize("precision", [3, 6])
def test_precision_supports_position_only_data(precision):
    frame = pd.DataFrame({constants.GPS_TIME: [841419729.8999999]})
    rounded = _round_time_precision(frame, precision)
    assert rounded[constants.GPS_TIME].iloc[0] == round(841419729.8999999, precision)


@pytest.mark.parametrize("precision", [None, 0])
def test_disabled_precision_preserves_original_values(precision):
    frame = pd.DataFrame(
        {
            constants.GPS_TIME: [841419729.8999999],
            constants.DATA_SPEC.tx_time: [841419729.8999999],
        }
    )
    expected = frame.copy()
    pd.testing.assert_frame_equal(_round_time_precision(frame, precision), expected)
