import pytest
from nptyping import Float64, NDArray, Shape, Str0
from numpy import array_equal
from pandas import DataFrame

from gnatss.constants import TIME_J2000
from gnatss.ops.qc import _compute_ticks_and_labels


@pytest.mark.parametrize(
    "data, time_col, expected_ticks, expected_labels",
    [
        (
            [700008333.0, 700008334.0, 700008340.0],
            TIME_J2000,
            [
                700008333.0,
                700008334.0,
                700008335.0,
                700008336.0,
                700008337.0,
                700008338.0,
                700008339.0,
                700008340.0,
                700008341.0,
            ],
            [
                "2022-03-08T10:45:33",
                "2022-03-08T10:45:34",
                "2022-03-08T10:45:35",
                "2022-03-08T10:45:36",
                "2022-03-08T10:45:37",
                "2022-03-08T10:45:38",
                "2022-03-08T10:45:39",
                "2022-03-08T10:45:40",
                "2022-03-08T10:45:41",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700008349.0],
            TIME_J2000,
            [700008330.0, 700008335.0, 700008340.0, 700008345.0, 700008350.0],
            [
                "2022-03-08T10:45:30",
                "2022-03-08T10:45:35",
                "2022-03-08T10:45:40",
                "2022-03-08T10:45:45",
                "2022-03-08T10:45:50",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700008788.0],
            TIME_J2000,
            [
                7.0000830e08,
                7.0000836e08,
                7.0000842e08,
                7.0000848e08,
                7.0000854e08,
                7.0000860e08,
                7.0000866e08,
                7.0000872e08,
                7.0000878e08,
                7.0000884e08,
            ],
            [
                "2022-03-08T10:45:00",
                "2022-03-08T10:46:00",
                "2022-03-08T10:47:00",
                "2022-03-08T10:48:00",
                "2022-03-08T10:49:00",
                "2022-03-08T10:50:00",
                "2022-03-08T10:51:00",
                "2022-03-08T10:52:00",
                "2022-03-08T10:53:00",
                "2022-03-08T10:54:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700009240.0],
            TIME_J2000,
            [7.000083e08, 7.000086e08, 7.000089e08, 7.000092e08, 7.000095e08],
            [
                "2022-03-08T10:45:00",
                "2022-03-08T10:50:00",
                "2022-03-08T10:55:00",
                "2022-03-08T11:00:00",
                "2022-03-08T11:05:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700012850.0],
            TIME_J2000,
            [7.000074e08, 7.000092e08, 7.000110e08, 7.000128e08, 7.000146e08],
            [
                "2022-03-08T10:30:00",
                "2022-03-08T11:00:00",
                "2022-03-08T11:30:00",
                "2022-03-08T12:00:00",
                "2022-03-08T12:30:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700035390.0],
            TIME_J2000,
            [
                7.000056e08,
                7.000092e08,
                7.000128e08,
                7.000164e08,
                7.000200e08,
                7.000236e08,
                7.000272e08,
                7.000308e08,
                7.000344e08,
                7.000380e08,
            ],
            [
                "2022-03-08T10:00",
                "2022-03-08T11:00",
                "2022-03-08T12:00",
                "2022-03-08T13:00",
                "2022-03-08T14:00",
                "2022-03-08T15:00",
                "2022-03-08T16:00",
                "2022-03-08T17:00",
                "2022-03-08T18:00",
                "2022-03-08T19:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700062340.0],
            TIME_J2000,
            [6.999912e08, 7.000128e08, 7.000344e08, 7.000560e08, 7.000776e08],
            [
                "2022-03-08T06:00",
                "2022-03-08T12:00",
                "2022-03-08T18:00",
                "2022-03-09T00:00",
                "2022-03-09T06:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 700332700.0],
            TIME_J2000,
            [
                6.999264e08,
                7.000128e08,
                7.000992e08,
                7.001856e08,
                7.002720e08,
                7.003584e08,
            ],
            [
                "2022-03-07T12:00",
                "2022-03-08T12:00",
                "2022-03-09T12:00",
                "2022-03-10T12:00",
                "2022-03-11T12:00",
                "2022-03-12T12:00",
            ],
        ),
        (
            [700008333.0, 700008334.0, 708648333.0],
            TIME_J2000,
            [
                6.997536e08,
                7.003584e08,
                7.009632e08,
                7.015680e08,
                7.021728e08,
                7.027776e08,
                7.033824e08,
                7.039872e08,
                7.045920e08,
                7.051968e08,
                7.058016e08,
                7.064064e08,
                7.070112e08,
                7.076160e08,
                7.082208e08,
                7.088256e08,
            ],
            [
                "2022-03-05T12:00",
                "2022-03-12T12:00",
                "2022-03-19T12:00",
                "2022-03-26T12:00",
                "2022-04-02T12:00",
                "2022-04-09T12:00",
                "2022-04-16T12:00",
                "2022-04-23T12:00",
                "2022-04-30T12:00",
                "2022-05-07T12:00",
                "2022-05-14T12:00",
                "2022-05-21T12:00",
                "2022-05-28T12:00",
                "2022-06-04T12:00",
                "2022-06-11T12:00",
                "2022-06-18T12:00",
            ],
        ),
    ],
)
def test__compute_ticks_and_labels(data, time_col, expected_ticks, expected_labels) -> None:
    ticks, labels = _compute_ticks_and_labels(DataFrame({time_col: data}))
    assert isinstance(ticks, NDArray[Shape[f"{len(expected_ticks)}"], Float64])
    assert array_equal(ticks, expected_ticks)
    assert isinstance(labels, NDArray[Shape[f"{len(expected_labels)}"], Str0])
    assert array_equal(labels, expected_labels)
