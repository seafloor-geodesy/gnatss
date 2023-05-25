"""ops_data.py
Datasets for ops test
"""

import numpy as np

# Datasets
GPS_DATASET = [
    {
        "time": 712215465.0,
        "x": -2575288.225,
        "y": -3682570.227,
        "z": 4511064.31,
        "xx": 0.0007642676487,
        "xy": 1.464868147e-07,
        "xz": -2.28909888e-06,
        "yx": 1.464868147e-07,
        "yy": 0.001003773991,
        "yz": -3.469954175e-06,
        "zx": -2.28909888e-06,
        "zy": -3.469954175e-06,
        "zz": 0.0008874766165,
    },
    {
        "time": 712215480.0,
        "x": -2575279.439,
        "y": -3682576.054,
        "z": 4511065.58,
        "xx": 0.0007632827522,
        "xy": 4.247534112e-07,
        "xz": -2.52535522e-06,
        "yx": 4.247534112e-07,
        "yy": 0.001002904626,
        "yz": -3.809993913e-06,
        "zx": -2.52535522e-06,
        "zy": -3.809993913e-06,
        "zz": 0.0008868617746,
    },
    {
        "time": 712215495.0,
        "x": -2575269.558,
        "y": -3682579.137,
        "z": 4511068.196,
        "xx": 0.0007636394742,
        "xy": 2.351279563e-07,
        "xz": -2.343107339e-06,
        "yx": 2.351279563e-07,
        "yy": 0.001003126674,
        "yz": -3.537116345e-06,
        "zx": -2.343107339e-06,
        "zy": -3.537116345e-06,
        "zz": 0.0008869383932,
    },
    {
        "time": 712215510.0,
        "x": -2575262.122,
        "y": -3682585.47,
        "z": 4511066.817,
        "xx": 0.0007622707048,
        "xy": -2.964120039e-08,
        "xz": -2.149297811e-06,
        "yx": -2.964120039e-08,
        "yy": 0.001003263125,
        "yz": -3.258009041e-06,
        "zx": -2.149297811e-06,
        "zy": -3.258009041e-06,
        "zz": 0.0008847786993,
    },
    # This should be missing yy and z
    {
        "time": 712215525.0,
        "x": -2575253.07,
        "y": -3682595.498,
        "xx": 0.0007639192564,
        "xy": -2.477204992e-07,
        "xz": -2.233920204e-06,
        "yx": -2.477204992e-07,
        "yz": -3.374849755e-06,
        "zx": -2.233920204e-06,
        "zy": -3.374849755e-06,
        "zz": 0.0008861840659,
    },
]

TRAVEL_TIMES_DATASET = [
    7.12215465e08,
    7.12215480e08,
    7.12215495e08,
    7.12215510e08,
    7.12215525e08,
    7.12215540e08,
]

TT_DELAY_SECONDS = [
    np.array([2.281219, 2.377755, 2.388229]),
    np.array([2.288577, 2.371308, 2.383921]),
    np.array([2.301908, 2.363096, 2.382359]),
]

TWTT_MODEL = [
    np.array([2.08140459, 2.05803764, 1.94834694]),
    np.array([2.08875182, 2.0515515, 1.94400265]),
    np.array([2.10208738, 2.04335558, 1.9424679]),
]

TT_RESIDUAL = [
    np.array([-0.00018559, -0.00028264, -0.00011794]),
    np.array([-1.74815363e-04, -2.43499730e-04, -8.16499168e-05]),
    np.array([-0.00017938, -0.00025958, -0.0001089]),
]

TRANSMIT_VECTORS = [
    np.array(
        [
            [-375.921, 1193.236, -900.247],
            [550.164, -141.056, -1414.852],
            [1160.014, 812.795, -277.465],
        ]
    ),
    np.array(
        [
            [-383.357, 1199.569, -898.868],
            [542.728, -134.723, -1413.473],
            [1152.578, 819.128, -276.086],
        ]
    ),
    np.array(
        [
            [-392.409, 1209.597, -897.82],
            [533.676, -124.695, -1412.425],
            [1143.526, 829.156, -275.038],
        ]
    ),
]

REPLY_VECTORS = [
    np.array(
        [
            [-376.828, 1194.152, -900.553],
            [549.157, -140.103, -1415.148],
            [1158.996, 813.753, -277.761],
        ]
    ),
    np.array(
        [
            [-384.879, 1199.434, -898.625],
            [541.175, -134.871, -1413.261],
            [1151.021, 818.979, -275.88],
        ]
    ),
    np.array(
        [
            [-393.932, 1211.999, -895.664],
            [532.113, -122.293, -1410.229],
            [1141.949, 831.558, -272.829],
        ]
    ),
]

A_PARTIALS = [
    np.array(
        [
            [-0.00032953, 0.00104515, -0.00078836, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.00048672, -0.00012449, -0.00125301, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00108455, 0.00076072, -0.00025968],
        ]
    ),
    np.array(
        [
            [-0.00033513, 0.00104654, -0.00078414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.00048141, -0.00011974, -0.00125553, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00107975, 0.00076784, -0.00025873],
        ]
    ),
    np.array(
        [
            [-0.00034085, 0.00104969, -0.00077743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.00047527, -0.00011014, -0.00125874, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0010721, 0.00077905, -0.00025701],
        ]
    ),
]

WEIGHT_MATRIX = [
    np.array(
        [
            [1.28820086e09, -3.10127429e08, -4.18300222e08],
            [-3.10115288e08, 1.33928090e09, -3.37561871e08],
            [-4.18276506e08, -3.37555947e08, 1.43612753e09],
        ]
    ),
    np.array(
        [
            [1.29215004e09, -3.11218615e08, -4.23212929e08],
            [-3.11206431e08, 1.33929681e09, -3.32301611e08],
            [-4.23188934e08, -3.32295780e08, 1.43595791e09],
        ]
    ),
    np.array(
        [
            [1.29461648e09, -3.13126871e08, -4.29881914e08],
            [-3.13114613e08, 1.33597475e09, -3.25935063e08],
            [-4.29857541e08, -3.25929343e08, 1.43244696e09],
        ]
    ),
]

LSQ_RESULTS = [
    (
        np.array(
            [
                0.03355802,
                -0.10643389,
                0.08028343,
                -0.0754853,
                0.01930713,
                0.19432906,
                -0.07018954,
                -0.04923202,
                0.01680588,
            ]
        ),
        np.array(
            [
                -0.02861046,
                -0.01043511,
                0.23732722,
                -0.02861046,
                -0.01043511,
                0.23732722,
                -0.02861046,
                -0.01043511,
                0.23732722,
            ]
        ),
        np.array(
            [
                [
                    1.01300373e14,
                    3.39081491e13,
                    2.77216506e13,
                    1.74637965e13,
                    1.03745581e14,
                    5.32055935e11,
                    5.91209340e12,
                    -8.48676442e13,
                    -1.44061220e14,
                ],
                [
                    2.65978977e13,
                    -5.85333432e12,
                    -1.81798787e13,
                    -4.12331028e11,
                    1.01432892e13,
                    -8.47140107e11,
                    -4.45504628e12,
                    -1.13280299e13,
                    -1.97675782e13,
                ],
                [
                    -7.08143380e12,
                    -2.19333360e13,
                    -3.56890453e13,
                    -7.84640599e12,
                    -2.99178328e13,
                    -1.34547272e12,
                    -8.37739581e12,
                    2.04563199e13,
                    3.40103624e13,
                ],
                [
                    5.95567930e12,
                    8.24186988e11,
                    -4.11265721e12,
                    -9.29718452e12,
                    -1.36979236e13,
                    -8.10285900e11,
                    -2.51686483e12,
                    4.42792616e12,
                    1.11965168e13,
                ],
                [
                    4.95613597e13,
                    1.05365044e13,
                    -6.99405486e13,
                    -3.70283610e13,
                    -3.37058736e14,
                    5.76696044e13,
                    -4.16558493e13,
                    1.26925129e14,
                    3.06106291e14,
                ],
                [
                    -2.61062996e12,
                    -7.26683057e11,
                    5.35126326e12,
                    6.74655576e10,
                    2.81668851e13,
                    -6.04438225e12,
                    3.16097097e12,
                    -1.08903752e13,
                    -2.60633224e13,
                ],
                [
                    -5.40542012e12,
                    7.03313330e11,
                    3.19183624e12,
                    9.59896969e11,
                    -8.13233894e12,
                    1.18083330e12,
                    3.67962207e12,
                    -1.70682990e12,
                    1.03678161e13,
                ],
                [
                    4.49228496e12,
                    -5.60113927e11,
                    -2.62030764e12,
                    -4.73022783e11,
                    -0.00000000e00,
                    -1.83741270e11,
                    -2.56735738e12,
                    -1.21685386e12,
                    -1.42872478e13,
                ],
                [
                    -9.41573234e12,
                    1.29655193e12,
                    5.65459629e12,
                    2.62329932e12,
                    -3.39646033e13,
                    4.39347309e12,
                    7.84694245e12,
                    -1.06932665e13,
                    1.44724181e12,
                ],
            ]
        ),
        np.array(
            [
                [
                    1.56250000e-01,
                    1.17187500e-02,
                    -2.85156250e-01,
                    -2.05078125e-01,
                    -1.56250000e00,
                    2.22534180e-01,
                    -1.37695312e-01,
                    2.03125000e-01,
                    1.03125000e00,
                ],
                [
                    3.12500000e-02,
                    0.00000000e00,
                    -8.59375000e-02,
                    -5.81665039e-02,
                    -3.49609375e-01,
                    4.96826172e-02,
                    -5.56640625e-02,
                    1.07421875e-01,
                    3.28125000e-01,
                ],
                [
                    -2.34375000e-02,
                    -7.81250000e-03,
                    7.81250000e-03,
                    1.26953125e-02,
                    1.99218750e-01,
                    -2.39257812e-02,
                    -1.46484375e-02,
                    5.85937500e-02,
                    1.56250000e-02,
                ],
                [
                    3.90625000e-03,
                    3.29589844e-03,
                    -2.34375000e-02,
                    -2.34375000e-02,
                    -9.96093750e-02,
                    2.16064453e-02,
                    -3.46679688e-02,
                    1.17187500e-01,
                    1.69921875e-01,
                ],
                [
                    -3.90625000e-02,
                    -2.53906250e-02,
                    7.81250000e-03,
                    7.81250000e-02,
                    6.25000000e-01,
                    1.56250000e-02,
                    -1.87500000e-01,
                    8.28125000e-01,
                    2.50000000e-01,
                ],
                [
                    2.92968750e-03,
                    4.39453125e-03,
                    -9.76562500e-03,
                    -1.56021118e-02,
                    -1.05468750e-01,
                    5.85937500e-03,
                    5.37109375e-03,
                    -4.49218750e-02,
                    4.29687500e-02,
                ],
                [
                    -1.95312500e-02,
                    1.83105469e-03,
                    2.39257812e-02,
                    2.56347656e-03,
                    7.81250000e-02,
                    -9.76562500e-03,
                    -1.95312500e-03,
                    5.37109375e-03,
                    -4.88281250e-02,
                ],
                [
                    1.85546875e-02,
                    -1.22070312e-03,
                    -1.80664062e-02,
                    1.83105469e-04,
                    -6.13203909e-02,
                    1.14440918e-02,
                    4.39453125e-03,
                    -1.70898438e-03,
                    2.53906250e-02,
                ],
                [
                    -2.53906250e-02,
                    2.44140625e-03,
                    4.29687500e-02,
                    1.22070312e-02,
                    1.44531250e-01,
                    -8.78906250e-03,
                    -9.76562500e-04,
                    1.75781250e-02,
                    -1.22314453e-01,
                ],
            ]
        ),
    ),
    (
        np.array(
            [
                0.03887405,
                -0.1238373,
                0.04104728,
                -0.06679783,
                0.01573827,
                0.16682842,
                -0.04093999,
                -0.04484418,
                0.01164074,
            ]
        ),
        np.array(
            [
                -0.0127554,
                -0.04220725,
                0.19492235,
                -0.0127554,
                -0.04220725,
                0.19492235,
                -0.0127554,
                -0.04220725,
                0.19492235,
            ]
        ),
        np.array(
            [
                [
                    -9.43966760e13,
                    -1.21881880e13,
                    -4.09865540e12,
                    -8.78511519e12,
                    -6.85039928e13,
                    9.25976160e10,
                    -1.34183448e13,
                    6.07476557e13,
                    2.30630554e13,
                ],
                [
                    1.06586677e12,
                    2.21318739e12,
                    2.33235216e12,
                    -6.51082178e11,
                    -2.01418117e12,
                    -6.85497952e10,
                    -2.98081992e11,
                    4.88068566e11,
                    -1.64576240e11,
                ],
                [
                    4.17663048e13,
                    8.16284918e12,
                    4.86454232e12,
                    2.88567362e12,
                    2.65894125e13,
                    -1.31063766e11,
                    5.33697449e12,
                    -2.53112691e13,
                    -1.00764626e13,
                ],
                [
                    4.57865178e13,
                    6.83114327e12,
                    -1.87038890e13,
                    -5.10309562e12,
                    2.14987518e13,
                    -5.79398623e12,
                    -1.31711987e12,
                    7.42442047e12,
                    -3.54993816e13,
                ],
                [
                    1.59797440e14,
                    1.60209245e13,
                    -4.37555816e13,
                    3.10739711e13,
                    4.20746918e13,
                    3.23318914e12,
                    -2.90279645e12,
                    1.38445552e13,
                    -9.65018400e13,
                ],
                [
                    2.31610720e12,
                    1.09135997e12,
                    -2.99869047e12,
                    -4.92021581e12,
                    4.23063606e12,
                    -2.52994750e12,
                    -2.28185572e11,
                    1.52640178e12,
                    -4.40819973e12,
                ],
                [
                    -7.02265316e12,
                    -6.30401466e11,
                    2.16002424e12,
                    3.47191370e12,
                    1.73075210e13,
                    -3.19377949e11,
                    2.81721876e12,
                    -3.25360493e12,
                    2.10120183e12,
                ],
                [
                    4.74552969e12,
                    3.30235192e11,
                    -1.58742702e12,
                    -4.11515229e12,
                    -1.64643798e13,
                    -7.67057108e09,
                    -4.58942490e12,
                    3.07773923e12,
                    -1.00190169e13,
                ],
                [
                    -1.52239873e13,
                    -1.65078728e12,
                    4.30331311e12,
                    2.27658284e12,
                    2.33671646e13,
                    -1.35561439e12,
                    -1.86314713e12,
                    -4.44428026e12,
                    -2.09648254e13,
                ],
            ]
        ),
        np.array(
            [
                [
                    -0.140625,
                    -0.00585938,
                    -0.12011719,
                    -0.05078125,
                    0.1484375,
                    -0.01179504,
                    0.01171875,
                    0.0546875,
                    0.05078125,
                ],
                [
                    0.05834961,
                    0.00390625,
                    -0.01025391,
                    0.01538086,
                    0.01806641,
                    0.00318909,
                    0.00213623,
                    -0.00537109,
                    -0.02410889,
                ],
                [
                    0.140625,
                    0.00488281,
                    0.03613281,
                    0.04345703,
                    -0.04296875,
                    0.01002502,
                    0.0,
                    -0.01953125,
                    -0.05273438,
                ],
                [
                    0.2734375,
                    0.01074219,
                    -0.078125,
                    0.10546875,
                    0.06640625,
                    0.01367188,
                    -0.00976562,
                    0.03515625,
                    -0.265625,
                ],
                [
                    0.96875,
                    0.01953125,
                    -0.265625,
                    0.29296875,
                    0.3828125,
                    0.02783203,
                    -0.01611328,
                    0.10351562,
                    -0.796875,
                ],
                [
                    0.01513672,
                    0.00170898,
                    -0.00537109,
                    0.01269531,
                    -0.01220703,
                    0.00292969,
                    -0.00100708,
                    0.00463867,
                    -0.02636719,
                ],
                [
                    -0.30664062,
                    -0.02478027,
                    0.09887695,
                    -0.06640625,
                    -0.15429688,
                    -0.00469971,
                    -0.00439453,
                    -0.03125,
                    0.20214844,
                ],
                [
                    0.265625,
                    0.0223999,
                    -0.08740234,
                    0.05957031,
                    0.14648438,
                    0.00394535,
                    0.00585938,
                    0.02587891,
                    -0.18359375,
                ],
                [
                    -0.50195312,
                    -0.0390625,
                    0.15820312,
                    -0.1015625,
                    -0.2109375,
                    -0.00830078,
                    -0.00292969,
                    -0.05859375,
                    0.30078125,
                ],
            ]
        ),
    ),
    (
        np.array(
            [
                0.03354962,
                -0.10332023,
                0.07652187,
                -0.06769533,
                0.01568785,
                0.17928928,
                -0.0640658,
                -0.04655393,
                0.01535822,
            ]
        ),
        np.array(
            [
                -0.04330739,
                0.11056801,
                0.10867776,
                -0.04330739,
                0.11056801,
                0.10867776,
                -0.04330739,
                0.11056801,
                0.10867776,
            ]
        ),
        np.array(
            [
                [
                    -9.68544122e13,
                    2.14281207e14,
                    6.54910000e13,
                    4.18440067e14,
                    1.26421195e15,
                    -1.81892310e13,
                    2.47284512e13,
                    2.68866810e14,
                    1.26797113e14,
                ],
                [
                    3.48886242e13,
                    9.60949371e13,
                    2.45248067e13,
                    8.55412941e13,
                    1.97123426e14,
                    -1.45076699e12,
                    3.97876227e12,
                    2.40020097e13,
                    2.15584572e13,
                ],
                [
                    8.95708505e13,
                    3.58001946e13,
                    4.40018657e12,
                    -6.79591165e13,
                    -2.88113596e14,
                    6.01590342e12,
                    -5.46959294e12,
                    -8.54721101e13,
                    -2.64835405e13,
                ],
                [
                    6.61924439e13,
                    8.75925362e12,
                    -2.11930304e13,
                    -5.08723700e13,
                    -1.45542784e14,
                    -5.34931980e12,
                    -8.35219806e12,
                    -3.47782401e13,
                    -2.40021424e13,
                ],
                [
                    3.16459348e14,
                    3.90425492e13,
                    1.28350888e14,
                    -6.90438992e13,
                    -1.09979919e14,
                    7.35874249e12,
                    1.48519655e13,
                    -6.93674768e13,
                    -1.24882855e14,
                ],
                [
                    -2.69757832e12,
                    -1.08946962e11,
                    -1.92327076e13,
                    -1.31668305e13,
                    -4.53301957e13,
                    -2.66366614e12,
                    -4.45313937e12,
                    -7.06176039e12,
                    1.86464198e12,
                ],
                [
                    -5.04004556e12,
                    -6.16212783e12,
                    -6.11042079e12,
                    -4.80010216e12,
                    -1.54534106e13,
                    -4.60226823e11,
                    -2.11081076e12,
                    3.30948902e12,
                    1.22663403e12,
                ],
                [
                    -5.16583196e12,
                    7.32051003e11,
                    3.25328382e12,
                    -2.81268581e12,
                    -4.67286507e12,
                    -6.53126005e11,
                    2.13871626e12,
                    -2.76676454e12,
                    5.34881088e11,
                ],
                [
                    -3.66829082e13,
                    -2.34859068e13,
                    -1.56278408e13,
                    -2.85491320e13,
                    -7.86272793e13,
                    -3.89956418e12,
                    -2.32221822e12,
                    5.41868124e12,
                    6.73815595e12,
                ],
            ]
        ),
        np.array(
            [
                [
                    -8.90625000e-01,
                    7.81250000e-01,
                    6.40625000e-01,
                    -1.12500000e00,
                    2.50000000e00,
                    -4.29687500e-02,
                    -9.37500000e-02,
                    -2.81250000e-01,
                    3.12500000e-02,
                ],
                [
                    3.90625000e-03,
                    6.25000000e-02,
                    1.52343750e-01,
                    -2.50000000e-01,
                    3.12500000e-01,
                    3.17382812e-03,
                    -1.95312500e-02,
                    -1.56250000e-02,
                    -1.95312500e-02,
                ],
                [
                    3.90625000e-01,
                    -2.57812500e-01,
                    -7.81250000e-02,
                    1.40625000e-01,
                    -6.25000000e-01,
                    1.75781250e-02,
                    2.24609375e-02,
                    6.25000000e-02,
                    -5.07812500e-02,
                ],
                [
                    -4.60937500e-01,
                    -1.75781250e-01,
                    -2.34375000e-01,
                    3.20312500e-01,
                    9.37500000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    2.30468750e-01,
                    2.03125000e-01,
                ],
                [
                    -3.75000000e00,
                    -2.34375000e-02,
                    -1.25000000e00,
                    1.07812500e00,
                    1.00000000e00,
                    -4.78515625e-02,
                    -1.07421875e-01,
                    7.96875000e-01,
                    1.39062500e00,
                ],
                [
                    1.46484375e-01,
                    -7.04345703e-02,
                    1.95312500e-02,
                    2.53906250e-02,
                    -5.46875000e-02,
                    5.37109375e-03,
                    9.76562500e-03,
                    1.56250000e-02,
                    -4.44335938e-02,
                ],
                [
                    4.88281250e-02,
                    -7.42187500e-02,
                    3.90625000e-03,
                    1.56250000e-02,
                    2.14843750e-02,
                    -1.09863281e-03,
                    2.44140625e-04,
                    5.56640625e-02,
                    -2.27050781e-02,
                ],
                [
                    1.02539062e-01,
                    7.58056641e-02,
                    2.63671875e-02,
                    -3.17382812e-02,
                    -1.53320312e-01,
                    6.22558594e-03,
                    6.83593750e-03,
                    -1.03515625e-01,
                    -2.79541016e-02,
                ],
                [
                    5.07812500e-01,
                    -8.59375000e-02,
                    9.57031250e-02,
                    -3.90625000e-02,
                    -3.90625000e-01,
                    1.41601562e-02,
                    1.80664062e-02,
                    -7.22656250e-02,
                    -1.77734375e-01,
                ],
            ]
        ),
    ),
]
