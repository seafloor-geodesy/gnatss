import typer
from pandas import DataFrame
from nptyping import Float, NDArray, Shape
from scipy.spatial.transform import Rotation
from typing import List
from gnatss.configs.solver import ArrayCenter
from pymap3d import ecef2enu


def rotation(
    df: DataFrame,
    atd_offsets: NDArray[Shape["3"], Float],
    array_center: ArrayCenter,
    input_rph_columns: List[str],
    output_rotation_columns: List[str],
    output_corrected_rotation_columns: List[str],
    input_ecef_columns: List[str],
    output_enu_columns: List[str],
) -> DataFrame:
    r = Rotation.from_euler("xyz", df[input_rph_columns], degrees=True)
    offsets = r.as_matrix() @ atd_offsets

    # NEU

    df[output_rotation_columns[0]] = offsets[:, 0]
    df[output_rotation_columns[1]] = offsets[:, 1]
    df[output_rotation_columns[2]] = -offsets[:, 2]

    enu = df[input_ecef_columns].apply(
        lambda row: ecef2enu(
            *row.values,
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    df = df.assign(
        **dict(zip(output_enu_columns, zip(*enu)))
    )

    # typer.echo(f"rotation\n output_rotation_columns: {output_rotation_columns}"
    #            f"output_enu_columns: {output_enu_columns}")

    # df[]
    """
    df["td_e0"] = df.ant_e0 + df.d_e0
    df["td_n0"] = df.ant_n0 + df.d_n0
    df["td_u0"] = df.ant_u0 + df.d_u0
    df["td_e1"] = df.ant_e1 + df.d_e1
    df["td_n1"] = df.ant_n1 + df.d_n1
    df["td_u1"] = df.ant_u1 + df.d_u1
    """

    for corrected_rot, rot, enu in zip(output_corrected_rotation_columns, output_rotation_columns, output_enu_columns):
        df[corrected_rot] = df.loc[:, [rot, enu]].sum(axis=1)

    return df
