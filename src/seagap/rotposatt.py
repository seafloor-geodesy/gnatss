import numpy as np


def rotposatt(
    position_in: np.ndarray,
    body_frame_offsets: np.ndarray,
    roll: float,
    pitch: float,
    heading: float,
    position_format: Literal["ENU", "XYZ"] = "ENU",
) -> np.ndarray:
    """
    Computes position of the surface platform transducer
    given the antenna position, the instantaneous roll,
    pitch, and heading, and the body frame offsets of the
    platform.

    Parameters
    ----------
    position_in : np.ndarray[3,1]
        Position array in meters, either XYZ or ENU
    body_frame_offsets : np.ndarray[3,1]
        Body frame offsets of the surface platform in meters
        Requires X, Y, and Z values, with Z positive downwards
    roll : float
        Roll of the surface platform in degrees
    pitch : float
        Pitch of the surface platform in degrees
    heading : float
        Heading of the surface platform in degrees
    position_format : string
        Format of the input/output positions
        Acceptable values are "ENU" (default) or "XYZ"

    """

    # Calculate offset between antenna and transducer
    # in ENU components.
    # This requires applying rotation matrices to the
    # body-frame offsets.
    # Assume: offset = R_heading * R_pitch * R_roll * body_frame_offsets

    # Calculate rotation matrix for roll:
    si = np.sin(np.deg2rad(-roll))
    ci = np.cos(np.deg2rad(-roll))
    R_roll = np.array({[1, 0, 0], [0, ci, si], [0, -si, ci]})

    # Calculate rotation matrix for pitch:
    si = np.sin(np.deg2rad(-pitch))
    ci = np.cos(np.deg2rad(-pitch))
    R_pitch = np.array({[ci, 0, -si], [0, 1, 0], [si, 0, ci]})

    # Calculate rotation matrix for roll:
    si = np.sin(np.deg2rad(heading - 360))
    ci = np.cos(np.deg2rad(heading - 360))
    R_heading = np.array({[ci, si, 0], [-si, ci, 0], [0, 0, 1]})

    # Rotate body frame offsets
    # First rotate for roll

    offset = np.matmul(R_roll, body_frame_offsets)

    # Next rotate for pitch

    offset = np.matmul(R_pitch, offset)

    # Next rotate for heading

    offset = np.matmul(R_heading, offset)

    return position_in + offset
