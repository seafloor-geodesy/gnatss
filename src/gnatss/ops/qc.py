from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import constants
from ..configs.main import Configuration
from ..utilities.geo import calc_enu_comp
from ..utilities.time import AstroTime

# The 8 Colorblind from Bang Wong’s
# Nature Methods paper https://www.nature.com/articles/nmeth.1618.pdf
CB_COLORS: List[str] = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def _sec_to_iso(sec: float, fmt: str = "unix_j2000") -> str:
    """Convert seconds to ISO format via AstroTime

    Parameters
    ----------
    sec : float
        The seconds to convert
    fmt : str, default "unix_j2000"
        The astropy time format

    Returns
    -------
    str
       The ISO formatted time string
       as 'YYYY-MM-DDThh:mm:ss.dddddd'
    """
    astro_time = AstroTime(sec, format=fmt)
    return astro_time.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _compute_ticks_and_labels(
    data: pd.DataFrame, n_ticks: int = 25, time_col: str = "time"
) -> Tuple:
    """Compute ticks and labels for x-axis

    Parameters
    ----------
    data : pd.DataFrame
        The data to compute ticks and labels for
        assuming that the data has a time column
        and that the time column is in seconds
    n_ticks : int, default 25
        The desired number of ticks on the x-axis
    time_col : str, default "time"
        The name of the time column

    Returns
    -------
    ticks : np.ndarray
        The ticks values
    labels : np.ndarray
        The tick labels
    """
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    step = (time_max - time_min) / n_ticks
    ticks = np.arange(time_min, time_max + step, step)
    labels = np.apply_along_axis(_sec_to_iso, 0, ticks)
    return ticks, labels


def plot_residuals(
    residuals_df: pd.DataFrame,
    outliers_df: Optional[pd.DataFrame] = None,
    n_ticks: int = 15,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot residuals

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Residuals dataframe
    outliers_df : pd.DataFrame, optional
        Outliers dataframe
    n_ticks : int, default 15
        The desired number of ticks on the x-axis
    figsize : tuple, default (10, 5)
        The figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Setup plot
    fig, axs = plt.subplots(figsize=figsize)

    # Plot residuals
    time_labels = [constants.TIME_J2000, constants.TIME_ISO]
    col_names = []
    for i, col in enumerate(residuals_df.columns):
        if col not in time_labels:
            col_names.append(col)
            residuals_df.plot.scatter(
                x="time", y=col, s=2, ax=axs, c=CB_COLORS[i + 1], zorder=i + 2
            )

    # Add outliers to plot
    if isinstance(outliers_df, pd.DataFrame) and len(outliers_df) > 0:
        for col in outliers_df.columns:
            if col not in time_labels:
                outliers_df.plot.scatter(
                    x="time", y=col, s=3, c=CB_COLORS[0], ax=axs, zorder=999
                )
        col_names += ["To be removed"]

    # Add legend
    axs.legend(labels=col_names)

    # Modify x and y labels
    plt.ylabel("Residual (cm)")
    plt.xlabel("Time")

    # Add light grid
    plt.grid(zorder=1, color="0.95")

    # Calculate ticks and labels
    ticks, labels = _compute_ticks_and_labels(
        residuals_df, n_ticks=n_ticks, time_col=constants.TIME_J2000
    )
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")

    fig.tight_layout()
    return fig


def plot_enu_comps(
    residuals_df: pd.DataFrame,
    config: Configuration,
    n_ticks: int = 15,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot averaged ENU components

    Parameters
    ----------
    residuals_df : pd.DataFrame
        The residuals dataframe
    config : Config
        The configuration object
    n_ticks : int, default 15
        The desired number of ticks on the x-axis
    figsize : tuple, default (10, 5)
        The figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Get transponder ids, azimuths, and elevations
    transponders_id = [t.pxp_id for t in config.solver.transponders]
    azimuths = np.array([t.azimuth for t in config.solver.transponders])
    elevations = np.array([t.elevation for t in config.solver.transponders])

    # Just grab the time and transponder columns
    cleaned_df = residuals_df[[constants.TIME_J2000] + transponders_id]

    # Computes the ENU components averaged over all transponders
    comps_df = pd.DataFrame.from_records(
        np.apply_along_axis(
            calc_enu_comp, 1, cleaned_df[transponders_id], az=azimuths, el=elevations
        ),
        columns=constants.GPS_LOCAL_TANGENT,
    )
    comps_df.loc[:, constants.TIME_J2000] = cleaned_df[constants.TIME_J2000]

    n_axis = len(constants.GPS_LOCAL_TANGENT)
    fig, axs = plt.subplots(nrows=n_axis, figsize=figsize)

    # Calculate ticks and labels
    ticks, labels = _compute_ticks_and_labels(
        comps_df, n_ticks=n_ticks, time_col=constants.TIME_J2000
    )

    for i, col in enumerate(constants.GPS_LOCAL_TANGENT):
        default_kwargs = dict(
            x="time",
            y=col,
            ax=axs[i],
            s=2,
            ylabel=f"{col.title()} (cm)",
            zorder=2,
            c=CB_COLORS[0],  # Black
        )
        labelbottom = True

        # Set empty ticks for all subplots except for last one
        if i < n_axis - 1:
            default_kwargs.update(dict(xticks=ticks, xlabel=""))
            labelbottom = False

        # Plot components to axis
        comps_df.plot.scatter(**default_kwargs)

        # Set label bottoms
        axs[i].tick_params(labelbottom=labelbottom)

        # Add light grid
        axs[i].grid(zorder=1, color="0.95")

    plt.xlabel("Time")

    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")

    fig.tight_layout()
    return fig
