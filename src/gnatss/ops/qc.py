from __future__ import annotations

from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptyping import Float64, NDArray, Shape, String

from .. import constants
from ..configs.main import Configuration
from ..utilities.geo import calc_enu_comp
from ..utilities.time import AstroTime
from .io import _to_file_fs

# The 8 Colorblind from Bang Wong's
# Nature Methods paper https://www.nature.com/articles/nmeth.1618.pdf
CB_COLORS: list[str] = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]
SECS_IN_MINUTE = 60
SECS_IN_HOUR = 3600
SECS_IN_DAY = 86400

MAX_PLOT_XAXIS_TICKS = 15


def _compute_ticks_and_labels(
    data: pd.DataFrame, time_col: str = "time"
) -> tuple[NDArray[Shape["*"], Float64], NDArray[Shape["*"], String]]:
    """Compute ticks and labels for x-axis ensuring that number of ticks stays between 4 and 15.
    Standard intervals of 1s, 5s, 30s, 1min, 5min, 30min, 1hr, 6hr, 24hr are used whenever possible.
    Datetime formats with 1s, and 1hr precision are used accordingly.
    Parameters
    ----------
    data : pd.DataFrame
        The data to compute ticks and labels for
        assuming that the data has a time column
        and that the time column is in seconds
    time_col : str, default "time"
        The name of the time column

    Returns
    -------
    ticks : (N,) ndarray
        The ticks values
    labels : (N,) ndarray
        The tick labels
    """
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    time_delta = time_max - time_min

    # step defines the interval in seconds between x-axis ticks
    step = None

    # Standard intervals of 1s, 5s, 30s, 1min, 5min, 30min, 1hr, 6hr, 24hr
    # are used whenever possible.
    if time_delta <= 15:
        step = 1
    elif time_delta <= 75:
        step = 5
    elif time_delta <= 450:
        step = 30
    elif time_delta <= 15 * SECS_IN_MINUTE:
        step = 1 * SECS_IN_MINUTE
    elif time_delta <= 75 * SECS_IN_MINUTE:
        step = 5 * SECS_IN_MINUTE
    elif time_delta <= 450 * SECS_IN_MINUTE:
        step = 30 * SECS_IN_MINUTE
    elif time_delta <= 15 * SECS_IN_HOUR:
        step = 1 * SECS_IN_HOUR
    elif time_delta <= 90 * SECS_IN_HOUR:
        step = 6 * SECS_IN_HOUR
    elif time_delta <= 15 * SECS_IN_DAY:
        step = 1 * SECS_IN_DAY
    else:
        # step is made to suitable day multiple
        step = ceil((time_delta / MAX_PLOT_XAXIS_TICKS) / SECS_IN_DAY) * SECS_IN_DAY

    # Find largest multiple of step which is not greater than time_min
    initial_tick = floor(time_min / step) * step

    # Find smallest multiple of step which is greater than time_max
    final_tick = (floor(time_max / step) * step) + step

    ticks = np.arange(float(initial_tick), float(final_tick + step), float(step))

    # Datetime formats with 1s and 1hr precision are used accordingly.
    if time_delta > 450 * SECS_IN_MINUTE:
        # Format ticks with hourly precision
        labels = np.apply_along_axis(
            lambda x: AstroTime(x, format="unix_j2000").strftime("%Y-%m-%dT%H:00"),
            0,
            ticks,
        )
    else:
        # Format ticks with second precision
        labels = np.apply_along_axis(
            lambda x: AstroTime(x, format="unix_j2000").strftime("%Y-%m-%dT%H:%M:%S"),
            0,
            ticks,
        )

    return ticks, labels


def plot_residuals(
    residuals_df: pd.DataFrame,
    outliers_df: pd.DataFrame | None = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot residuals

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Residuals dataframe
    outliers_df : pd.DataFrame, optional
        Outliers dataframe
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
                outliers_df.plot.scatter(x="time", y=col, s=3, c=CB_COLORS[0], ax=axs, zorder=999)
        col_names += ["To be removed"]

    # Add legend
    axs.legend(labels=col_names)

    # Modify x and y labels
    plt.ylabel("Residual (cm)")
    plt.xlabel("Time")

    # Add light grid
    plt.grid(zorder=1, color="0.95")

    # Calculate ticks and labels
    ticks, labels = _compute_ticks_and_labels(residuals_df, time_col=constants.TIME_J2000)
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")

    fig.tight_layout()
    return fig


def plot_enu_comps(
    residuals_df: pd.DataFrame,
    config: Configuration,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot averaged ENU components

    Parameters
    ----------
    residuals_df : pd.DataFrame
        The residuals dataframe
    config : Config
        The configuration object
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
    cleaned_df = residuals_df[[constants.TIME_J2000, *transponders_id]]

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
    ticks, labels = _compute_ticks_and_labels(comps_df, time_col=constants.TIME_J2000)

    for i, col in enumerate(constants.GPS_LOCAL_TANGENT):
        default_kwargs = dict(  # noqa: C408
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
            default_kwargs.update(
                {
                    "xticks": ticks,
                    "xlabel": "",
                }
            )
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


def export_qc_plots(config, result_dict):
    from gnatss.ops.qc import plot_enu_comps, plot_residuals

    output_path = config.output.path
    resdf = result_dict.get("residuals")
    outliers_df = result_dict.get("outliers")

    res_png = output_path + "residuals.png"
    enu_comp_png = output_path + "residuals_enu_components.png"

    # Plot the figures
    res_figure = plot_residuals(resdf, outliers_df)
    enu_figure = plot_enu_comps(resdf, config)

    # Save the figures
    # export residuals qc plots
    _to_file_fs(config.output._fsmap.fs, res_png, res_figure.savefig)
    # export residuals enu components qc plots
    _to_file_fs(config.output._fsmap.fs, enu_comp_png, enu_figure.savefig)
