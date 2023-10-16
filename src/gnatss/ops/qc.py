from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import constants
from ..utilities.time import AstroTime


def plot_residuals(
    residuals_df: pd.DataFrame,
    outliers_df: Optional[pd.DataFrame] = None,
    n_ticks: int = 25,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot residuals

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Residuals dataframe
    outliers_df : pd.DataFrame, optional
        Outliers dataframe
    n_ticks : int, default 25
        The number of ticks on the x-axis.
    figsize : tuple, default (10, 5)
        The figure size

    Returns
    -------
    plt.Figure
        _description_
    """

    def sec_to_iso(sec, fmt="unix_j2000"):
        astro_time = AstroTime(sec, format=fmt)
        return astro_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Setup plot
    fig, axs = plt.subplots(figsize=figsize)

    # Plot residuals
    time_labels = [constants.TIME_J2000, constants.TIME_ISO]
    col_names = []
    for i, col in enumerate(residuals_df.columns):
        if col not in time_labels:
            col_names.append(col)
            residuals_df.plot.scatter(
                x="time", y=col, s=2, ax=axs, c=f"C{i}", zorder=i + 1
            )

    # Add outliers to plot
    if isinstance(outliers_df, pd.DataFrame) and len(outliers_df) > 0:
        for col in outliers_df.columns:
            if col not in time_labels:
                outliers_df.plot.scatter(
                    x="time", y=col, s=3, c="lightgray", ax=axs, zorder=999
                )
        col_names += ["To be removed"]

    # Add legend
    axs.legend(labels=col_names)

    # Modify x and y labels
    plt.ylabel("Residual (cm)")
    plt.xlabel("Time")

    # Add light grid
    plt.grid(zorder=0, color="0.95")

    # Calculate ticks and labels
    time_min = residuals_df["time"].min()
    time_max = residuals_df["time"].max()
    step = (time_max - time_min) / n_ticks
    ticks = np.arange(time_min, time_max + step, step)
    labels = np.apply_along_axis(sec_to_iso, 0, ticks)
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")

    return fig
