"""
Plot strain crosses in the style of Colgan _et al._ (2016).
"""

import matplotlib
import matplotlib.pyplot as plt
import xarray
import numpy as np

from typing import Literal


def strain_quiver(
    ax: matplotlib.axes._axes.Axes,
    principal_strain_rates: xarray.Dataset,
    show_every: int = 10,
    arrowscale: float = 10.0,
    arrowwidth: float = 0.003,
    cmap: str = "coolwarm",
    unit: Literal["a", "m", "d", "s"] = "a",
    key: bool = True,
    keypos_x: float = 0.075,
    keypos_y: float = 0.9,
    keycolor: str = "k",
):
    """
    Given a matplolib axis and a principal strain rates xarray Dataset (output
    from strain.principal with `output="directions"`), will plot a quiver plot
    of the second and first principal strain rates in the style of Colgan _et
    al._ (2016; https://doi.org/10.1002/2015RG000504).

    Args:
        ax (matplotlib.axes._axes.Axes): Matplotlib axis to plot on
        principal_strain_rates (xarray.Dataset): Principal strain rates xarray Dataset
        show_every (int, optional): Set to plot every Nth quiver. Default is 10.
        arrowscale (float, optional): Scale of arrows. Default is 10.
        arrowwidth (float, optional): Width of arrows. Default is 0.003.
        cmap (str, optional): Colormap to use. Default is 'coolwarm'.
        unit (Literal["a", "m", "d", "s"], optional): Unit of arrows. Default is 'a'.
        key (bool, optional): Plot quiver key. Default is True.
        keypos_x (float, optional): X position of quiver key. Default is 0.075.
        keypos_y (float, optional): Y position of quiver key. Default is 0.9.
        keycolor (str, optional): Colour of quiver key. Default is 'k'.

    Returns:
        None
    """

    # Sanity check data
    if not isinstance(principal_strain_rates, xarray.Dataset):
        raise ValueError("principal_strain_rates must be an xarray Dataset")
    if unit not in ["a", "m", "d", "s"]:
        raise ValueError("unit must be one of ['a', 'm', 'd', 's']")

    # Plot every Nth quiver as per https://stackoverflow.com/questions/33576572/python-quiver-options
    skip = slice(None, None, show_every)

    # Derive binary directions for extensional and compressional quivers (for colouring)
    with np.errstate(
        invalid="ignore"
    ):  # ignore runtime warnings in comparisons against np.nans
        principal_strain_rates["e1colour"] = (
            ("y", "x"),
            np.where(principal_strain_rates.e_1 > 0, 1, -1),
        )
        principal_strain_rates["e2colour"] = (
            ("y", "x"),
            np.where(principal_strain_rates.e_2 > 0, 1, -1),
        )

    # Plot second principal strain quivers
    e2_qvr = ax.quiver(
        principal_strain_rates.x[skip],
        principal_strain_rates.y[skip],
        principal_strain_rates.e_2U[skip, skip],
        principal_strain_rates.e_2V[skip, skip],
        principal_strain_rates.e2colour[skip, skip],
        scale=arrowscale,
        cmap=cmap,
        clim=(-1, +1),
        pivot="mid",
        headaxislength=0,
        headlength=0,
        width=arrowwidth,
        zorder=3,
    )

    # Plot first principal strain quivers
    e1_qvr = ax.quiver(
        principal_strain_rates.x[skip],
        principal_strain_rates.y[skip],
        principal_strain_rates.e_1U[skip, skip],
        principal_strain_rates.e_1V[skip, skip],
        principal_strain_rates.e1colour[skip, skip],
        scale=arrowscale,
        cmap=cmap,
        clim=(-1, +1),
        pivot="mid",
        headaxislength=0,
        headlength=0,
        width=arrowwidth,
        zorder=3,
    )

    # Plot quiver key
    if key == True:
        quivlength = arrowscale / 10
        unit = s = f"$\\mathregular{{{{{unit}}}^{{-1}}}}$"
        plt.quiverkey(
            e1_qvr,
            X=keypos_x,
            Y=keypos_y,
            U=quivlength,
            label=f"{quivlength} {unit}",
            labelpos="N",
            color=keycolor,
            labelcolor=keycolor,
        )
