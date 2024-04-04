#!/usr/bin/env python

import os
import argparse
import timeit

import numpy as np
import rasterio as rs

from ._logarithmic import *


def principal_strain_rate_magnitudes(e_xx, e_yy, e_xy):
    """
    Given e_xx, e_yy, and e_xy, return principal strain rates following
    methods in Nye (1959) and Harper et al. (1998) - i.e. not using
    eigenvectors. Quicker to compute, only returns magnitude values.

    Parameters:
        e_xx: Array of strain rate in xx direction
        e_yy: Array of strain rate in yy direction
        e_xy: Array of strain rate in xy direction

    Returns:
        e_1: Array of first principal strain rate
        e_2: Array of second principal strain rate
    """

    # Calculate first and second principal strain rates
    e_1 = 0.5 * (e_xx + e_yy) + np.sqrt(0.25 * np.square(e_xx - e_yy) + np.square(e_xy))
    e_2 = 0.5 * (e_xx + e_yy) - np.sqrt(0.25 * np.square(e_xx - e_yy) + np.square(e_xy))

    return e_1, e_2


def flow_direction(vx, vy):
    """
    Calculates a grid of flow directions (in radians) so that the grid-oriented
    strain rates can be rotated to align with local flow directions

    Parameters:
        vx: array of velocity in x direction
        vy: array of velocity in y direction

    Returns:
        angle: array of flow direction in radians
    """
    angle = np.degrees(np.arctan(vy / vx))
    angle = np.where(vx > 0, angle, angle + 180)

    # from 0 - 360 degrees, following Alley et al. 2018
    # angle = np.where(angle < 0, angle + 360, angle)

    # # from -180 to 180, following Bindschadler et al. 1996
    angle = np.where(angle > 180, angle - 360, angle)

    angle = np.deg2rad(angle)

    return angle


def rotated_strain_rates(e_xx, e_yy, e_xy, angle):
    """
    Given e_xx, e_yy, and e_xy, return various rotated strain rates
    following Bindschadler et al. (1996).

    Parameters:
        e_xx: Array of strain rate in xx direction
        e_xx: Array of strain rate in yy direction
        e_xy: Array of strain rate in xy direction
        angle: array of flow direction in radians

    Returns:
        e_lon: Longitudinal strain rate
        e_trn: Transverse strain rate
        e_shr: Shear strain rate
    """

    # Calculate longitudinal and transverse strain rate (Bindschadler et al. 1996)
    e_lon = (
        (e_xx * np.cos(angle) ** 2)
        + (2 * e_xy * np.cos(angle) * np.sin(angle))
        + (e_yy * np.sin(angle) ** 2)
    )
    e_trn = (
        (e_xx * np.sin(angle) ** 2)
        - (2 * e_xy * np.cos(angle) * np.sin(angle))
        + (e_yy * np.cos(angle) ** 2)
    )
    e_shr = ((e_yy - e_xx) * np.cos(angle) * np.sin(angle)) + (
        e_xy * (np.cos(angle) ** 2 - np.sin(angle) ** 2)
    )

    return e_lon, e_trn, e_shr


def effective_strain_rate(e_xx, e_yy, e_xy):
    """
    Given e_xx, e_yy, and e_xy, return effective strain
    rate (Cuffey & Paterson p.59).

    Parameters:
        e_xx: Array of strain rate in xx direction
        e_yy: Array of strain rate in yy direction
        e_xy: Array of strain rate in xy direction

    Returns:
        e_E: Effective strain rate
    """

    # Calculate effective strain rate (Cuffey & Paterson p.59)
    e_E = np.sqrt(0.5 * (e_xx**2 + e_yy**2) + e_xy**2)

    return e_E


def main(
    vx_fpath, vy_fpath, length_scale, pixel_size=None, no_data=np.nan, tol=10e-4, ydir=1
):
    """main"""

    print("\nOpening velocities as arrays...")
    with rs.open(vx_fpath) as src:
        vx = src.read(1)
    with rs.open(vy_fpath) as src:
        vy = src.read(1)

        # if pixel_size isn't defined, use rasterio to extract resolution
        if pixel_size is None:
            try:
                pixel_size_x = src.res[0]
                pixel_size_y = src.res[1]
                if not pixel_size_x == pixel_size_y:
                    raise ValueError("Pixel resolutions not equal")
                pixel_size = pixel_size_x
            except ValueError:
                print(
                    f"Error: pixel resolutions in x and y resolutions are not equal ({pixel_size_x} and {pixel_size_y}). Consider setting pixel size manually with -p."
                )
                exit()

    # Set no data values to NaN
    vx = np.where(vx == no_data, np.nan, vx)
    vy = np.where(vy == no_data, np.nan, vy)

    print("\nCalculating strain rates...")
    start = timeit.default_timer()
    e_xx, e_yy, e_xy = log_strain_rates(vx, vy, pixel_size, length_scale, tol, ydir)
    end = timeit.default_timer()
    print(f"Strain rates calculated. Elapsed time: {end - start} seconds.")

    print("\nGetting flow direction...")
    angle = flow_direction(vx, vy)

    print("\nGetting principal strain rates...")
    e_1, e_1U, e_1V, e_2, e_2U, e_2V = principal_strain_rate_directions(
        e_xx, e_yy, e_xy
    )
    # e_1, e_2, e_M = principal_strain_rate_magnitudes(e_xx, e_yy, e_xy, angle)

    print("\nGetting rotated strain rates...")
    e_lon, e_trn, e_shr = rotated_strain_rates(e_xx, e_yy, e_xy, angle)

    print("\nGetting effective strain rate...")
    e_E = effective_strain_rate(e_xx, e_yy, e_xy)

    print("\nWriting geotiffs...")

    def geotiffwrite(dirpath, array, name, lengthscale, profile):
        fpath = os.path.join(dirpath, f"log_{name}_{lengthscale}m.tif")
        with rs.open(fpath, "w", **profile) as dst:
            dst.write(array, 1)

    outdir = os.path.dirname(vx_fpath)

    with rs.open(vx_fpath) as src:
        profile = src.profile
        profile.update(dtype=rs.float32, compress="lzw", predictor=3)

    geotiffwrite(outdir, e_1, "e_1", length_scale, profile)
    geotiffwrite(outdir, e_2, "e_2", length_scale, profile)
    # geotiffwrite(outdir, e_M, "e_M", length_scale, profile)
    geotiffwrite(outdir, e_lon, "e_lon", length_scale, profile)
    geotiffwrite(outdir, e_trn, "e_trn", length_scale, profile)
    geotiffwrite(outdir, e_shr, "e_shr", length_scale, profile)
    geotiffwrite(outdir, e_E, "e_E", length_scale, profile)

    print("\nComplete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        "vx_fpath", help="path of velocity *.tif in x direction", type=str
    )
    parser.add_argument(
        "vy_fpath", help="path of velocity *.tif in y direction", type=str
    )
    parser.add_argument(
        "length_scale", help="half-length-scale in distance units", type=int
    )

    # Optional arguments.
    parser.add_argument(
        "-p",
        "--pixel_size",
        help="pixel resolution in distance units",
        dest="pixel_size",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--no_data",
        help="no data value",
        dest="no_data",
        type=float,
        default=np.nan,
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        help="tolerance for error in adaptive time-stepping scheme",
        dest="tol",
        type=float,
        default=10e-4,
    )
    parser.add_argument(
        "-y",
        "--ydir",
        help="1 if positive y-velocities go up; -1 if down",
        dest="ydir",
        type=int,
        default=1,
    )

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__),
    )

    args = parser.parse_args()
    main(
        args.vx_fpath,
        args.vy_fpath,
        args.length_scale,
        args.pixel_size,
        args.no_data,
        args.tol,
        args.ydir,
    )
