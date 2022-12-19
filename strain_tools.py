#!/usr/bin/env python

"""
Glacier surface logarithmic strain rate calculator

This script implements a Python version of the matlab script presented
in Alley et al. (2018) to calculate logarithmic strain rates from remotely
sensed glacier surface velocity fields (in the x- and y- directions).

It also adapts code from Chudley et al. (2021) to calculate the directions
of the surface-parallel principal strain rates.

To import this file as a module and use in a pre-existing Python script, the
main() function provides a suitable example of a workable implementation.

This file can also be used as a command-line interface, e.g.:

    $ log_strain.py vx.tif vy.tif 750 --pixel_size 200 --no_data -9999.0

where the first three mandatory arguments are the x-velocity field, the
y-velocity field, and the length scale (in distance units). Output strain
fields are saved as *.tif files in the same directory as the input files.

Notes:

 - Consider effect of pixel size and length_scale. Large length scales
    relative to the pixel size will result in slow processing.
 - This script has yet to implement the ice thickness function in the
    original script.

Citations:

Alley et al. (2018). Continent-wide estimates of Antarctic strain rates from
Landsat 8-derived velocity grids. Journal of Glaciology, 64(244)
321-332. https://doi.org/10.1017/jog.2018.23

Chudley et al. (2021). Controls on water storage and drainage in crevasses on
the Greenland Ice Sheet. Journal of Geophysical Research: Earth Surface,
126, e2021JF006287. https://doi.org/10.1029/2021JF006287

TODO:
 - Implement ice thickness function from original Alley et al. (2018) script.

Tom Chudley | chudley.1@osu.edu | Byrd Polar & Climate Research Insitute
November 2021
"""

__author__ = "Tom Chudley"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import argparse
import timeit

import numpy as np
import rasterio as rs
from numba import njit


# ========================================================================== #
# STRAIN FUNCTIONS
# ========================================================================== #


@njit()
def loc_interp2(rowCoord, colCoord, Vx_array, Vy_array):
    """
    Carries out a bilinear interpolation at index [rowCoord, colCoord]
    within a local square. Python adaptation of Alley et al. (2018) matlab script.

    Variable names remain the same as in matlab script for continuity.

    Parameters:
        rowCoord: row index of desired interpolated point.
        colCoord: column index of desired interpolated point.
        Vx_array: array of local square of velocity values in x-direction
        Vy_array: array of local square of velocity values in y-direction

    Returns:
        inXvel: Interpolated value in x-direction.
        inYvel: Interpolated value in y-direction.
    """

    rowCoordfloor = int(np.floor(rowCoord))
    rowCoordceil = int(np.ceil(rowCoord))
    colCoordfloor = int(np.floor(colCoord))
    colCoordceil = int(np.ceil(colCoord))

    ULxVel = Vx_array[rowCoordfloor, colCoordfloor]
    URxVel = Vx_array[rowCoordfloor, colCoordceil]
    LLxVel = Vx_array[rowCoordceil, colCoordfloor]
    LRxVel = Vx_array[rowCoordceil, colCoordceil]

    ULyVel = Vy_array[rowCoordfloor, colCoordfloor]
    URyVel = Vy_array[rowCoordfloor, colCoordceil]
    LLyVel = Vy_array[rowCoordceil, colCoordfloor]
    LRyVel = Vy_array[rowCoordceil, colCoordceil]

    topInterp = colCoord - colCoordfloor
    sideInterp = rowCoord - rowCoordfloor

    topxVel = ULxVel + (URxVel - ULxVel) * topInterp
    botxVel = LLxVel + (LRxVel - LLxVel) * topInterp

    topyVel = ULyVel + (URyVel - ULyVel) * topInterp
    botyVel = LLyVel + (LRyVel - LLyVel) * topInterp

    intXVel = topxVel + (botxVel - topxVel) * sideInterp
    intYVel = topyVel + (botyVel - topyVel) * sideInterp

    return intXVel, intYVel


@njit()
def log_strain_rates(vx, vy, pixel_size, length_scale, tol=10e-4, ydir=1):
    """
    Calculates the logarithmic strain rate for a given glacier surface
    velocity field. Python adaptation of Alley et al. (2018) matlab script.

    Variable names remain the same as in matlab script for continuity.

    Parameters:
        vx: Array of velocity in x direction
        vy: Array of velocity in y direction
        pixel_size: Input pixel size in measurement units for velocity (and
            thickness) grids
        length_scale: Set the half-length-scale to the desired value in
            distance units. Length scale will be rounded to the nearest
            integer number of pixels.
        tol: Set the tolerance for the adaptive time-stepping scheme (see
            supplemental information to Alley et al. 2018). Value is the
            percent difference between the two stake position estimates
            divided by 100. Default of 10^-4 should be adequate for most
            applications
        ydir: Set to 1 if the positive y-direction is in the upwards direction
            on the screen, and -1 if the positive y-direction is downwards

    Returns:
        exGrid: Array of strain rate in xx direction
        eyGrid: Array of strain rate in yy direction
        exyGrid: Array of strain rate in xy direction

    """
    # ====================================================================== #
    # INITIALISE CALCULATIONS
    # ====================================================================== #

    # Convert integer variables to appropriate data types for numba
    ydir = np.array([ydir], dtype=np.float64)[0]
    pixel_size = np.array([pixel_size], dtype=np.float64)[0]

    # Finds the nearest number of pixels to the given length scale; if 0, set to 1
    r = int(np.round(length_scale / pixel_size))
    if r == 0:
        r = int(1)

    # Set a maximum value for r (useful for ice thickness functionality only:
    # here, using a single length scale, so set it as length_scale/pixel size)
    maxR = r

    time_max = 0.1 * maxR * pixel_size / 0.01
    time_max = np.array([time_max]).astype(np.float64)[0]

    # Local square dimensions
    locMult = 2
    # This sets the dimensions of the local square extracted at each time step
    # the default dimensions are 2*locMult*length_scale

    # Create arrays that the calculated values will be written into
    curXVels = np.zeros(5)
    curYVels = np.zeros(5)
    checkXVels = np.zeros(5)
    checkYVels = np.zeros(5)
    checkRowCoords = np.zeros(5)
    checkColCoords = np.zeros(5)
    errorCriteriaX = np.zeros(5)
    errorCriteriaY = np.zeros(5)
    newRowCoords = np.zeros(5)
    newColCoords = np.zeros(5)

    # Create empty strain output grids
    exGrid = np.empty(vx.shape) * np.nan
    exyGrid = np.empty(vx.shape) * np.nan
    eyGrid = np.empty(vx.shape) * np.nan

    [rows, cols] = vx.shape

    # Define the lengths of the segments at the beginning of each calculation
    l0a1 = r
    l0a2 = r
    l0b1 = r * np.sqrt(2)
    l0b2 = r * np.sqrt(2)
    l0c1 = r
    l0c2 = r
    l0d1 = r * np.sqrt(2)
    l0d2 = r * np.sqrt(2)

    # Local square dimensions
    locDim = 2 * locMult * r
    locCent = int(np.ceil(locDim / 2))

    # Assign local coordinates to the stakes around each strain square
    rowCoords = np.array([locCent, locCent - r, locCent, locCent + r, locCent]).astype(
        np.float64
    )
    colCoords = np.array([locCent, locCent, locCent + r, locCent, locCent - r]).astype(
        np.float64
    )
    # (In order, these are the center point, the top point, the right-hand
    # point, the bottom point, and the left-hand point. In other words, it
    # starts from the center and then moves clockwise from the top point.)

    # ====================================================================== #
    # LOOP THROUGH VELOCITY GRIDS
    # ====================================================================== #

    # Loop through pixels as i,j.
    for i in range(locMult * maxR, rows - locMult * maxR, 1):
        for j in range(locMult * maxR, cols - locMult * maxR, 1):

            # Skip processing if centre cell is nan
            if np.isnan(vx[i, j]):
                continue

            # -------------------------------------------------------------- #
            # Initialize calculations for each center pixel

            # Extract velocity array around the center point (i,j) that
            # represents twice the dimensions of the strain square in order
            # to make later calculations. This is the "local grid"
            sqVx = vx[
                (i - (locMult * r)): (i + (locMult * r) + 1),
                (j - (locMult * r)): (j + (locMult * r) + 1),
            ]
            sqVy = vy[
                (i - (locMult * r)): (i + (locMult * r) + 1),
                (j - (locMult * r)): (j + (locMult * r) + 1),
            ]

            # Extract the x- and y-velocities at the original stake positions
            for k in range(5):
                rowCoord = rowCoords.astype(np.int64)[k]
                colCoord = colCoords.astype(np.int64)[k]
                curXVels[k] = sqVx[rowCoord][colCoord]
                curYVels[k] = sqVy[rowCoord][colCoord]

            # Go to the next pixel in the for-loop if any of the current
            # velocities are NaNs
            if np.isnan(np.sum(curXVels)) or np.isnan(np.sum(curXVels)):
                continue

            # Extract an array around the center point (i,j) that represents
            # just the strain square in order to calculate the average velocity
            # at the center point and determine a reasonable time interval
            sqVxmean = vx[(i - r): (i + r + 1), (j - r): (j + r + 1)]
            sqVymean = vy[(i - r): (i + r + 1), (j - r): (j + r + 1)]
            [sqRows, sqCols] = sqVx.shape

            # Calculate mean velocity
            meanX = np.nanmean(sqVxmean)
            meanY = np.nanmean(sqVymean)
            meanVel = np.sqrt(meanX ** 2 + meanY ** 2)

            # Let the stakes move by approximately one tenth of the length scale
            time = 0.1 * r * pixel_size / meanVel
            time = np.array([time, time_max]).min()

            # Initialize time step as the time it takes to move a twentieth of
            # the pixel length, according to the average velocity
            dtOrig = pixel_size / meanVel * 0.05
            dt = dtOrig
            t = dtOrig * 0  # Initialize time tracker to = 0

            # Set the current rows and columns to the coordinates of the strain
            # square
            curRows = rowCoords
            curCols = colCoords

            # Set the initial strain experienced by each strain segment to zero
            stota1 = 0
            stota2 = 0
            stotb1 = 0
            stotb2 = 0
            stotc1 = 0
            stotc2 = 0
            stotd1 = 0
            stotd2 = 0

            # Set the current length of each strain segment to the original
            # lengths comprising the strain square
            lLasta1 = l0a1
            lLasta2 = l0a2
            lLastb1 = l0b1
            lLastb2 = l0b2
            lLastc1 = l0c1
            lLastc2 = l0c2
            lLastd1 = l0d1
            lLastd2 = l0d2

            # ---------------------------------------------------------------- #
            # Let the stakes move and measure strain rates

            while t < time:
                # Move the stakes according to the x- and y-velocities and the
                # time step. Calculate the new column and row positions.
                t += dt
                newRowCoords = curRows - ydir * curYVels * dt / pixel_size
                newColCoords = curCols + curXVels * dt / pixel_size

                # Check to be sure that the stakes haven't moved outside of
                # the local grid
                if not (
                    np.max(newRowCoords) <= sqRows - 1
                    and np.max(newColCoords) <= sqCols - 1
                    and np.min(newRowCoords) >= 0
                    and np.min(newColCoords >= 0)
                ):
                    break

                # Get velocities at new stake positions
                for k in range(5):
                    checkXVels[k], checkYVels[k] = loc_interp2(
                        curRows[k], curCols[k], sqVx, sqVy
                    )

                # Calculate the column and row positions according to the
                # improved Euler method to check for accuracy.
                checkRowCoords = (
                    curRows - ydir * 0.5 *
                    (curYVels + checkYVels) * dt / pixel_size
                )
                checkColCoords = (
                    curCols + 0.5 * (curXVels + checkXVels) * dt / pixel_size
                )
                errorCriteriaY = np.abs(
                    (checkRowCoords - newRowCoords) / checkRowCoords
                )
                errorCriteriaX = np.abs(
                    (checkColCoords - newColCoords) / checkColCoords
                )

                if np.isnan(np.sum(checkXVels)) or np.isnan(np.sum(checkYVels)):
                    break

                # ---------------------------------------------------------- #
                # Adaptive time stepping

                if np.any(errorCriteriaX >= tol) or np.any(errorCriteriaY >= tol):
                    t = t - dt  # Reverse the time to what it was before
                    dt = dt / 2  # Make a smaller time step
                    # We leave the current rows, columns, and velocities the same

                else:
                    # -------------------------------------------------------#
                    # Make final calculations

                    # Check to be sure that the stakes haven't moved outside of
                    # the local grid
                    if (
                        np.max(newRowCoords) <= sqRows - 1
                        and np.max(newColCoords) <= sqCols - 1
                        and np.min(newRowCoords) >= 0
                        and np.min(newColCoords >= 0)
                    ):
                        # Calculate the current length
                        lfa1 = np.sqrt(
                            np.square(newRowCoords[0] - newRowCoords[1])
                            + np.square(newColCoords[0] - newColCoords[1])
                        )
                        lfa2 = np.sqrt(
                            np.square(newRowCoords[0] - newRowCoords[3])
                            + np.square(newColCoords[0] - newColCoords[3])
                        )
                        lfb1 = np.sqrt(
                            np.square(newRowCoords[1] - newRowCoords[4])
                            + np.square(newColCoords[1] - newColCoords[4])
                        )
                        lfb2 = np.sqrt(
                            np.square(newRowCoords[2] - newRowCoords[3])
                            + np.square(newColCoords[2] - newColCoords[3])
                        )
                        lfc1 = np.sqrt(
                            np.square(newRowCoords[0] - newRowCoords[4])
                            + np.square(newColCoords[0] - newColCoords[4])
                        )
                        lfc2 = np.sqrt(
                            np.square(newRowCoords[0] - newRowCoords[2])
                            + np.square(newColCoords[0] - newColCoords[2])
                        )
                        lfd1 = np.sqrt(
                            np.square(newRowCoords[1] - newRowCoords[2])
                            + np.square(newColCoords[1] - newColCoords[2])
                        )
                        lfd2 = np.sqrt(
                            np.square(newRowCoords[3] - newRowCoords[4])
                            + np.square(newColCoords[3] - newColCoords[4])
                        )

                        # Calculate the current strains and strain rates
                        stra1 = np.log(lfa1 / lLasta1)
                        stra2 = np.log(lfa2 / lLasta2)
                        strb1 = np.log(lfb1 / lLastb1)
                        strb2 = np.log(lfb2 / lLastb2)
                        strc1 = np.log(lfc1 / lLastc1)
                        strc2 = np.log(lfc2 / lLastc2)
                        strd1 = np.log(lfd1 / lLastd1)
                        strd2 = np.log(lfd2 / lLastd2)

                        # Update the new rows and columns as current
                        curRows = newRowCoords
                        curCols = newColCoords

                        # Update the current lengths as the previous lengths
                        lLasta1 = lfa1
                        lLasta2 = lfa2
                        lLastb1 = lfb1
                        lLastb2 = lfb2
                        lLastc1 = lfc1
                        lLastc2 = lfc2
                        lLastd1 = lfd1
                        lLastd2 = lfd2

                        # Update the running total of strain for each segment
                        stota1 = stota1 + stra1
                        stota2 = stota2 + stra2
                        stotb1 = stotb1 + strb1
                        stotb2 = stotb2 + strb2
                        stotc1 = stotc1 + strc1
                        stotc2 = stotc2 + strc2
                        stotd1 = stotd1 + strd1
                        stotd2 = stotd2 + strd2

                        # Calculate final strain rate components
                        ea1f = stota1 / t
                        ea2f = stota2 / t
                        eb1f = stotb1 / t
                        eb2f = stotb2 / t
                        ec1f = stotc1 / t
                        ec2f = stotc2 / t
                        ed1f = stotd1 / t
                        ed2f = stotd2 / t

                        # Reset the time step
                        dt = dtOrig

                        # Set the current velocities to those calculated at
                        # the end of the time step
                        curXVels = checkXVels
                        curYVels = checkYVels

                    else:
                        break
                    # If a stake has moved outside the strain square, leave
                    # the current rows and columns and current velocities the
                    # same as the previous time step, and simply move on to
                    # calculating the strain rate components. This will be the
                    # final value for this strain square. Change the time to
                    # kick it out of the while loop

            # -------------------------------------------------------------- #
            # Create strain rate grids

            # Don't calculate values if the stakes have been allowed to move
            # for less than half of the designated time increment
            if t > 0.5 * time:
                # Average the strain rate components
                ea = (ea1f + ea2f) / 2
                eb = (eb1f + eb2f) / 2
                ec = (ec1f + ec2f) / 2
                ed = (ed1f + ed2f) / 2

                # Calculate coordinate-oriented strain values
                exGrid[i, j] = 0.25 * (eb + ed - ea) + 0.75 * ec
                eyGrid[i, j] = 0.75 * ea + 0.25 * (eb + ed - ec)
                exyGrid[i, j] = 0.5 * eb - 0.5 * ed

    return exGrid, eyGrid, exyGrid


@njit()
def principal_strain_rate_directions(e_xx, e_yy, e_xy):
    """
    Calculates the directions of principal strains as output by the
    logarithmic_strain function. Implementation of Chudley et al. (2021).

    Parameters:
        e_xx: Array of strain rate in xx direction
        e_yy: Array of strain rate in yy direction
        e_xy: Array of strain rate in xy direction

    Returns:
        e_1: Array of first principal strain rate
        e_1U: Array of U component of first principal strain rate
        e_1V: Array of V component of first principal strain rate
        e_2: Array of second principal strain rate
        e_2U: Array of U component of second principal strain rate
        e_2V: Array of V component of second principal strain rate
    """

    # Create strain tensors
    zeros = np.zeros(e_xy.shape)
    stack = np.dstack((e_xx, e_xy, zeros, e_xy, e_yy,
                      zeros, zeros, zeros, zeros))
    # NB current shape of strain tensor at [I, J] is (9,).
    # Will reshape to (3,3) at pixel loop.

    # Create empty arrays to write results to
    def initiate_empty():
        return np.full((e_xy.shape), np.nan)

    e_1 = initiate_empty()  # e1  = first principal strain
    e_1U = initiate_empty()  # e1U = U component of first principal strain rate
    e_1V = initiate_empty()  # e1V = V component of first principal strain rate

    e_2 = initiate_empty()  # e1  = second principal strain
    e_2U = initiate_empty()  # e1U = U component of second principal strain rate
    e_2V = initiate_empty()  # e1V = V component of second principal strain rate

    # Loop through strain tensors
    for I in range(zeros.shape[0]):
        for J in range(zeros.shape[1]):

            # Extract and reshape strain tensor at [I, J] into (3,3) shape
            strain = stack[I, J, :]
            strain = np.reshape(strain, (-1, 3))

            # If cell is empty, then skip
            if np.isnan(strain[0, 0]) == True or np.isnan(strain[0, 1]) == True:
                pass

            # Else, calculate principal strain rates
            else:

                # principal strain rates from eigenvalues, and principal
                #  strain rate directions from eigenvectors
                eigvals, eigvecs = np.linalg.eigh(strain[0:2, 0:2])

                # NB np.lingalg.eigh (cf. np.linalg.eig) returns sorted
                # eigenvalues and assumes a symmetric matrix, both useful
                # in this case. Uses a faster algorithm as well.

                # extract values - eigenvalues are sorted in ascending order.
                e_1[I, J] = eigvals[1]  # First principal strain rate
                e_1U[I, J] = eigvecs[0, 1]
                e_1V[I, J] = eigvecs[1, 1]

                e_2[I, J] = eigvals[0]  # Second principal strain rate
                e_2U[I, J] = eigvecs[0, 0]
                e_2V[I, J] = eigvecs[1, 0]

    # As raw output, eigenvectors are normalised:
    # here, we extract U and V magnitudes for plotting.
    e_1U = e_1 * e_1U
    e_1V = e_1 * e_1V
    e_2U = e_2 * e_2U
    e_2V = e_2 * e_2V

    return e_1, e_1U, e_1V, e_2, e_2U, e_2V


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
    e_1 = 0.5 * (e_xx + e_yy) + np.sqrt(0.25 *
                                        np.square(e_xx - e_yy) + np.square(e_xy))
    e_2 = 0.5 * (e_xx + e_yy) - np.sqrt(0.25 *
                                        np.square(e_xx - e_yy) + np.square(e_xy))

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
    e_E = np.sqrt(0.5 * (e_xx ** 2 + e_yy ** 2) + e_xy ** 2)

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
    e_xx, e_yy, e_xy = log_strain_rates(
        vx, vy, pixel_size, length_scale, tol, ydir)
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
        profile.update(dtype=rs.float32, compress='lzw', predictor=3)

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
