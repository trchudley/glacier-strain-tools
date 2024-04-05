"""
Hidden numba-wrapped functions to calculate strain rate magnitude following Alley _et 
al._ (2018) and direction following Chudley et al. (2021).
"""

import numpy as np
from numba import njit

from typing import Tuple


# ========================================================================== #
# HIDDEN STRAIN FUNCTIONS
# ========================================================================== #


@njit()
def _log_strain_rates(vx, vy, pixel_size, length_scale, tol=10e-4, ydir=1):
    """
    Calculates the directions of principal strains from e_xx, e_yy, and e_xy strain
    rates. Implementation of Alley _et al. (2018).

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
                (i - (locMult * r)) : (i + (locMult * r) + 1),
                (j - (locMult * r)) : (j + (locMult * r) + 1),
            ]
            sqVy = vy[
                (i - (locMult * r)) : (i + (locMult * r) + 1),
                (j - (locMult * r)) : (j + (locMult * r) + 1),
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
            sqVxmean = vx[(i - r) : (i + r + 1), (j - r) : (j + r + 1)]
            sqVymean = vy[(i - r) : (i + r + 1), (j - r) : (j + r + 1)]
            [sqRows, sqCols] = sqVx.shape

            # Calculate mean velocity
            meanX = np.nanmean(sqVxmean)
            meanY = np.nanmean(sqVymean)
            meanVel = np.sqrt(meanX**2 + meanY**2)

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
                    checkXVels[k], checkYVels[k] = _loc_interp2(
                        curRows[k], curCols[k], sqVx, sqVy
                    )

                # Calculate the column and row positions according to the
                # improved Euler method to check for accuracy.
                checkRowCoords = (
                    curRows - ydir * 0.5 * (curYVels + checkYVels) * dt / pixel_size
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
def _loc_interp2(rowCoord, colCoord, Vx_array, Vy_array):
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
def _principal_strain_rate_directions(
    e_xx: np.ndarray, e_yy: np.ndarray, e_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the directions of principal strains as output by the logarithmic_strain
    function. Implementation of Chudley et al. (2021).

    Args:
        e_xx (np.ndarray): Array of strain rate in xx direction
        e_yy (np.ndarray): Array of strain rate in yy direction
        e_xy (np.ndarray): Array of strain rate in xy direction

    Returns:
        Tuple of:
            e_1 (np.ndarray): Array of first principal strain rate
            e_1U (np.ndarray): Array of U component of first principal strain rate
            e_1V (np.ndarray): Array of V component of first principal strain rate
            e_2 (np.ndarray): Array of second principal strain rate
            e_2U (np.ndarray): Array of U component of second principal strain rate
            e_2V (np.ndarray): Array of V component of second principal strain rate
    """

    # Create strain tensors
    zeros = np.zeros(e_xy.shape)
    stack = np.dstack((e_xx, e_xy, zeros, e_xy, e_yy, zeros, zeros, zeros, zeros))
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
