"""
Core functions to calculate strain rate magnitude following Alley _et al._ (2018) and 
direction following Chudley et al. (2021), as well as derivative strain rates (e.g.
shear strain rate).
"""

import warnings

import numpy as np
import xarray as xr

from typing import Tuple

from ._numba import _log_strain_rates, _principal_strain_rate_directions

def _sanitise_unit_time(unit_time: str):
    """
    Sanitise unit time input.

    Args:
        unit_time (str): Time unit representing 'year' or 'day' as a string.

    Returns:
        str: Either 'a' or 'd'
    """

    if unit_time.lower() in ["a", "annual", "annually", "y", "yr", "year", "yearly", "per year"]:
        return "a"
    elif unit_time.lower() in ["m", "monthly", "month", "per month"]:
        return "m"
    elif unit_time.lower() in ["d", "day", "daily", "per day"]:
        return "d"
    elif unit_time.lower() in ["s", "second", "secondly", "per second"]:
        return "s"
    else: 
        raise ValueError(
            f"Time unit must be 'a' for annual, 'm' for monthly, 'd' for daily, or 's' for second. Currently {unit_time}"
        )

def log_strain_rates(
    vx: np.ndarray | xr.DataArray,
    vy: np.ndarray | xr.DataArray,
    pixel_size: float,
    length_scale: float,
    tol: float = 10e-4,
    ydir: int = 1,
    unit_time: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.DataArray:
    """
    Calculates the directions of principal strains from e_xx, e_yy, and e_xy strain
    rates. Implementation of Chudley et al. (2021).

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        vx (np.ndarray | xr.DataArray): Array of velocity in x direction
        vy (np.ndarray | xr.DataArray): Array of velocity in y direction
        pixel_size (float): Input pixel size in measurement units for velocity (and
            thickness) grids
        length_scale (float): Set the half-length-scale to the desired value in
            distance units. Length scale will be rounded to the nearest
            integer number of pixels.
        tol (float, optional): Set the tolerance for the adaptive time-stepping scheme
            (see supplemental information to Alley et al. 2018). Value is the
            percent difference between the two stake position estimates
            divided by 100. Default of 10^-4 should be adequate for most
            applications. Defaults to 10e-4.
        ydir (int, optional): Set to 1 if the positive y-direction is in the upwards
            direction on the screen, and -1 if the positive y-direction is downwards.
            Defaults to 1.
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.DataArray: Returns the e_xx,
            e_yy, and e_xy strain rates, either as a tuple of three numpy arrays or
            as an xarray DataArray, depending on the input type.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    if type(vx) != type(vy):
        raise ValueError(
            f"Input velocity fields must be of same type. Currently {type(vx)} and {type(vy)}"
        )
    elif type(vx) == xr.DataArray:

        # Check that ydir matches the xarray coordinate values
        if vx.y.values[1] - vx.y.values[0] < 0:
            ydir_est = 1
        else:
            ydir_est = -1
        if ydir_est != ydir:
            warnings.warn(
                f"`ydir` estimated from xarray coordinate values ({ydir_est}) does not match that provided by manual/default `ydir` variable ({ydir}). Double-check this manually as output values may be incorrect."
            )

        dummy_xds = vx * 0
        vx = vx.values
        vy = vy.values
        output = "xarray"
    elif type(vx) == np.ndarray:
        output = "numpy"
    else:
        raise ValueError(
            f"Input velocity fields must be of type np.ndarray or xr.DataArray. Currently {type(vx)}"
        )

    # actually calculate this
    e_xx, e_yy, e_xy = _log_strain_rates(
        vx.squeeze(), vy.squeeze(), pixel_size, length_scale, tol, ydir
    )

    if output == "xarray":
        xds = xr.Dataset(
            data_vars={
                "e_xx": dummy_xds + e_xx,
                "e_yy": dummy_xds + e_yy,
                "e_xy": dummy_xds + e_xy,
            }
        )
        xds.data_vars["e_xx"].attrs["long_name"] = "Normal Strain Rate ($xx$)"
        xds.data_vars["e_yy"].attrs["long_name"] = "Normal Strain Rate ($yy$)"
        xds.data_vars["e_xy"].attrs["long_name"] = "Shear Strain Rate ($xy$)"
        if unit_time is not None:
            for var in xds.data_vars:
                xds[var].attrs["units"] = f"{unit_time}$^{{-1}}$"
        return xds
    else:
        return e_xx, e_yy, e_xy


def principal_strain_rate_directions(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: str = None,
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | xr.DataArray
):
    """
    Calculates the directions of principal strains from e_xx, e_yy, and e_xy strain
    rates. Implementation of Chudley et al. (2021).

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            | xr.DataArray: Returns the e_1, e_1U, e_1V, e_2, e_2U, and e_2V first and
            second principal strain rates. *U and *V denote the U and V components of
            the principal strain rate. Returns either tuple of six numpy arrays or an
            xarray DataArray, depending on the input type.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    if not type(e_xx) == type(e_yy) == type(e_xy):
        raise ValueError(
            f"Input strain rate fields must be of same type. Currently {type(e_xx)}, {type(e_yy)}, and {type(e_xy)}"
        )
    elif type(e_xx) == xr.DataArray:
        dummy_xds = e_xx * 0
        e_xx = e_xx.values
        e_yy = e_yy.values
        e_xy = e_xy.values
        output = "xarray"
    elif type(e_xx) == np.ndarray:
        output = "numpy"
    else:
        raise ValueError(
            f"Input strain rate fields must be of type np.ndarray or xr.DataArray. Currently {type(e_xx)}."
        )

    e_1, e_1U, e_1V, e_2, e_2U, e_2V = _principal_strain_rate_directions(
        e_xx, e_yy, e_xy
    )

    if output == "xarray":
        xds = xr.Dataset(
            data_vars={
                "e_1": dummy_xds + e_1,
                "e_1U": dummy_xds + e_1U,
                "e_1V": dummy_xds + e_1V,
                "e_2": dummy_xds + e_2,
                "e_2U": dummy_xds + e_2U,
                "e_2V": dummy_xds + e_2V,
            }
        )
        xds.data_vars["e_1"].attrs["long_name"] = "First Principal Strain Rate"
        xds.data_vars["e_1U"].attrs["long_name"] = "U Component of First Principal Strain Rate"
        xds.data_vars["e_1V"].attrs["long_name"] = "V Component of First Principal Strain Rate"
        xds.data_vars["e_2"].attrs["long_name"] = "Second Principal Strain Rate"
        xds.data_vars["e_2U"].attrs["long_name"] = "U Component of Second Principal Strain Rate"
        xds.data_vars["e_2V"].attrs["long_name"] = "V Component of Second Principal Strain Rate"
        if unit_time is not None:
            for var in xds.data_vars:
                xds[var].attrs["units"] = f"{unit_time}$^{{-1}}$"
        return xds
    else:
        return e_1, e_1U, e_1V, e_2, e_2U, e_2V


def principal_strain_rate_magnitudes(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: str = None,
) -> Tuple[np.ndarray, np.ndarray] | xr.DataArray:
    """
    Given e_xx, e_yy, and e_xy, return principal strain rates following
    methods in Nye (1959) and Harper et al. (1998) - i.e. not using
    eigenvectors. Quicker to compute, only returns magnitude values.

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray] | xr.DataArray: Returns the e_1 and e_2 strain
            rates, either as a tuple of two numpy arrays or as an xarray Dataset,
            depending on the input type.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Calculate first and second principal strain rates
    e_1 = 0.5 * (e_xx + e_yy) + np.sqrt(0.25 * np.square(e_xx - e_yy) + np.square(e_xy))
    e_2 = 0.5 * (e_xx + e_yy) - np.sqrt(0.25 * np.square(e_xx - e_yy) + np.square(e_xy))

    if type(e_xx) == xr.DataArray:
        xds = xr.Dataset(
            data_vars={
                "e_1": e_1,
                "e_2": e_2,
            }
        )
        xds.data_vars["e_1"].attrs["long_name"] = "First Principal Strain Rate"
        xds.data_vars["e_2"].attrs["long_name"] = "Second Principal Strain Rate"
        if unit_time is not None:
            for var in xds.data_vars:
                xds[var].attrs["units"] = f"{unit_time}$^{{-1}}$"
        return xds
    else:
        return e_1, e_2


def flow_direction(vx: np.ndarray | xr.DataArray, vy: np.ndarray | xr.DataArray):
    """
    Calculates a grid of flow directions (in radians) so that the grid-oriented
    strain rates can be rotated to align with local flow directions

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        vx (np.ndarray | xr.DataArray): array of velocity in x direction
        vy (np.ndarray | xr.DataArray): array of velocity in y direction

    Returns:
        angle (np.ndarray): Flow direction in radians. Returns either a numpy array
            or an xarray DataArray, depending on the input type.
    """

    # # Sanitise inputs
    # if type(vx) != type(vy):
    #     raise ValueError(
    #         f"Input velocity fields must be of same type. Currently {type(vx)} and {type(vy)}"
    #     )
    # elif type(vx) == xr.DataArray:
    #     dummy_xds = vx * 0
    #     vx = vx.values
    #     vy = vy.values
    #     output = "xarray"
    # elif type(vx) == np.ndarray:
    #     output = "numpy"
    # else:
    #     raise ValueError(
    #         f"Input velocity fields must be of type np.ndarray or xr.DataArray. Currently {type(vx)}"
    #     )

    angle = np.degrees(np.arctan(vy / vx))
    angle = np.where(vx > 0, angle, angle + 180)

    # from 0 - 360 degrees, following Alley et al. 2018
    # angle = np.where(angle < 0, angle + 360, angle)

    # # from -180 to 180, following Bindschadler et al. 1996
    angle = np.where(angle > 180, angle - 360, angle)

    angle = np.deg2rad(angle)

    if type(angle) == xr.DataArray:
        angle = angle.rename('angle')
        angle.attrs["long_name"] = "Flow Direction"
        angle.attrs["units"] = "radians"
        return angle
    else:
        return angle


def rotated_strain_rates(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    angle: np.ndarray | xr.DataArray,
    unit_time: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset:
    """
    Given e_xx, e_yy, and e_xy, return various rotated strain rates
    following Bindschadler et al. (1996).

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Parameters:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        angle (np.ndarray | xr.DataArray): array of flow direction in radians
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset: Longitudinal, transverse
            and shear strain rates. Returns either a numpy array or an xarray Dataset,
            depending on the input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Calculate longitudinal and transverse strain rate (Bindschadler et al. 1996)

    with np.errstate(divide="ignore", invalid="ignore"):
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

    if type(e_xx) == xr.DataArray:
        xds = xr.Dataset(
            data_vars={
                "e_lon": e_lon,
                "e_trn": e_trn,
                "e_shr": e_shr,
            },
        )
        xds.e_lon.attrs["long_name"] = "Longitudinal Strain Rate"
        xds.e_trn.attrs["long_name"] = "Transverse Strain Rate"
        xds.e_shr.attrs["long_name"] = "Shear Strain Rate"
        if unit_time is not None:
            for var in xds.data_vars:
                xds[var].attrs["units"] = f"{unit_time}$^{{-1}}$"
        return xds
    else:
        return e_lon, e_trn, e_shr


def effective_strain_rate(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: str = None,
) -> np.ndarray | xr.DataArray:
    """
    Given e_xx, e_yy, and e_xy, return effective strain rate (Cuffey & Paterson p.59).

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Parameters:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        np.ndarray | xr.DataArray: Effective strain rate. Returns either a numpy array
            or an xarray DataArray, depending on the input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Calculate effective strain rate (Cuffey & Paterson p.59)
    e_E = np.sqrt(0.5 * (e_xx**2 + e_yy**2) + e_xy**2)

    if type(e_E) == xr.DataArray:
        e_E = e_E.rename('e_E')
        e_E.attrs["long_name"] = "Effective Strain Rate"
        if unit_time is not None:
            e_E.attrs["units"] = f"{unit_time}$^{{-1}}$"
        return e_E
    else:
        return e_E


def strain_rate_uncertainty(
    ve_x: np.ndarray | xr.DataArray,
    ve_y: np.ndarray | xr.DataArray,
    length_scale: float,
    unit_time: str = None,
) -> np.ndarray | xr.DataArray:
    """
    Calculate strain rate uncertainty following Poinar and Andrews (2021, eq. 4):

    $\\delta_{\\dot{\epsilon}} = \\frac{1}{\\Delta x} \sqrt{(\\delta u)^2 + (\\delta v)^2}$

    Where Δx is the baseline distance between observation points (i.e. the lengthscale),
    and δu and δv are the velocity uncertainties in the x and y directions.

    Args:
        ve_x (np.ndarray | xr.DataArray): Velocity uncertainty in x direction.
        ve_y (np.ndarray | xr.DataArray): Velocity uncertainty in y direction.
        length_scale (float): Half length scale of strain rate calculation.
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        np.ndarray | xr.DataArray: Strain rate uncertainty. Returns numpy array or
            xarray DataArray depending on input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)
    if type(ve_x) != type(ve_y):
        raise ValueError("Input velocity uncertainties must be of same type")
    elif type(ve_x) == xr.DataArray:
        ve_x_arr = ve_x.values
        ve_y_arr = ve_y.values
        output = "xarray"
    elif type(ve_x) == np.ndarray:
        ve_x_arr = ve_x
        ve_y_arr = ve_y
        output = "numpy"
    else:
        raise ValueError(
            "Input velocity uncertainties must be of type np.ndarray or xr.DataArray"
        )

    # Calculate strain rate uncertainty as a numpy array
    uncertainty = (1 / (2 * length_scale) ) * np.sqrt( (ve_x_arr ** 2) + (ve_y_arr ** 2) ) 

    if output == "xarray":
        xda = (ve_x * 0 + uncertainty).rename('uncertainty')
        try:
            xda.attrs["long_name"] = "Strain Rate Uncertainty"
            if unit_time is not None:
                xda.attrs["units"] = f"{unit_time}$^{{-1}}$"
        except:
            pass
        return xda

    else:
        return uncertainty
