"""
Core functions to calculate strain rates from velocity fields.
"""

import warnings

import numpy as np
import xarray as xr

from typing import overload, TypeGuard, Tuple, Optional, Literal

from ._numba import _log_strain_rates, _principal_strain_rate_eigenvalues
from ._utils import _all_numpy, _all_xarray, _sanitise_unit_time


@overload
def logarithmic(
    vx: np.ndarray,
    vy: np.ndarray,
    pixel_size: float,
    length_scale: float,
    tol: float = 10e-4,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def logarithmic(
    vx: xr.DataArray,
    vy: xr.DataArray,
    pixel_size: float,
    length_scale: float,
    tol: float = 10e-4,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> xr.Dataset: ...


def logarithmic(
    vx: np.ndarray | xr.DataArray,
    vy: np.ndarray | xr.DataArray,
    pixel_size: float,
    length_scale: float,
    tol: float = 10e-4,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset:
    r"""
    Calculates the logarithmic strain rates ($\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, 
    and $\dot{\varepsilon}_{xy}$) from provided $v_x$ and $v_y$ velocity fields. Implementation of Alley et
    al. (2018).

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
            Defa1ults to 1.
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

    if _all_xarray(vx, vy):

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

    elif _all_numpy(vx, vy):
        output = "numpy"

    else:
        raise ValueError(
            f"Input velocity fields must be the same type and either np.ndarray or xr.DataArray."
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


@overload
def nominal(
    vx: np.ndarray,
    vy: np.ndarray,
    pixel_size: float,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def nominal(
    vx: xr.DataArray,
    vy: xr.DataArray,
    pixel_size: float,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> xr.Dataset: ...


def nominal(
    vx: np.ndarray | xr.DataArray,
    vy: np.ndarray | xr.DataArray,
    pixel_size: float,
    ydir: int = 1,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset:
    r"""
    Calculates the nominal strain rates ($\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, 
    and $\dot{\varepsilon}_{xy}$) from provided $v_x$ and $v_y$ velocity fields. Nominal strain rates
    are calculated using the finite difference of the velocity field: here, using numpy.diff().

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        vx (np.ndarray | xr.DataArray): Array of velocity in x direction
        vy (np.ndarray | xr.DataArray): Array of velocity in y direction
        pixel_size (float): Input pixel size in measurement units for velocity (and
            thickness) grids
        ydir (int, optional): Set to 1 if the positive y-direction is in the upwards
            direction on the screen, and -1 if the positive y-direction is downwards.
            Defa1ults to 1.
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

    if _all_xarray(vx, vy):

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
        # vx = vx.values
        # vy = vy.values
        output = "xarray"

    elif _all_numpy(vx, vy):
        output = "numpy"

    else:
        raise ValueError(
            f"Input velocity fields must be the same type and either np.ndarray or xr.DataArray."
        )

    # Calculate strain rates e_xx, e_yy, and e_xy from np.diff

    # Calculate velocity gradients using finite differences

    if output == "xarray":

        # Calculate velocity gradients using xarray diff and coordinate spacing
        # dvx/dx - derivative of x-velocity in x-direction
        dvx_dx = vx.diff(dim="x") / vx.x.diff(dim="x")

        # dvy/dy - derivative of y-velocity in y-direction
        dvy_dy = vy.diff(dim="y") / vy.y.diff(dim="y")

        # dvx/dy - derivative of x-velocity in y-direction
        dvx_dy = vx.diff(dim="y") / vx.y.diff(dim="y")

        # dvy/dx - derivative of y-velocity in x-direction
        dvy_dx = vy.diff(dim="x") / vy.x.diff(dim="x")

        # Normal strain rates
        e_xx = dvx_dx
        e_yy = dvy_dy

        # Shear strain rate (average of the two cross-derivatives)
        # xarray automatically handles alignment
        e_xy = 0.5 * (dvx_dy + dvy_dx)

    else:
        # dvx/dx - derivative of x-velocity in x-direction
        dvx_dx = np.diff(vx, axis=1) / pixel_size  # axis=1 is x-direction

        # dvy/dy - derivative of y-velocity in y-direction
        dvy_dy = np.diff(vy, axis=0) / pixel_size  # axis=0 is y-direction

        # dvx/dy - derivative of x-velocity in y-direction
        dvx_dy = np.diff(vx, axis=0) / pixel_size

        # dvy/dx - derivative of y-velocity in x-direction
        dvy_dx = np.diff(vy, axis=1) / pixel_size

        # Normal strain rates
        e_xx = dvx_dx
        e_yy = dvy_dy

        # Shear strain rate (average of the two cross-derivatives)
        # Need to handle different shapes from diff operations
        # Trim to common dimensions
        min_y = min(dvx_dy.shape[0], dvy_dx.shape[0])
        min_x = min(dvx_dy.shape[1], dvy_dx.shape[1])

        e_xy = 0.5 * (dvx_dy[:min_y, :min_x] + dvy_dx[:min_y, :min_x])

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


@overload
def _principal_eigenvalues(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.Dataset: ...


@overload
def _principal_eigenvalues(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def _principal_eigenvalues(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | xr.Dataset
):
    r"""
    Calculates the directions of principal strains from $\dot{\varepsilon}_{xx}$, 
    $\dot{\varepsilon}_{yy}$, and $\dot{\varepsilon}_{xy}$ strain
    rates.

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

    # Type checking
    if _all_xarray(e_xx, e_yy, e_xy):
        dummy_xds = e_xx * 0
        if "units" in e_xx.attrs:
            dummy_xds.attrs["units"] = e_xx.attrs["units"]
        e_xx = e_xx.values
        e_yy = e_yy.values
        e_xy = e_xy.values
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy):
        output = "numpy"
    else:
        raise ValueError(
            f"Input strain rate fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    e_1, e_1U, e_1V, e_2, e_2U, e_2V = _principal_strain_rate_eigenvalues(
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
        xds.data_vars["e_1U"].attrs[
            "long_name"
        ] = "U Component of First Principal Strain Rate"
        xds.data_vars["e_1V"].attrs[
            "long_name"
        ] = "V Component of First Principal Strain Rate"
        xds.data_vars["e_2"].attrs["long_name"] = "Second Principal Strain Rate"
        xds.data_vars["e_2U"].attrs[
            "long_name"
        ] = "U Component of Second Principal Strain Rate"
        xds.data_vars["e_2V"].attrs[
            "long_name"
        ] = "V Component of Second Principal Strain Rate"
        if unit_time is not None:
            for var in xds.data_vars:
                xds[var].attrs["units"] = f"{unit_time}$^{{-1}}$"
        elif "units" in dummy_xds.attrs:
            for var in xds.data_vars:
                xds[var].attrs["units"] = dummy_xds.attrs["units"]
        return xds
    else:
        return e_1, e_1U, e_1V, e_2, e_2U, e_2V


@overload
def _principal_magnitudes(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def _principal_magnitudes(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.Dataset: ...


def _principal_magnitudes(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray] | xr.Dataset:
    r"""
    Given $\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, and $\dot{\varepsilon}_{xy}$, 
    return principal strain rates following the method of Nye (1959):

    $$
    \dot{\varepsilon}_1, \dot{\varepsilon}_2 = \frac{1}{2} (\dot{\varepsilon}_{xx} + 
    \dot{\varepsilon}_{yy}) \pm \sqrt{ \frac{1}{4} (\dot{\varepsilon}_{xx} - 
    \dot{\varepsilon}_{yy})^2 + \dot{\varepsilon}_{xy}^2 }
    $$
    
    Quicker to compute, only returns magnitude values.

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

    # Type check
    if _all_xarray(e_xx, e_yy, e_xy):
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy):
        output = "numpy"
    else:
        raise ValueError("All inputs must be either numpy arrays or xarray DataArrays.")

    # Calculate principal strain rates
    e_1 = 0.5 * (e_xx + e_yy) + np.sqrt(0.25 * (e_xx - e_yy) ** 2 + e_xy**2)
    e_2 = 0.5 * (e_xx + e_yy) - np.sqrt(0.25 * (e_xx - e_yy) ** 2 + e_xy**2)

    # Return
    if output == "xarray":

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
        elif "units" in e_xx.attrs:
            for var in xds.data_vars:
                xds[var].attrs["units"] = e_xx.attrs["units"]
        return xds

    else:
        return e_1, e_2


@overload
def principal(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    unit_time: Optional[str] = None,
    output: Literal["directions", "magnitudes"] = "directions",
) -> (
    Tuple[np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
): ...


@overload
def principal(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    unit_time: Optional[str] = None,
    output: Literal["directions", "magnitudes"] = "directions",
) -> xr.Dataset: ...


def principal(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
    output: Literal["eigenvectors", "magnitudes"] = "eigenvectors",
) -> (
    Tuple[np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | xr.Dataset
):
    r"""
    Calculates the first and second principal strain rates from $\dot{\varepsilon}_{xx}$, 
    $\dot{\varepsilon}_{yy}$, and $\dot{\varepsilon}_{xy}$ strain rates.

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    If `output = "eigenvectors"`, principal strain directions are calculated as
    eigenvectors and the directional strain rates are returned as e_1, e_1U,
    e_1V, e_2, e_2U, e_2V. If `output = "magnitudes"`, principal strain rate
    magnitudes are calculated following methods in Nye (1959) and Harper et al.
    (1998) and returned as e_1 and e_2. This is quicker to compute, but only
    returns magnitude values.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.
        output (str, optional): Set to "eigenvectors" or "magnitudes". Defaults to
            "eigenvectors".

    Returns:
        Tuple[np.ndarray, np.ndarray] |  Tuple[np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray]| xr.Dataset: Returns the first and
            second principal strain rates. If `output = "eigenvectors"`, returns the e_1,
            e_1U, e_1V, e_2, e_2U, and e_2V first and second principal strain rates.
            *U and *V denote the U and V components of the principal strain rate.
            Returns either tuple of six numpy arrays or an xarray Dataset, depending
            on the input type. If `output = "magnitudes"`, returns the e_1 and e_2
            strain rates. Reurns either tuple of two numpy arrays or an xarray
            Dataset, depending on the input type.
    """

    if output == "eigenvectors":
        return _principal_eigenvalues(e_xx, e_yy, e_xy, unit_time=unit_time)
    elif output == "magnitudes":
        return _principal_magnitudes(e_xx, e_yy, e_xy, unit_time=unit_time)


@overload
def longitudinal(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    angle: np.ndarray,
    unit_time: Optional[str] = None,
) -> np.ndarray: ...


@overload
def longitudinal(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    angle: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.DataArray: ...


def longitudinal(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    angle: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> np.ndarray | xr.DataArray:
    r"""
    Given $\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, and
    $\dot{\varepsilon}_{xy}$, return longitudinal (along-flow) strain
    rate following Bindschadler et al. (1996):

    $$
    \dot{\varepsilon}_{lon} = \dot{\varepsilon}_{xx} \cos^2 \theta +
    2 \dot{\varepsilon}_{xy} \cos \theta \sin \theta +
    \dot{\varepsilon}_{yy} \sin^2 \theta
    $$

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        angle (np.ndarray | xr.DataArray): Array of flow direction in radians
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        np.ndarray | xr.DataArray: Longitudinal strain rate. Returns either a
            numpy array or an xarray DataArray, depending on the input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(e_xx, e_yy, e_xy, angle):
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy, angle):
        output = "numpy"
    else:
        raise ValueError(
            f"Input fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    # Calculate longitudinal strain rate (Bindschadler et al. 1996)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_lon = (
            (e_xx * np.cos(angle) ** 2)
            + (2 * e_xy * np.cos(angle) * np.sin(angle))
            + (e_yy * np.sin(angle) ** 2)
        )

    if output == "xarray":
        e_lon = e_lon.rename("e_lon")
        e_lon.attrs["long_name"] = "Longitudinal Strain Rate"
        if unit_time is not None:
            e_lon.attrs["units"] = f"{unit_time}$^{{-1}}$"
        elif "units" in e_xx.attrs:
            e_lon.attrs["units"] = e_xx.attrs["units"]
        return e_lon
    else:
        return e_lon


@overload
def transverse(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    angle: np.ndarray,
    unit_time: Optional[str] = None,
) -> np.ndarray: ...


@overload
def transverse(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    angle: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.DataArray: ...


def transverse(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    angle: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> np.ndarray | xr.DataArray:
    r"""
    Given $\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, and
    $\dot{\varepsilon}_{xy}$, return transverse (across-flow) strain
    rate following Bindschadler et al. (1996):

    $$
    \dot{\varepsilon}_{trn} = \dot{\varepsilon}_{xx} \sin^2 \theta -
    2 \dot{\varepsilon}_{xy} \cos \theta \sin \theta +
    \dot{\varepsilon}_{yy} \cos^2 \theta
    $$

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        angle (np.ndarray | xr.DataArray): Array of flow direction in radians
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        np.ndarray | xr.DataArray: Transverse strain rate. Returns either a
            numpy array or an xarray DataArray, depending on the input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(e_xx, e_yy, e_xy, angle):
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy, angle):
        output = "numpy"
    else:
        raise ValueError(
            f"Input fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    # Calculate transverse strain rate (Bindschadler et al. 1996)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_trn = (
            (e_xx * np.sin(angle) ** 2)
            - (2 * e_xy * np.cos(angle) * np.sin(angle))
            + (e_yy * np.cos(angle) ** 2)
        )

    if output == "xarray":
        e_trn = e_trn.rename("e_trn")
        e_trn.attrs["long_name"] = "Transverse Strain Rate"
        if unit_time is not None:
            e_trn.attrs["units"] = f"{unit_time}$^{{-1}}$"
        elif "units" in e_xx.attrs:
            e_trn.attrs["units"] = e_xx.attrs["units"]
        return e_trn
    else:
        return e_trn


@overload
def shear(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    angle: np.ndarray,
    unit_time: Optional[str] = None,
) -> np.ndarray: ...


@overload
def shear(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    angle: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.DataArray: ...


def shear(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    angle: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> np.ndarray | xr.DataArray:
    r"""
    Given $\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, and
    $\dot{\varepsilon}_{xy}$, return shear strain rate in the flow-aligned
    coordinate system following Bindschadler et al. (1996):

    $$
    \dot{\varepsilon}_{shr} = (\dot{\varepsilon}_{yy} -
    \dot{\varepsilon}_{xx}) \cos \theta \sin \theta +
    \dot{\varepsilon}_{xy} (\cos^2 \theta - \sin^2 \theta)
    $$

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        angle (np.ndarray | xr.DataArray): Array of flow direction in radians
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        np.ndarray | xr.DataArray: Shear strain rate. Returns either a numpy
            array or an xarray DataArray, depending on the input.
    """

    # Sanitise inputs
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(e_xx, e_yy, e_xy, angle):
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy, angle):
        output = "numpy"
    else:
        raise ValueError(
            f"Input fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    # Calculate shear strain rate (Bindschadler et al. 1996)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_shr = ((e_yy - e_xx) * np.cos(angle) * np.sin(angle)) + (
            e_xy * (np.cos(angle) ** 2 - np.sin(angle) ** 2)
        )

    if output == "xarray":
        e_shr = e_shr.rename("e_shr")
        e_shr.attrs["long_name"] = "Shear Strain Rate"
        if unit_time is not None:
            e_shr.attrs["units"] = f"{unit_time}$^{{-1}}$"
        elif "units" in e_xx.attrs:
            e_shr.attrs["units"] = e_xx.attrs["units"]
        return e_shr
    else:
        return e_shr


@overload
def rotated(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    angle: np.ndarray,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def rotated(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    angle: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.Dataset: ...


def rotated(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    angle: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset:
    r"""
    Wrapper function returning all rotated strain-rate components (longitudinal 
    ($\dot{\varepsilon}_{lon}$), transverse ($\dot{\varepsilon}_{trn}$),
    and shear ($\dot{\varepsilon}_{shr}$)) as a single xarray Dataset or tuple of
    numpy arrays.

    Recommend using `longitudinal`, `transverse`, and `shear` directly when only
    one component is required.

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
        e_xx (np.ndarray | xr.DataArray): Array of strain rate in xx direction
        e_yy (np.ndarray | xr.DataArray): Array of strain rate in yy direction
        e_xy (np.ndarray | xr.DataArray): Array of strain rate in xy direction
        angle (np.ndarray | xr.DataArray): Array of flow direction in radians
        unit_time (str, optional): Set to apply a time unit to the output strain
            rates. Set to 'a' for annual or 'd' for daily. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] | xr.Dataset: Longitudinal,
            transverse, and shear strain rates. Returns either a tuple of three
            numpy arrays or an xarray Dataset, depending on the input.
    """

    e_lon = longitudinal(e_xx, e_yy, e_xy, angle, unit_time=unit_time)
    e_trn = transverse(e_xx, e_yy, e_xy, angle, unit_time=unit_time)
    e_shr = shear(e_xx, e_yy, e_xy, angle, unit_time=unit_time)

    if isinstance(e_lon, xr.DataArray):
        xds = xr.Dataset(
            data_vars={
                "e_lon": e_lon,
                "e_trn": e_trn,
                "e_shr": e_shr,
            },
        )
        return xds
    else:
        return e_lon, e_trn, e_shr


@overload
def effective(
    e_xx: np.ndarray,
    e_yy: np.ndarray,
    e_xy: np.ndarray,
    unit_time: Optional[str] = None,
) -> np.ndarray: ...


@overload
def effective(
    e_xx: xr.DataArray,
    e_yy: xr.DataArray,
    e_xy: xr.DataArray,
    unit_time: Optional[str] = None,
) -> xr.DataArray: ...


def effective(
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray,
    e_xy: np.ndarray | xr.DataArray,
    unit_time: Optional[str] = None,
    form: Literal["full", "planar"] = "full",
) -> np.ndarray | xr.DataArray:
    r"""
    Given $\dot{\varepsilon}_{xx}$, $\dot{\varepsilon}_{yy}$, and 
    $\dot{\varepsilon}_{xy}$, return effective strain rate. 

    Default form is "full" for 3D effective strain rate, which is calculated as

    $$
    \dot{\varepsilon}_{E} = \sqrt{ \frac{1}{2} [ \dot{\varepsilon}_{xx}^2 + 
    \dot{\varepsilon}_{yy}^2 + (-\dot{\varepsilon}_{xx} - \dot{\varepsilon}_{yy})^2 ] + \dot{\varepsilon}_{xy}^2 }.
    $$

    Alternative form is "planar" for 2D effective strain rate, which is calculated as

    $$
    \dot{\varepsilon}_{E} = \sqrt{ \frac{1}{2} \left( \dot{\varepsilon}_{xx}^2 + \dot{\varepsilon}_{yy}^2 \right)+ \dot{\varepsilon}_{xy}^2 },
    $$

    ignoring the vertical strain rate comopnent.

    Accepts numpy arrays or xarray DataArrays. Output type will match the input.

    Args:
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

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(e_xx, e_yy, e_xy):
        output = "xarray"
    elif _all_numpy(e_xx, e_yy, e_xy):
        output = "numpy"
    else:
        raise ValueError(
            f"Input fields must be all the same type and either np.ndarray or xr.DataArray."
        )
    
    if form == "full":
        e_E = np.sqrt(
            0.5 * (e_xx**2 + e_yy**2 + (-e_xx - e_yy) ** 2) + e_xy**2
        )
    elif form == "planar":
        e_E = np.sqrt(0.5 * (e_xx**2 + e_yy**2) + e_xy**2)
    else:
        raise ValueError(f"Invalid form '{form}'. Must be 'full' or 'planar'.")


    if output == "xarray":
        e_E = e_E.rename("e_E")
        e_E.attrs["long_name"] = "Effective Strain Rate"
        if unit_time is not None:
            e_E.attrs["units"] = f"{unit_time}$^{{-1}}$"
        elif "units" in e_xx.attrs:
            e_E.attrs["units"] = e_xx.attrs["units"]
        return e_E
    else:
        return e_E


@overload
def uncertainty(
    ve_x: np.ndarray,
    ve_y: np.ndarray,
    length_scale: float,
    unit_time: Optional[str] = None,
) -> np.ndarray: ...


@overload
def uncertainty(
    ve_x: xr.DataArray,
    ve_y: xr.DataArray,
    length_scale: float,
    unit_time: Optional[str] = None,
) -> xr.DataArray: ...


def uncertainty(
    ve_x: np.ndarray | xr.DataArray,
    ve_y: np.ndarray | xr.DataArray,
    length_scale: float,
    unit_time: str | None = None,
) -> np.ndarray | xr.DataArray:
    r"""
    Calculate strain rate uncertainty following Poinar and Andrews (2021, eq. 4):

    $$
    \delta_{\dot{\epsilon}} = \frac{1}{\Delta x} \sqrt{(\delta u)^2 + (\delta v)^2}
    $$

    Where $\Delta x$ is the baseline distance between observation points (i.e. the length scale),
    and $\delta u$ and $\delta v$ are the velocity uncertainties in the $x$ and $y$ directions.

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

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(ve_x, ve_y):
        output = "xarray"
    elif _all_numpy(ve_x, ve_y):
        output = "numpy"
    else:
        raise ValueError(
            f"Input fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    # Calculate strain rate uncertainty as a numpy array
    uncertainty = (1 / (2 * length_scale)) * np.sqrt((ve_x**2) + (ve_y**2))

    # Output
    if output == "xarray":
        uncertainty = uncertainty.rename("uncertainty")
        try:
            uncertainty.attrs["long_name"] = "Strain Rate Uncertainty"
            if unit_time is not None:
                uncertainty.attrs["units"] = f"{unit_time}$^{{-1}}$"
        except:
            pass
        return uncertainty

    else:
        return uncertainty
