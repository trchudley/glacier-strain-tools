"""
Core functions to calculate glacier stresses from strain rates.
"""

import warnings

import numpy as np
import xarray as xr

from typing import overload, Literal, Optional

from ._utils import _sanitise_unit_time, _all_numpy, _all_xarray


@overload
def deviatoric(
    strain_rate: np.ndarray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> np.ndarray: ...


@overload
def deviatoric(
    strain_rate: xr.DataArray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> xr.DataArray: ...


def deviatoric(
    strain_rate: np.ndarray | xr.DataArray,
    effective_strain_rate: np.ndarray | xr.DataArray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_time: Optional[str] = "a",
    unit_temp: Literal["C", "K"] = "K",
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> np.ndarray | xr.DataArray:
    """
    Calculate deviatoric stress for a given strain rate field.

    Strain rates assumed to be in units of a-1 unless otherwise specified.

    Parameters following Wells-Moran _et al._ (2024).

    :param strain_rate: Strain rate.
    :type strain_rate: np.ndarray | xr.DataArray
    :param effective_strain_rate: Effective strain rate.
    :type effective_strain_rate: np.ndarray | xr.DataArray
    :param temperature: Temperature.
    :type temperature: np.ndarray | xr.DataArray | float
    :param n: Number of dimensions.
    :type n: Literal[3, 4]
    :param unit_time: Set to apply a time unit to the input strain rates.
        Set to 'a' for annual, 'm' for monthly, or 'd' for daily. Defaults to
        None, which assumes units of a-1.
    :type unit_time: Optional[str], optional
    :param unit_temp: Set to apply a temperature unit to the input temperature.
        Set to 'C' for Celsius or 'K' for Kelvin. Defaults to 'K', will sanity check.
    :type unit_temp: Literal["C", "K"], optional
    :param unit_output: Set to apply a unit to the output stress.
        Set to 'Pa', 'kPa', or 'MPa'. Defaults to 'Pa'.
    :type unit_output: Literal["Pa", "kPa", "MPa"], optional
    :param long_name: Set to apply a long name to the output DataArray.
        Defaults to 'Deviatoric Stress'.
    :type long_name: Optional[str], optional

    Returns:
        np.ndarray | xr.DataArray: Deviatoric stress. Returns either a numpy array
            or an xarray DataArray, depending on the input.
    """

    # Sanitise inputs
    # Time
    if unit_time is not None:
        unit_time = _sanitise_unit_time(unit_time)

    # Output
    if unit_output not in ["Pa", "kPa", "MPa"]:
        raise ValueError("unit_output must be 'Pa', 'kPa', or 'MPa'")

    # Check if all inputs are (i) the same and (ii) either ndarray or xarray
    if _all_xarray(strain_rate, effective_strain_rate):
        output = "xarray"
    elif _all_numpy(strain_rate, effective_strain_rate):
        output = "numpy"
    else:
        raise ValueError(
            "Input strain_rate and effective_strain_rate must be either both numpy arrays or xarray DataArrays"
        )

    # Temperature
    if unit_temp.upper() not in ["C", "K"]:
        raise ValueError("unit_temp must be 'C' or 'K'")
    try:
        mean_temp = float(np.nanmean(temperature))
        warning_str = " mean"
    except Exception:
        mean_temp = float(temperature)
        warning_str = ""

    if unit_temp.upper() == "C" and mean_temp > 20:
        warnings.warn(
            f"Are you sure about the assumed temperature unit? (`unit_temp = 'C'`) "
            f"The{warning_str} temperature is {mean_temp:.1f} °C, which seems too high for ice. "
            f"Perhaps input is in Kelvin?"
        )
    elif unit_temp.upper() == "K" and mean_temp < 100:
        warnings.warn(
            f"Are you sure about the assumed temperature unit? (`unit_temp = 'K'`) "
            f"Mean temperature is {mean_temp:.1f} K, which seems too low for ice. "
            f"Perhaps input is in Celsius?"
        )
    else:
        pass

    # If units are reported to be d-1 or s-1, convert to a-1
    if unit_time == "a":
        pass
    elif unit_time == "m":
        strain_rate = strain_rate * 365 / 12
        effective_strain_rate = effective_strain_rate * 365 / 12
    elif unit_time == "d":
        strain_rate = strain_rate * 365
        effective_strain_rate = effective_strain_rate * 365
    elif unit_time == "s":
        strain_rate = strain_rate * 365 * 24 * 60 * 60
        effective_strain_rate = effective_strain_rate * 365 * 24 * 60 * 60

    elif output == "xarray":

        # if no units set, simply continue
        units = [a.attrs.get("units") for a in (strain_rate, effective_strain_rate)]
        if all(u is None for u in units):
            warnings.warn(
                "No units found in xarray DataArrays (strain_rate, effective_strain_rate). Assuming units of 'a-1'. If not correct, override by setting 'unit_time' parameter"
            )
        # If different units, raise error
        elif len(set(units)) > 1:
            raise ValueError(
                f"Inferred units from xarray DataArrays (strain_rate, effective_strain_rate) do not match: {units}. Ensure they match, and/or override by setting 'unit_time' parameter"
            )

        if "units" in strain_rate.attrs and strain_rate.attrs["units"] == "d^{-1}":
            strain_rate = strain_rate * 365
            effective_strain_rate = effective_strain_rate * 365
        elif "units" in strain_rate.attrs and strain_rate.attrs["units"] == "m^{-1}":
            strain_rate = strain_rate * 365 / 12
            effective_strain_rate = effective_strain_rate * 365 / 12
        elif "units" in strain_rate.attrs and strain_rate.attrs["units"] == "s^{-1}":
            strain_rate = strain_rate * 365 * 24 * 60 * 60
            effective_strain_rate = effective_strain_rate * 365 * 24 * 60 * 60
        else:
            # warn about unit time
            warnings.warn(
                "`unit_time` not set and no readable units found in xarray DataArray: assuming strain rates are in units of a-1",
            )
    else:
        # warn about unit time
        warnings.warn(
            "`unit_time` not set: assuming strain rates are in units of a-1",
        )

    # Initialise parameters from Table 1 of Wells-Moran _et al._ (2024).

    if unit_temp.upper() == "C":
        temperature = temperature + 273.15

    # Prefactor
    if n == 3:
        A0 = 2.290e4  # kPa^-3 a^-1, Duval et al. (1983)
    elif n == 4:
        A0 = 12.614  # kPa^-4 a^-1, Goldsby and Kohlstedt (2001)
    else:
        raise ValueError("n must be 3 or 4")

    # # Convert A0 from kPa^-3 a^-1 to Pa^-3 a^-1
    A0 = A0 * 1e3**-3

    # Activation energy from Duval et al. (1983)
    Q_c = 60e3  # J mol^-1

    # Ideal gas constant
    R = 8.314  # J K^-1 mol^-1

    # Calculate flow rate parameter A (Equation 6)
    A = A0 * np.exp(-Q_c / (R * temperature))

    # Calculate dynamic viscosity η (Equation 4)
    # η = 1/(2A^(1/n)) * ε̇_E^((1-n)/n)
    eta = (1 / (2 * A ** (1 / n))) * effective_strain_rate ** ((1 - n) / n)

    # Calculate deviatoric stress (Equation 3)
    # 2ηε̇_ij = τ_ij
    tau = 2 * eta * strain_rate

    if unit_output == "kPa":
        tau = tau * 1e-3
    elif unit_output == "MPa":
        tau = tau * 1e-3 * 1e-3

    if output == "xarray":

        tau.attrs["long_name"] = long_name
        tau.attrs["units"] = unit_output
        tau.attrs["stress_exponent"] = n

        return tau

    elif output == "numpy":
        return tau


@overload
def cauchy(
    deviatoric_stress: np.ndarray,
    deviatoric_first_principal: np.ndarray,
    deviatoric_second_principal: np.ndarray,
    long_name: Optional[str] = "Cauchy Stress",
) -> np.ndarray: ...


@overload
def cauchy(
    deviatoric_stress: xr.DataArray,
    deviatoric_first_principal: xr.DataArray,
    deviatoric_second_principal: xr.DataArray,
    long_name: Optional[str] = "Cauchy Stress",
) -> xr.DataArray: ...


def cauchy(
    deviatoric_stress: np.ndarray | xr.DataArray,
    deviatoric_first_principal: np.ndarray | xr.DataArray,
    deviatoric_second_principal: np.ndarray | xr.DataArray,
    long_name: Optional[str] = "Cauchy Stress",
) -> np.ndarray | xr.DataArray:
    r"""
    Calculate Cauchy stress for a given deviatoric stress using the first and
    second principal deviatoric stresses. These are related through the isotropic
    pressure such that:

    $ \sigma_{ij} =  \tau_{ij} + p \delta_{ij}, $

    where $ p = \frac{1}{3} (\sigma_{1} + \sigma_{2} + \sigma_{zz}) $. Assuming
    that $ \sigma_{zz} = 0 $, $ p = \frac{1}{3} (\sigma_{1} + \sigma_{2}) $ which
    $ = \tau{1} + \tau{2} $. Hence,

    $ \sigma_{ij} = \tau_{ij} + \tau{1} + \tau{2}. $

    Will accept a numpy array or an xarray Dataarray, and will return the same
    data type.

    :param deviatoric_stress: Deviatoric stress.
    :type deviatoric_stress: np.ndarray | xr.DataArray
    :param deviatoric_first_principal: First principal deviatoric stress.
    :type deviatoric_first_principal: np.ndarray | xr.DataArray
    :param deviatoric_second_principal: Second principal deviatoric stress.
    :type deviatoric_second_principal: np.ndarray | xr.DataArray
    :param long_name: Optional long name to attach to output xarray DataArray.
        Defaults to "Cauchy Stress".
    :type long_name: str

    :return: Cauchy stress.
    :rtype: np.ndarray | xr.DataArray
    """

    if isinstance(deviatoric_stress, xr.DataArray):
        output = "xarray"
    elif isinstance(deviatoric_stress, np.ndarray):
        output = "numpy"
    else:
        raise ValueError(
            f"Input deviatoric stress must be either np.ndarray or xr.DataArray."
        )

    cauchy_stress = (
        deviatoric_stress + deviatoric_first_principal + deviatoric_second_principal
    )

    if output == "xarray":

        cauchy_stress.attrs["long_name"] = long_name
        cauchy_stress.attrs["units"] = deviatoric_first_principal.attrs["units"]

        return cauchy_stress

    else:
        return cauchy_stress
