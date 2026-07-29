"""
Core functions to calculate glacier stresses from strain rates.
"""

import warnings

import numpy as np
import xarray as xr

from typing import overload, Literal, Optional

from ._utils import _sanitise_unit_time, _all_numpy, _all_xarray, _normalise_rate_unit


@overload
def deviatoric(
    strain_rate: np.ndarray,
    effective_strain_rate: np.ndarray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_time: Optional[str] = None,
    unit_temp: Literal["C", "K"] = "K",
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> np.ndarray: ...


@overload
def deviatoric(
    strain_rate: xr.DataArray,
    effective_strain_rate: xr.DataArray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_time: Optional[str] = None,
    unit_temp: Literal["C", "K"] = "K",
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> xr.DataArray: ...


def deviatoric(
    strain_rate: np.ndarray | xr.DataArray,
    effective_strain_rate: np.ndarray | xr.DataArray,
    temperature: np.ndarray | xr.DataArray | float,
    n: Literal[3, 4] = 3,
    unit_time: Optional[str] = None,
    unit_temp: Literal["C", "K"] = "K",
    unit_output: Literal["Pa", "kPa", "MPa"] = "Pa",
    long_name: Optional[str] = "Deviatoric Stress",
) -> np.ndarray | xr.DataArray:
    r"""
    Calculate deviatoric stress for a given strain rate field following the
    isotropic form of Glen's flow law:

    $$
    \tau_{ij} = A^{-\frac{1}{n}} \dot{\varepsilon}_E^{\frac{1-n}{n}} \dot{\varepsilon}_{ij},
    $$

    where

    $$
    A = A_* \exp \left[ -\frac{Q_c}{R} \left( \frac{1}{T_h} - \frac{1}{T_*} \right) \right].
    $$

    Parameters follow Cuffey and Paterson (2010):

    \begin{align*}
    A_* &= 3.5 \times 10^{-25} \quad \mathrm{Pa^{-3}\,s^{-1}} \\
    n   &= 3 \\ 
    T_* &= 263 \quad \mathrm{K} \\
    T_h &= T \quad \mathrm{K} \\
    Q_c &= 
    \begin{cases} 
    Q^- = 60 \quad \mathrm{kJ\,mol^{-1}}, & \text{if } T_h \lt T_* \\[3pt]
    Q^+ = 115 \quad \mathrm{kJ\,mol^{-1}}, & \text{if } T_h \geq T_* 
    \end{cases}
    \end{align*}

    Temperature can be provided as a scalar value or as a 2D array/dataarray of the same shape
    as the strain rate arrays.

    Input strain rates are assumed to be in units of a-1 unless otherwise specified.

    Args:
        strain_rate (np.ndarray | xr.DataArray): Strain rate.
        effective_strain_rate (np.ndarray | xr.DataArray): Effective strain rate.
        temperature (np.ndarray | xr.DataArray | float): Temperature.
        n (Literal[3, 4], optional): Value of n in Glen's flow law. Currently only n=3 is implemented. Defaults to 3.
        unit_time (Optional[str], optional): Set to apply a time unit to the input strain rates.
            Set to 'a' for annual, 'm' for monthly, 'd' for daily, or 's' for per-second.
            Defaults to None, which assumes units of a-1 (or infers from xarray attrs where available).
        unit_temp (Literal["C", "K"], optional): Set to apply a temperature unit to the input temperature.
            Set to 'C' for Celsius or 'K' for Kelvin. Defaults to 'K', will sanity check.
        unit_output (Literal["Pa", "kPa", "MPa"], optional): Set to apply a unit to the output stress.
            Set to 'Pa', 'kPa', or 'MPa'. Defaults to 'Pa'.
        long_name (Optional[str], optional): Set to apply a long name to the output DataArray. Defaults to 'Deviatoric Stress'. 
    
    Returns:
        np.ndarray | xr.DataArray: Deviatoric stress. Returns either a numpy array
            or an xarray DataArray, depending on the input.
    """
    
    # Sanitise inputs
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

    # Convert inputs to a-1 if an explicit unit is provided.
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

        units = [a.attrs.get("units") for a in (strain_rate, effective_strain_rate)]
        norm_units = [_normalise_rate_unit(u) for u in units]

        # if no units set, simply continue
        if all(u is None for u in units):
            warnings.warn(
                "No units found in xarray DataArrays (strain_rate, effective_strain_rate). Assuming units of 'a-1'. If not correct, override by setting 'unit_time' parameter"
            )

        # If units cannot be parsed, warn and assume annual.
        elif any(u is None for u in norm_units):
            warnings.warn(
                "`unit_time` not set and no readable units found in xarray DataArray: assuming strain rates are in units of a-1",
            )

        # If different units, raise error
        elif len(set(norm_units)) > 1:
            raise ValueError(
                f"Inferred units from xarray DataArrays (strain_rate, effective_strain_rate) do not match: {units}. Ensure they match, and/or override by setting 'unit_time' parameter"
            )

        elif norm_units[0] == "d":
            strain_rate = strain_rate * 365
            effective_strain_rate = effective_strain_rate * 365
        elif norm_units[0] == "m":
            strain_rate = strain_rate * 365 / 12
            effective_strain_rate = effective_strain_rate * 365 / 12
        elif norm_units[0] == "s":
            strain_rate = strain_rate * 365 * 24 * 60 * 60
            effective_strain_rate = effective_strain_rate * 365 * 24 * 60 * 60
        elif norm_units[0] == "a":
            pass
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


    if unit_temp.upper() == "C":
        temperature = temperature + 273.15

    if n == 4:
        raise NotImplementedError(
            "n=4 is not yet implemented. Please use n=3 for now."
        )
    elif n != 3:
        raise ValueError("n must be 3 or 4")
    
    # Initialise values
    Astar = 3.5e-25  # Pa^-3 s^-1
    Tstar = 263  # K
    Qminus = 60e3  # J mol^-1
    Qplus = 115e3  # J mol^-1

    # Convert Astar to Pa^-3 a^-1 to match strain-rate units after conversion above.
    seconds_per_year = 365 * 24 * 60 * 60
    Astar = Astar * seconds_per_year

    # If temperature is an int/float/scalar, set Qc as Qminus or Qplus
    if np.isscalar(temperature) or (isinstance(temperature, np.ndarray) and temperature.ndim == 0):
        if temperature < Tstar:
            Q_c = Qminus
        else:
            Q_c = Qplus
    elif isinstance(temperature, (np.ndarray, xr.DataArray)):
        # If temperature is a 2D array, set Qc based on each element
        Q_c = np.where(temperature < Tstar, Qminus, Qplus)
    else: 
        raise ValueError("Temperature must be a scalar or a 2D array/dataarray of same shape as the strain rate arrays.")

    # Surface-pressure form of Cuffey & Paterson (2010):
    # A = Astar * exp[ -(Qc/R) * (1/T_h - 1/Tstar) ], where T_h = T at the surface.
    A = Astar * np.exp(-(Q_c / 8.314) * ((1 / temperature) - (1 / Tstar)))
    tau = (A ** -(1 / n)) * effective_strain_rate ** ((1 - n) / n) * strain_rate  # Pa

    # # Alternative implementation of Wells-Moran _et al._ (2024) - keeping for record.

    # # Prefactor
    # if n == 3:
    #     A0 = 2.290e4  # kPa^-3 a^-1, Duval et al. (1983)
    # elif n == 4:
    #     A0 = 12.614  # kPa^-4 a^-1, Goldsby and Kohlstedt (2001)
    # else:
    #     raise ValueError("n must be 3 or 4")

    # # # Convert A0 from kPa^-3 a^-1 to Pa^-3 a^-1
    # A0 = A0 * 1e3**-3

    # # Activation energy from Duval et al. (1983)
    # Q_c = 60e3  # J mol^-1

    # # Ideal gas constant
    # R = 8.314  # J K^-1 mol^-1

    # # Calculate flow rate parameter A (Equation 6)
    # A = A0 * np.exp(-Q_c / (R * temperature))

    # # Calculate dynamic viscosity η (Equation 4)
    # # η = 1/(2A^(1/n)) * ε̇_E^((1-n)/n)
    # eta = (1 / (2 * A ** (1 / n))) * effective_strain_rate ** ((1 - n) / n)

    # # Calculate deviatoric stress (Equation 3)
    # # 2ηε̇_ij = τ_ij
    # tau = 2 * eta * strain_rate

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
    deviatoric_xx: np.ndarray | xr.DataArray,
    deviatoric_yy: np.ndarray | xr.DataArray,
    long_name: Optional[str] = "Cauchy Stress",
) -> np.ndarray | xr.DataArray:
    r"""
    Calculate Cauchy stress for a given normal deviatoric stress using the first and
    second principal deviatoric stresses. These are related through the isotropic
    pressure as

    $$
    \sigma_{ij} = \tau_{ij} + p \delta_{ij},
    $$

    which can simplify for normal stress components to

    $$
    \sigma_{ij} = \tau_{ij} + \tau_{xx} + \tau_{yy} = \tau_{ij} + \tau_{1} + \tau_{2}.
    $$

    As $\tau_{xx} + \tau_{yy} = \tau_{1} + \tau_{2}$, the parameter `deviatoric_xx`
    and `deviatoric_yy` can accept either $\tau_{xx}$ and $\tau_{yy}$ or $\tau_{1}$ and $\tau_{2}$, 
    so long as they are consistent.

    Note that this is conversion is not necessary for shear stress components (where $i/neqj$), 
    where $\sigma_{ij} = \tau_{ij}$.

    The function accepts numpy arrays or an xarray Dataarrays, and will return the same
    data type.

    Args:
        deviatoric_stress (np.ndarray | xr.DataArray): Deviatoric stress.
        deviatoric_xx (np.ndarray | xr.DataArray): X-component of deviatoric stress.
        deviatoric_yy (np.ndarray | xr.DataArray): Y-component of deviatoric stress.
        long_name: (str, optional): Optional long name to attach to output xarray DataArray.
            Defaults to "Cauchy Stress".

    Returns:
        np.ndarray | xr.DataArray: Cauchy stress. Returns either a numpy array
            or an xarray DataArray, depending on the input.
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
        deviatoric_stress + deviatoric_xx + deviatoric_yy
    )

    if output == "xarray":

        cauchy_stress.attrs["long_name"] = long_name
        cauchy_stress.attrs["units"] = deviatoric_xx.attrs["units"]

        return cauchy_stress

    else:
        return cauchy_stress
