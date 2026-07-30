import numpy as np
import xarray as xr

from typing import Dict, Optional, Tuple, TypeGuard, overload


# Keep a narrow whitelist of geospatial attrs that can be important for
# downstream georeferencing while dropping source-provenance clutter.
_GEOSPATIAL_ATTR_KEYS = {
    "grid_mapping",
    "spatial_ref",
    "crs",
    "epsg",
    "epsg_code",
    "projection",
    "proj4",
    "proj4text",
    "GeoTransform",
    "transform",
}


def _all_numpy(
    *arrays: np.ndarray | xr.DataArray,
) -> TypeGuard[Tuple[np.ndarray, ...]]:
    """Check if all arrays are numpy ndarrays."""
    return all(isinstance(arr, np.ndarray) for arr in arrays)


def _all_xarray(
    *arrays: np.ndarray | xr.DataArray,
) -> TypeGuard[Tuple[xr.DataArray, ...]]:
    """Check if all arrays are xarray DataArrays."""
    return all(isinstance(arr, xr.DataArray) for arr in arrays)


def _sanitise_unit_time(unit_time: str) -> str:
    """
    Sanitise unit time input.

    Args:
        unit_time (str): Time unit representing 'year' or 'day' as a string.

    Returns:
        str: Either 'a' or 'm' or 'd' or 's'
    """

    if unit_time.lower() in [
        "a",
        "annual",
        "annually",
        "y",
        "yr",
        "year",
        "yearly",
        "per year",
    ]:
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


def _normalise_rate_unit(unit: Optional[str]) -> Optional[str]:
    """Normalise common time-rate unit strings to one of {'a','m','d','s'}.

    Supports plain-text forms (e.g. ``a^{-1}``, ``yr^-1``) and LaTeX-styled
    forms used in this package (e.g. ``a$^{-1}$``).
    """

    if unit is None:
        return None

    u = unit.strip().lower().replace(" ", "")
    u = u.replace("$", "").replace("{", "").replace("}", "")

    annual = {
        "a^-1",
        "a-1",
        "1/a",
        "annual",
        "annually",
        "yr^-1",
        "yr-1",
        "1/yr",
        "y^-1",
        "y-1",
        "1/y",
        "year^-1",
        "year-1",
        "1/year",
    }
    monthly = {"m^-1", "m-1", "1/m", "month^-1", "month-1", "1/month"}
    daily = {"d^-1", "d-1", "1/d", "day^-1", "day-1", "1/day"}
    secondly = {
        "s^-1",
        "s-1",
        "1/s",
        "sec^-1",
        "sec-1",
        "1/sec",
        "second^-1",
        "second-1",
        "1/second",
    }

    if u in annual:
        return "a"
    if u in monthly:
        return "m"
    if u in daily:
        return "d"
    if u in secondly:
        return "s"
    return None


def _extract_geospatial_attrs(*arrays: xr.DataArray) -> Dict[str, object]:
    """Collect a minimal set of geospatial attrs from xarray inputs."""

    attrs: Dict[str, object] = {}
    for arr in arrays:
        for key in _GEOSPATIAL_ATTR_KEYS:
            if key in arr.attrs and key not in attrs:
                attrs[key] = arr.attrs[key]
    return attrs


def _strip_nonessential_attrs(
    xda: xr.DataArray,
    *,
    geospatial_attrs: Optional[Dict[str, object]] = None,
    long_name: Optional[str] = None,
    units: Optional[str] = None,
) -> xr.DataArray:
    """Return a DataArray with only desired semantic and geospatial attrs."""

    clean_attrs: Dict[str, object] = dict(geospatial_attrs or {})
    if long_name is not None:
        clean_attrs["long_name"] = long_name
    if units is not None:
        clean_attrs["units"] = units
    xda.attrs = clean_attrs
    return xda


def _strip_nonessential_dataset_attrs(
    xds: xr.Dataset,
    *,
    geospatial_attrs: Optional[Dict[str, object]] = None,
) -> xr.Dataset:
    """Keep only essential geospatial attrs on a Dataset and data vars."""

    clean_geo = dict(geospatial_attrs or {})
    xds.attrs = clean_geo
    for var_name in xds.data_vars:
        xds[var_name].attrs = dict(clean_geo)
    return xds


@overload
def flow_direction(vx: np.ndarray, vy: np.ndarray) -> np.ndarray: ...


@overload
def flow_direction(vx: xr.DataArray, vy: xr.DataArray) -> xr.DataArray: ...


def flow_direction(
    vx: np.ndarray | xr.DataArray, vy: np.ndarray | xr.DataArray
) -> np.ndarray | xr.DataArray:
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

    if _all_xarray(vx, vy):
        output = "xarray"
    elif _all_numpy(vx, vy):
        output = "numpy"
    else:
        raise ValueError(
            f"Input velocity fields must be all the same type and either np.ndarray or xr.DataArray."
        )

    # angle = np.degrees(np.arctan(vy / vx))
    # angle = np.where(vx > 0, angle, angle + 180)

    # # # from -180 to 180, following Bindschadler et al. 1996
    # angle = np.where(angle > 180, angle - 360, angle)

    # angle = np.deg2rad(angle)

    angle = np.arctan2(vy, vx) 

    if output == "xarray":

        geospatial_attrs = _extract_geospatial_attrs(vx, vy)
        angle = vx * 0 + angle

        angle = angle.rename("angle")
        angle = _strip_nonessential_attrs(
            angle,
            geospatial_attrs=geospatial_attrs,
            long_name="Flow Direction",
            units="radians",
        )

        return angle

    else:
        return angle


def flip(xds: xr.DataArray) -> xr.DataArray:
    """
    Flips y dimension if y[1] < y[0]. This is required for feeding xarray
    DataArrays to the `matplotlib` `quiver()` function.

    :param xds: xarray DataArray
    :type xds: xr.DataArray

    :return: Flipped xarray DataArray
    :rtype: xr.DataArray
    """
    if xds.y.values[1] - xds.y.values[0] < 0:
        return xds.reindex(y=xds.y[::-1])
    else:
        return xds
