import numpy as np
import warnings
import xarray as xr

from numpy.testing import assert_allclose

from strain_tools import strain, stress


def test_deviatoric_temperature_unit_consistency():
    shape = (5, 4)
    strain_rate = np.full(shape, 1.0e-3)
    effective_strain_rate = np.full(shape, 1.0e-3)

    tau_c = stress.deviatoric(
        strain_rate,
        effective_strain_rate,
        temperature=-5.0,
        unit_temp="C",
        unit_time="a",
    )
    tau_k = stress.deviatoric(
        strain_rate,
        effective_strain_rate,
        temperature=268.15,
        unit_temp="K",
        unit_time="a",
    )

    assert_allclose(tau_c, tau_k, rtol=1e-12, atol=0.0)


def test_deviatoric_expected_order_of_magnitude_for_typical_inputs():
    shape = (3, 3)
    strain_rate = np.full(shape, 1.0e-3)
    effective_strain_rate = np.full(shape, 1.0e-3)

    tau = stress.deviatoric(
        strain_rate,
        effective_strain_rate,
        temperature=268.15,
        unit_temp="K",
        unit_time="a",
    )

    mean_tau = float(np.nanmean(tau))
    assert 1.0e4 < mean_tau < 1.0e6


def test_deviatoric_time_unit_conversion_matches_equivalent_annual_input():
    shape = (4, 4)
    seconds_per_year = 365 * 24 * 60 * 60

    strain_rate_annual = np.full(shape, 1.2e-3)
    effective_annual = np.full(shape, 1.2e-3)

    strain_rate_seconds = strain_rate_annual / seconds_per_year
    effective_seconds = effective_annual / seconds_per_year

    tau_a = stress.deviatoric(
        strain_rate_annual,
        effective_annual,
        temperature=265.0,
        unit_temp="K",
        unit_time="a",
    )
    tau_s = stress.deviatoric(
        strain_rate_seconds,
        effective_seconds,
        temperature=265.0,
        unit_temp="K",
        unit_time="s",
    )

    assert_allclose(tau_a, tau_s, rtol=1e-12, atol=0.0)


def test_xarray_workflow_units_propagate_without_deviatoric_unit_warning():
    # Build a linear velocity field on descending y coordinates (ydir=1 convention).
    ny, nx = 6, 7
    pixel_size = 100.0
    y = np.arange(ny)[::-1] * pixel_size
    x = np.arange(nx) * pixel_size
    xx, yy = np.meshgrid(x, y)

    vx = 2.0e-3 * xx + 1.0e-3 * yy + 10.0
    vy = -1.0e-3 * xx + 3.0e-3 * yy - 5.0

    vx_da = xr.DataArray(vx, dims=("y", "x"), coords={"y": y, "x": x})
    vy_da = xr.DataArray(vy, dims=("y", "x"), coords={"y": y, "x": x})

    # Units should originate here and propagate via attrs.
    lsr = strain.nominal(vx_da, vy_da, pixel_size=pixel_size, unit_time="a")
    e_E = strain.effective(lsr.e_xx, lsr.e_yy, lsr.e_xy)

    assert lsr.e_xx.attrs.get("units") == "a$^{-1}$"
    assert e_E.attrs.get("units") == "a$^{-1}$"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = stress.deviatoric(
            lsr.e_xx,
            e_E,
            temperature=268.15,
            unit_temp="K",
            unit_time=None,
        )

    warning_messages = [str(w.message) for w in caught]
    assert not any("no readable units found" in msg for msg in warning_messages)
