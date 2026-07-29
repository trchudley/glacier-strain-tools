import numpy as np
import xarray as xr

from numpy.testing import assert_allclose

from strain_tools import strain


def _linear_velocity_field(nx=7, ny=6, pixel_size=2.0):
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    xx, yy = np.meshgrid(x, y)

    # vx = ax*x + ay*y + c0, vy = bx*x + by*y + c1
    ax, ay = 3.0e-3, 2.0e-3
    bx, by = -1.0e-3, 4.0e-3
    c0, c1 = 10.0, -7.5

    vx = ax * xx + ay * yy + c0
    vy = bx * xx + by * yy + c1
    return vx, vy, (ax, ay, bx, by)


def test_nominal_numpy_linear_field_derivatives_are_constant():
    pixel_size = 2.0
    vx, vy, (ax, ay, bx, by) = _linear_velocity_field(pixel_size=pixel_size)

    e_xx, e_yy, e_xy = strain.nominal(vx, vy, pixel_size=pixel_size)

    assert e_xx.shape == (vx.shape[0], vx.shape[1] - 1)
    assert e_yy.shape == (vx.shape[0] - 1, vx.shape[1])
    assert e_xy.shape == (vx.shape[0] - 1, vx.shape[1] - 1)

    assert_allclose(e_xx, ax)
    assert_allclose(e_yy, by)
    assert_allclose(e_xy, 0.5 * (bx + ay))


def test_nominal_xarray_linear_field_derivatives_are_constant_on_valid_cells():
    pixel_size = 2.0
    ax, ay = 3.0e-3, 2.0e-3
    bx, by = -1.0e-3, 4.0e-3
    c0, c1 = 10.0, -7.5

    # Descending y coordinate is consistent with default ydir=1 behavior.
    ny, nx = 6, 7
    y = np.arange(ny)[::-1] * pixel_size
    x = np.arange(nx) * pixel_size

    xx, yy = np.meshgrid(x, y)
    vx_np = ax * xx + ay * yy + c0
    vy_np = bx * xx + by * yy + c1

    vx = xr.DataArray(vx_np, dims=("y", "x"), coords={"y": y, "x": x})
    vy = xr.DataArray(vy_np, dims=("y", "x"), coords={"y": y, "x": x})

    out = strain.nominal(vx, vy, pixel_size=pixel_size, unit_time="a")

    assert set(out.data_vars) == {"e_xx", "e_yy", "e_xy"}
    assert out["e_xx"].attrs["units"] == "a$^{-1}$"

    exx = out["e_xx"].values
    eyy = out["e_yy"].values
    exy = out["e_xy"].values

    assert_allclose(exx[np.isfinite(exx)], ax)
    assert_allclose(eyy[np.isfinite(eyy)], by)
    assert_allclose(exy[np.isfinite(exy)], 0.5 * (bx + ay))


def test_rotated_component_functions_match_axis_aligned_expectation():
    e_xx = np.full((4, 5), 2.0e-3)
    e_yy = np.full((4, 5), -3.0e-3)
    e_xy = np.full((4, 5), 1.5e-3)

    # angle=0 should map lon->xx, trn->yy, shr->xy
    angle = np.zeros_like(e_xx)

    e_lon = strain.longitudinal(e_xx, e_yy, e_xy, angle)
    e_trn = strain.transverse(e_xx, e_yy, e_xy, angle)
    e_shr = strain.shear(e_xx, e_yy, e_xy, angle)

    assert_allclose(e_lon, e_xx)
    assert_allclose(e_trn, e_yy)
    assert_allclose(e_shr, e_xy)


def test_rotated_wrapper_matches_split_component_functions():
    rng = np.random.default_rng(42)
    e_xx = rng.normal(size=(5, 6)) * 1.0e-3
    e_yy = rng.normal(size=(5, 6)) * 1.0e-3
    e_xy = rng.normal(size=(5, 6)) * 1.0e-3
    angle = rng.uniform(-np.pi, np.pi, size=(5, 6))

    e_lon = strain.longitudinal(e_xx, e_yy, e_xy, angle)
    e_trn = strain.transverse(e_xx, e_yy, e_xy, angle)
    e_shr = strain.shear(e_xx, e_yy, e_xy, angle)

    e_lon_old, e_trn_old, e_shr_old = strain.rotated(e_xx, e_yy, e_xy, angle)

    assert_allclose(e_lon_old, e_lon)
    assert_allclose(e_trn_old, e_trn)
    assert_allclose(e_shr_old, e_shr)
