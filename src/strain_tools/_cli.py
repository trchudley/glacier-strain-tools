"""
Command line interface for strain rate generator - will run via project.toml setup from 
command line using `strain_tools` command.
"""

import os, argparse, timeit

import rioxarray as rxr

# import rasterio as rs

from ._strain_rates import *


def cli():

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


def main(
    vx_fpath, vy_fpath, length_scale, pixel_size=None, no_data=np.nan, tol=10e-4, ydir=1
):
    """main"""

    print("\nOpening velocities...")
    vx = rxr.open_rasterio(vx_fpath).squeeze()
    vy = rxr.open_rasterio(vy_fpath).squeeze()

    # if pixel_size isn't defined, use rioxarray to extract resolution
    if pixel_size is None:
        pixel_size_x = vx.rio.resolution()[0]
        pixel_size_y = vx.rio.resolution()[1]
        if not abs(pixel_size_x) == abs(pixel_size_y):
            raise ValueError(
                f"Error: pixel resolutions in x and y resolutions are not equal ({pixel_size_x} and {pixel_size_y}). Consider setting pixel size manually with -p."
            )
        pixel_size = pixel_size_x

    # Set no data values to NaN
    vx = vx.where(vx != no_data, np.nan)
    vy = vy.where(vy != no_data, np.nan)

    print("\nCalculating strain rates...")
    start = timeit.default_timer()
    lsr = log_strain_rates(vx, vy, pixel_size, length_scale, tol, ydir)
    end = timeit.default_timer()
    print(f"\nStrain rates calculated. Elapsed time: {end - start} seconds.")

    print("\nGetting principal strain rates...")
    psr = principal_strain_rate_directions(lsr.e_xx, lsr.e_yy, lsr.e_xy)

    print("\nGetting flow direction...")
    angle = flow_direction(vx, vy)

    print("\nGetting rotated strain rates...")
    rsr = rotated_strain_rates(lsr.e_xx, lsr.e_yy, lsr.e_xy, angle)

    print("\nGetting effective strain rate...")
    e_E = effective_strain_rate(lsr.e_xx, lsr.e_yy, lsr.e_xy)

    print("\nWriting geotiffs...")

    def geotiffwrite(dirpath, xda, name, lengthscale):
        fpath = os.path.join(dirpath, f"log_{name}_{lengthscale}m.tif")
        xda.rio.to_raster(fpath, compress="ZSTD", predictor=3, zlevel=1)

    outdir = os.path.dirname(vx_fpath)

    # with rs.open(vx_fpath) as src:
    #     profile = src.profile
    #     profile.update(dtype=rs.float32, compress="lzw", predictor=3)

    geotiffwrite(outdir, psr.e_1, "e_1", length_scale)
    geotiffwrite(outdir, psr.e_2, "e_2", length_scale)
    # geotiffwrite(outdir, e_M, "e_M", length_scale)
    geotiffwrite(outdir, rsr.e_lon, "e_lon", length_scale)
    geotiffwrite(outdir, rsr.e_trn, "e_trn", length_scale)
    geotiffwrite(outdir, rsr.e_shr, "e_shr", length_scale)
    geotiffwrite(outdir, e_E, "e_E", length_scale)

    print("\nComplete.")
