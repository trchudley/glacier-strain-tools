# glacier-strain-tools

_Tools for deriving surface-parallel strain rates from glacier velocity fields._

## Overview

This tool implements (i) a Python version of the matlab script presented in Alley *et al.* (2018) to calculate logarithmic strain rates from remotely sensed glacier surface velocity fields (in the _x_- and _y_- directions); (ii) code adapted from Chudley *et al.* (2021) to calculate the magnitude and directions of the surface-parallel principal strain rates; (iii) functions for determining further strain rate components (longitudinal, transverse, shear, effective).

Contact: Tom Chudley, thomas.r.chudley@durham.ac.uk

## Installation

This module was created using a conda installation of Python with the following packages:

 - numpy
 - rioxarray
 - numba

It is recommended to install these dependencies into your conda environment before downloading this repository (an `environment.yml` is provided to aid with this) and installing `strain_tools` from the top-level directory with `pip install .`.

## Command line use

For simple testing and use on single fields, this tool can be used in the command line, although the implementation is simple and use within Python will ikely be preferred for most custom cases.

```$ strain_tools vx.tif vy.tif 750 --pixel_size 200 --no_data -9999.0```

where the first three mandatory arguments are the _x_-velocity field geotiff, the _y_-velocity field geotiff, and the length scale (in distance units). Output strain fields are saved as \*.tif files in the same directory as the input files. Optional flags are available -- see `strain_tools.py -h` -- but particularly important ones are the spatial resolution (`--pixel_size`) as an integer (the script will try and determine this manually but will often throw an error, so it may be prefereable to manually set this), and the value of no_data pixels (`--no_data`) if this value will not automatically be loaded in as NaN values by rasterio.

## Python functions

It is preferable to use this tool as an imported Python module. Following installation, it can be used in combination with numpy arrays or xarray Dataarrays of _x_ and _y_ velocities within a Python script:

```
import rioxarray as rxr
import numpy as np

import strain_tools

# Set variables
vx_fpath = "path/to/vx.tif"
vy_fpath = "path/to/vy.tif"
no_data = -9999.0   # pixel value of nodata cells, if applicable
pixel_size = 200    # metres
length_scale = 750  # metres

# Get vx and vy arrays with rioxarray
vx = rxr.open_rasterio(vx_fpath).squeeze()
vy = rxr.open_rasterio(vy_fpath).squeeze()

# Set no data values to NaN
vx = vx.where(vx != no_data, np.nan)
vy = vy.where(vy != no_data, np.nan)

# Use strain_tools functions to calculate strain rate components
lsr = strain_tools.log_strain_rates(vx, vy, pixel_size, length_scale)
psr = strain_tools.principal_strain_rate_directions(lsr.e_xx, lsr.e_yy, lsr.e_xy)
angle = strain_tools.flow_direction(vx, vy)
rsr = strain_tools.rotated_strain_rates(lsr.e_xx, lsr.e_yy, lsr.e_xy, angle)
e_E = strain_tools.effective_strain_rate(lsr.e_xx, lsr.e_yy, lsr.e_xy)
```

Functions will accept a numpy array or an xarray Dataarray, and will return the same data type. 

## List of functions

Functions are fully documented within docstrings, which can be visualised using the Python `help()` function.

### log_strain_rates

```(e_xx, e_yy, e_xy) = log_strain_rates(vx, vy, pixel_size, length_scale, tol=10e-4, ydir=1)```

Given vx and vy fields alongside the pixel size and desired half-length-jscale, returns the logarithmic strain rates (e<sub>xx</sub>, e<sub>yy</sub>, e<sub>xy</sub>) for a given glacier surface velocity field. Python adaptation of Alley et al. (2018) matlab script, itself based on principles from Nye (1959).

### principal_strain_rate_directions

```(e_1, e_1U, e_1V, e_2, e_2U, e_2V) = principal_strain_rate_directions(e_xx, e_yy, e_xy)```

Given e<sub>xx</sub>, e<sub>yy</sub>, and e<sub>xy</sub>, return the magnitude and directions of principal strain rates as output by the logarithmic_strain function. Implementation of Python script from Chudley _et al._ (2021).

The *U* and *V* components of the principal strain rate fields can be visualised within a matplotlib quiverplot. See the included jupyter notebook for an example of this.


### principal_strain_rate_magnitudes

```(e_1, e_2) = principal_strain_rate_magnitudes(e_xx, e_yy, e_xy, angle)```

Given e<sub>xx</sub>, e<sub>yy</sub>, and e<sub>xy</sub>, return principal surface-parallel strain rates following Nye (1959) and Harper _et al._ (1998) - i.e. not using eigenvectors. Quicker to compute, only returns magnitude values.

### flow_direction

```angle = flow_direction(vx, vy)```

Calculates a grid of flow directions (in radians offset from x=0) so that the grid-oriented strain rates can be rotated to align with local flow directions

### rotated_strain_rates

```(e_lon, e_trn, e_shr) = rotated_strain_rates(e_xx, e_yy, e_xy, angle)```

Given e<sub>xx</sub>, e<sub>yy</sub>, and e<sub>xy</sub>, returns longitudinal, transverse, and shear strain rates following Bindschadler _et al._ (1996). 

### effective_strain_rate

```e_E = effective_strain_rate(e_xx, e_yy, e_xy)```

Given e<sub>xx</sub>, e<sub>yy</sub>, and e<sub>xy</sub>, returns effective strain rate following Cuffey & Paterson (2010, p.59).

### strain_rate_uncertainty(ve_x, ve_y, length_scale)

Calculates the strain rate uncertainty field from a provided vx and vy error field, as well as the half-length-scale, following Poinar and Andrews (2021, eq. 4).

## Usage Notes

 - Consider effect of pixel size and length_scale. Large length scales relative to the pixel size will result in slow processing.
 - This script has yet to implement the ice thickness function in the original Alley script.

## To Do

 - Implement ice thickness function from original Alley et al. (2018) script.


## References

Alley *et al.* (2018). Continent-wide estimates of Antarctic strain rates from
Landsat 8-derived velocity grids. *Journal of Glaciology*, *64*(244)
321–332. https://doi.org/10.1017/jog.2018.23

Bindschadler _et al._ (1996). Surface velocity and mass balance of Ice Streams D and E, West Antarctica. *Journal of Glaciology*, *42*(142), 461–475.  https://doi.org/10.1017/s0022143000003452

Chudley *et al.* (2021). Controls on water storage and drainage in crevasses on
the Greenland Ice Sheet. *Journal of Geophysical Research: Earth Surface*,
*126*, e2021JF006287. https://doi.org/10.1029/2021JF006287

Cuffey & Paterson (2010). _The Physics of Glaciers_. Academic Press.

Harper _et al._ (1998). Crevasse patterns and the strain-rate tensor: A high-resolution comparison. _Journal of Glaciology_, _44_(146), 68-76. https://doi.org/10.3189/S0022143000002367

Nye, J. (1959). A Method of Determining the Strain-Rate Tensor at the Surface of a Glacier. _Journal of Glaciology_, _3_(25), 409-419. https://doi.org/10.3189/S0022143000017093
