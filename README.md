# glacier-strain-tools

_Tools for deriving surface-parallel strain rates from glacier velocity fields._

## Overview

This tool implements convenient Python functions for calculating strain rates and stresses from remotely sensed velocity fields. Functions include:

- Calculate logarithmic or nominal strain rates from remote sensed velocities.
- Calculate derivative strain rates (e.g. principal, longitudinal, transverse, shear, effective).
- Calculate predicted uncertainties in strain rates.
- Calculate deviatoric and cauchy stress from strain rates following Glen's flow law.

Contact: Tom Chudley, tom.chudley@bristol.ac.uk

## Installation

This module was created using a conda installation of Python with the following packages:

- numpy
- rioxarray
- numba
- matplotlib

It is recommended to install these dependencies into your conda environment before downloading this repository (an `environment.yml` is provided to aid with this) and installing `strain_tools` from the top-level directory with `pip install .`.

## Documentation

An explanatory notebook showing how to derive strain rates and stresses from an example velocity field is available at [`example_notebook.ipynb`](./example_notebook.ipynb). For further information, function docstrings are complete. Detailed user-friendly documentation is in production.

## Command line use

For simple testing and use on single fields, this tool can be used in the command line, although the implementation is simple and use within Python will ikely be preferred for most custom cases.

```$ strain_tools vx.tif vy.tif 750 --pixel_size 200 --no_data -9999.0```

where the first three mandatory arguments are the _x_-velocity field geotiff, the _y_-velocity field geotiff, and the length scale (in distance units). Output strain fields are saved as \*.tif files in the same directory as the input files. Optional flags are available -- see `strain_tools.py -h` -- but particularly important ones are the spatial resolution (`--pixel_size`) as an integer (the script will try and determine this manually but will often throw an error, so it may be prefereable to manually set this), and the value of no_data pixels (`--no_data`) if this value will not automatically be loaded in as NaN values by rasterio.


## References

Alley *et al.* (2018). Continent-wide estimates of Antarctic strain rates from Landsat 8-derived velocity grids. *Journal of Glaciology*, *64*(244) 321–332. https://doi.org/10.1017/jog.2018.23

Bindschadler _et al._ (1996). Surface velocity and mass balance of Ice Streams D and E, West Antarctica. *Journal of Glaciology*, *42*(142), 461–475.  https://doi.org/10.1017/s0022143000003452

Chudley *et al.* (2021). Controls on water storage and drainage in crevasses on the Greenland Ice Sheet. *Journal of Geophysical Research: Earth Surface*, *126*, e2021JF006287. https://doi.org/10.1029/2021JF006287

Cuffey & Paterson (2010). _The Physics of Glaciers_. Academic Press.

Gardner _et al._ (2025). ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities: Version 2. Data archived at National Snow and Ice Data Center; https://doi:10.5067/6II6VW8LLWJ7.

Harper _et al._ (1998). Crevasse patterns and the strain-rate tensor: A high-resolution comparison. _Journal of Glaciology_, _44_(146), 68-76. https://doi.org/10.3189/S0022143000002367

Luetzenburg _et al._ (2025). PROMICE-2022 Ice Mask V2. Data archived at the GEUS Dataverse. https://doi.org/10.22008/FK2/O8CLRE.

Nye, J. (1959). A Method of Determining the Strain-Rate Tensor at the Surface of a Glacier. _Journal of Glaciology_, _3_(25), 409-419. https://doi.org/10.3189/S0022143000017093

Wells-Moran, S. _et al._ (2025). Fracture criteria and tensile strength for natural glacier ice calibrated from remote sensing observations of Antarctic ice shelves. _Journal of Glaciology_, _71_, e47. https://doi.org/10.1017/jog.2024.104
