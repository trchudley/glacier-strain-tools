"""
Glacier surface logarithmic strain rate calculator

This script implements a Python version of the matlab script presented
in Alley et al. (2018) to calculate logarithmic strain rates from remotely
sensed glacier surface velocity fields (in the x- and y- directions).

It also adapts code from Chudley et al. (2021) to calculate the directions
of the surface-parallel principal strain rates.

To import this file as a module and use in a pre-existing Python script, the
main() function provides a suitable example of a workable implementation.

This file can also be used as a command-line interface, e.g.:

    $ log_strain.py vx.tif vy.tif 750 --pixel_size 200 --no_data -9999.0

where the first three mandatory arguments are the x-velocity field, the
y-velocity field, and the length scale (in distance units). Output strain
fields are saved as *.tif files in the same directory as the input files.

Notes:

 - Consider effect of pixel size and length_scale. Large length scales
    relative to the pixel size will result in slow processing.
 - This script has yet to implement the ice thickness function in the
    original script.

Citations:

Alley et al. (2018). Continent-wide estimates of Antarctic strain rates from
Landsat 8-derived velocity grids. Journal of Glaciology, 64(244)
321-332. https://doi.org/10.1017/jog.2018.23

Chudley et al. (2021). Controls on water storage and drainage in crevasses on
the Greenland Ice Sheet. Journal of Geophysical Research: Earth Surface,
126, e2021JF006287. https://doi.org/10.1029/2021JF006287


Tom Chudley | thomas.r.chudley@durhmam.ac.uk | Durham University
April 2024
"""

from importlib.metadata import version

from ._utils import *
