# Command Line Interface

For simple testing and use on single fields, this tool can be used in the command line, although the implementation is simple and use within Python will ikely be preferred for most custom cases.

```$ strain_tools vx.tif vy.tif 750 --pixel_size 200 --no_data -9999.0```

where the first three mandatory arguments are the _x_-velocity field geotiff, the _y_-velocity field geotiff, and the length scale (in distance units). Output strain fields are saved as \*.tif files in the same directory as the input files. Optional flags are available -- see `strain_tools.py -h` -- but particularly important ones are the spatial resolution (`--pixel_size`) as an integer (the script will try and determine this manually but will often throw an error, so it may be prefereable to manually set this), and the value of no_data pixels (`--no_data`) if this value will not automatically be loaded in as NaN values by rasterio.