# Installation

## Install via package manager

### `conda` / `mamba`

The recommended method to install `strain_tools` from the `conda-forge` using `conda`:

```bash
conda install glacier-strain-tools -c conda-forge
```

or `mamba:`

```bash
mamba install glacier-strain-tools -c conda-forge
```

### `pip` 

It is also possible to install pdemtools with `pip`:

```bash
pip install glacier-strain-tools
```

However, there are [known errors when installing the GDAL package with `pip`](https://github.com/OSGeo/gdal/issues/2827), meaning that the GDAL dependency may create an error when installing through `pip`. If this occurs, please ensure GDAL is installed on your system by other means prior to installing through `pip`, or revert to using conda.


## Developer Install

For development purposes, `strain_tools` can be cloned from the [Github repository](https://github.com/trchudley/glacier-strain-tools/) and installed in your Python environment locally using `pip`.

The module was developed using a `conda` installation of Python with the following `conda-forge`-hosted packages:

- `numpy`
- `rioxarray`
- `numba`
- `matplotlib`

It is recommended to install these dependencies into your conda environment from `conda-forge` before downloading this repository (an `environment.yml` is provided to aid with this). Once you have done this, install `strain_tools` from the top-level directory with `pip install -e .`.

A complete install looks like this:

```bash
# Clone the strain_tools github repository
git clone git@github.com:trchudley/glacier-strain-tools.git

# Move to strain_tools directory
cd glacier-strain-tools

# Initiate a new conda environment with dependencies
conda env create -f environment.yml -n straintools_env
conda activate straintools_env

# Install strain_tools in editable mode
pip install -e .
```
