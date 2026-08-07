# glacier-strain-tools

[![conda-forge version](https://anaconda.org/conda-forge/glacier-strain-tools/badges/version.svg)](https://anaconda.org/conda-forge/glacier-strain-tools) [![PyPI version](https://badge.fury.io/py/glacier-strain-tools.svg)](https://pypi.org/project/glacier-strain-tools/) [![Unit Tests](https://github.com/trchudley/glacier-strain-tools/actions/workflows/unit-test.yml/badge.svg)](https://github.com/trchudley/glacier-strain-tools/actions/workflows/unit-test.yml) 

Tools for deriving surface-parallel strain rates and directions from glacier velocity fields.

## Installation

Install from `pip` or `conda`/`mamba` (preferred):

```bash
conda install -c conda-forge glacier-strain-tools
```

If you want the latest development version:

```bash
git clone https://github.com/trchudley/glacier-strain-tools.git
cd glacier-strain-tools
python -m pip install -e .
```

## Minimal Example: Logarithmic Strain Rate

```python
import rioxarray as rxr
from strain_tools import strain

# Load velocity fields
vx = rxr.open_rasterio('data/vx.tif').squeeze()
vy = rxr.open_rasterio('data/vy.tif').squeeze()

# Inputs:
# pixel_size   : image resolution in distance units (e.g. meters)
# length_scale : half-length scale for the virtual stake method (same distance units)
logarithmic_strains = strain.logarithmic(
	vx=vx,
	vy=vy,
	pixel_size=200.0,
	length_scale=500.0,
	unit_time="a",  # velocity field is in units m a-1
)
```

## What is Included

- Logarithmic and nominal strain-rate calculations
- Principal, longitudinal, transverse, shear, and effective strain rates
- Strain-rate uncertainty estimates
- Deviatoric and Cauchy stress conversion from strain rates

## Documentation

Full documentation, theory notes, and API reference are available at the GitHub repo:

[https://github.com/trchudley/glacier-strain-tools](tom.chudley@bristol.ac.uk).

## Contact

Tom Chudley
[tom.chudley@bristol.ac.uk](mailto:tom.chudley@bristol.ac.uk)
