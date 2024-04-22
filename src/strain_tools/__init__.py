from importlib.metadata import version

from ._strain_rates import (
    log_strain_rates,
    principal_strain_rate_directions,
    principal_strain_rate_magnitudes,
    flow_direction,
    rotated_strain_rates,
    effective_strain_rate,
    strain_rate_uncertainty,
)

from . import _cli

__version__ = "0.2.0"


# # # for testing only - reload modules ---------
# from importlib import reload

# from . import _strain_rates

# reload(_strain_rates)

# from ._strain_rates import (
#     log_strain_rates,
#     principal_strain_rate_directions,
#     principal_strain_rate_magnitudes,
#     flow_direction,
#     rotated_strain_rates,
#     effective_strain_rate,
#     strain_rate_uncertainty,
# )
# # # --------------------------
