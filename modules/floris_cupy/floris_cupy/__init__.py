# floris_cupy: load numpy/cupy compat shim early
from floris_cupy import _np_compat  # noqa: F401 – side-effects: sets HSA env var

from importlib.metadata import version
from pathlib import Path


__version__ = version("floris_cupy")


from .floris_model import FlorisModel
from .flow_visualization import (
    plot_rotor_values,
    visualize_cut_plane,
    visualize_quiver,
)
from .heterogeneous_map import HeterogeneousMap
from .par_floris_model import ParFlorisModel
from .parallel_floris_model import ParallelFlorisModel
from .uncertain_floris_model import ApproxFlorisModel, UncertainFlorisModel
from .wind_data import (
    TimeSeries,
    WindRose,
    WindRoseWRG,
    WindTIRose,
)
