
from typing import Any, Dict


from floris_cupy._np_compat import np
from attrs import define, field

from floris_cupy.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris_cupy.utilities import cosd, sind

# --- floris_cupy: numexpr replaced with numpy/cupy eval shim ---
import builtins as _builtins

class _NeShim:
    """Drop-in for `import numexpr as ne`.
    Evaluates the expression string using the *caller's* local namespace so
    that all array variables (which may now be cupy arrays) are resolved
    correctly.  The result type matches whatever numpy/cupy returns.
    """
    import inspect as _inspect

    @staticmethod
    def evaluate(expr, local_dict=None, global_dict=None, **_kw):
        import inspect
        frame = inspect.currentframe().f_back
        locs = dict(frame.f_locals)
        if local_dict:
            locs.update(local_dict)
        globs = frame.f_globals if global_dict is None else global_dict
        # make math builtins available (exp, sqrt, pi …)
        try:
            import cupy as _xp
        except Exception:
            import numpy as _xp
        import math
        _math_ns = {k: getattr(_xp, k, getattr(math, k, None))
                    for k in ("exp", "sqrt", "log", "pi", "abs")}
        _math_ns["sqrt"] = _xp.sqrt
        _math_ns["exp"]  = _xp.exp
        locs.update({k: v for k, v in _math_ns.items() if k not in locs})
        return eval(expr, globs, locs)   # noqa: S307

ne = _NeShim()
# --- end numexpr shim ---



@define
class CrespoHernandez(BaseModel):
    """
    CrespoHernandez is a wake-turbulence model that is used to compute
    additional variability introduced to the flow field by operation of a wind
    turbine. Implementation of the model follows the original formulation and
    limitations outlined in :cite:`cht-crespo1996turbulence`.

    Note: The values for default parameters provided here differ from those in
    :cite:`cht-crespo1996turbulence. Following their recommendations, the
    default parameters would instead be:
        - initial: -0.0325*
        - constant: 0.73
        - ai: 0.8325
        - downstream: -0.32
    * The "initial" parameter is given as -0.0325 in :cite:`cht-crespo1996turbulence`,
    but the negative exponent is not clear in the scans of the paper found on the internet,
    and several subsequent paper cite the exponent as positive (0.0325). This discrepancy
    is noted in :cite:`zehtabiyan_rezaie_CH_2023`. Moreover, :cite:`zehtabiyan_rezaie_CH_2023`
    argues that positive values for this exponent are not representative of the physical
    phenomena occurring. For more details, see https://github.com/NREL/floris/issues/773.
    Nonetheless, the default value here is set to 0.1 for consistency with previous
    FLORIS versions. The default value may be updated in a future release.

    Args:
        parameter_dictionary (dict): Model-specific parameters.
            Default values are used when a parameter is not included
            in `parameter_dictionary`. Possible key-value pairs include:

            -   **initial** (*float*): The exponent on the initial ambient
                turbulence intensity.
            -   **constant** (*float*): The constant used to scale the
                wake-added turbulence intensity.
            -   **ai** (*float*): The axial induction factor exponent used
                in in the calculation of wake-added turbulence.
            -   **downstream** (*float*): The exponent applied to the
                distance downstream of an upstream turbine normalized by
                the rotor diameter used in the calculation of wake-added
                turbulence.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cht-
    """

    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    def prepare_function(self) -> dict:
        pass

    def function(
        self,
        ambient_TI: float,
        x: np.ndarray,
        x_i: np.ndarray,
        rotor_diameter: float,
        axial_induction: np.ndarray,
    ) -> None:
        # Replace zeros and negatives with 1 to prevent nans/infs
        delta_x = x - x_i

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        #        Keep downstream components          Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # turbulence intensity calculation based on Crespo et. al.
        constant = self.constant
        ai = self.ai
        initial = self.initial
        downstream = self.downstream
        ti = ne.evaluate(
            "constant"
            " * axial_induction ** ai"
            " * ambient_TI ** initial"
            " * (delta_x / rotor_diameter) ** downstream"
        )
        # Mask the 1 values from above with zeros
        return ti * downstream_mask
