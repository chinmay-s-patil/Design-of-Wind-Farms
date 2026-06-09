
from typing import Any, Dict


from floris_cupy._np_compat import np
from attrs import (
    define,
    field,
    fields,
)

from floris_cupy.core import (

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

    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class JensenVelocityDeficit(BaseModel):
    """
    The Jensen model computes the wake velocity deficit based on the classic
    Jensen/Park model :cite:`jvm-jensen1983note`.

    -   **we** (*float*): The linear wake decay constant that
        defines the cone boundary for the wake as well as the
        velocity deficit. D/2 +/- we*x is the cone boundary for the
        wake.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    we: float = field(converter=float, default=0.05)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:
        """
        This function prepares the inputs from the various FLORIS data structures
        for use in the Jensen model. This should only be used to 'initialize'
        the inputs. For any data that should be updated successively,
        do not use this function and instead pass that data directly to
        the model function.
        """
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i,
        rotor_diameter_i,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:

        # u is 4-dimensional (n wind speeds, n turbines, grid res 1, grid res 2)
        # velocities is 3-dimensional (n turbines, grid res 1, grid res 2)

        # TODO: check the rotations with multiple directions or non-0/270
        # grid.rotate_fields(flow_field.wind_directions)

        # Calculate and apply wake mask
        # x = grid.x_sorted # mesh_x_rotated - x_coord_rotated

        # This is the velocity deficit seen by the i'th turbine due to wake effects
        # from upstream turbines.
        # Indeces of velocity_deficit corresponding to unwaked turbines will have 0's
        # velocity_deficit = np.zeros(np.shape(flow_field.u_initial))

        rotor_radius = rotor_diameter_i / 2.0

        # Numexpr - do not change below without corresponding changes above.
        dx = ne.evaluate("x - x_i")
        dy = ne.evaluate("y - y_i - deflection_field_i")
        dz = ne.evaluate("z - z_i")

        we = self.we

        # Construct a boolean mask to include all points downstream of the turbine
        downstream_mask = ne.evaluate("dx > 0 + NUM_EPS")

        # Construct a boolean mask to include all points within the wake boundary
        # as defined by the Jensen model. This is a linear wake expansion that makes
        # a shape like a cone and starts at the turbine disc.
        # The left side of the inequality below evaluates the distance from the wake centerline
        # for all points including positive and negative values. The inequality compares distance
        # from the centerline and it must be below the line defined by the wake
        # expansion parameter, "we".
        boundary_mask = ne.evaluate("sqrt(dy ** 2 + dz ** 2) < we * dx + rotor_radius")

        # Calculate C for points within the mask and fill points outside with 0
        c = np.where(
            np.logical_and(downstream_mask, boundary_mask),
            ne.evaluate("(rotor_radius / (rotor_radius + we * dx + NUM_EPS)) ** 2"),  # This is "C"
            0.0,
        )

        velocity_deficit = ne.evaluate("2 * axial_induction_i * c")

        return velocity_deficit
