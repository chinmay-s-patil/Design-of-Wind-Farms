"""
_np_compat.py - numpy / cupy compatibility shim for floris_cupy.

Import this module instead of numpy directly:

    from floris_cupy._np_compat import np

If cupy is importable and the HSA gfx override is in place, `np` will be
cupy; otherwise it falls back to numpy silently.
"""
import os

# ROCm / RDNA2 gfx override – needed for RX 6xxx (gfx1030) cards.
# Harmless if already set via .bashrc.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

try:
    import cupy as np          # noqa: F401  (re-exported as `np`)
    _BACKEND = "cupy"
except Exception:              # cupy not installed / ROCm not found
    import numpy as np         # noqa: F401
    _BACKEND = "numpy"
