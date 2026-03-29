"""Backend selection: NumPy/SciPy (CPU) or CuPy (GPU)."""
import numpy as np
import scipy

_GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy as csp
    _GPU_AVAILABLE = True
except ImportError:
    pass


def get_array_module(use_gpu=False):
    """Return (xp, xsp) -- the array module and scipy-like module.

    Args:
        use_gpu (bool): If True, return CuPy modules for GPU acceleration.

    Returns:
        xp: numpy or cupy array module.
        xsp: scipy or cupyx.scipy module.

    Raises:
        ImportError: If use_gpu=True but CuPy is not installed.
    """
    if use_gpu:
        if not _GPU_AVAILABLE:
            raise ImportError(
                "CuPy is required for GPU acceleration. "
                "Install it with: pip install cupy-cuda12x  "
                "(adjust the cuda suffix for your CUDA version)"
            )
        return cp, csp
    return np, scipy


def to_device(arr, xp):
    """Move a numpy array to the target device."""
    if xp is np:
        return np.asarray(arr)
    return xp.asarray(arr)


def to_numpy(arr):
    """Move an array back to numpy (no-op if already numpy)."""
    if isinstance(arr, np.ndarray):
        return arr
    return arr.get()


def gpu_available():
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE
