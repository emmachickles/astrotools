"""CuPy/NumPy backend abstraction for GPU-accelerated photometry."""

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def gpu_available():
    """Check if CuPy is installed and a CUDA device is present."""
    if not _CUPY_AVAILABLE:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def get_array_module(device="gpu"):
    """Return the array module and resolved device string.

    Parameters
    ----------
    device : str
        ``"gpu"`` to request CuPy (falls back to NumPy if unavailable),
        or ``"cpu"`` to force NumPy.

    Returns
    -------
    xp : module
        ``cupy`` or ``numpy``.
    actual_device : str
        ``"gpu"`` or ``"cpu"``.
    """
    if device == "gpu" and gpu_available():
        return cp, "gpu"
    return np, "cpu"


def to_device(array, xp):
    """Transfer a NumPy array to the target backend.

    Parameters
    ----------
    array : numpy.ndarray
        Input array (must be a NumPy array).
    xp : module
        Target array module (``cupy`` or ``numpy``).

    Returns
    -------
    device_array
        Array on the target device.
    """
    if xp is np:
        return np.asarray(array)
    return cp.asarray(array)


def to_numpy(array):
    """Transfer any array back to CPU as a NumPy array.

    Parameters
    ----------
    array : array-like
        NumPy or CuPy array.

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(array, np.ndarray):
        return array
    if _CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
