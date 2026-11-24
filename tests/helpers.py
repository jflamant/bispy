"""Shared helpers for the test suite (imported by the test modules)."""

import numpy as np
import quaternion

import bispy as bsp


def as_float(x):
    """Quaternion array -> float array with a trailing axis of length 4."""
    x = np.asarray(x)
    if x.dtype == "quaternion":
        return quaternion.as_float_array(x)
    return x


def real_bivariate(u, v):
    """Quaternion array u + i v (j and k components zero) from two real arrays."""
    return bsp.utils.sympSynth(np.asarray(u, dtype=float), np.asarray(v, dtype=float))


def jk_part(x):
    """Largest j or k component: zero for a real bivariate signal."""
    return np.max(np.abs(as_float(x)[..., 2:]))


def two_chirps(N=256):
    """Sum of two polarized linear chirps (as in the tutorial), as an H-embedding."""
    t = np.linspace(0, 1, N)
    a, b, c = 250 * np.pi, 50 * np.pi, 150 * np.pi
    env = bsp.utils.windows.hanning(N)
    x1 = bsp.signals.bivariateAMFM(env, np.pi / 4, np.pi / 6 - t, b * t + a * t**2)
    x2 = bsp.signals.bivariateAMFM(env, np.pi / 4 * 10 * t, 0, c * t + a * t**2)
    return t, x1 + x2
