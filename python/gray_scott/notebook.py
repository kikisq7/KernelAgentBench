"""Harness-side pieces of parts/gpu/gray-scott.ipynb: parameters and initial condition.

Mirrors kernels/gray_scott/notebook.jl bit for bit. Fields are NumPy float64
arrays of shape (ny, nx) in C order, so element [y, x] sits at x + nx*y exactly
like Julia's U[x, y].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 20260918


@dataclass(frozen=True)
class GrayScottParams:
    """Defaults of simulate_gs_* in the notebook. The noise term (0.0 there) is dropped."""

    D_u: float = 0.16
    D_v: float = 0.08
    F: float = 0.035
    k: float = 0.06
    dt: float = 1.0
    dx: float = 1.0


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def initialize_fields(
    nx: int, ny: int, *, seed: int = SEED, init_multi: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """The notebook's ``U = rand(0.0:0.001:0.1, n, n)`` with a portable generator."""
    if nx < 16 or ny < 16:
        raise ValueError("grid must be at least 16x16")
    index = np.arange(nx * ny, dtype=np.uint64)
    with np.errstate(over="ignore"):
        draws = _splitmix64(np.uint64(seed) + index * np.uint64(0x9E3779B97F4A7C15))
    U = ((draws % np.uint64(101)).astype(np.float64) / 1000.0).reshape(ny, nx)
    V = np.zeros((ny, nx))

    # Set up an initial small disturbance (Julia's n_mid-5:n_mid+5, 0-based here)
    nx_mid, ny_mid = nx // 2 - 1, ny // 2 - 1
    U[ny_mid - 5 : ny_mid + 6, nx_mid - 5 : nx_mid + 6] = 1.0 * init_multi
    V[ny_mid - 5 : ny_mid + 6, nx_mid - 5 : nx_mid + 6] = 0.5 * init_multi
    return U, V
