"""Editable candidate. It starts identical to baseline.py, which is the scored
reference for both speed and correctness. Keep prepare_candidate, run_candidate,
and candidate_fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@njit
def laplacian(A, x, y, dx):
    DAx = (A[y, x - 1] - (2 * A[y, x]) + A[y, x + 1]) / dx
    DAy = (A[y - 1, x] - (2 * A[y, x]) + A[y + 1, x]) / dx
    return DAx + DAy


@njit(parallel=True)
def calculate_gs_cpu_mt(U, V, D_u, D_v, F, k, dt, dx, nx, ny):
    # Allocate U and V temporaries for updates
    DU = np.empty((ny, nx))
    DV = np.empty((ny, nx))

    # Compute diffusion and concentration fields changes
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Compute diffusion
            ΔU = laplacian(U, x, y, dx)
            ΔV = laplacian(V, x, y, dx)

            # Compute concentration fields changes
            UVV = U[y, x] * (V[y, x] ** 2)
            DU[y, x] = D_u * ΔU - UVV + F * (1 - U[y, x])
            DV[y, x] = D_v * ΔV + UVV - (F + k) * V[y, x]

    # Update concentration fields
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            U[y, x] += DU[y, x] * dt
            V[y, x] += DV[y, x] * dt


@dataclass
class CandidateState:
    U: np.ndarray
    V: np.ndarray


def prepare_candidate(U: np.ndarray, V: np.ndarray) -> CandidateState:
    return CandidateState(U.copy(), V.copy())


def run_candidate(state: CandidateState, p, steps: int) -> CandidateState:
    ny, nx = state.U.shape
    for _ in range(steps):
        calculate_gs_cpu_mt(state.U, state.V, p.D_u, p.D_v, p.F, p.k, p.dt, p.dx, nx, ny)
    return state


def candidate_fields(state: CandidateState) -> tuple[np.ndarray, np.ndarray]:
    return state.U, state.V
