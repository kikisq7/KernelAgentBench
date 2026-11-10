"""Editable candidate. It starts identical to baseline.py, which is the scored
reference for both speed and correctness. Keep prepare_candidate, run_candidate,
and candidate_fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 16


@triton.jit
def laplacian(A, x, y, dx, nx, mask):
    center = tl.load(A + x + nx * y, mask=mask)
    DAx = (tl.load(A + (x - 1) + nx * y, mask=mask) - (2 * center)
           + tl.load(A + (x + 1) + nx * y, mask=mask)) / dx
    DAy = (tl.load(A + x + nx * (y - 1), mask=mask) - (2 * center)
           + tl.load(A + x + nx * (y + 1), mask=mask)) / dx
    return DAx + DAy


# P holds [D_u, D_v, F, k, dt, dx] as float64: Triton passes Python floats as fp32.
@triton.jit
def calculate_gs_triton(DU, DV, U, V, P, nx, ny, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)[None, :]
    y = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[:, None]
    D_u = tl.load(P + 0)
    D_v = tl.load(P + 1)
    F = tl.load(P + 2)
    k = tl.load(P + 3)
    dx = tl.load(P + 5)

    # Skip ghost cells on the boundaries
    mask = (0 < x) & (x < nx - 1) & (0 < y) & (y < ny - 1)
    i = x + nx * y

    # Compute diffusion
    DeltaU = laplacian(U, x, y, dx, nx, mask)
    DeltaV = laplacian(V, x, y, dx, nx, mask)

    # Update concentration fields
    u = tl.load(U + i, mask=mask)
    v = tl.load(V + i, mask=mask)
    UVV = u * (v * v)
    tl.store(DU + i, D_u * DeltaU - UVV + F * (1 - u), mask=mask)
    tl.store(DV + i, D_v * DeltaV + UVV - (F + k) * v, mask=mask)


@triton.jit
def update_gs_triton(U, V, DU, DV, P, nx, ny, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)[None, :]
    y = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[:, None]
    dt = tl.load(P + 4)

    # Skip ghost cells on the boundaries
    mask = (0 < x) & (x < nx - 1) & (0 < y) & (y < ny - 1)
    i = x + nx * y

    # Update concentration fields
    tl.store(U + i, tl.load(U + i, mask=mask) + tl.load(DU + i, mask=mask) * dt, mask=mask)
    tl.store(V + i, tl.load(V + i, mask=mask) + tl.load(DV + i, mask=mask) * dt, mask=mask)


@dataclass
class CandidateState:
    U: torch.Tensor
    V: torch.Tensor
    DU: torch.Tensor
    DV: torch.Tensor


def prepare_candidate(U: np.ndarray, V: np.ndarray) -> CandidateState:
    # Transfer to the GPU
    U_gpu = torch.from_numpy(np.ascontiguousarray(U)).to("cuda")
    V_gpu = torch.from_numpy(np.ascontiguousarray(V)).to("cuda")
    return CandidateState(U_gpu, V_gpu, torch.empty_like(U_gpu), torch.empty_like(V_gpu))


def run_candidate(state: CandidateState, p, steps: int) -> CandidateState:
    ny, nx = state.U.shape
    P = torch.tensor([p.D_u, p.D_v, p.F, p.k, p.dt, p.dx], dtype=torch.float64, device="cuda")
    grid = (triton.cdiv(nx, BLOCK), triton.cdiv(ny, BLOCK))
    for _ in range(steps):
        calculate_gs_triton[grid](
            state.DU, state.DV, state.U, state.V, P, nx, ny, BLOCK=BLOCK, num_warps=8
        )
        update_gs_triton[grid](
            state.U, state.V, state.DU, state.DV, P, nx, ny, BLOCK=BLOCK, num_warps=8
        )
    return state


def candidate_fields(state: CandidateState) -> tuple[np.ndarray, np.ndarray]:
    return state.U.cpu().numpy(), state.V.cpu().numpy()
