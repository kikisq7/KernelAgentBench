from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import cuda

from python.gray_scott.reference import GrayScottParams


@dataclass
class CandidateState:
    u: Any
    v: Any
    u_next: Any
    v_next: Any


@cuda.jit
def candidate_kernel(u_next, v_next, u, v, diffusion_u, diffusion_v, feed, kill, dt):
    y, x = cuda.grid(2)
    height, width = u.shape
    if 0 < y < height - 1 and 0 < x < width - 1:
        u_center = u[y, x]
        v_center = v[y, x]
        lap_u = u[y - 1, x] + u[y + 1, x] + u[y, x - 1] + u[y, x + 1] - 4 * u_center
        lap_v = v[y - 1, x] + v[y + 1, x] + v[y, x - 1] + v[y, x + 1] - 4 * v_center
        reaction = u_center * v_center * v_center
        u_next[y, x] = u_center + dt * (diffusion_u * lap_u - reaction + feed * (1 - u_center))
        v_next[y, x] = v_center + dt * (diffusion_v * lap_v + reaction - (feed + kill) * v_center)


def prepare_candidate(u: np.ndarray, v: np.ndarray) -> CandidateState:
    u_device, v_device = cuda.to_device(u), cuda.to_device(v)
    return CandidateState(
        u=u_device,
        v=v_device,
        u_next=cuda.to_device(u),
        v_next=cuda.to_device(v),
    )


def run_candidate(state: CandidateState, params: GrayScottParams, steps: int) -> CandidateState:
    threads = (16, 16)
    blocks = (
        (state.u.shape[0] + threads[0] - 1) // threads[0],
        (state.u.shape[1] + threads[1] - 1) // threads[1],
    )
    scalars = (
        np.float32(params.diffusion_u),
        np.float32(params.diffusion_v),
        np.float32(params.feed),
        np.float32(params.kill),
        np.float32(params.dt),
    )
    for _ in range(steps):
        candidate_kernel[blocks, threads](
            state.u_next,
            state.v_next,
            state.u,
            state.v,
            *scalars,
        )
        state.u, state.u_next = state.u_next, state.u
        state.v, state.v_next = state.v_next, state.v
    return state


def candidate_fields(state: CandidateState) -> tuple[Any, Any]:
    return state.u, state.v
