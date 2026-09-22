from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class GrayScottParams:
    diffusion_u: float = 0.16
    diffusion_v: float = 0.08
    feed: float = 0.035
    kill: float = 0.065
    dt: float = 1.0


def initialize_fields(
    shape: tuple[int, int],
    seed: int = 20260918,
) -> tuple[FloatArray, FloatArray]:
    height, width = shape
    if height < 5 or width < 5:
        raise ValueError("shape must be at least 5x5")
    u = np.ones(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    radius_y, radius_x = max(1, height // 10), max(1, width // 10)
    center_y, center_x = (height - 1) // 2, (width - 1) // 2
    for y in range(max(1, center_y - radius_y), min(height - 1, center_y + radius_y + 1)):
        for x in range(max(1, center_x - radius_x), min(width - 1, center_x + radius_x + 1)):
            # Indices are converted to Julia's one-based coordinates so every
            # language starts from the same mathematical field.
            noise = np.float32(
                0.01 * np.sin(((y + 1) * 73856093 + (x + 1) * 19349663 + seed) * 0.001)
            )
            u[y, x] = np.float32(0.5) + noise
            v[y, x] = np.float32(0.25) - noise
    return u, v


def cpu_simulate(
    initial_u: FloatArray,
    initial_v: FloatArray,
    params: GrayScottParams,
    steps: int,
) -> tuple[FloatArray, FloatArray]:
    u, v = initial_u.copy(), initial_v.copy()
    for _ in range(steps):
        u_next, v_next = u.copy(), v.copy()
        u_center = u[1:-1, 1:-1]
        v_center = v[1:-1, 1:-1]
        lap_u = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - np.float32(4) * u_center
        lap_v = v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:] - np.float32(4) * v_center
        reaction = u_center * v_center * v_center
        u_next[1:-1, 1:-1] = u_center + params.dt * (
            params.diffusion_u * lap_u - reaction + params.feed * (1 - u_center)
        )
        v_next[1:-1, 1:-1] = v_center + params.dt * (
            params.diffusion_v * lap_v + reaction - (params.feed + params.kill) * v_center
        )
        u, v = u_next, v_next
    return u, v


def relative_l2_error(actual: FloatArray, expected: FloatArray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(actual - expected) / denominator)
