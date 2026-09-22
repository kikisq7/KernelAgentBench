from __future__ import annotations

from pathlib import Path

import numpy as np

from python.gray_scott.reference import GrayScottParams, cpu_simulate, initialize_fields


def test_reference_is_deterministic_and_preserves_boundaries() -> None:
    u, v = initialize_fields((31, 29))
    result_u, result_v = cpu_simulate(u, v, GrayScottParams(), 5)
    repeated_u, repeated_v = cpu_simulate(u, v, GrayScottParams(), 5)
    np.testing.assert_array_equal(result_u, repeated_u)
    np.testing.assert_array_equal(result_v, repeated_v)
    np.testing.assert_array_equal(result_u[[0, -1], :], u[[0, -1], :])
    np.testing.assert_array_equal(result_u[:, [0, -1]], u[:, [0, -1]])
    np.testing.assert_array_equal(result_v[[0, -1], :], v[[0, -1], :])
    np.testing.assert_array_equal(result_v[:, [0, -1]], v[:, [0, -1]])
    assert np.isfinite(result_u).all()
    assert np.isfinite(result_v).all()


def test_candidate_source_compiles_without_gpu_dependencies() -> None:
    candidate = Path(__file__).parents[1] / "gray_scott" / "candidate.py"
    compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
