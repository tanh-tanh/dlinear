"""P khớp moving_avg của DLinear, tính chất của P và N. Kiểm trong dlinearv2.ipynb mục 4 và 01 mục 2."""
import unittest

import numpy as np
import torch

from src.models import moving_avg
from src.operators import build_N, build_P, nlinear_effective


def probe_P(L, k):
    """Kênh i của đầu vào là e_i  =>  output kênh i là cột i của P."""
    E = torch.eye(L, dtype=torch.float64).unsqueeze(0)       # [1, L, L]
    with torch.no_grad():
        return moving_avg(k, 1)(E)[0].numpy()


class TestP(unittest.TestCase):
    def test_small_example(self):
        expected = np.array([[3, 1, 1, 0, 0],
                             [2, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 2],
                             [0, 0, 1, 1, 3]])
        np.testing.assert_allclose(build_P(5, 5) * 5, expected, atol=1e-12)

    def test_matches_moving_avg(self):
        for L, k in [(5, 3), (5, 5), (10, 7), (96, 25), (336, 5), (336, 15), (336, 25), (336, 49)]:
            with self.subTest(L=L, k=k):
                np.testing.assert_allclose(build_P(L, k), probe_P(L, k), atol=1e-14)

    def test_rows_sum_to_one(self):
        for k in [5, 15, 25, 49]:
            np.testing.assert_allclose(build_P(336, k).sum(1), 1.0, atol=1e-13)

    def test_N_eigenvalues_at_least_half(self):
        # vᵀNv = ‖Pv‖² + ‖(I−P)v‖² ≥ ½‖v‖²
        for k in [5, 15, 25, 49]:
            self.assertGreaterEqual(np.linalg.eigvalsh(build_N(build_P(336, k))).min(), 0.5 - 1e-12)


class TestNLinearEffective(unittest.TestCase):
    def test_rows_sum_to_one(self):
        W = np.random.default_rng(0).normal(size=(4, 9))
        np.testing.assert_allclose(nlinear_effective(W).sum(1), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
