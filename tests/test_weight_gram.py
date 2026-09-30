import unittest

import torch

from src.solvers import weight_gram


class TestWeightGram(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.p, self.H = 50, 4, 5
        self.Xt = torch.randn(self.n, self.p, dtype=torch.float64)
        self.w = torch.rand(self.n, self.H, dtype=torch.float64) + 0.1
        # chunk=2 với H=5 → khối cuối lẻ (chỉ có h=4)
        self.A = weight_gram(self.Xt, self.w, chunk=2)

    def test_shape(self):
        self.assertEqual(tuple(self.A.shape), (self.H, self.p, self.p))

    def test_matches_per_column(self):
        for h in range(self.H):
            A_ref = self.Xt.T @ (self.w[:, h, None] * self.Xt)
            err = (self.A[h] - A_ref).abs().max().item()
            self.assertLess(err, 1e-12, msg=f"h={h}, err={err:.2e}")


if __name__ == "__main__":
    unittest.main()