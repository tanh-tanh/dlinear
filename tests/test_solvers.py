"""Solver dạng đóng trên dữ liệu giả có đáp án biết trước (Nlinear.ipynb), và ba đẳng thức của RQ1 ở cỡ nhỏ."""
import unittest

import numpy as np

from src.metrics import sse
from src.operators import build_N, build_P, make_Z, nlinear_transform
from src.solvers import (fit_with_bias, lstsq_solver, nlinear_constrained, ols, ridge,
                         ridge_path, ridge_pen)


def toy(n=200, L=12, H=4, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    W_true, b_true = rng.normal(size=(H, L)), rng.normal(size=H)
    X = rng.normal(size=(n, L))
    return X, X @ W_true.T + b_true + noise * rng.normal(size=(n, H)), W_true, b_true


def random_walk_windows(n=400, L=24, H=6, seed=1):
    s = np.cumsum(np.random.default_rng(seed).normal(size=n + L + H))
    X = np.stack([s[i: i + L] for i in range(n)])
    Y = np.stack([s[i + L: i + L + H] for i in range(n)])
    return X, Y


class TestBasicSolvers(unittest.TestCase):
    def test_ols_recovers_truth(self):
        X, Y, W_true, b_true = toy()
        W, b = fit_with_bias(ols, X, Y)
        np.testing.assert_allclose(W, W_true, atol=1e-12)
        np.testing.assert_allclose(b, b_true, atol=1e-12)

    def test_ridge_zero_is_ols(self):
        X, Y, _, _ = toy(noise=0.1)
        np.testing.assert_allclose(ridge(X, Y, 0.0), ols(X, Y), atol=1e-12)

    def test_ridge_path_matches_ridge(self):
        X, Y, _, _ = toy(noise=0.1)
        W_at = ridge_path(X, Y)
        for lam in [0.0, 1.0, 1e3]:
            np.testing.assert_allclose(W_at(lam), ridge(X, Y, lam), atol=1e-10)


class TestNLinear(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = random_walk_windows()

    def test_constrained_rows_sum_to_one(self):
        V, _ = fit_with_bias(nlinear_constrained, self.X, self.Y)
        np.testing.assert_allclose(V.sum(1), 1.0, atol=1e-10)

    def test_sse_increase_is_r2_over_s(self):
        X, Y = self.X, self.Y
        W_ols, b_ols = fit_with_bias(ols, X, Y)
        V, b_V = fit_with_bias(nlinear_constrained, X, Y)
        Xc = X - X.mean(0)
        u = np.linalg.solve(Xc.T @ Xc, np.ones(X.shape[1]))
        r = W_ols.sum(1) - 1.0
        lhs = sse(V, b_V, X, Y) - sse(W_ols, b_ols, X, Y)
        self.assertAlmostEqual(lhs / (r @ r / u.sum()), 1.0, places=8)

    def test_transform_equals_constrained(self):
        X, Y = self.X, self.Y
        Xt, Yt, x_last = nlinear_transform(X, Y)
        W_n, b_n = fit_with_bias(ols, Xt, Yt)
        V, b_V = fit_with_bias(nlinear_constrained, X, Y)
        np.testing.assert_allclose(Xt @ W_n.T + b_n + x_last, X @ V.T + b_V, atol=1e-8)


class TestDLinear(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = random_walk_windows()
        self.P = build_P(self.X.shape[1], 5)

    def test_lambda_zero_equals_linear(self):
        X, Y = self.X, self.Y
        Z = make_Z(X, self.P)
        self.assertEqual(np.linalg.matrix_rank(Z - Z.mean(0)), X.shape[1])
        W_D, b_D = fit_with_bias(lstsq_solver, Z, Y)
        W_L, b_L = fit_with_bias(ols, X, Y)
        np.testing.assert_allclose(Z @ W_D.T + b_D, X @ W_L.T + b_L, atol=1e-8)

    def test_weight_decay_is_ridge_in_metric_N(self):
        X, Y = self.X, self.Y
        Z = make_Z(X, self.P)
        N_inv = np.linalg.inv(build_N(self.P))
        for lam in [1.0, 1e2]:
            W_a, b_a = fit_with_bias(lambda A, B: ridge(A, B, lam), Z, Y)
            W_c, b_c = fit_with_bias(lambda A, B: ridge_pen(A, B, lam * N_inv), X, Y)
            np.testing.assert_allclose(Z @ W_a.T + b_a, X @ W_c.T + b_c, atol=1e-8)
            W_b, b_b = fit_with_bias(lambda A, B: ridge(A, B, lam), X, Y)
            self.assertGreater(np.abs(Z @ W_a.T + b_a - (X @ W_b.T + b_b)).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
