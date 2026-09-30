"""Kiểm IRLS (MAE, Huber) trong src/solvers.py.

Hai phép kiểm theo outline:
  1. Huber với δ lớn phải về đúng nghiệm MSE (có và không có ràng buộc NLinear).
  2. MAE chạy từ nhiều điểm khởi tạo phải về cùng một nghiệm.
Thêm: nghiệm MAE của IRLS khớp nghiệm chính xác từ quy hoạch tuyến tính; có phạt λ thì khớp
ridge (δ lớn) và quy hoạch bậc hai (MAE).

Ngưỡng được chọn theo sai số đo thử (float64, dữ liệu dưới đây):
  - δ lớn, so với OLS/KKT: ~1e-15          → ngưỡng 1e-10
  - MAE, 3 điểm khởi tạo với nhau: ~1e-14   → ngưỡng 1e-8
  - MAE so với quy hoạch tuyến tính: ~6e-9  → ngưỡng 1e-6 (sai lệch tỷ lệ với δ)
  - λ > 0, δ lớn so với ridge/KKT: ~1e-15   → ngưỡng 1e-10
  - MAE + λ so với quy hoạch bậc hai: ~3e-8 → ngưỡng 1e-6
Các test đo độ chính xác của nghiệm truyền tol = 1e-12; tol mặc định (1e-9) là tiêu chí dừng thực tế.
Và hai kiểu số khác: float32 thuần (MAE kém ~2e-6 → ngưỡng 1e-4), pha float32/float64 (W lệch
~7e-12 so với float64 → ngưỡng 1e-8).
"""
import unittest

import numpy as np
import torch

from src.solvers import MAE_DELTA, irls

try:
    from scipy.optimize import linprog, minimize
except ImportError:  # scipy chỉ cần cho hai test so với bộ giải độc lập
    linprog = minimize = None

DT = torch.float64


def make_data(n=200, L=4, H=3, seed=0):
    """Dữ liệu giả: Y = X W + bias + nhiễu Laplace (đuôi dày, hợp với MAE)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, L, generator=g, dtype=DT)
    W_true = torch.randn(H, L, generator=g, dtype=DT)
    noise = torch.distributions.Laplace(0.0, 1.0).sample((n, H)).to(DT)
    Y = X @ W_true.T + 0.5 + noise
    Xt = torch.cat([X, torch.ones(n, 1, dtype=DT)], dim=1)   # cột 1 cho bias
    return Xt, Y


def rel_err(A, B):
    return (torch.linalg.norm(A - B) / torch.linalg.norm(B)).item()


class TestHuberLargeDelta(unittest.TestCase):
    """δ lớn hơn mọi phần dư → mọi trọng số = 1 → vòng 1 ra nghiệm MSE, vòng 2 xác nhận dừng."""

    def setUp(self):
        torch.manual_seed(0)
        self.Xt, self.Y = make_data()
        self.H, self.p = self.Y.shape[1], self.Xt.shape[1]
        self.W0 = torch.zeros(self.H, self.p, dtype=DT)       # cố ý xa nghiệm

    def test_one_iteration_equals_ols(self):
        W, n_iter = irls(self.Xt, self.Y, 1e6, self.W0)
        self.assertEqual(n_iter, 2)
        W_ols = torch.linalg.lstsq(self.Xt, self.Y).solution.T
        err = rel_err(W, W_ols)
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")

    def test_constrained_equals_kkt(self):
        W, n_iter = irls(self.Xt, self.Y, 1e6, self.W0, constrained=True)
        self.assertEqual(n_iter, 2)

        # Nghiệm tham chiếu độc lập: giải hệ KKT cho từng h
        #   [XᵀX  a][w]   [Xᵀy]
        #   [aᵀ   0][μ] = [ 1 ]
        a = torch.ones(self.p, dtype=DT)
        a[-1] = 0
        G = self.Xt.T @ self.Xt
        K = torch.zeros(self.p + 1, self.p + 1, dtype=DT)
        K[:self.p, :self.p] = G
        K[:self.p, -1] = a
        K[-1, :self.p] = a
        W_ref = torch.stack([
            torch.linalg.solve(K, torch.cat([self.Xt.T @ self.Y[:, h], torch.ones(1, dtype=DT)]))[:self.p]
            for h in range(self.H)
        ])

        err = rel_err(W, W_ref)
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")
        row_sums = (W * a).sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(self.H, dtype=DT), atol=1e-12))


class TestPenalty(unittest.TestCase):
    """λ > 0. Hàm mục tiêu Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ Pen w_h, không phạt bias.

    δ lớn: ρ_δ(r)/δ = r²/(2δ) nên nghiệm là ridge của MSE với phạt 2δλ·Pen.
    """

    D = 1e6

    def setUp(self):
        self.Xt, self.Y = make_data()
        self.H, self.p = self.Y.shape[1], self.Xt.shape[1]
        self.W0 = torch.zeros(self.H, self.p, dtype=DT)
        g = torch.Generator().manual_seed(2)
        M = torch.randn(self.p - 1, self.p - 1, generator=g, dtype=DT)
        self.M = M @ M.T + torch.eye(self.p - 1, dtype=DT)       # đối xứng, xác định dương

    def pad(self, P):
        """[L, L] → [p, p], hàng và cột bias bằng 0."""
        out = torch.zeros(self.p, self.p, dtype=DT)
        out[:-1, :-1] = P
        return out

    def ridge_ref(self, Pen):
        return torch.linalg.solve(self.Xt.T @ self.Xt + Pen, self.Xt.T @ self.Y).T

    def test_large_delta_equals_ridge(self):
        lam = 25 / self.D                                         # λ_mse = 2δλ = 50
        W, n_iter = irls(self.Xt, self.Y, self.D, self.W0, lam=lam)
        self.assertEqual(n_iter, 2)
        err = rel_err(W, self.ridge_ref(self.pad(50 * torch.eye(self.p - 1, dtype=DT))))
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")

    def test_large_delta_penalty_matrix(self):
        lam = 5 / self.D
        W, n_iter = irls(self.Xt, self.Y, self.D, self.W0, lam=lam, pen=self.M)
        self.assertEqual(n_iter, 2)
        err = rel_err(W, self.ridge_ref(self.pad(10 * self.M)))
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")

    def test_constrained_large_delta_equals_kkt(self):
        lam = 25 / self.D
        W, n_iter = irls(self.Xt, self.Y, self.D, self.W0, lam=lam, constrained=True)
        self.assertEqual(n_iter, 2)
        a = torch.ones(self.p, dtype=DT)
        a[-1] = 0
        K = torch.zeros(self.p + 1, self.p + 1, dtype=DT)
        K[:self.p, :self.p] = self.Xt.T @ self.Xt + self.pad(50 * torch.eye(self.p - 1, dtype=DT))
        K[:self.p, -1] = a
        K[-1, :self.p] = a
        W_ref = torch.stack([
            torch.linalg.solve(K, torch.cat([self.Xt.T @ self.Y[:, h], torch.ones(1, dtype=DT)]))[:self.p]
            for h in range(self.H)
        ])
        err = rel_err(W, W_ref)
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")

    @unittest.skipIf(minimize is None, "cần scipy")
    def test_mae_matches_qp(self):
        """MAE + λ = quy hoạch bậc hai: min Σt + λ wᵀ Pen w  s.t.  −t ≤ y − Xβ ≤ t (SLSQP).

        So hai chiều trên F = Σ|r| + λ wᵀ Pen w. Nếu irls dùng sai quy ước λ (vd lệch 2 lần) thì
        F của nó kém hẳn; nếu SLSQP dừng sớm thì irls tốt hơn hẳn. SLSQP kết thúc với "Positive
        directional derivative" (hết cải thiện được ở giới hạn chính xác), không phải lỗi.
        Đo được |ΔF|/F ≈ 1e-8–3e-8 ở tol mặc định.
        """
        lam = 5.0
        X, Pen = self.Xt.numpy(), self.pad(self.M).numpy()
        n, p = X.shape
        W, _ = irls(self.Xt, self.Y, MAE_DELTA, self.W0, lam=lam, pen=self.M)
        W = W.numpy()
        A_ub = np.block([[X, np.eye(n)], [-X, np.eye(n)]])        # A z ≥ [y; −y]  ⇔  t ≥ |y − Xβ|
        for h in range(self.H):
            y = self.Y[:, h].numpy()
            b0 = np.linalg.lstsq(X, y, rcond=None)[0]
            res = minimize(
                lambda z: z[p:].sum() + lam * z[:p] @ Pen @ z[:p],
                np.r_[b0, np.abs(y - X @ b0) + 1e-3],
                jac=lambda z: np.r_[2 * lam * Pen @ z[:p], np.ones(n)],
                method="SLSQP",
                constraints=[{"type": "ineq", "fun": lambda z: A_ub @ z - np.r_[y, -y], "jac": lambda z: A_ub}],
                options={"maxiter": 2000, "ftol": 1e-15},
            )
            F = lambda b: np.abs(y - X @ b).sum() + lam * b @ Pen @ b
            gap = (F(W[h]) - F(res.x[:p])) / F(res.x[:p])
            self.assertLess(abs(gap), 1e-6, msg=f"h={h}: F irls so với QP lệch tương đối {gap:.2e}")

        # phạt có tác dụng thật: nghiệm khác hẳn MAE không phạt
        W_free, _ = irls(self.Xt, self.Y, MAE_DELTA, self.W0)
        self.assertGreater(rel_err(torch.from_numpy(W), W_free), 1e-2)


class TestMAE(unittest.TestCase):
    """δ nhỏ → IRLS giải MAE."""

    def setUp(self):
        torch.manual_seed(0)
        self.Xt, self.Y = make_data()
        self.H, self.p = self.Y.shape[1], self.Xt.shape[1]

    def test_multi_init_agree(self):
        g = torch.Generator().manual_seed(1)
        inits = [
            torch.zeros(self.H, self.p, dtype=DT),
            torch.linalg.lstsq(self.Xt, self.Y).solution.T,
            3 * torch.randn(self.H, self.p, generator=g, dtype=DT),
        ]
        # tol chặt hơn mặc định: kiểm nghiệm duy nhất tới ~1e-14, không kiểm tiêu chí dừng
        sols = [irls(self.Xt, self.Y, MAE_DELTA, W0, tol=1e-12)[0] for W0 in inits]
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                err = rel_err(sols[i], sols[j])
                self.assertLess(err, 1e-8, msg=f"init {i} vs {j}: rel err = {err:.2e}")

    @unittest.skipIf(linprog is None, "cần scipy")
    def test_matches_linear_program(self):
        """MAE chính xác = quy hoạch tuyến tính: min Σt  s.t.  −t ≤ y − Xw ≤ t."""
        X = self.Xt.numpy()
        n, p = X.shape
        W_lp = []
        for h in range(self.H):
            y = self.Y[:, h].numpy()
            res = linprog(
                c=np.r_[np.zeros(p), np.ones(n)],
                A_ub=np.block([[X, -np.eye(n)], [-X, -np.eye(n)]]),
                b_ub=np.r_[y, -y],
                bounds=[(None, None)] * p + [(0, None)] * n,
                method="highs",
            )
            W_lp.append(res.x[:p])
        W_lp = torch.tensor(np.array(W_lp), dtype=DT)

        W, _ = irls(self.Xt, self.Y, MAE_DELTA, torch.zeros(self.H, self.p, dtype=DT))
        mae_irls = (self.Y - self.Xt @ W.T).abs().mean(dim=0)
        mae_lp = (self.Y - self.Xt @ W_lp.T).abs().mean(dim=0)
        gap = ((mae_irls - mae_lp) / mae_lp).max().item()
        self.assertLess(gap, 1e-6, msg=f"MAE IRLS kém LP tương đối {gap:.2e}")

    def test_float32_no_false_alarm(self):
        """float32: J lệch vài ε do làm tròn, tăng cỡ đó không phải lỗi code (không RuntimeError).

        So bằng MAE tính lại bằng float64, không so W: gần nghiệm MAE rất phẳng nên W của
        float32 lệch ~7e-4 mà MAE chỉ kém ~2e-6 tương đối.
        """
        W0 = torch.zeros(self.H, self.p, dtype=DT)
        W64, _ = irls(self.Xt, self.Y, MAE_DELTA, W0, tol=1e-12)
        W32, _ = irls(self.Xt.float(), self.Y.float(), MAE_DELTA, W0.float(), tol=1e-12)
        mae64 = (self.Y - self.Xt @ W64.T).abs().mean()
        mae32 = (self.Y - self.Xt @ W32.double().T).abs().mean()
        gap = ((mae32 - mae64) / mae64).item()
        self.assertLess(gap, 1e-4, msg=f"MAE float32 kém float64 tương đối {gap:.2e}")

    def test_mixed_precision_matches_float64(self):
        """A_h lập bằng float32, r/g/J bằng float64 → cùng nghiệm với float64 thuần (đo: ~7e-12)."""
        W0 = torch.zeros(self.H, self.p, dtype=DT)
        W64, _ = irls(self.Xt, self.Y, MAE_DELTA, W0, tol=1e-12)
        Wm, _ = irls(self.Xt, self.Y, MAE_DELTA, W0, tol=1e-12, gram_dtype=torch.float32)
        self.assertEqual(Wm.dtype, DT)
        err = rel_err(Wm, W64)
        self.assertLess(err, 1e-8, msg=f"mixed vs float64: rel err = {err:.2e}")

    def test_warns_when_not_converged(self):
        """MAE cần hàng trăm vòng; chạm max_iter phải có cảnh báo, không im lặng."""
        W0 = torch.zeros(self.H, self.p, dtype=DT)
        with self.assertWarns(UserWarning):
            _, n_iter = irls(self.Xt, self.Y, MAE_DELTA, W0, max_iter=3)
        self.assertEqual(n_iter, 3)


if __name__ == "__main__":
    unittest.main()