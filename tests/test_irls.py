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
import warnings

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

    def test_nlinear_penalty_equals_transform_ridge(self):
        """NLinear có weight decay: ràng buộc tổng hàng = 1 và phạt diag(1, …, 1, 0) (không phạt lag cuối)
        bằng ridge trên chuỗi đã trừ giá trị cuối (thế w_L = 1 − Σ_{j<L} w_j). δ lớn → MSE. Đo: ~1e-15."""
        from src.operators import nlinear_effective, nlinear_transform
        from src.solvers import fit_with_bias, ridge
        L = self.p - 1
        D = torch.diag(torch.cat([torch.ones(L - 1, dtype=DT), torch.zeros(1, dtype=DT)]))
        lam = 25 / self.D                                         # λ_mse = 2δλ = 50
        W, n_iter = irls(self.Xt, self.Y, self.D, self.W0, lam=lam, pen=D, constrained=True)
        self.assertEqual(n_iter, 2)
        X, Y = self.Xt[:, :-1].numpy(), self.Y.numpy()
        Xn, Yn, _ = nlinear_transform(X, Y)
        Wn, b = fit_with_bias(lambda A, B: ridge(A, B, 50.0), Xn, Yn)
        W_ref = torch.from_numpy(np.concatenate([nlinear_effective(Wn), b[:, None]], axis=1))
        err = rel_err(W, W_ref)
        self.assertLess(err, 1e-10, msg=f"rel err = {err:.2e}")
        # phạt diag khác phạt I (không để test đạt tầm thường)
        W_I, _ = irls(self.Xt, self.Y, self.D, self.W0, lam=lam, constrained=True)
        self.assertGreater(rel_err(W_I, W_ref), 1e-3)

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


class TestPatience(unittest.TestCase):
    """patience: chỉ dừng khi J giảm < tol trong patience vòng liên tiếp."""

    def setUp(self):
        self.Xt, self.Y = make_data()
        self.H, self.p = self.Y.shape[1], self.Xt.shape[1]
        self.W0 = torch.zeros(self.H, self.p, dtype=DT)

    def run_J(self, W0, **kw):
        """irls, trả (W, số vòng, J cuối do irls tính)."""
        Js = []
        W, n = irls(self.Xt, self.Y, MAE_DELTA, W0, callback=lambda it, W, J: Js.append(J.item()), **kw)
        return W, n, Js[-1]

    def test_patience_one_is_default(self):
        W_def, n_def = irls(self.Xt, self.Y, MAE_DELTA, self.W0)
        W_p1, n_p1 = irls(self.Xt, self.Y, MAE_DELTA, self.W0, patience=1)
        self.assertEqual(n_p1, n_def)
        self.assertTrue(torch.equal(W_p1, W_def))

    def test_warm_start_does_not_stop_after_one_iteration(self):
        """Warm từ nghiệm λ = 1 sang λ = 1,1: vòng đầu J giảm < tol nên patience = 1 dừng ngay
        (đo: 1 vòng), patience = 5 đi tiếp (đo: 5 vòng) và J không lớn hơn."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_near, _ = irls(self.Xt, self.Y, MAE_DELTA, self.W0, lam=1.0)
        _, n1, J1 = self.run_J(W_near, lam=1.1, tol=1e-6, patience=1)
        _, n5, J5 = self.run_J(W_near, lam=1.1, tol=1e-6, patience=5)
        self.assertGreater(n5, n1)
        self.assertLessEqual(J5, J1)


class TestGramFallback(unittest.TestCase):
    """Chế độ pha: bước h nào làm J_h tăng (A_h float32 quá xấu điều kiện) thì lập lại A_h bằng float64.

    Trên ETTh2 H = 720 MAE, κ(A_3) = 8,3e7 làm bước h = 3 đi sai (BAO_CAO_TIEU_CHI_DUNG mục 8).
    Ở đây tái hiện bằng hai cột gần cộng tuyến. Đo được: không fallback → J tăng ở vòng 48; có
    fallback → 30 bước h lập lại, MAE khớp float64 tới ~3e-13.
    """

    def ill_data(self):
        g = torch.Generator().manual_seed(0)
        n, L, H = 300, 6, 3
        X = torch.randn(n, L, generator=g, dtype=DT)
        X[:, 1] = X[:, 0] + X[:, 1] / 100                   # hai cột gần cộng tuyến
        W_true = torch.randn(H, L, generator=g, dtype=DT)
        torch.manual_seed(0)
        Y = X @ W_true.T + 0.5 + torch.distributions.Laplace(0.0, 1.0).sample((n, H)).to(DT)
        return torch.cat([X, torch.ones(n, 1, dtype=DT)], 1), Y

    def test_fallback_fixes_ill_conditioned_step(self):
        Xt, Y = self.ill_data()
        W0 = torch.zeros(Y.shape[1], Xt.shape[1], dtype=DT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(RuntimeError):
                irls(Xt, Y, MAE_DELTA, W0, tol=1e-12, gram_dtype=torch.float32, gram_fallback=False)
            W, _ = irls(Xt, Y, MAE_DELTA, W0, tol=1e-12, gram_dtype=torch.float32)
            self.assertGreater(irls.last_fallbacks, 0)
            W64, _ = irls(Xt, Y, MAE_DELTA, W0, tol=1e-12)
        mae, mae64 = (Y - Xt @ W.T).abs().mean(), (Y - Xt @ W64.T).abs().mean()
        gap = ((mae - mae64) / mae64).item()
        self.assertLess(abs(gap), 1e-8, msg=f"MAE có fallback lệch float64 tương đối {gap:.2e}")

    def test_no_fallback_when_well_conditioned(self):
        """Dữ liệu điều kiện tốt: fallback không chạy, kết quả trùng từng bit với gram_fallback=False."""
        Xt, Y = make_data()
        W0 = torch.zeros(Y.shape[1], Xt.shape[1], dtype=DT)
        W1, n1 = irls(Xt, Y, MAE_DELTA, W0, gram_dtype=torch.float32)
        self.assertEqual(irls.last_fallbacks, 0)
        W2, n2 = irls(Xt, Y, MAE_DELTA, W0, gram_dtype=torch.float32, gram_fallback=False)
        self.assertEqual(n1, n2)
        self.assertTrue(torch.equal(W1, W2))


class TestDLinearEquivalence(unittest.TestCase):
    """DLinear + weight decay ≡ Linear với phạt N⁻¹, với mọi hàm mất mát (NHAT_KY mục 1.5).

    Cách 1: giải trực tiếp trong 2L chiều, Z = [X Pᵀ, X (I − P)ᵀ] cộng cột 1, pen = I trên 2L hệ số
    (irls lấy cỡ pen từ Xt: p − 1 = 2L), ghép W_eff = W_t P + W_s (I − P). Cách 2: L chiều, pen = N⁻¹.
    Mỗi vòng IRLS là một bài ridge có trọng số, và ridge có trọng số trong 2L chiều với phạt I
    trùng đại số với L chiều với phạt N⁻¹, nên hai dãy lặp trùng nhau từng vòng. Đo được: Huber
    W_eff lệch 1e-15, bias 0; MAE F lệch 0, W lệch 2e-15. Phạt có tác dụng thật: nghiệm N⁻¹ khác
    nghiệm phạt I ~0,11 và khác λ = 0 ~0,3–0,4 (kiểm ở check_penalty_matters).
    """

    L, K, H, N, LAM = 24, 5, 3, 400, 5.0

    @classmethod
    def setUpClass(cls):
        from src.operators import build_N, build_P, dlinear_effective, make_Z
        g = torch.Generator().manual_seed(3)
        # chuỗi có tương quan theo thời gian để P có tác dụng thật
        X = torch.cumsum(torch.randn(cls.N, cls.L, generator=g, dtype=DT), dim=1) / 3
        W_true = torch.randn(cls.H, cls.L, generator=g, dtype=DT) / cls.L
        noise = torch.distributions.Laplace(0.0, 0.5).sample((cls.N, cls.H)).to(DT)
        Y = X @ W_true.T + 0.3 + noise
        cls.P = build_P(cls.L, cls.K)
        cls.Ninv = np.linalg.inv(build_N(cls.P))
        ones = torch.ones(cls.N, 1, dtype=DT)
        cls.Xt = torch.cat([X, ones], 1)
        cls.Zt = torch.cat([torch.from_numpy(make_Z(X.numpy(), cls.P)), ones], 1)
        cls.Y = Y
        cls.eff = staticmethod(dlinear_effective)

    def solve_both(self, delta):
        kw = dict(lam=self.LAM, tol=1e-12, patience=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # MAE có thể chạm max_iter ở tol = 1e-12
            G, _ = irls(self.Zt, self.Y, delta, torch.zeros(self.H, 2 * self.L + 1, dtype=DT), **kw)
            W, _ = irls(self.Xt, self.Y, delta, torch.zeros(self.H, self.L + 1, dtype=DT), pen=self.Ninv, **kw)
        G = G.numpy()
        W_eff = self.eff(G[:, :self.L], G[:, self.L:2 * self.L], self.P)
        W1 = np.concatenate([W_eff, G[:, -1:]], axis=1)              # [H, L + 1], bias ở cột cuối
        return W1, W.numpy(), G

    def F(self, W, delta):
        """Hàm mục tiêu có phạt, dạng tổng, trong không gian L chiều: Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ N⁻¹ w_h."""
        R = self.Y.numpy() - self.Xt.numpy() @ W.T
        a = np.abs(R)
        q = np.minimum(a, delta)
        return (q * (a - q / 2)).sum() / delta + self.LAM * np.einsum("hi,ij,hj->", W[:, :-1], self.Ninv, W[:, :-1])

    def check_penalty_matters(self, W2, delta):
        """Không để test đạt tầm thường: phạt N⁻¹ phải cho nghiệm khác hẳn phạt I."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            WI, _ = irls(self.Xt, self.Y, delta, torch.zeros(self.H, self.L + 1, dtype=DT), lam=self.LAM,
                         tol=1e-12, patience=5)
        WI = WI.numpy()
        self.assertGreater(np.linalg.norm(W2 - WI) / np.linalg.norm(WI), 1e-2)

    def test_huber(self):
        W1, W2, _ = self.solve_both(1.0)
        self.check_penalty_matters(W2, 1.0)
        err = np.linalg.norm(W1[:, :-1] - W2[:, :-1]) / np.linalg.norm(W2[:, :-1])
        self.assertLess(err, 1e-8, msg=f"W_eff lệch tương đối {err:.2e}")
        err_b = np.abs(W1[:, -1] - W2[:, -1]).max()
        self.assertLess(err_b, 1e-8, msg=f"bias lệch {err_b:.2e}")

    def test_mae(self):
        """Mặt MAE phẳng: so giá trị hàm mục tiêu có phạt, hai chiều (|ΔF|), không so W.

        Thêm: F của cách 1 tính bằng phạt 2L chiều của chính nó (‖W_t‖² + ‖W_s‖²) cũng phải bằng,
        vì ở nghiệm, (W_t, W_s) là cách tách W_eff có chuẩn nhỏ nhất.
        """
        W1, W2, G = self.solve_both(MAE_DELTA)
        self.check_penalty_matters(W2, MAE_DELTA)
        F1, F2 = self.F(W1, MAE_DELTA), self.F(W2, MAE_DELTA)
        gap = (F1 - F2) / F2
        self.assertLess(abs(gap), 1e-6, msg=f"F (2L chiều) so với F (N⁻¹) lệch tương đối {gap:.2e}")
        pen_2L = self.LAM * (G[:, :-1] ** 2).sum()
        pen_L = self.LAM * np.einsum("hi,ij,hj->", W1[:, :-1], self.Ninv, W1[:, :-1])
        gap_pen = (F1 - pen_L + pen_2L - F2) / F2
        self.assertLess(abs(gap_pen), 1e-6, msg=f"F với phạt 2L chiều lệch tương đối {gap_pen:.2e}")


class TestNLinearMAE(unittest.TestCase):
    """Câu 11.6: NLinear với MAE = quy hoạch tuyến tính có ràng buộc aᵀw = 1, a = [1, …, 1, 0]."""

    @unittest.skipIf(linprog is None, "cần scipy")
    def test_matches_constrained_lp(self):
        Xt, Y = make_data()
        X = Xt.numpy()
        n, p = X.shape
        H = Y.shape[1]
        a = np.r_[np.ones(p - 1), 0.0]
        W_lp = []
        for h in range(H):
            y = Y[:, h].numpy()
            res = linprog(
                c=np.r_[np.zeros(p), np.ones(n)],
                A_ub=np.block([[X, -np.eye(n)], [-X, -np.eye(n)]]),
                b_ub=np.r_[y, -y],
                A_eq=np.r_[a, np.zeros(n)][None, :], b_eq=[1.0],
                bounds=[(None, None)] * p + [(0, None)] * n,
                method="highs",
            )
            self.assertTrue(res.success, res.message)
            W_lp.append(res.x[:p])
        W_lp = torch.tensor(np.array(W_lp), dtype=DT)

        W, _ = irls(Xt, Y, MAE_DELTA, torch.zeros(H, p, dtype=DT), constrained=True, tol=1e-12)
        mae_irls = (Y - Xt @ W.T).abs().mean(dim=0)
        mae_lp = (Y - Xt @ W_lp.T).abs().mean(dim=0)
        gap = ((mae_irls - mae_lp) / mae_lp).max().item()
        self.assertLess(gap, 1e-6, msg=f"MAE IRLS kém LP tương đối {gap:.2e}")
        row_sums = W[:, :-1].sum(dim=1)
        self.assertLess((row_sums - 1).abs().max().item(), 1e-10)
        # ràng buộc có tác dụng thật: nghiệm khác MAE không ràng buộc
        W_free, _ = irls(Xt, Y, MAE_DELTA, torch.zeros(H, p, dtype=DT), tol=1e-12)
        self.assertGreater((W_free[:, :-1].sum(dim=1) - 1).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()