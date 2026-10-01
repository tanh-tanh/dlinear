"""Solver cho hàm mục tiêu MSE (dạng đóng, numpy) và MAE/Huber (IRLS, torch).

Quy ước: X [n, L], Y [n, H], trả W [H, L] sao cho dự báo = X @ W.T (+ b).

MSE (numpy)
    λ phạt trên TỔNG bình phương lỗi: ‖Y − X Wᵀ‖² + λ‖W‖². Đổi sang weight_decay của
    PyTorch (loss là MSE trung bình trên n·H phần tử): weight_decay ≈ 2λ / (n·H).
    Bias xử lý bằng fit_with_bias (trừ trung bình cột).

MAE, Huber (torch, chạy được trên GPU)
    irls(Xt, Y, delta, W0): Xt có cột 1 ở cuối cho bias, nên W có p = L + 1 cột.
    KHÔNG dùng fit_with_bias cho IRLS: với trọng số, trung bình để căn giữa phải là
    trung bình có trọng số và khác nhau theo từng bước h.
    MAE = irls với delta = MAE_DELTA (1e-6): Huber ở δ nhỏ chia cho δ cho đúng MAE (sai lệch
    tỷ lệ với δ). λ phạt trên TỔNG: Σ|r| + λ‖W‖² với MAE, Σ ρ_δ(r)/δ + λ‖W‖² với Huber
    (xem docstring của irls). Không phạt bias.

Chuyển từ Nlinear.ipynb, dlinearv2.ipynb và 01_three_predictions.ipynb.
"""
import warnings

import numpy as np
import torch

MAE_DELTA = 1e-6   # δ nhỏ hơn (vd 1e-8) làm IRLS hội tụ quá chậm và dừng sớm


# ---------------------------------------------------------------------------
# MSE: nghiệm dạng đóng (numpy)
# ---------------------------------------------------------------------------

def ols(X, Y):
    return np.linalg.solve(X.T @ X, X.T @ Y).T


def ridge(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y).T


def ridge_pen(X, Y, Pen):
    """Ridge với ma trận phạt tùy ý: (XᵀX + Pen)⁻¹XᵀY. DLinear có weight decay: Pen = λ·N⁻¹."""
    return np.linalg.solve(X.T @ X + Pen, X.T @ Y).T


def lstsq_solver(X, Y):
    """Nghiệm chuẩn nhỏ nhất, chịu được ma trận suy biến (Z của DLinear có hạng L, không phải 2L)."""
    return np.linalg.lstsq(X, Y, rcond=None)[0].T


def nlinear_constrained(X, Y):
    """OLS với ràng buộc mỗi hàng của W cộng bằng 1 (NLinear ở λ = 0), giải bằng nhân tử Lagrange.

    V = W_ols − (1/s)·r uᵀ, với r = W_ols 1_L − 1_H, u = (XᵀX)⁻¹1_L, s = 1_Lᵀu.
    """
    W_ols = ols(X, Y)
    ones_L, ones_H = np.ones(X.shape[1]), np.ones(Y.shape[1])
    u = np.linalg.solve(X.T @ X, ones_L)
    s = ones_L @ u
    return W_ols - np.outer(W_ols @ ones_L - ones_H, u) / s


def fit_with_bias(solver, X, Y):
    """Xử lý bias bằng cách trừ trung bình: giải W trên dữ liệu đã trừ trung bình, b = ȳ − W x̄.

    Bias không bị phạt. Truyền hàm solver, không gọi: fit_with_bias(ols, X, Y).
    Chỉ dùng cho MSE; IRLS dùng cột 1 (xem docstring của module).
    """
    x_bar, y_bar = X.mean(axis=0), Y.mean(axis=0)
    W = solver(X - x_bar, Y - y_bar)
    return W, y_bar - W @ x_bar


def ridge_path(X, Y):
    """Quét λ nhanh: phân rã XᵀX = Q diag(σ) Qᵀ một lần, mỗi λ chỉ còn một phép nhân ma trận.

    X, Y nên đã trừ trung bình (như trong fit_with_bias). Trả hàm lam -> W [H, L].
    """
    sig, Q = np.linalg.eigh(X.T @ X)
    QtC = Q.T @ (X.T @ Y)

    def W_at(lam):
        return ((Q / (sig + lam)) @ QtC).T
    return W_at


# ---------------------------------------------------------------------------
# MAE, Huber: IRLS (torch)
# ---------------------------------------------------------------------------

def weight_gram(Xt, w, chunk=8):
    """A[h] = Xtᵀ diag(w[:, h]) Xt cho mọi h.

    Xt [n, p], w [n, H] → A [H, p, p]. Tính theo khối `chunk` bước h, vì tensor trung gian
    của cả H bước cùng lúc có kích thước n·H·p (khoảng 7,4 GB với ETTh1, H = 96, float32).
    """
    n, p = Xt.shape
    H = w.shape[1]
    A = torch.empty(H, p, p, dtype=Xt.dtype, device=Xt.device)
    for h0 in range(0, H, chunk):
        h1 = min(h0 + chunk, H)
        A[h0:h1] = torch.einsum('nh,na,nb->hab', w[:, h0:h1], Xt, Xt)
    return A


def huber_objective(R, delta):
    """Trung bình Huber của phần dư R: r²/2 nếu |r| ≤ δ, δ(|r| − δ/2) nếu ngược lại.

    Viết gọn qua q = min(|r|, δ): ρ(r) = q·(|r| − q/2).
    """
    a = R.abs()
    q = a.clamp(max=delta)
    return (q * (a - q / 2)).mean()


def irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9,
         callback=None, gram_dtype=None, chunk=8, patience=1, gram_fallback=True):
    """Cực tiểu Huber(δ) có phạt ridge bằng IRLS; δ = MAE_DELTA cho MAE, δ rất lớn cho MSE.

    Hàm mục tiêu, dạng tổng như quy ước MSE ‖Y − XWᵀ‖² + λ‖W‖²:
        F(W) = Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ Pen w_h
    với ρ_δ(r) = r²/2 nếu |r| ≤ δ, δ(|r| − δ/2) nếu ngược lại; w_h là L hệ số đầu của hàng h
    (không phạt bias). Chia cho δ để δ → 0 cho đúng MAE: Σ|r| + λ‖W‖². δ lớn thì
    F ≈ ‖r‖²/(2δ) + λ‖W‖², tức ridge của MSE với λ_mse = 2δλ.

    Tham số
        Xt  [n, p]  đầu vào, cột cuối toàn 1 (bias là hệ số thứ p).
        Y   [n, H]  đích.
        W0  [H, p]  điểm khởi tạo; nên dùng nghiệm MSE (hội tụ nhanh nhất).
        lam  λ ≥ 0.
        pen  [L, L] đối xứng nửa xác định dương, mặc định I. DLinear có weight decay: pen = N⁻¹
             (weight decay của DLinear chỉ phụ thuộc W_eff nên tương đương này đúng với mọi hàm mất mát).
        constrained  True cho NLinear: mỗi hàng có tổng L hệ số đầu bằng 1, bias tự do.
        tol  dừng khi J giảm tương đối ít hơn tol sau một vòng. Với 1e-9, MAE trên ETTh1 H = 96
             cho MSE/MAE test lệch nghiệm ở vòng 1000 khoảng 2e-6 / 5e-6 (docs/BAO_CAO_IRLS_DTYPE.md).
        callback  nếu có, gọi callback(it, W, J_new) mỗi vòng, trước phép kiểm J không tăng.
        gram_dtype  kiểu số để lập A_h (vd torch.float32 khi Xt là float64). Mặc định: kiểu của Xt.
        chunk  số bước h lập A_h cùng lúc, chuyển thẳng cho weight_gram. Chỉ đổi tốc độ và bộ nhớ.
        patience  chỉ dừng khi J giảm tương đối ít hơn tol trong patience vòng liên tiếp; một vòng
             giảm ≥ tol đặt lại bộ đếm. Cần khi warm start: vòng đầu giảm rất ít dù chưa hội tụ
             (docs/BAO_CAO_TIEU_CHI_DUNG.md). J tăng trong ngưỡng rise_tol vẫn dừng ngay.
        gram_fallback  chỉ có tác dụng khi gram_dtype khác kiểu của Xt: bước h nào làm J_h tăng
             (A_h lập bằng gram_dtype quá xấu điều kiện) thì lập lại A_h đó bằng kiểu của Xt và giải
             lại. Không bao giờ chạy khi các bước đều tốt, nên không đổi kết quả trong trường hợp đó.
             Số bước h đã lập lại ghi ở irls.last_fallbacks.

    Trả (W [H, p], số vòng đã chạy). Báo RuntimeError nếu J tăng quá mức nhiễu làm tròn
    (IRLS là thuật toán MM nên J không được tăng; tăng nghĩa là code sai, hoặc hệ A_h quá xấu
    điều kiện so với kiểu số). Cảnh báo nếu chạm max_iter. J = δ·F/(n·H), bằng
    huber_objective khi λ = 0.

    Mỗi vòng: phần dư r → trọng số w = δ / max(|r|, δ) → giải H hệ theo batch, dạng hiệu chỉnh
    ŵ_h = w_h + A_h⁻¹ g_h, với A_h = Xtᵀ diag(w_h) Xt + 2λδ·Pen và
    g_h = Xtᵀ diag(w_h) r_h − 2λδ·Pen w_h (bằng A_h⁻¹ c_h, c_h = Xtᵀ diag(w_h) y_h).
    Với NLinear, v_h = ŵ_h − (aᵀŵ_h − 1)/(aᵀu_h) · u_h, u_h = A_h⁻¹a, a = [1, …, 1, 0].

    Pha độ chính xác (gram_dtype = float32, Xt float64): lập A_h, phần tốn 96·n·p² phép tính,
    bằng float32; r, g, J và phép giải bằng float64. Điểm dừng g = 0 tính bằng float64 nên
    nghiệm đạt độ chính xác float64; A_h lệch chỉ làm chậm hội tụ (cần κ(A_h)·ε₃₂ ≪ 1).
    """
    H, p = W0.shape
    W = W0.clone()
    Xg = Xt if gram_dtype is None else Xt.to(gram_dtype)

    # Phạt trong hệ của IRLS: trọng số w = δ·(1/max(|r|, δ)) tức hàm mục tiêu đã nhân δ, nên phạt
    # λ·wᵀ Pen w thành 2λδ·Pen trong A_h. Hàng và cột bias bằng 0.
    Pen = torch.zeros(p, p, dtype=Xt.dtype, device=Xt.device)
    if lam:
        Pen[:-1, :-1] = (torch.eye(p - 1, dtype=Xt.dtype, device=Xt.device) if pen is None
                         else torch.as_tensor(pen, dtype=Xt.dtype, device=Xt.device))
        Pen *= 2 * lam * delta

    def objective(W, per_h=False):
        """J (như trước, không đổi cách tính); per_h=True trả thêm J_h [H], Σ_h J_h = J (tới làm tròn)."""
        R = Y - Xt @ W.T
        WPW = (W @ Pen) * W
        J = huber_objective(R, delta) + WPW.sum() / (2 * Y.numel())
        if not per_h:
            return J
        a_ = R.abs()
        q = a_.clamp(max=delta)
        return J, ((q * (a_ - q / 2)).sum(0) + WPW.sum(-1) / 2) / Y.numel()

    # Fallback theo h (chỉ ở chế độ pha): nếu A_h lập bằng gram_dtype quá xấu điều kiện (κ(A_h)·ε ≳ 1),
    # bước h đó đi sai và J_h tăng. Vì J tách theo h, lập lại riêng các A_h đó bằng kiểu của Xt rồi
    # giải lại; các h khác giữ nguyên. Toán của IRLS không đổi (docs/BAO_CAO_TIEU_CHI_DUNG.md mục 8).
    fallback = gram_fallback and Xg.dtype != Xt.dtype
    if fallback:
        J_old, Jh_old = objective(W, per_h=True)
    else:
        J_old = objective(W)
    stall = 0                                       # số vòng liên tiếp J giảm < tol
    # J tính bằng kiểu của Xt nên lệch vài ε do làm tròn; float32 (ε ≈ 1,2e-7) cần ngưỡng rộng hơn.
    # J tăng trong ngưỡng này coi là đã chạm độ chính xác của kiểu số và dừng (bước dừng bên dưới).
    rise_tol = max(1e-10, 100 * torch.finfo(Xt.dtype).eps)

    if constrained:
        a = torch.ones(p, dtype=Xt.dtype, device=Xt.device)
        a[-1] = 0                                   # không ràng buộc bias

    def step(A, g, W):
        """W + A⁻¹g cho từng hàng; với NLinear thì chiếu lên ràng buộc aᵀw = 1."""
        if constrained:
            B = torch.stack([g, a.expand(len(W), -1)], dim=-1)     # [h, p, 2]
            sol = torch.linalg.solve(A, B)
            w_hat, u = W + sol[..., 0], sol[..., 1]
            coef = ((w_hat * a).sum(-1, keepdim=True) - 1) / (u * a).sum(-1, keepdim=True)
            return w_hat - coef * u
        return W + torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)

    n_fallback = 0
    for it in range(max_iter):
        # 1–2. phần dư và trọng số từ W hiện tại
        R = Y - Xt @ W.T
        w = delta / R.abs().clamp(min=delta)

        # 3. lập H hệ phương trình
        A = weight_gram(Xg, w.to(Xg.dtype), chunk).to(Xt.dtype) + Pen
        g = (w * R).T @ Xt - W @ Pen

        # 4. giải (và hiệu chỉnh theo ràng buộc nếu là NLinear)
        W_prev = W
        W = step(A, g, W)

        # 5. kiểm J không tăng, kiểm dừng
        if fallback:
            J_new, Jh_new = objective(W, per_h=True)
            bad = torch.nonzero(Jh_new > Jh_old * (1 + rise_tol)).flatten()
            if len(bad):
                A_bad = weight_gram(Xt, w[:, bad], chunk) + Pen
                W = W.clone()
                W[bad] = step(A_bad, g[bad], W_prev[bad])
                J_new, Jh_new = objective(W, per_h=True)
                n_fallback += len(bad)
            Jh_old = Jh_new
        else:
            J_new = objective(W)
        if callback is not None:
            callback(it, W, J_new)
        if J_new > J_old * (1 + rise_tol):
            raise RuntimeError(f"J tăng ở vòng {it}: {J_old.item():.6e} → {J_new.item():.6e}")
        if J_new > J_old:                           # tăng trong ngưỡng: chạm độ chính xác, dừng ngay
            break
        stall = stall + 1 if (J_old - J_new) / J_old < tol else 0
        if stall >= patience:
            break
        J_old = J_new
    else:
        warnings.warn(f"IRLS chưa hội tụ sau {max_iter} vòng (delta={delta})")

    irls.last_fallbacks = n_fallback
    return W, it + 1