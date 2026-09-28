"""Solver dạng đóng cho hàm mục tiêu MSE.

Quy ước: X [n, L], Y [n, H], trả W [H, L] sao cho dự báo = X @ W.T (+ b).
λ phạt trên TỔNG bình phương lỗi: ‖Y − X Wᵀ‖² + λ‖W‖². Đổi sang weight_decay của PyTorch
(loss là MSE trung bình trên n·H phần tử): weight_decay ≈ 2λ / (n·H).

Chuyển từ Nlinear.ipynb, dlinearv2.ipynb và 01_three_predictions.ipynb.
Chưa có: IRLS cho MAE và Huber (việc của tuần 2).
"""
import numpy as np


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
