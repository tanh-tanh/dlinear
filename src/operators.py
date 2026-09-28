"""Toán tử tuyến tính của DLinear và NLinear.

DLinear: trend = P x, seasonal = (I − P) x, với P là moving_avg của LTSF-Linear
(đệm (k−1)/2 bản sao giá trị đầu và cuối, AvgPool1d bước 1).
NLinear: trừ giá trị cuối x_L khỏi đầu vào, cộng lại vào đầu ra.

Chuyển từ dlinearv2.ipynb (mục 4, 6) và 01_three_predictions.ipynb (mục 2, 6).
"""
import numpy as np


def build_P(L, k):
    """Ma trận trung bình trượt P [L, L], khớp moving_avg của LTSF-Linear tới sai số máy."""
    assert k % 2 == 1, "k chẵn thì output dài L-1, P không vuông"
    pad = (k - 1) // 2
    P = np.zeros((L, L))
    for t in range(L):
        for m in range(k):
            j = min(max(t + m - pad, 0), L - 1)
            P[t, j] += 1.0 / k
    return P


def build_N(P):
    """N = PᵀP + (I−P)ᵀ(I−P). Weight decay của DLinear = ridge trên X với phạt λ·N⁻¹."""
    I = np.eye(P.shape[0])
    return P.T @ P + (I - P).T @ (I - P)


def make_Z(X, P):
    """Ma trận thiết kế của DLinear: ghép ngang [X Pᵀ, X (I−P)ᵀ] -> [n, 2L].

    Không cộng hai khối: tổng của chúng chỉ cho lại X.
    """
    I = np.eye(P.shape[0])
    return np.concatenate([X @ P.T, X @ (I - P).T], axis=1)


def dlinear_effective(W_t, W_s, P):
    """Ánh xạ tương đương của DLinear trên x: W_eff = W_t P + W_s (I − P)."""
    return W_t @ P + W_s @ (np.eye(P.shape[0]) - P)


def nlinear_transform(X, Y):
    """Trừ giá trị cuối: X_t = (X − x_L)[:, :-1] (bỏ cột cuối luôn bằng 0), Y_t = Y − x_L.

    Dự báo gốc = dự báo trên (X_t) + x_L.
    """
    x_last = X[:, -1:]
    return (X - x_last)[:, :-1], Y - x_last, x_last


def nlinear_effective(W):
    """Từ trọng số W [H, L−1] học trên chuỗi đã trừ giá trị cuối, trả W_eff [H, L] trên x.

    W_eff = [W, 1_H − W 1] nên mỗi hàng của W_eff cộng bằng 1.
    """
    return np.concatenate([W, (1.0 - W.sum(1))[:, None]], axis=1)
