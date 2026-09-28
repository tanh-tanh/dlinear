"""Sai số cho mô hình tuyến tính dự báo = X @ W.T + b.

Mọi hàm lấy trung bình toàn cục trên mọi phần tử (cách LTSF-Linear tính trên mảng nối),
không lấy trung bình theo batch.
"""
import numpy as np


def predict(W, b, X):
    return X @ W.T + b


def sse(W, b, X, Y):
    """Tổng (không phải trung bình) bình phương phần dư."""
    return np.sum((Y - predict(W, b, X)) ** 2)


def mse(W, b, X, Y):
    return np.mean((Y - predict(W, b, X)) ** 2)


def mae(W, b, X, Y):
    return np.mean(np.abs(Y - predict(W, b, X)))
