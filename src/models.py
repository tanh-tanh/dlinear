"""DLinear bằng PyTorch (bản tự viết, nhánh dùng chung trọng số), dùng cho baseline SGD.

Chuyển từ dlinearv2.ipynb, mục 1. Cấu trúc giống models/DLinear.py của LTSF-Linear với individual=False.
"""
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):          # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, 1)

    def forward(self, x):          # x: [B, L, C]
        trend = self.moving_avg(x)
        return x - trend, trend    # seasonal, trend


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size=25):
        super().__init__()
        self.series_decomp = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):          # x: [B, L, C] -> [B, H, C]
        seasonal, trend = self.series_decomp(x)
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend = self.Linear_Trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)


class Linear(nn.Module):
    """models/Linear.py của LTSF-Linear, individual=False: một nn.Linear(L, H) dùng chung cho mọi kênh."""

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):          # x: [B, L, C] -> [B, H, C]
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class NLinear(nn.Module):
    """models/NLinear.py của LTSF-Linear, individual=False: trừ x_L, nn.Linear(L, H) đủ L cột, cộng lại x_L.

    Cột cuối của trọng số nhân với x_L − x_L = 0 nên không có gradient, nhưng vẫn giữ để khởi tạo
    tiêu thụ cùng số ngẫu nhiên như repo gốc.
    """

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):          # x: [B, L, C] -> [B, H, C]
        seq_last = x[:, -1:, :].detach()
        x = self.Linear((x - seq_last).permute(0, 2, 1)).permute(0, 2, 1)
        return x + seq_last


MODELS = {"Linear": Linear, "DLinear": DLinear, "NLinear": NLinear}


def effective_weights(model):
    """(W_eff [H, L], b [H]) float64 numpy sao cho dự báo của mỗi kênh = W_eff x + b.

    DLinear: W_eff = W_t P + W_s (I − P), b = b_t + b_s. NLinear: W_eff = W, trừ cột cuối là
    1 − tổng L − 1 cột đầu (nlinear_effective).
    """
    import numpy as np
    from src.operators import build_P, dlinear_effective, nlinear_effective

    def wb(lin):
        return lin.weight.detach().double().cpu().numpy(), lin.bias.detach().double().cpu().numpy()

    if isinstance(model, DLinear):
        (W_s, b_s), (W_t, b_t) = wb(model.Linear_Seasonal), wb(model.Linear_Trend)
        P = build_P(W_s.shape[1], model.series_decomp.moving_avg.kernel_size)
        return dlinear_effective(W_t, W_s, P), b_t + b_s
    W, b = wb(model.Linear)
    if isinstance(model, NLinear):
        return nlinear_effective(W[:, :-1]), b
    return W, b
