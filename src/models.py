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
