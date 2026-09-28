"""Pipeline dữ liệu ETT theo đúng cách chia của LTSF-Linear (Dataset_ETT_hour / Dataset_ETT_minute).

Chuyển từ Nlinear.ipynb / 01_three_predictions.ipynb. Mọi hàm ở đây chỉ dùng numpy.
"""
import numpy as np
import pandas as pd

# Mốc chia theo số hàng của dữ liệu giờ: 12 tháng train, 4 tháng val, 4 tháng test.
# ETTm1/ETTm2 lấy mẫu 15 phút nên mọi mốc nhân 4.
_HOUR = 30 * 24
STEPS_PER_HOUR = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 4, "ETTm2": 4}


def split_borders(name):
    """Trả (train_end, val_end, test_end) theo số hàng, đúng như LTSF-Linear."""
    m = STEPS_PER_HOUR[name]
    return 12 * _HOUR * m, 16 * _HOUR * m, 20 * _HOUR * m


def load_ett(path, L, name="ETTh1"):
    """Đọc CSV, chuẩn hóa từng kênh bằng mean/std của train, chia train/val/test.

    val và test lùi L hàng để cửa sổ đầu tiên của chúng có đủ L giá trị quá khứ.
    std dùng ddof=0, giống StandardScaler của sklearn mà LTSF-Linear dùng.
    """
    train_end, val_end, test_end = split_borders(name)
    data = pd.read_csv(path).iloc[:, 1:].values          # bỏ cột date -> [N, 7]
    mean = np.mean(data[:train_end], axis=0)
    std = np.std(data[:train_end], axis=0)
    data = (data - mean) / std
    train = data[0: train_end]
    val = data[train_end - L: val_end]
    test = data[val_end - L: test_end]
    return train, val, test


def load_etth1(path, L):
    """Giữ tên cũ trong notebook: ETTh1, chia 8640/2880/2880."""
    return load_ett(path, L, "ETTh1")


def make_windows(series, L, H):
    """Cắt chuỗi 1 chiều thành các cặp (x: L bước, y: H bước kế tiếp), bước trượt 1."""
    X, Y = [], []
    for i in range(len(series) - L - H + 1):
        X.append(series[i: i + L])
        Y.append(series[i + L: i + L + H])
    return np.array(X), np.array(Y)


def make_dataset(split, L, H):
    """Gộp 7 kênh thành một tập mẫu (dùng chung trọng số cho mọi kênh, individual=False).

    Hàng được xếp theo kênh: toàn bộ cửa sổ của kênh 0, rồi kênh 1, ...
    """
    Xs, Ys = [], []
    for c in range(split.shape[1]):
        X_c, Y_c = make_windows(split[:, c], L, H)
        Xs.append(X_c)
        Ys.append(Y_c)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)
