"""Huấn luyện baseline bằng SGD (Adam), theo cấu hình DLinear/ETTh1 của LTSF-Linear.

Chuyển từ dlinearv2.ipynb, mục 2–3. evaluate() lấy trung bình toàn cục trên mọi phần tử
(đã sửa lỗi trung bình theo batch ở 01_three_predictions.ipynb, Phụ lục A).
"""
import torch
from torch.utils.data import Dataset


class ETTDataset(Dataset):
    """Cửa sổ trượt trên mảng đã chuẩn hóa [N, C]: x [L, C], y [H, C]."""

    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, i):
        x = self.data[i: i + self.seq_len]
        y = self.data[i + self.seq_len: i + self.seq_len + self.pred_len]
        return x, y


def evaluate(model, loader, reduction="mse"):
    """MSE hoặc MAE trung bình trên toàn bộ phần tử của loader."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            d = model(x) - y
            total += (d ** 2).sum().item() if reduction == "mse" else d.abs().sum().item()
            count += d.numel()
    model.train()
    return total / count


def train_one_epoch(model, loader, criterion, optimizer):
    train_loss = 0.0
    for x, y in loader:
        loss = criterion(model(x), y)
        train_loss += loss.item() / len(loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss


def lr_factor(epoch_idx, lradj="type1"):
    """Hệ số lr cho epoch thứ epoch_idx + 1 (epoch_idx đếm từ 0).

    'type1': đúng lradj='type1' của LTSF-Linear. Repo gọi adjust_learning_rate(epoch + 1) sau mỗi
    epoch với lr = lr0 · 0.5^(epoch − 1), nên epoch 1 và 2 cùng chạy lr0, epoch 3 chạy lr0/2, ...
    'step': StepLR(step_size=1, gamma=0.5) của notebook gốc, giảm một nửa ngay sau epoch 1
    (checkpoints/dlinear_sgd_ETTh1_L336_H96.pt được huấn luyện theo lịch này).
    """
    if lradj == "type1":
        return 0.5 ** max(epoch_idx - 1, 0)
    if lradj == "step":
        return 0.5 ** epoch_idx
    raise ValueError(f"lradj không hỗ trợ: {lradj}")


def train_sgd(model, train_loader, val_loader, ckpt_path, lr=5e-3, epochs=10, patience=3, lradj="type1"):
    """Adam, lịch lr theo lradj, dừng sớm theo MSE validation, nạp lại trọng số tốt nhất."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, lradj))
    best_val, counter, history = float("inf"), 0, []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, "mse")
        scheduler.step()
        history.append((epoch + 1, train_loss, val_loss, optimizer.param_groups[0]["lr"]))
        if val_loss < best_val:
            best_val, counter = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter += 1
            if counter >= patience:
                break
    model.load_state_dict(torch.load(ckpt_path))
    return history
