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


def train_sgd(model, train_loader, val_loader, ckpt_path, lr=5e-3, epochs=10, patience=3):
    """Adam, lr giảm một nửa sau mỗi epoch, dừng sớm theo MSE validation, nạp lại trọng số tốt nhất.

    Lưu ý: LTSF-Linear (lradj='type1') giữ lr gốc cho cả epoch 1 và 2 rồi mới giảm một nửa;
    StepLR ở đây giảm ngay sau epoch 1, giống notebook gốc.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
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
