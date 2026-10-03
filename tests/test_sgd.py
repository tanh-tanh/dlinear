"""Kiểm lịch learning rate của train_sgd so với lradj='type1' của LTSF-Linear."""
import unittest

import torch

import numpy as np

from src.sgd import MaskedETTDataset, masked_mse, lr_factor


def ltsf_type1_lrs(lr0, epochs):
    """Mô phỏng exp_main.py: epoch 1 chạy lr0, sau mỗi epoch gọi adjust_learning_rate(epoch + 1)."""
    lrs, lr = [], lr0
    for epoch in range(epochs):
        lrs.append(lr)
        lr = lr0 * (0.5 ** ((epoch + 1 - 1) // 1))     # utils/tools.py, type1
    return lrs


def scheduler_lrs(lradj, lr0, epochs):
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_factor(e, lradj))
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return lrs


class TestLRSchedule(unittest.TestCase):
    def test_type1_matches_ltsf(self):
        lrs = scheduler_lrs("type1", 5e-3, 10)
        for a, b in zip(lrs, ltsf_type1_lrs(5e-3, 10)):
            self.assertAlmostEqual(a, b, places=15)
        self.assertEqual(lrs[:3], [5e-3, 5e-3, 2.5e-3])

    def test_step_is_old_notebook_schedule(self):
        self.assertEqual(scheduler_lrs("step", 5e-3, 3), [5e-3, 2.5e-3, 1.25e-3])

    def test_unknown(self):
        with self.assertRaises(ValueError):
            lr_factor(0, "type9")


class TestMaskedLoss(unittest.TestCase):
    def test_all_kept_equals_mse(self):
        torch.manual_seed(0)
        p, y = torch.randn(4, 6, 3), torch.randn(4, 6, 3)
        keep = torch.ones(4, 3, dtype=torch.bool)
        self.assertAlmostEqual(masked_mse(p, y, keep).item(), torch.nn.MSELoss()(p, y).item(), places=6)

    def test_dropped_pairs_ignored(self):
        """Bỏ cặp (cửa sổ, kênh) = MSE trên các hàng còn lại của ma trận make_dataset."""
        torch.manual_seed(1)
        p, y = torch.randn(4, 6, 3, dtype=torch.float64), torch.randn(4, 6, 3, dtype=torch.float64)
        keep = torch.tensor([[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype=torch.bool)
        rows = [((p[b, :, c] - y[b, :, c]) ** 2).mean() for b in range(4) for c in range(3) if keep[b, c]]
        self.assertAlmostEqual(masked_mse(p, y, keep).item(), torch.stack(rows).mean().item(), places=12)
        y2 = y.clone()
        y2[~keep.unsqueeze(1).expand_as(y)] = 1e6           # giá trị ở cặp bị bỏ không ảnh hưởng
        self.assertAlmostEqual(masked_mse(p, y2, keep).item(), masked_mse(p, y, keep).item(), places=12)

    def test_dataset_returns_keep(self):
        data = np.arange(40, dtype=float).reshape(20, 2)
        keep = np.zeros((20 - 5 - 3 + 1, 2), dtype=bool)
        keep[3, 1] = True
        ds = MaskedETTDataset(data, 5, 3, keep)
        x, y, k = ds[3]
        self.assertEqual(tuple(x.shape), (5, 2))
        self.assertEqual(k.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
