"""Kiểm lịch learning rate của train_sgd so với lradj='type1' của LTSF-Linear."""
import unittest

import torch

from src.sgd import lr_factor


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


if __name__ == "__main__":
    unittest.main()
