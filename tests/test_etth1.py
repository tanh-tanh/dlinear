"""Kiểm hồi quy: src/ phải cho lại đúng các số đã ghi trong notebook trên ETTh1, L = 336, H = 96."""
import os
import unittest

import numpy as np

from src.data import load_etth1, make_dataset, split_borders
from src.metrics import mse, sse
from src.solvers import fit_with_bias, nlinear_constrained, ols

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ETTh1.csv")
CKPT = os.path.join(ROOT, "checkpoints", "dlinear_sgd_ETTh1_L336_H96.pt")
L, H = 336, 96


class TestSplits(unittest.TestCase):
    def test_borders(self):
        self.assertEqual(split_borders("ETTh1"), (8640, 11520, 14400))
        self.assertEqual(split_borders("ETTh2"), (8640, 11520, 14400))
        self.assertEqual(split_borders("ETTm1"), (34560, 46080, 57600))


@unittest.skipUnless(os.path.isfile(DATA), "thiếu data/ETTh1.csv")
class TestETTh1ClosedForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, _, test = load_etth1(DATA, L)
        cls.X_tr, cls.Y_tr = make_dataset(train, L, H)
        cls.X_te, cls.Y_te = make_dataset(test, L, H)
        cls.W_ols, cls.b_ols = fit_with_bias(ols, cls.X_tr, cls.Y_tr)
        cls.V, cls.b_V = fit_with_bias(nlinear_constrained, cls.X_tr, cls.Y_tr)

    def test_shapes(self):
        self.assertEqual(self.X_tr.shape, (57463, L))
        self.assertEqual(self.X_te.shape, (19495, L))      # 2785 cửa sổ × 7 kênh

    def test_ols_test_mse(self):                           # Nlinear.ipynb, 01 mục 1
        self.assertAlmostEqual(mse(self.W_ols, self.b_ols, self.X_te, self.Y_te), 0.37023527067813916, places=10)

    def test_nlinear_test_mse(self):                       # Nlinear.ipynb, cell 12
        self.assertAlmostEqual(mse(self.V, self.b_V, self.X_te, self.Y_te), 0.3701986328402697, places=10)

    def test_nlinear_sse_gap(self):                        # Nlinear.ipynb, cell 10
        gap = sse(self.V, self.b_V, self.X_tr, self.Y_tr) - sse(self.W_ols, self.b_ols, self.X_tr, self.Y_tr)
        self.assertAlmostEqual(gap / 17002.829247226007, 1.0, places=9)


@unittest.skipUnless(os.path.isfile(DATA) and os.path.isfile(CKPT), "thiếu dữ liệu hoặc checkpoint SGD")
class TestSGDCheckpoint(unittest.TestCase):
    def test_same_test_windows_and_mse(self):
        """Checkpoint SGD đánh giá trên đúng 2785 cửa sổ test mà OLS dùng, trung bình toàn cục."""
        import torch
        from torch.utils.data import DataLoader
        from src.models import DLinear
        from src.sgd import ETTDataset, evaluate

        _, _, test = load_etth1(DATA, L)
        test_ds = ETTDataset(test, L, H)
        self.assertEqual(len(test_ds) * 7, 19495)
        model = DLinear(L, H)
        model.load_state_dict(torch.load(CKPT))
        loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        self.assertAlmostEqual(evaluate(model, loader, "mse"), 0.3766, places=4)   # dlinearv2.ipynb, cell 14
        self.assertAlmostEqual(evaluate(model, loader, "mae"), 0.3996, places=4)


if __name__ == "__main__":
    unittest.main()
