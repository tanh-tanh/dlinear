"""Kiểm W_eff, bias của ba mô hình torch: dự báo X W_effᵀ + b phải trùng forward của mô hình."""
import unittest

import numpy as np
import torch

from src.models import MODELS, effective_weights


class TestEffectiveWeights(unittest.TestCase):
    def test_matches_forward(self):
        L, H, C = 48, 12, 3
        torch.manual_seed(0)
        x = torch.randn(5, L, C, dtype=torch.float64)
        for name, cls in MODELS.items():
            with self.subTest(model=name):
                torch.manual_seed(1)
                m = cls(L, H).double()
                with torch.no_grad():                       # trọng số khác khởi tạo, để kiểm đủ các số hạng
                    for p in m.parameters():
                        p.add_(0.1 * torch.randn_like(p))
                W, b = effective_weights(m)
                y = m(x).detach().numpy()                   # [B, H, C]
                Xn = x.permute(0, 2, 1).reshape(-1, L).numpy()
                pred = (Xn @ W.T + b).reshape(5, C, H).transpose(0, 2, 1)
                np.testing.assert_allclose(pred, y, rtol=0, atol=1e-12)

    def test_nlinear_rows_sum_to_one(self):
        W, _ = effective_weights(MODELS["NLinear"](20, 4))
        np.testing.assert_allclose(W.sum(1), 1.0, atol=1e-12)

    def test_same_init_as_ltsf(self):
        """Linear và NLinear tạo đúng một nn.Linear(L, H): cùng seed thì cùng trọng số khởi tạo."""
        torch.manual_seed(2021)
        a = MODELS["Linear"](30, 5)
        torch.manual_seed(2021)
        b = MODELS["NLinear"](30, 5)
        torch.manual_seed(2021)
        ref = torch.nn.Linear(30, 5)
        for m in (a, b):
            self.assertTrue(torch.equal(m.Linear.weight, ref.weight))
            self.assertTrue(torch.equal(m.Linear.bias, ref.bias))


if __name__ == "__main__":
    unittest.main()
