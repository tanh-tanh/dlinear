"""Gradient descent toàn batch, lr cố định, trên MSE của Linear (docs/NHIEM_VU_3.md, B.3).

Tách tác dụng của dừng sớm khỏi nhiễu mini-batch: không có Adam, không có batch, chỉ GD trên
J(W, b) = ‖Y − X Wᵀ − b‖² / (n·H), xuất phát từ khởi tạo nn.Linear mặc định của LTSF-Linear (seed 2021).

Mọi thứ tính qua ma trận Gram của Xt = [X, 1] (float64, GPU): bước GD là Wt ← Wt − η·(2/(nH))(Wt G − Cᵀ),
MSE val/test tính đúng bằng ‖Y‖² − 2 tr(Wt C) + tr(Wt G Wtᵀ). η = 1/λ_max(2G/(nH)) (bước an toàn).
Ghi MSE val/test theo số bước (lưới log), chọn số bước theo val MSE.

    python scripts/gd_early_stop.py --H 720 --steps 1000000

Kết quả: results/gd_early_stop/Linear_H{H}.json.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_5060ti import load_cell, save_json                  # noqa: E402  (tắt TF32 khi import)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gd_early_stop"
L, SEED = 336, 2021
DEV = "cuda"


def gram(Xt, Y):
    Xt, Y = torch.as_tensor(Xt, device=DEV), torch.as_tensor(Y, device=DEV)
    return Xt.T @ Xt, Xt.T @ Y, float((Y ** 2).sum()), Y.numel()


def mse(Wt, G, C, yy, n):
    return float((yy - 2 * (Wt * C.T).sum() + ((Wt @ G) * Wt).sum()) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-log", type=int, default=400)
    args = ap.parse_args()
    H = args.H
    d = load_cell("ETTh2", H)
    G, C, yy, n = gram(d["Xt"], d["Y"])
    Gv, Cv, yyv, nv = gram(d["Xt_va"], d["Y_va"])
    Gt, Ct, yyt, nt = gram(d["Xt_te"], d["Y_te"])
    torch.manual_seed(SEED)
    lin = torch.nn.Linear(L, H)
    Wt = torch.cat([lin.weight.detach().double(), lin.bias.detach().double()[:, None]], 1).to(DEV)
    Hs = 2 * G / n
    lam_max = float(torch.linalg.eigvalsh(Hs)[-1])
    eta = 1.0 / lam_max
    log_at = set(np.unique(np.r_[0, np.geomspace(1, args.steps, args.n_log).astype(int)]).tolist())
    hist = []
    t0 = time.perf_counter()
    for s in range(args.steps + 1):
        if s in log_at:
            W, b = Wt[:, :-1], Wt[:, -1]
            hist.append({"step": s, "train_mse": mse(Wt, G, C, yy, n), "val_mse": mse(Wt, Gv, Cv, yyv, nv),
                         "test_mse": mse(Wt, Gt, Ct, yyt, nt), "row_sum_median": float(W.sum(1).median()),
                         "bias_norm": float(b.norm())})
        if s < args.steps:
            Wt = Wt - eta * (2.0 / n) * (Wt @ G - C.T)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    best = min(hist, key=lambda r: r["val_mse"])
    print(f"H={H}: η = {eta:.3e}, {args.steps} bước, {sec:.0f} s; bước tốt nhất theo val {best['step']}: "
          f"val {best['val_mse']:.6f} test {best['test_mse']:.6f}; cuối: val {hist[-1]['val_mse']:.6f} "
          f"test {hist[-1]['test_mse']:.6f}")
    save_json(OUT / f"Linear_H{H}.json", {"dataset": "ETTh2", "H": H, "model": "Linear", "seed": SEED,
                                          "init": "nn.Linear mặc định", "eta": eta, "lam_max_hessian": lam_max,
                                          "steps": args.steps, "seconds": sec, "best": best,
                                          "best_at_last_step": best["step"] == args.steps, "history": hist})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
