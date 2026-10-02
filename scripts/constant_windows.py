"""Độ nhạy với đoạn hằng của ETTh2 (docs/NHIEM_VU_3.md, C.3).

Cửa sổ hằng: cửa sổ (một kênh, sau chuẩn hóa) có max − min của x hoặc của y nhỏ hơn 1e-12.
Đếm cửa sổ hằng trong train/val/test theo H và kênh; rồi giải MSE dạng đóng (Linear, DLinear, NLinear,
lưới runner.MSE_LAMS) với train bỏ các cửa sổ hằng, val và test giữ nguyên, λ* theo val MSE.
"Trước" lấy từ results/lambda_path.csv (cùng lưới, train đủ).

    python scripts/constant_windows.py

Kết quả: results/constant_windows/summary.json, path.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R                                   # noqa: E402
from bench_5060ti import save_json                   # noqa: E402

OUT = R.ROOT / "results" / "constant_windows"
DS = "ETTh2"
EPS = 1e-12
N_CH = 7


def constant_mask(X, Y):
    return (np.ptp(X, axis=1) < EPS) | (np.ptp(Y, axis=1) < EPS)


def per_channel(mask):
    """Số cửa sổ hằng theo kênh: hàng của make_dataset xếp theo kênh, mỗi kênh cùng số cửa sổ."""
    return mask.reshape(N_CH, -1).sum(1).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {}
    with R.PATH_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == DS and r["objective"] == "MSE":
                before.setdefault((int(r["H"]), r["model"]), []).append(r)
    counts, cells, path_rows = [], [], []
    for H in R.HORIZONS:
        cell = R.Cell(DS, H)
        d = cell.d
        splits = {"train": (d["X"], d["Y"]), "val": (d["Xt_va"][:, :-1], d["Y_va"]),
                  "test": (d["Xt_te"][:, :-1], d["Y_te"])}
        masks = {}
        for split, (X, Y) in splits.items():
            m = masks[split] = constant_mask(X, Y)
            counts.append({"H": H, "split": split, "n_windows": int(len(m)), "n_constant": int(m.sum()),
                           "per_channel": per_channel(m), "n_constant_x": int((np.ptp(X, 1) < EPS).sum()),
                           "n_constant_y": int((np.ptp(Y, 1) < EPS).sum())})
            print(f"H={H} {split}: {int(m.sum())}/{len(m)} cửa sổ hằng, theo kênh {per_channel(m)}")
        keep = ~masks["train"]
        cell.set_train(d["X"][keep], d["Y"][keep])
        for model in R.MODELS:
            rows = []
            for lam in R.MSE_LAMS:
                m = cell.evaluate(cell.closed_form(model, lam), model, "MSE", lam)
                rows.append({"H": H, "model": model, "lam": lam, **{k: m[k] for k in
                             ("train_obj", "val_mse", "val_mae", "test_mse", "test_mae")}})
            path_rows += rows
            after = min(rows, key=lambda r: r["val_mse"])
            b = min(before[(H, model)], key=lambda r: float(r["val_mse"]))
            cells.append({"H": H, "model": model, "n_train_dropped": int(masks["train"].sum()),
                          "before": {"lam_star": float(b["lam"]), "val_mse": float(b["val_mse"]),
                                     "test_mse": float(b["test_mse"]), "test_mae": float(b["test_mae"]),
                                     "n_lams": len(before[(H, model)])},
                          "after": {"lam_star": after["lam"], "val_mse": after["val_mse"],
                                    "test_mse": after["test_mse"], "test_mae": after["test_mae"],
                                    "n_lams": len(rows)}})
            print(f"  {model:<7}: λ* {float(b['lam']):g} → {after['lam']:g}, test MSE {float(b['test_mse']):.6f} → "
                  f"{after['test_mse']:.6f}", flush=True)
        del cell
    with (OUT / "path.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(path_rows[0]))
        w.writeheader()
        w.writerows(path_rows)
    save_json(OUT / "summary.json", {"dataset": DS, "eps": EPS, "selection": "val_mse",
                                     "definition": "max − min của x hoặc của y < eps, theo kênh, sau chuẩn hóa",
                                     "counts": counts, "cells": cells})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
