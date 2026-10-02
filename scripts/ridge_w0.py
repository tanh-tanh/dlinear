"""Ridge co về W₀ thay vì 0 (docs/NHIEM_VU_3.md, B.2). Chỉ MSE, dạng đóng.

    min ‖Y − X Wᵀ − b‖² + λ·Σ_h (w_h − w0_h)ᵀ Pen (w_h − w0_h),   bias không phạt.

Đặt V = W − W₀: ridge trên (X, Y − X W₀ᵀ) với cùng Pen. Trên dữ liệu đã trừ trung bình (như fit_with_bias):
V = (A + λ·Pen)⁻¹ (C − A W₀ᵀ), với A = X̃ᵀX̃, C = X̃ᵀỸ của runner.Cell; W = V + W₀, b = ȳ − W x̄. Với W₀ = 0
đây đúng là biểu thức của runner nên phải trùng results/lambda_path.csv (kiểm tới 1e-10). λ = 0 cho OLS
với mọi W₀ (runner.Cell.closed_form).

W₀: zero (0), mean (mọi phần tử 1/L), last (1 ở lag cuối). Mô hình: Linear (Pen = I), DLinear (Pen = N⁻¹, k = 25).
Lưới λ = runner.MSE_LAMS (141 giá trị dương và 0); λ* tối thiểu val MSE.

    python scripts/ridge_w0.py

Kết quả: results/ridge_w0/path.csv (mọi λ), results/ridge_w0/summary.json (λ*, kiểm zero).
Mỗi ô ghi riêng results/ridge_w0/cells/{ds}_H{H}.csv nên chạy lại = chạy tiếp.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R                                   # noqa: E402
from bench_5060ti import save_json                   # noqa: E402

OUT = R.ROOT / "results" / "ridge_w0"
MODELS = ["Linear", "DLinear"]
W0S = ["zero", "mean", "last"]
CHECK = ["val_mse", "val_mae", "test_mse", "test_mae"]
TOL = 1e-10


def make_W0(name, H, L):
    if name == "zero":
        return np.zeros((H, L))
    if name == "mean":
        return np.full((H, L), 1.0 / L)
    W0 = np.zeros((H, L))
    W0[:, -1] = 1.0
    return W0


def solve(cell, model, lam, W0):
    """W [H, L + 1] (bias ở cột cuối)."""
    if lam == 0:
        return cell.closed_form(model, 0.0)
    V = np.linalg.solve(cell.A + lam * cell.pens[model], cell.C - cell.A @ W0.T).T
    W = V + W0
    b = cell.y_bar - W @ cell.x_bar
    return np.concatenate([W, b[:, None]], axis=1)


def evaluate(cell, W):
    """MSE, MAE val và test; đúng biểu thức của runner.Cell.evaluate nhưng bỏ phần train (không cần ở đây)."""
    out = {}
    for tag, name in (("va", "val"), ("te", "test")):
        Rs = cell.d[f"Y_{tag}"] - cell.d[f"Xt_{tag}"] @ W.T
        out[f"{name}_mse"] = float((Rs ** 2).mean())
        out[f"{name}_mae"] = float(np.abs(Rs).mean())
    return out


def run_cell(ds, H):
    f = OUT / "cells" / f"{ds}_H{H}.csv"
    if f.exists():
        with f.open(encoding="utf-8", newline="") as fh:
            return [{**r, "H": int(r["H"]), "lam": float(r["lam"]), **{k: float(r[k]) for k in CHECK}}
                    for r in csv.DictReader(fh)]
    cell = R.Cell(ds, H)
    rows = []
    for model in MODELS:
        for w0 in W0S:
            W0 = make_W0(w0, H, R.L)
            rows += [{"dataset": ds, "H": H, "model": model, "W0": w0, "lam": lam,
                      **evaluate(cell, solve(cell, model, lam, W0))} for lam in R.MSE_LAMS]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(f)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = {}
    with R.PATH_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["objective"] == "MSE":
                main_rows[(r["dataset"], int(r["H"]), r["model"], R.lam_key(r["lam"]))] = r
    path_rows, summary, max_dev, missing = [], [], 0.0, 0
    for ds in R.DATASETS:
        for H in R.HORIZONS:
            cell_rows = run_cell(ds, H)
            for model in MODELS:
                for w0 in W0S:
                    rows = [r for r in cell_rows if (r["model"], r["W0"]) == (model, w0)]
                    if w0 == "zero":
                        for r in rows:
                            ref = main_rows.get((ds, H, model, R.lam_key(r["lam"])))
                            if ref is None:
                                missing += 1
                            else:
                                max_dev = max(max_dev, *(abs(r[k] - float(ref[k])) for k in CHECK))
                    best = min(rows, key=lambda r: r["val_mse"])
                    summary.append({**best, "lam_star": best["lam"],
                                    "lam_at_edge": best["lam"] in (R.MSE_LAMS[0], R.MSE_LAMS[-1])})
                    path_rows += rows
                    print(f"{ds} H={H:>3} {model:<7} W0={w0:<4}: λ* = {best['lam']:>10.4g} val MSE {best['val_mse']:.6f} "
                          f"test MSE {best['test_mse']:.6f}", flush=True)
    with (OUT / "path.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(path_rows[0]))
        w.writeheader()
        w.writerows(path_rows)
    ok = missing == 0 and max_dev <= TOL
    print(f"kiểm zero so với lambda_path.csv: lệch lớn nhất {max_dev:.2e}, thiếu {missing} dòng → {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    save_json(OUT / "summary.json", {"zero_check": {"max_abs_dev": max_dev, "missing": missing, "tol": TOL,
                                                    "pass": ok, "metrics": CHECK},
                                     "selection": "val_mse", "lams": R.MSE_LAMS, "cells": summary})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
