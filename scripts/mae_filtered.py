"""MAE λ = 0 trên train đã lọc cửa sổ hằng, ETTh2 (docs/NHIEM_VU_4.md, Phần C).

Train đã lọc: đúng hàm constant_mask của scripts/constant_windows.py (NHIEM_VU_3 C.3); val, test giữ nguyên.
IRLS như grid: chế độ pha, chunk theo runner.Cell, tol = runner.STOP["MAE"]["tol_lam0"] (3e-11), patience 1,
khởi tạo từ nghiệm MSE λ = 0 trên cùng train đã lọc (runner.Cell.closed_form sau set_train).

    python scripts/mae_filtered.py                        # Linear, 4 H
    python scripts/mae_filtered.py --model NLinear --H 96 192 336

Kết quả: results/mae_filtered/{model}_H{H}.json (mỗi ô một file; chạy lại bỏ qua ô đã có).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R                                   # noqa: E402  (tắt TF32, đặt expandable_segments)
from bench_5060ti import save_json                   # noqa: E402
from constant_windows import constant_mask           # noqa: E402
from src.solvers import irls                         # noqa: E402

OUT = R.ROOT / "results" / "mae_filtered"
DS = "ETTh2"


def run(model, H):
    f = OUT / f"{model}_H{H}.json"
    if f.exists():
        print(f"  bỏ qua {model} H={H} (đã có)")
        return
    cell = R.Cell(DS, H)
    d = cell.d
    mask = constant_mask(d["X"], d["Y"])
    cell.set_train(d["X"][~mask], d["Y"][~mask])
    W_mse = cell.closed_form(model, 0.0)
    mse_row = cell.evaluate(W_mse, model, "MSE", 0.0)
    tol, pat = R.STOP["MAE"]["tol_lam0"], R.STOP["MAE"]["patience"]
    W, n, conv, sec = cell.run_irls(model, W_mse, R.DELTAS["MAE"], 0.0, tol, pat)
    mae_row = cell.evaluate(W, model, "MAE", 0.0)
    res = {"dataset": DS, "H": H, "model": model, "lam": 0.0, "objective": "MAE", "delta": R.DELTAS["MAE"],
           "filtered_train": True, "n_train_rows": int(cell.n), "n_dropped": int(mask.sum()),
           "init": "MSE λ = 0 trên train đã lọc", "tol": tol, "patience": pat, "chunk": cell.chunk,
           "gram_dtype": "float32", "n_iter": n, "converged": conv, "n_fallback": irls.last_fallbacks,
           "seconds": sec, "mae": mae_row, "mse_lam0": mse_row}
    save_json(f, res)
    print(f"  {model} H={H}: {n} vòng, {sec:.0f} s, fallback {irls.last_fallbacks}; test MSE "
          f"MSE-λ0 {mse_row['test_mse']:.6f} → MAE-λ0 {mae_row['test_mse']:.6f}", flush=True)
    cell.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["Linear"], choices=["Linear", "NLinear"])
    ap.add_argument("--H", nargs="+", type=int, default=R.HORIZONS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.model:
        for H in args.H:
            run(model, H)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
