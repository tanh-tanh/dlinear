"""Chọn λ* từ results/lambda_path.csv, sinh results/results.csv.

Mỗi (dataset, H, mô hình, mục tiêu) cho hai mức λ: λ = 0 và λ*. λ* có hai cách chọn (cột selection):
    val_objective  λ* tối thiểu metric val của chính mục tiêu (MSE → val_mse, MAE → val_mae, Huber → val_huber)
    val_mse        λ* tối thiểu val_mse với mọi mục tiêu
Mỗi cách chọn cho một bảng đủ 216 ô (3 mô hình × 3 mục tiêu × 2 mức λ × 3 dataset × 4 horizon), nên
results.csv có 432 dòng khi grid chạy xong. λ* có thể bằng 0. Test không dùng để chọn gì.

    python scripts/select_lambda.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_CSV = ROOT / "results" / "lambda_path.csv"
OUT_CSV = ROOT / "results" / "results.csv"
OWN = {"MSE": "val_mse", "MAE": "val_mae", "Huber": "val_huber"}
KEEP = ["dataset", "H", "model", "objective", "lam", "lam_convention", "delta", "k", "train_obj",
        "val_mse", "val_mae", "val_huber", "test_mse", "test_mae", "n_iter", "converged", "init",
        "derived_from", "tol", "patience", "gpu", "git_commit"]


def main():
    with PATH_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], int(r["H"]), r["model"], r["objective"]), []).append(r)
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        path = sorted(groups[key], key=lambda r: float(r["lam"]))
        lams = [float(r["lam"]) for r in path]
        zero = next((r for r in path if float(r["lam"]) == 0), None)
        for sel in ("val_objective", "val_mse"):
            metric = OWN[key[3]] if sel == "val_objective" else "val_mse"
            best = min(path, key=lambda r: float(r[metric]))
            edge = float(best["lam"]) in (lams[0], lams[-1])
            for level, r in (("lam0", zero), ("lam_star", best)):
                if r is None:
                    continue
                out.append({**{c: r[c] for c in KEEP}, "selection": sel, "selection_metric": metric,
                            "level": level, "lam_at_edge": edge if level == "lam_star" else "",
                            "n_lams": len(path), "lam_min": lams[0], "lam_max": lams[-1]})
    cols = ["selection", "selection_metric", "level", "lam_at_edge", "n_lams", "lam_min", "lam_max"] + KEEP
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"{len(groups)} đường λ → {len(out)} dòng ({len(out) // 2} ô mỗi cách chọn) → {OUT_CSV.relative_to(ROOT)}")
    for r in out:
        if r["level"] == "lam_star":
            print(f"  {r['dataset']} H={r['H']:>3} {r['model']:<7} {r['objective']:<5} {r['selection']:<13} "
                  f"λ* = {float(r['lam']):>10.4g}{' (đầu mút)' if r['lam_at_edge'] else '':<10} "
                  f"val {float(r[r['selection_metric']]):.6f}  test MSE {float(r['test_mse']):.6f} "
                  f"MAE {float(r['test_mae']):.6f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
