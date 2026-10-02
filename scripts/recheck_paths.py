"""Chạy lại bằng code hiện tại (có fallback float64 theo h) một số đường λ đã chạy trước khi có fallback,
rồi so với results/lambda_path.csv (docs/NHIEM_VU_3.md, C.1). Không đụng file chính.

    python scripts/recheck_paths.py            # chạy (chạy lại = chạy tiếp) rồi so
    python scripts/recheck_paths.py --compare  # chỉ so

Ghi results/recheck/lambda_path_recheck.csv, W ở results/recheck/weights/, bảng so ở results/recheck/compare.json.
Khởi tạo λ = 0 lấy nghiệm MSE λ = 0 của file chính (chép sang thư mục recheck, chỉ đọc). Với DLinear,
λ = 0 ≡ Linear: chạy lại Linear λ = 0 rồi chép như runner, để λ = 0 cũng được kiểm.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R                                   # noqa: E402  (tắt TF32, đặt expandable_segments)
from bench_5060ti import save_json                   # noqa: E402
from src.solvers import irls                         # noqa: E402

PATHS = [("ETTh2", 336, "Linear"), ("ETTh2", 336, "NLinear"), ("ETTh1", 720, "DLinear")]
OBJ = "MAE"
MAIN_CSV, MAIN_W = R.PATH_CSV, R.WEIGHTS
OUT = R.ROOT / "results" / "recheck"
METRICS = ["val_mse", "val_mae", "val_huber", "test_mse", "test_mae"]
TOL = 1e-5


def lam0_fresh(cell, model, done, writer):
    """Chạy irls ở λ = 0 (khởi tạo từ MSE λ = 0), ghi như một dòng của runner."""
    key = (cell.ds, cell.H, model, OBJ, R.lam_key(0))
    if key in done:
        return
    W0 = R.np.load(R.weight_path(cell.ds, cell.H, model, "MSE", 0))
    t, p = R.STOP[OBJ]["tol_lam0"], R.STOP[OBJ]["patience"]
    W, n, conv, sec = cell.run_irls(model, W0, R.DELTAS[OBJ], 0.0, t, p)
    row = R.base_row(cell, model, OBJ, 0.0)
    row.update(init="ols", n_iter=n, converged=conv, seconds=sec, tol=t, patience=p, chunk=cell.chunk,
               gram_dtype="float32", n_fallback=irls.last_fallbacks, **cell.evaluate(W, model, OBJ, 0.0))
    writer.write(row, W)
    done[key] = row
    print(f"  {cell.ds} H={cell.H} {model} {OBJ} λ = 0: {n} vòng, {sec:.1f} s, fallback {irls.last_fallbacks}")


def run():
    R.PATH_CSV, R.WEIGHTS = OUT / "lambda_path_recheck.csv", OUT / "weights"
    R.WEIGHTS.mkdir(parents=True, exist_ok=True)
    done = R.read_done()
    writer = R.Writer()
    for ds, H, model in PATHS:
        need = ["Linear", model] if model == "DLinear" else [model]
        for m in need:                                 # khởi tạo MSE λ = 0 từ file chính
            src = MAIN_W / R.weight_path(ds, H, m, "MSE", 0).name
            shutil.copyfile(src, R.weight_path(ds, H, m, "MSE", 0))
        print(f"[{ds} H={H} {model} {OBJ}]", flush=True)
        cell = R.Cell(ds, H)
        if model == "DLinear":
            lam0_fresh(cell, "Linear", done, writer)
        R.run_irls_path(cell, model, OBJ, done, writer)
        cell.free()
        del cell
        R.torch.cuda.empty_cache()
    R.PATH_CSV, R.WEIGHTS = MAIN_CSV, MAIN_W


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compare():
    old, new = read(MAIN_CSV), read(OUT / "lambda_path_recheck.csv")
    out, ok_all = [], True
    for ds, H, model in PATHS:
        sel = lambda rows: {R.lam_key(r["lam"]): r for r in rows                       # noqa: E731
                            if (r["dataset"], int(r["H"]), r["model"], r["objective"]) == (ds, H, model, OBJ)}
        o, n = sel(old), sel(new)
        lams = sorted(set(o) | set(n), key=float)
        rows, max_dev = [], 0.0
        for lam in lams:
            if lam not in o or lam not in n:
                rows.append({"lam": float(lam), "only_in": "old" if lam in o else "new"})
                ok_all = False
                continue
            dev = {m: float(n[lam][m]) - float(o[lam][m]) for m in METRICS}
            max_dev = max(max_dev, *(abs(v) for v in dev.values()))
            rows.append({"lam": float(lam), **{f"d_{m}": v for m, v in dev.items()},
                         "n_iter_old": int(o[lam]["n_iter"]), "n_iter_new": int(n[lam]["n_iter"]),
                         "n_fallback_new": int(n[lam]["n_fallback"]) if n[lam].get("n_fallback") else 0,
                         "derived_from": n[lam]["derived_from"], "commit_old": o[lam]["git_commit"]})
        stars = {}
        for metric in ("val_mae", "val_mse"):
            stars[metric] = {"old": float(min(o.values(), key=lambda r: float(r[metric]))["lam"]),
                             "new": float(min(n.values(), key=lambda r: float(r[metric]))["lam"])}
        same_star = all(v["old"] == v["new"] for v in stars.values())
        ok = max_dev <= TOL and same_star and all("only_in" not in r for r in rows)
        ok_all &= ok
        out.append({"dataset": ds, "H": H, "model": model, "objective": OBJ, "max_abs_dev": max_dev,
                    "lam_star": stars, "same_lam_star": same_star, "pass": ok, "rows": rows})
        print(f"{ds} H={H} {model}: lệch lớn nhất {max_dev:.2e}, λ* {stars}, "
              f"fallback mới {sum(r.get('n_fallback_new', 0) for r in rows)} bước h → {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        for r in rows:
            if "only_in" in r:
                print(f"    λ = {r['lam']:g}: chỉ có ở {r['only_in']}")
            else:
                print(f"    λ = {r['lam']:>8g}: Δtest MSE {r['d_test_mse']:+.1e} Δval MAE {r['d_val_mae']:+.1e} "
                      f"vòng {r['n_iter_old']}→{r['n_iter_new']} fallback {r['n_fallback_new']}")
    save_json(OUT / "compare.json", {"tol": TOL, "pass": ok_all, "paths": out})
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if not args.compare:
        run()
    sys.exit(0 if compare() else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
