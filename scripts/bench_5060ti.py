"""Đo IRLS trên RTX 5060 Ti: tốc độ, bộ nhớ, chi phí grid, warm start khi quét λ.

Mọi lần chạy ở chế độ pha (Xt, Y, W0 float64 trên GPU, gram_dtype=torch.float32), TF32 tắt.
Linear, khởi tạo OLS (fit_with_bias(ols), bias ghép vào cột cuối), trừ khi ghi khác.
Mọi sai số (J, MSE/MAE val/test) tính lại bằng float64 trên CPU.

    python scripts/bench_5060ti.py --step chunk                  # 3.2: quét chunk, ETTh1 H = 96
    python scripts/bench_5060ti.py --step full                   # 3.2: Huber và MAE chạy đủ, so với máy cũ
    python scripts/bench_5060ti.py --step ettm1                  # 3.3: ETTm1 H = 720, 3 vòng MAE
    python scripts/bench_5060ti.py --step cell --data ETTh1 --H 720   # 3.4: ô bất kỳ, 3 vòng MAE
    python scripts/bench_5060ti.py --step warmstart --loss mae   # 4: quét λ cho DLinear, lạnh và warm
    python scripts/bench_5060ti.py --summary                     # in bảng, ước lượng grid

`chunk` mặc định lấy từ results/bench_5060ti/chunk.json (nhanh nhất theo một vòng irls);
đổi bằng --chunk. Kết quả: results/bench_5060ti/*.json (và .npy cho W của bước full).
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_ett, make_dataset, split_borders            # noqa: E402
from src.operators import build_N, build_P                             # noqa: E402
from src.solvers import (MAE_DELTA, fit_with_bias, irls, ols, ridge_pen,  # noqa: E402
                         weight_gram)

# Quy tắc 1: không TF32, để float32 thật sự là float32 (23 bit định trị)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

L = 336
HUBER_DELTA = 1.0
LOSSES = {"mae": (MAE_DELTA, 1e-8), "huber": (HUBER_DELTA, 1e-9)}   # (δ, tol)
CHUNKS = [8, 16, 32, 64, 96]
LAMS = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
K_DLINEAR = 25
DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
HORIZONS = [96, 192, 336, 720]
FIT_CELLS = ["ETTh1 H=96", "ETTh1 H=720", "ETTm1 H=96", "ETTm1 H=720"]
N_COMBOS = 5          # Linear λ=0, Linear λ*, DLinear λ*, NLinear λ=0, NLinear λ*
OUT = ROOT / "results" / "bench_5060ti"
OLD = ROOT / "results" / "irls_dtype"
DEV = "cuda"
MiB = 2 ** 20


# ---------------------------------------------------------------------------
# tiện ích
# ---------------------------------------------------------------------------

def hw_info():
    """Quy tắc 5: phần cứng và phiên bản, ghi vào mọi file kết quả."""
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gib": props.total_memory / 2 ** 30,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "driver": driver,
        "os": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def git_commit():
    """Hash HEAD, thêm '-dirty' nếu cây làm việc có thay đổi chưa commit (trong src/, scripts/, tests/)."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        h = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--", "src", "scripts", "tests")
        return h + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def save_json(path, res):
    """Ghi JSON kèm phần cứng (quy tắc 5), commit và thời điểm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res = {"hardware": hw_info(), "git_commit": git_commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), **res}
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


def save(name, res):
    save_json(OUT / f"{name}.json", res)


def load_json(name):
    f = OUT / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def best_chunk():
    r = load_json("chunk")
    if r is None:
        sys.exit("Chưa có chunk.json: chạy --step chunk trước, hoặc truyền --chunk.")
    return r["best_chunk"]


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def add_ones(X):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1)


def load_cell(name, H, need_eval=True):
    """Trả dict các mảng float64 numpy: Xt, Y (train) và Xt_va, Y_va, Xt_te, Y_te."""
    train, val, test = load_ett(ROOT / "data" / f"{name}.csv", L, name)
    X, Y = make_dataset(train, L, H)
    d = {"X": X, "Y": Y, "Xt": add_ones(X)}
    if need_eval:
        for tag, split in (("va", val), ("te", test)):
            Xs, Ys = make_dataset(split, L, H)
            d[f"Xt_{tag}"], d[f"Y_{tag}"] = add_ones(Xs), Ys
    return d


def ols_init(X, Y):
    W, b = fit_with_bias(ols, X, Y)
    return np.concatenate([W, b[:, None]], axis=1)


def huber_np(R, delta):
    a = np.abs(R)
    q = np.minimum(a, delta)
    return float((q * (a - q / 2)).mean())


def metrics64(W, d, delta):
    """W [H, p] numpy float64 → sai số float64. J_train là phần Huber, chưa gồm phạt."""
    out = {"J_train": huber_np(d["Y"] - d["Xt"] @ W.T, delta)}
    for tag in ("va", "te"):
        if f"Xt_{tag}" in d:
            R = d[f"Y_{tag}"] - d[f"Xt_{tag}"] @ W.T
            out[f"mse_{tag}"] = float((R ** 2).mean())
            out[f"mae_{tag}"] = float(np.abs(R).mean())
            out[f"huber1_{tag}"] = huber_np(R, HUBER_DELTA)
    return out


def to_gpu(*arrs):
    return [torch.tensor(a, dtype=torch.float64, device=DEV) for a in arrs]


def warmup(Xg, Yg, W0g, chunk):
    """Khởi động cuBLAS/cuSOLVER và bộ cấp phát trên một lát nhỏ, không tính giờ."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irls(Xg[:2000], Yg[:2000], MAE_DELTA, W0g, max_iter=2, gram_dtype=torch.float32, chunk=chunk)
    torch.cuda.synchronize()


def run_irls(Xg, Yg, W0g, delta, tol, chunk, max_iter=1000, lam=0.0, pen=None, keep_W=False,
             patience=1, constrained=False, quiet=False):
    """Một lần irls ở chế độ pha có bấm giờ theo vòng. Trả dict và danh sách W theo vòng (nếu keep_W)."""
    trace, Ws = [], []

    def cb(it, W, J):
        J = J.item()                                   # đồng bộ GPU
        trace.append([it, time.perf_counter() - t0, J])
        if keep_W:
            Ws.append(W.detach().cpu().numpy())
        if it % 25 == 0 and not quiet:
            print(f"    vòng {it}: J = {J:.10e}  ({trace[-1][1]:.1f} s)", flush=True)

    torch.cuda.reset_peak_memory_stats()
    status = "ok"
    t0 = sync_time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        W, n_iter = irls(Xg, Yg, delta, W0g, lam=lam, pen=pen, max_iter=max_iter, tol=tol,
                         callback=cb, gram_dtype=torch.float32, chunk=chunk, patience=patience,
                         constrained=constrained)
    elapsed = sync_time() - t0
    if caught:
        status = "warning: " + str(caught[-1].message)
    t = [0.0] + [row[1] for row in trace]
    per_iter = np.diff(t)
    res = {"status": status, "converged": not caught, "n_iter": n_iter, "seconds": elapsed,
           "s_per_iter": elapsed / n_iter,
           "s_per_iter_median": float(np.median(per_iter[1:] if len(per_iter) > 1 else per_iter)),
           "per_iter_seconds": per_iter.tolist(),
           "peak_gpu_mb": torch.cuda.max_memory_allocated() / MiB,
           "J_final_irls": trace[-1][2], "trace": trace}
    return res, W.cpu().numpy(), Ws


# ---------------------------------------------------------------------------
# 3.2: chunk
# ---------------------------------------------------------------------------

def step_chunk(repeats=5):
    d = load_cell("ETTh1", 96, need_eval=False)
    n, p = d["Xt"].shape
    H = d["Y"].shape[1]
    W0 = ols_init(d["X"], d["Y"])
    Xg, Yg, W0g = to_gpu(d["Xt"], d["Y"], W0)
    Xg32 = Xg.float()
    R = Yg - Xg @ W0g.T
    w32 = (MAE_DELTA / R.abs().clamp(min=MAE_DELTA)).float()      # trọng số MAE tại OLS
    del R
    flop = n * p * p * H
    warmup(Xg, Yg, W0g, 8)

    rows = []
    for c in CHUNKS:
        print(f"chunk = {c}", flush=True)
        row = {"chunk": c}
        try:
            # weight_gram riêng (float32)
            weight_gram(Xg32, w32, c)                       # khởi động
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            for _ in range(repeats):
                t0 = sync_time()
                A = weight_gram(Xg32, w32, c)
                ts.append(sync_time() - t0)
                del A
            row["gram_s"] = ts
            row["gram_s_median"] = float(np.median(ts))
            row["gram_gflops"] = flop / row["gram_s_median"] / 1e9
            row["gram_peak_mb"] = torch.cuda.max_memory_allocated() / MiB
            row["gram_peak_extra_mb"] = (torch.cuda.max_memory_allocated() - base) / MiB

            # một vòng irls đầy đủ (chế độ pha, MAE). max_iter=1 gồm cả một lần tính J ban đầu.
            ts = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                irls(Xg, Yg, MAE_DELTA, W0g, max_iter=1, gram_dtype=torch.float32, chunk=c)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(repeats):
                    t0 = sync_time()
                    irls(Xg, Yg, MAE_DELTA, W0g, max_iter=1, gram_dtype=torch.float32, chunk=c)
                    ts.append(sync_time() - t0)
            row["iter_s"] = ts
            row["iter_s_median"] = float(np.median(ts))
            row["iter_peak_mb"] = torch.cuda.max_memory_allocated() / MiB
            print(f"  weight_gram {row['gram_s_median']:.4f} s ({row['gram_gflops']:.0f} GFLOP/s), "
                  f"một vòng irls {row['iter_s_median']:.4f} s, đỉnh {row['iter_peak_mb']:.0f} MB", flush=True)
        except torch.OutOfMemoryError as e:
            row["error"] = f"OOM: {e}".splitlines()[0]
            print("  " + row["error"])
        torch.cuda.empty_cache()
        rows.append(row)

    # Chênh lệch giữa các chunk cỡ nhiễu đo, còn bộ nhớ tăng tuyến tính theo chunk: lấy chunk nhỏ
    # nhất trong khoảng 2% so với vòng irls nhanh nhất.
    ok = [r for r in rows if "iter_s_median" in r]
    fastest = min(r["iter_s_median"] for r in ok)
    best = min(r["chunk"] for r in ok if r["iter_s_median"] <= 1.02 * fastest)
    print(f"chunk chọn: {best} (nhỏ nhất trong 2% của vòng nhanh nhất {fastest:.4f} s)")
    save("chunk", {"cell": "ETTh1 H=96", "n": n, "p": p, "H": H, "flop_per_gram": flop,
                   "repeats": repeats, "warmup": 1, "rows": rows, "best_chunk": best,
                   "best_rule": "chunk nhỏ nhất có iter_s_median ≤ 1,02 × nhanh nhất"})


# ---------------------------------------------------------------------------
# 3.2: chạy đầy đủ, so với máy cũ
# ---------------------------------------------------------------------------

def step_full(chunk, losses):
    d = load_cell("ETTh1", 96)
    W0 = ols_init(d["X"], d["Y"])
    Xg, Yg, W0g = to_gpu(d["Xt"], d["Y"], W0)
    warmup(Xg, Yg, W0g, chunk)
    for loss in losses:
        delta, tol = LOSSES[loss]
        print(f"[full {loss}] δ = {delta}, tol = {tol}, chunk = {chunk}", flush=True)
        res, W, Ws = run_irls(Xg, Yg, W0g, delta, tol, chunk, keep_W=True)
        m = metrics64(W, d, delta)
        # sai số theo từng vòng (sau khi đã bấm giờ), để so với máy cũ ở cùng số vòng
        for row, Wk in zip(res["trace"], Ws):
            mk = metrics64(Wk, d, delta)
            row += [mk["mse_te"], mk["mae_te"]]

        comp = {}
        old = json.loads((OLD / f"{loss}_mixed.json").read_text(encoding="utf-8"))
        k = res["n_iter"]
        if len(old["trace"]) >= k:
            o = old["trace"][k - 1]                        # [it, t, J, mae_train, mse_test, mae_test]
            comp = {"old_file": f"results/irls_dtype/{loss}_mixed.json", "old_gpu": old["gpu"],
                    "iter": k, "old_mse_te": o[4], "old_mae_te": o[5],
                    "d_mse_te": m["mse_te"] - o[4], "d_mae_te": m["mae_te"] - o[5],
                    "old_J": o[2], "d_J": res["J_final_irls"] - o[2],
                    "max_abs_diff_all_iters": max(
                        max(abs(a[3] - b[4]), abs(a[4] - b[5]))
                        for a, b in zip(res["trace"], old["trace"][:k])),
                    "old_s_per_iter": old["s_per_iter"]}
            comp["pass_1e-6"] = abs(comp["d_mse_te"]) <= 1e-6 and abs(comp["d_mae_te"]) <= 1e-6
        res.update({"loss": loss, "delta": delta, "tol": tol, "chunk": chunk, "cell": "ETTh1 H=96",
                    "init": "OLS", "mode": "mixed (float64 + gram float32)",
                    "metrics": m, "compare_old": comp})
        np.save(OUT / f"full_{loss}.npy", W)
        save(f"full_{loss}", res)
        show_full(res)


def show_full(r):
    m, c = r["metrics"], r["compare_old"]
    print(f"  {r['n_iter']} vòng, {r['seconds']:.1f} s ({r['s_per_iter']:.3f} s/vòng, trung vị "
          f"{r['s_per_iter_median']:.3f}), đỉnh {r['peak_gpu_mb']:.0f} MB, {r['status']}")
    print(f"  J train {m['J_train']:.10e}  MSE test {m['mse_te']:.6f}  MAE test {m['mae_te']:.6f}")
    if c:
        print(f"  so máy cũ ({c['old_gpu']}) ở vòng {c['iter']}: ΔMSE test {c['d_mse_te']:+.2e}, "
              f"ΔMAE test {c['d_mae_te']:+.2e}, max lệch mọi vòng {c['max_abs_diff_all_iters']:.2e} "
              f"→ {'ĐẠT' if c['pass_1e-6'] else 'LỆCH QUÁ 1e-6'}")


# ---------------------------------------------------------------------------
# 3.3, 3.4: một ô bất kỳ, vài vòng MAE
# ---------------------------------------------------------------------------

def step_cell(name, H, chunk, iters=3):
    t_load = time.perf_counter()
    d = load_cell(name, H, need_eval=False)
    W0 = ols_init(d["X"], d["Y"])
    t_load = time.perf_counter() - t_load
    n, p = d["Xt"].shape
    Xg, Yg, W0g = to_gpu(d["Xt"], d["Y"], W0)
    del d
    sizes = {k: {"shape": list(t.shape), "mib": t.numel() * t.element_size() / MiB}
             for k, t in (("Xt", Xg), ("Y", Yg), ("W0", W0g))}
    print(f"[cell {name} H={H}] n = {n}, p = {p}, nạp + OLS {t_load:.1f} s, "
          f"tensor: " + ", ".join(f"{k} {v['shape']} {v['mib']:.0f} MiB" for k, v in sizes.items()), flush=True)

    attempts, res = [], None
    c = chunk
    while c >= 1:
        try:
            warmup(Xg, Yg, W0g, c)
            print(f"  chunk = {c}", flush=True)
            res, _, _ = run_irls(Xg, Yg, W0g, MAE_DELTA, 1e-8, c, max_iter=iters)
            attempts.append({"chunk": c, "ok": True, "peak_gpu_mb": res["peak_gpu_mb"]})
            break
        except torch.OutOfMemoryError as e:
            attempts.append({"chunk": c, "ok": False, "peak_gpu_mb": torch.cuda.max_memory_allocated() / MiB,
                             "error": str(e).splitlines()[0]})
            print(f"  OOM ở chunk = {c} (đỉnh {attempts[-1]['peak_gpu_mb']:.0f} MB)", flush=True)
            torch.cuda.empty_cache()
            c //= 2
    out = {"cell": f"{name} H={H}", "dataset": name, "H": H, "L": L, "n": n, "p": p,
           "nH": n * H, "tensor_sizes": sizes, "load_and_ols_seconds": t_load,
           "loss": "mae", "delta": MAE_DELTA, "mode": "mixed (float64 + gram float32)",
           "attempts": attempts}
    if res is None:
        out["status"] = "OOM ở mọi chunk"
        print("  OOM ở mọi chunk")
    else:
        # s/vòng: trung vị các vòng từ vòng 2 (vòng 1 có thêm J ban đầu và chưa ấm hẳn)
        out.update({"chunk": c, **res})
        print(f"  {res['n_iter']} vòng, s/vòng (trung vị vòng 2..) {res['s_per_iter_median']:.3f}, "
              f"theo từng vòng {[round(x, 3) for x in res['per_iter_seconds']]}, "
              f"đỉnh {res['peak_gpu_mb']:.0f} MB", flush=True)
    save(f"cell_{name}_H{H}", out)


# ---------------------------------------------------------------------------
# 4: warm start khi quét λ (DLinear, pen = N⁻¹)
# ---------------------------------------------------------------------------

def next_lam(lam):
    return lam * 3 if str(int(lam)).startswith("1") else lam * 10 // 3


def best_of(rows, key):
    return min(rows, key=lambda r: r["metrics"][key])["lam"]


def step_warmstart(loss, chunk, extend=2):
    delta, tol = LOSSES[loss]
    own = "mae_va" if loss == "mae" else "huber1_va"
    d = load_cell("ETTh1", 96)
    Ninv = np.linalg.inv(build_N(build_P(L, K_DLINEAR)))
    Xg, Yg = to_gpu(d["Xt"], d["Y"])
    W_ols = ols_init(d["X"], d["Y"])
    warmup(Xg, Yg, torch.tensor(W_ols, device=DEV), chunk)

    def cold_init(lam):
        # quy ước mục 1.5 của NHAT_KY: δ lớn thì F ≈ ‖r‖²/(2δ) + λ wᵀPen w, tức ridge MSE với 2δλ·Pen
        if lam == 0:
            return W_ols
        W, b = fit_with_bias(lambda X, Y: ridge_pen(X, Y, 2 * delta * lam * Ninv), d["X"], d["Y"])
        return np.concatenate([W, b[:, None]], axis=1)

    def one(lam, W0, mode):
        print(f"  [{loss} {mode}] λ = {lam}", flush=True)
        r, W, _ = run_irls(Xg, Yg, torch.tensor(W0, device=DEV), delta, tol, chunk,
                           lam=lam, pen=Ninv if lam else None)
        m = metrics64(W, d, delta)
        m["norm_W"] = float(np.linalg.norm(W[:, :-1]))
        print(f"    {r['n_iter']} vòng, {r['seconds']:.1f} s, MSE val {m['mse_va']:.6f}, "
              f"MAE val {m['mae_va']:.6f}, MSE test {m['mse_te']:.6f}", flush=True)
        r.pop("trace")
        return {"lam": lam, "mode": mode, **r, "metrics": m, "init_metrics": metrics64(W0, d, delta)}, W

    lams = list(LAMS)
    rows = {"cold": [], "warm": []}
    W_warm = W_ols
    i, added = 0, 0
    while i < len(lams):
        lam = lams[i]
        rc, W_cold = one(lam, cold_init(lam), "cold")
        rows["cold"].append(rc)
        if lam == 0:                      # λ = 0: warm và lạnh cùng khởi tạo OLS, dùng lại kết quả
            rows["warm"].append({**rc, "mode": "warm"})
            W_warm = W_cold
        else:
            rw, W_warm = one(lam, W_warm, "warm")
            rows["warm"].append(rw)
        i += 1
        if i == len(lams) and added < extend:
            top = lams[-1]
            if any(best_of(rows[mo], key) == top for mo in rows for key in (own, "mse_va")):
                lams.append(next_lam(top))
                added += 1
                print(f"  λ tốt nhất nằm ở đầu mút {top}: mở rộng thêm λ = {lams[-1]}", flush=True)

    summ = {}
    for mo in rows:
        summ[mo] = {"total_iter": sum(r["n_iter"] for r in rows[mo]),
                    "total_seconds": sum(r["seconds"] for r in rows[mo]),
                    "best_lam_own": best_of(rows[mo], own),
                    "best_lam_mse": best_of(rows[mo], "mse_va")}
        for k in ("own", "mse"):
            summ[mo][f"best_lam_{k}_at_edge"] = summ[mo][f"best_lam_{k}"] in (lams[0], lams[-1])
    save(f"warmstart_{loss}", {
        "cell": "ETTh1 H=96", "model": f"DLinear (Linear + pen = N⁻¹, k = {K_DLINEAR})",
        "loss": loss, "delta": delta, "tol": tol, "chunk": chunk, "lams": lams, "extended": added,
        "own_metric": own,
        "cold_init": "ridge MSE dạng đóng, fit_with_bias(ridge_pen(Pen = 2δλ·N⁻¹)) theo mục 1.5 "
                     "của NHAT_KY_THAY_DOI; λ = 0 là OLS",
        "warm_init": "nghiệm IRLS của λ liền trước (λ = 0 từ OLS)",
        "note_lam0": "λ = 0: hàng warm chép từ hàng lạnh (cùng khởi tạo OLS), không chạy lại",
        "summary": summ, "rows": rows})
    show_warm(load_json(f"warmstart_{loss}"))


def show_warm(r):
    print(f"[warmstart {r['loss']}] δ = {r['delta']}, tol = {r['tol']}, chọn theo {r['own_metric']} và mse_va")
    print(f"  {'λ':>7} | {'vòng lạnh':>9} {'vòng warm':>9} | {'s lạnh':>7} {'s warm':>7} | "
          f"{'MSE val':>9} {'MAE val':>9} {'Hub val':>9} | {'MSE test':>9} {'MAE test':>9}")
    for c, w in zip(r["rows"]["cold"], r["rows"]["warm"]):
        m = w["metrics"]
        print(f"  {c['lam']:>7} | {c['n_iter']:>9} {w['n_iter']:>9} | {c['seconds']:>7.1f} {w['seconds']:>7.1f} | "
              f"{m['mse_va']:.6f} {m['mae_va']:.6f} {m['huber1_va']:.6f} | {m['mse_te']:.6f} {m['mae_te']:.6f}")
    for mo, s in r["summary"].items():
        print(f"  {mo}: tổng {s['total_iter']} vòng, {s['total_seconds']:.0f} s; λ* ({r['own_metric']}) = "
              f"{s['best_lam_own']}{' (đầu mút)' if s['best_lam_own_at_edge'] else ''}, λ* (mse_va) = "
              f"{s['best_lam_mse']}{' (đầu mút)' if s['best_lam_mse_at_edge'] else ''}")


# ---------------------------------------------------------------------------
# tổng hợp, ước lượng grid (3.4)
# ---------------------------------------------------------------------------

def n_windows(name, H):
    train_end = split_borders(name)[0]
    return 7 * (train_end - L - H + 1)


def grid_estimate():
    cells = [load_json(f"cell_{nm}_H{H}") for nm in DATASETS for H in HORIZONS]
    cells = [c for c in cells if c and "s_per_iter_median" in c]
    full = {loss: load_json(f"full_{loss}") for loss in LOSSES}
    if len(cells) < 2 or not all(full.values()):
        return None
    # khớp s/vòng ≈ a + b·n·H trên 4 ô góc (ETTh1, ETTm1 × H = 96, 720), kiểm trên các ô còn lại
    corners = [c for c in cells if c["cell"] in FIT_CELLS] or cells
    x = np.array([c["nH"] for c in corners], float)
    y = np.array([c["s_per_iter_median"] for c in corners])
    b, a = np.polyfit(x, y, 1)
    fit = [{"cell": c["cell"], "nH": c["nH"], "measured": c["s_per_iter_median"],
            "fitted": a + b * c["nH"], "used_in_fit": c in corners,
            "rel_err": (a + b * c["nH"]) / c["s_per_iter_median"] - 1} for c in cells]
    iters = {loss: full[loss]["n_iter"] for loss in LOSSES}
    # kịch bản MAE tol = 1e-9: số vòng từ quỹ đạo máy cũ (189, BAO_CAO_IRLS_DTYPE mục 5)
    old_mae = json.loads((OLD / "mae_mixed.json").read_text(encoding="utf-8"))
    J = [row[2] for row in old_mae["trace"]]
    J0 = old_mae["J0_in_dtype"]
    Js = [J0] + J
    it_1e9 = next(k + 1 for k in range(len(J)) if (Js[k] - Js[k + 1]) / Js[k] < 1e-9)
    rows = []
    for nm in DATASETS:
        for H in HORIZONS:
            nH = n_windows(nm, H) * H
            measured = next((c["s_per_iter_median"] for c in cells if c["cell"] == f"{nm} H={H}"), None)
            s = measured if measured is not None else a + b * nH
            rows.append({"cell": f"{nm} H={H}", "n": n_windows(nm, H), "nH": nH, "s_per_iter": s,
                         "measured": measured, "fitted": a + b * nH,
                         "hours_mae": s * iters["mae"] * N_COMBOS / 3600,
                         "hours_huber": s * iters["huber"] * N_COMBOS / 3600,
                         "hours_mae_tol1e-9": s * it_1e9 * N_COMBOS / 3600})
    tot = {k: sum(r[k] for r in rows) for k in ("hours_mae", "hours_huber", "hours_mae_tol1e-9")}
    by_ds = {nm: sum(r["hours_mae"] + r["hours_huber"] for r in rows if r["cell"].startswith(nm))
             for nm in DATASETS}
    # quét λ: số vòng của cả lưới đo ở bước warmstart (ETTh1 H = 96, DLinear), giả định như nhau
    # cho mọi ô và cho 3 mô hình có λ* (Linear, DLinear, NLinear)
    sum_s = sum(r["s_per_iter"] for r in rows)
    sweep = {}
    for loss in LOSSES:
        w = load_json(f"warmstart_{loss}")
        if w:
            sweep[loss] = {mo: {"iters": s_["total_iter"], "hours": s_["total_iter"] * sum_s * 3 / 3600}
                           for mo, s_ in w["summary"].items()}
    return {"model": "s/vòng ≈ a + b·n·H, bình phương tối thiểu trên 4 ô góc; ô nào đã đo thì "
                     "dùng số đo (trung vị vòng 2–3), còn lại dùng nội suy",
            "fit_cells": FIT_CELLS,
            "a": a, "b": b, "fit": fit, "iters": iters, "iters_mae_tol1e-9_old_trace": it_1e9,
            "n_combos": N_COMBOS, "assumption": "mọi ô cần số vòng như ETTh1 H = 96 (bước full); "
            "chưa tính quét λ, nạp dữ liệu và OLS", "rows": rows, "total_hours": tot,
            "hours_by_dataset": by_ds, "sum_s_per_iter_12_cells": sum_s,
            "lambda_sweep": sweep, "lambda_sweep_assumption": "3 mô hình có λ* × 12 ô, mỗi ô quét cả "
            "lưới với số vòng như bước warmstart; gồm cả λ = 0 (trùng lần chạy λ = 0 của grid)"}


def summary():
    r = load_json("chunk")
    if r:
        print(f"[chunk] ETTh1 H=96, n = {r['n']}, {r['hardware']['gpu']}")
        for row in r["rows"]:
            if "error" in row:
                print(f"  chunk {row['chunk']:>3}: {row['error']}")
            else:
                print(f"  chunk {row['chunk']:>3}: weight_gram {row['gram_s_median']:.4f} s "
                      f"({row['gram_gflops']:.0f} GFLOP/s, thêm {row['gram_peak_extra_mb']:.0f} MB), "
                      f"một vòng {row['iter_s_median']:.4f} s, đỉnh {row['iter_peak_mb']:.0f} MB")
        print(f"  tốt nhất: {r['best_chunk']}\n")
    for loss in LOSSES:
        f = load_json(f"full_{loss}")
        if f:
            print(f"[full {loss}]")
            show_full(f)
    print()
    for nm in DATASETS:
        for H in HORIZONS:
            c = load_json(f"cell_{nm}_H{H}")
            if c:
                s = (f"{c['s_per_iter_median']:.3f} s/vòng, đỉnh {c['peak_gpu_mb']:.0f} MB, chunk {c['chunk']}"
                     if "s_per_iter_median" in c else c.get("status"))
                print(f"[cell {nm} H={H}] n = {c['n']}, n·H = {c['nH']:.3e}: {s}")
    g = grid_estimate()
    if g:
        save("grid_estimate", g)
        print(f"\n[grid] s/vòng ≈ {g['a']:.4f} + {g['b']:.3e}·n·H; số vòng MAE {g['iters']['mae']}, "
              f"Huber {g['iters']['huber']} (MAE tol 1e-9: {g['iters_mae_tol1e-9_old_trace']})")
        for row in g["fit"]:
            print(f"  {'khớp' if row['used_in_fit'] else 'kiểm'} {row['cell']:<12}: đo {row['measured']:.3f}, "
                  f"nội suy {row['fitted']:.3f} ({row['rel_err']:+.1%})")
        for row in g["rows"]:
            print(f"  {row['cell']:<12} n = {row['n']:>6}  {row['s_per_iter']:.3f} s/vòng  "
                  f"MAE {row['hours_mae']:.2f} h  Huber {row['hours_huber']:.2f} h")
        t = g["total_hours"]
        print(f"  tổng: MAE {t['hours_mae']:.1f} h (tol 1e-9: {t['hours_mae_tol1e-9']:.1f} h), "
              f"Huber {t['hours_huber']:.1f} h; theo dataset {({k: round(v, 1) for k, v in g['hours_by_dataset'].items()})}")
        for loss, sw in g["lambda_sweep"].items():
            print(f"  quét λ {loss} (3 mô hình × 12 ô): " + ", ".join(
                f"{mo} {v['iters']} vòng/ô → {v['hours']:.1f} h" for mo, v in sw.items()))
    for loss in LOSSES:
        w = load_json(f"warmstart_{loss}")
        if w:
            print()
            show_warm(w)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["chunk", "full", "ettm1", "cell", "warmstart"])
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--loss", choices=list(LOSSES), help="full, warmstart: mặc định cả hai")
    ap.add_argument("--data", choices=DATASETS, default="ETTh1")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    losses = [args.loss] if args.loss else list(LOSSES)
    if args.summary:
        summary()
    elif args.step == "chunk":
        step_chunk()
    elif args.step == "full":
        step_full(args.chunk or best_chunk(), losses)
    elif args.step == "ettm1":
        step_cell("ETTm1", 720, args.chunk or best_chunk(), args.iters)
    elif args.step == "cell":
        step_cell(args.data, args.H, args.chunk or best_chunk(), args.iters)
    elif args.step == "warmstart":
        for loss in losses:
            step_warmstart(loss, args.chunk or best_chunk())
    else:
        ap.error("cần --step hoặc --summary")
