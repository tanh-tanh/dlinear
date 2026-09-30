"""Mục 11.2: IRLS nên chạy float32 hay float64 trên GPU?

Một ô: ETTh1, L = 336, H = 96, Linear (có bias, λ = 0), khởi tạo từ nghiệm OLS.
Mỗi lần chạy một mục tiêu (mae: δ = MAE_DELTA, huber: δ = HUBER_DELTA) ở một kiểu số.
Mọi sai số (J train, MSE/MAE test) và số điều kiện đều tính lại bằng float64, để hai kiểu
được chấm như nhau.

    python scripts/irls_dtype_check.py --loss mae --dtype float32     # float32 | float64 | mixed
    python scripts/irls_dtype_check.py --summary

mixed: dữ liệu và W float64, chỉ lập A_h bằng float32 (irls(..., gram_dtype=torch.float32)).

Kết quả: results/irls_dtype/<loss>_<dtype>.json (số đo, quỹ đạo J) và .npy (W [H, L+1]).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_etth1, make_dataset          # noqa: E402
from src.solvers import MAE_DELTA, fit_with_bias, huber_objective, irls, ols, weight_gram  # noqa: E402

L, H = 336, 96
HUBER_DELTA = 1.0            # dữ liệu đã chuẩn hóa theo kênh; 1.0 là mặc định của torch.nn.HuberLoss
DELTAS = {"mae": MAE_DELTA, "huber": HUBER_DELTA}
TOL = 1e-12                  # như lúc chạy báo cáo 11.2; mặc định của irls giờ là 1e-9
OUT = ROOT / "results" / "irls_dtype"
DTYPES = ["float32", "mixed", "float64"]


def load():
    train, _, test = load_etth1(ROOT / "data" / "ETTh1.csv", L)
    X, Y = make_dataset(train, L, H)
    X_te, Y_te = make_dataset(test, L, H)
    return X, Y, X_te, Y_te


def add_ones(X):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1)


def metrics64(W, Xt, Y, Xt_te, Y_te, delta):
    """W [H, p] → sai số tính bằng float64 trên CPU."""
    R = Y - Xt @ W.T
    R_te = Y_te - Xt_te @ W.T
    return {
        "J_train": huber_objective(torch.from_numpy(R), delta).item(),
        "mae_train": float(np.abs(R).mean()),
        "mse_test": float((R_te ** 2).mean()),
        "mae_test": float(np.abs(R_te).mean()),
    }


def cond_stats(A):
    """A [H, p, p] đối xứng xác định dương (float64) → κ theo từng h qua trị riêng."""
    ev = torch.linalg.eigvalsh(A)
    kappa = (ev[:, -1] / ev[:, 0]).cpu().numpy()
    return {"min": float(kappa.min()), "median": float(np.median(kappa)), "max": float(kappa.max()),
            "lambda_min_min": float(ev[:, 0].min()), "per_h": kappa.tolist()}


def run(loss, dtype_name, max_iter):
    dtype = torch.float64 if dtype_name == "mixed" else getattr(torch, dtype_name)
    gram_dtype = torch.float32 if dtype_name == "mixed" else None
    dev = "cuda"
    delta = DELTAS[loss]

    X, Y, X_te, Y_te = load()
    W_ols, b_ols = fit_with_bias(ols, X, Y)
    W0 = np.concatenate([W_ols, b_ols[:, None]], axis=1)          # [H, L+1], float64
    Xt, Xt_te = add_ones(X), add_ones(X_te)

    Xg = torch.tensor(Xt, dtype=dtype, device=dev)
    Yg = torch.tensor(Y, dtype=dtype, device=dev)
    W0g = torch.tensor(W0, dtype=dtype, device=dev)

    # Khởi động cuBLAS/cuSOLVER trên một lát nhỏ, không tính giờ
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irls(Xg[:2000], Yg[:2000], delta, W0g, max_iter=2, gram_dtype=gram_dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    trace = []                       # [vòng, giây từ lúc bắt đầu, J]; thêm sai số float64 sau khi bấm giờ
    Ws = []
    state = {}

    def cb(it, W, J):
        J = J.item()                 # đồng bộ GPU; irls cũng đồng bộ ngay sau đó khi so J
        trace.append([it, time.perf_counter() - t0, J])
        Ws.append(W.detach().double().cpu().numpy())
        if it % 25 == 0:
            print(f"  vòng {it}: J = {J:.10e}  ({trace[-1][1]:.0f} s)", flush=True)
        if J <= state.get("best_J", np.inf):
            state["best_J"], state["best"] = J, Ws[-1]

    status = "ok"
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _, n_iter = irls(Xg, Yg, delta, W0g, max_iter=max_iter, tol=TOL, callback=cb,
                             gram_dtype=gram_dtype)
        except RuntimeError as e:
            status, n_iter = f"RuntimeError: {e}", len(trace)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if status == "ok" and caught:
        status = "warning: " + str(caught[-1].message)
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    J0 = huber_objective(Yg - Xg @ W0g.T, delta).item()

    # Khi irls báo lỗi thì W cuối là W đã làm J tăng; chấm W có J thấp nhất trong quỹ đạo
    W = Ws[-1] if status == "ok" or status.startswith("warning") else state["best"]

    # Sai số theo từng vòng, float64 trên CPU: để so float32 với float64 ở cùng số vòng
    for row, Wk in zip(trace, Ws):
        m = metrics64(Wk, Xt, Y, Xt_te, Y_te, delta)
        row += [m["mae_train"], m["mse_test"], m["mae_test"]]

    # Số điều kiện, tính bằng float64: A_h ở nghiệm này, và XᵀX của MSE để so
    del Xg, Yg
    torch.cuda.empty_cache()
    X64 = torch.tensor(Xt, dtype=torch.float64, device=dev)
    Y64 = torch.tensor(Y, dtype=torch.float64, device=dev)
    R = Y64 - X64 @ torch.tensor(W, device=dev).T
    w = delta / R.abs().clamp(min=delta)
    kappa_h = cond_stats(weight_gram(X64, w))
    kappa_mse = cond_stats((X64.T @ X64).unsqueeze(0))

    res = {
        "loss": loss, "dtype": dtype_name, "delta": delta, "status": status,
        "n_iter": n_iter, "seconds": elapsed, "s_per_iter": elapsed / max(n_iter, 1),
        "peak_gpu_mb": peak_mb, "gpu": torch.cuda.get_device_name(0),
        "J0_in_dtype": J0,
        "metrics": metrics64(W, Xt, Y, Xt_te, Y_te, delta),
        "metrics_ols": metrics64(W0, Xt, Y, Xt_te, Y_te, delta),
        "weights": {"min": w.min().item(), "max": w.max().item(),
                    "frac_clamped": (w == 1).double().mean().item()},
        "kappa_A_h": kappa_h, "kappa_XtX": kappa_mse,
        "trace": trace,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{loss}_{dtype_name}.npy", W)
    (OUT / f"{loss}_{dtype_name}.json").write_text(json.dumps(res, indent=1))
    show(res)


def show(r):
    m, k = r["metrics"], r["kappa_A_h"]
    print(f"[{r['loss']} {r['dtype']}] {r['status']}")
    print(f"  {r['n_iter']} vòng, {r['seconds']:.1f} s ({r['s_per_iter']:.2f} s/vòng), đỉnh GPU {r['peak_gpu_mb']:.0f} MB")
    print(f"  J train {m['J_train']:.10e}  MAE train {m['mae_train']:.6f}  "
          f"MSE test {m['mse_test']:.6f}  MAE test {m['mae_test']:.6f}")
    print(f"  κ(A_h) min/median/max = {k['min']:.2e} / {k['median']:.2e} / {k['max']:.2e}   "
          f"κ(XᵀX) = {r['kappa_XtX']['max']:.2e}   w ∈ [{r['weights']['min']:.1e}, {r['weights']['max']:.1e}]")


def summary():
    """So float32 và mixed với float64 (tham chiếu): ở điểm dừng, và ở cùng số vòng."""
    for loss in DELTAS:
        runs = {}
        for dt in DTYPES:
            f = OUT / f"{loss}_{dt}.json"
            if f.exists():
                runs[dt] = json.loads(f.read_text())
                show(runs[dt])
        if "float64" not in runs:
            print()
            continue
        ref = runs["float64"]
        W64 = np.load(OUT / f"{loss}_float64.npy")
        for dt in ("float32", "mixed"):
            if dt not in runs:
                continue
            r = runs[dt]
            W = np.load(OUT / f"{loss}_{dt}.npy")
            print(f"  → {loss} {dt} so với float64: ‖W − W64‖/‖W64‖ = "
                  f"{np.linalg.norm(W - W64) / np.linalg.norm(W64):.2e}, "
                  f"thời gian float64 / {dt} = {ref['seconds'] / r['seconds']:.1f}× "
                  f"(mỗi vòng {ref['s_per_iter'] / r['s_per_iter']:.1f}×)")
            for key in ("J_train", "mse_test", "mae_test"):
                d = r["metrics"][key] - ref["metrics"][key]
                print(f"    {key:9s} {dt} − float64 = {d:+.3e}  (tương đối {d / ref['metrics'][key]:+.2e})")
            # float64 dừng ở cùng số vòng thì sao: tách lỗi do kiểu số khỏi lỗi do dừng sớm
            k = min(r["n_iter"], ref["n_iter"]) - 1
            row = ref["trace"][k]
            print(f"    float64 ở vòng {k + 1}: MSE test {row[4]:.6f}, MAE test {row[5]:.6f} "
                  f"(so với {dt}: {r['trace'][k][4]:.6f}, {r['trace'][k][5]:.6f})")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")        # console Windows mặc định cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=list(DELTAS))
    ap.add_argument("--dtype", choices=DTYPES)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        run(args.loss, args.dtype, args.max_iter)
