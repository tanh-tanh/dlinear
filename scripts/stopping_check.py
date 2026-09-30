"""Nhiệm vụ 2, Phần A: tiêu chí dừng của IRLS khi quét λ cho MAE.

Ô: DLinear (Linear có bias, pen = N⁻¹, k = 25), ETTh1 H = 96, chế độ pha, chunk 16, TF32 tắt.

    python scripts/stopping_check.py --step reference                    # A.2: tol = 0, 2000 vòng, λ ∈ {0, 30, 100, 300, 1000}
    python scripts/stopping_check.py --step sweep --loss mae --tol 1e-9 --patience 5   # A.3
    python scripts/stopping_check.py --step analyze                      # A.4: ba tiêu chí, chọn cấu hình

Kết quả: results/stopping/*.json. Tập test chỉ ghi để tham khảo, không dùng để chọn gì.
Bước reference và sweep bỏ qua file đã có (chạy tiếp được); --force để chạy lại.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_5060ti as B                    # noqa: E402  (tắt TF32 khi import)
from bench_5060ti import (LOSSES, MAE_DELTA, ROOT, load_cell, metrics64, ols_init,  # noqa: E402
                          run_irls, save_json, to_gpu, warmup)
from src.operators import build_N, build_P  # noqa: E402
from src.solvers import fit_with_bias, irls, ridge_pen  # noqa: E402

OUT = ROOT / "results" / "stopping"
CHUNK = 16
K = 25
LAMS = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
REF_LAMS = [0, 30, 100, 300, 1000]
REF_ITERS = 2000
REF_EVERY = 50
MAE_CONFIGS = [(1e-9, 1), (1e-9, 5), (1e-10, 1), (1e-10, 5)]
HUBER_CONFIGS = [(1e-9, 1), (1e-9, 5)]
GAP_TOL = 1e-5            # tiêu chí 3: lệch nghiệm tham chiếu


def load_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


class Cell:
    """Dữ liệu ETTh1 H = 96 trên GPU, N⁻¹, W_OLS; khởi động GPU một lần."""

    def __init__(self):
        self.d = load_cell("ETTh1", 96)
        self.Ninv = np.linalg.inv(build_N(build_P(B.L, K)))
        self.Xg, self.Yg = to_gpu(self.d["Xt"], self.d["Y"])
        self.W_ols = ols_init(self.d["X"], self.d["Y"])
        warmup(self.Xg, self.Yg, torch.tensor(self.W_ols, device=B.DEV), CHUNK)

    def cold_init(self, lam, delta):
        """Ridge MSE dạng đóng với 2δλ·N⁻¹ (quy ước mục 1.5); MAE: thực chất là OLS."""
        if lam == 0:
            return self.W_ols
        W, b = fit_with_bias(lambda X, Y: ridge_pen(X, Y, 2 * delta * lam * self.Ninv), self.d["X"], self.d["Y"])
        return np.concatenate([W, b[:, None]], axis=1)


# ---------------------------------------------------------------------------
# A.2: nghiệm tham chiếu
# ---------------------------------------------------------------------------

def step_reference(cell, lams, force=False):
    for lam in lams:
        f = OUT / f"reference_mae_lam{lam}.json"
        if f.exists() and not force:
            print(f"bỏ qua λ = {lam}: đã có {f.name}")
            continue
        print(f"[reference] λ = {lam}: tol = 0, max_iter = {REF_ITERS}, khởi tạo OLS", flush=True)
        Js, snaps = [], []

        def cb(it, W, J):
            Js.append(J.item())
            if (it + 1) % REF_EVERY == 0:
                snaps.append((it + 1, W.detach().cpu().numpy()))
                if (it + 1) % 500 == 0:
                    print(f"    vòng {it + 1}: J = {Js[-1]:.12e} ({time.perf_counter() - t0:.0f} s)", flush=True)

        t0 = B.sync_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            W, n_iter = irls(cell.Xg, cell.Yg, MAE_DELTA, torch.tensor(cell.W_ols, device=B.DEV),
                             lam=lam, pen=cell.Ninv if lam else None, max_iter=REF_ITERS, tol=0.0,
                             callback=cb, gram_dtype=torch.float32, chunk=CHUNK)
        elapsed = B.sync_time() - t0
        W = W.cpu().numpy()
        traj = []
        for k, Wk in snaps:
            m = metrics64(Wk, cell.d, MAE_DELTA)
            traj.append({"iter": k, "J": Js[k - 1], "mae_va": m["mae_va"], "mse_va": m["mse_va"],
                         "huber1_va": m["huber1_va"]})
        last = [t for t in traj if t["iter"] >= n_iter - 500]
        drift = {key: max(t[key] for t in last) - min(t[key] for t in last) for key in ("mae_va", "mse_va", "huber1_va")}
        m = metrics64(W, cell.d, MAE_DELTA)
        res = {"cell": "ETTh1 H=96", "model": f"DLinear (pen = N⁻¹, k = {K})", "loss": "mae",
               "delta": MAE_DELTA, "lam": lam, "tol": 0.0, "max_iter": REF_ITERS, "init": "OLS",
               "chunk": CHUNK, "mode": "mixed", "n_iter": n_iter, "seconds": elapsed,
               "warnings": [str(w.message) for w in caught], "J_final_irls": Js[-1],
               "J_rel_drop_last_500": (Js[-501] - Js[-1]) / Js[-501],
               "metrics": m, "drift_last_500": drift, "trajectory": traj}
        np.save(OUT / f"reference_mae_lam{lam}.npy", W)
        save_json(f, res)
        print(f"  {n_iter} vòng, {elapsed:.0f} s; MAE val {m['mae_va']:.8f}, MSE val {m['mse_va']:.8f}; "
              f"trôi 500 vòng cuối: MAE val {drift['mae_va']:.2e}, MSE val {drift['mse_va']:.2e}", flush=True)


# ---------------------------------------------------------------------------
# A.3: quét λ, lạnh và warm
# ---------------------------------------------------------------------------

def sweep_name(loss, tol, patience):
    return OUT / f"sweep_{loss}_tol{tol:.0e}_p{patience}.json"


def step_sweep(cell, loss, tol, patience, force=False, extend=2):
    f = sweep_name(loss, tol, patience)
    if f.exists() and not force:
        print(f"bỏ qua: đã có {f.name}")
        return
    delta = LOSSES[loss][0]
    own = "mae_va" if loss == "mae" else "huber1_va"

    def one(lam, W0, mode):
        r, W, _ = run_irls(cell.Xg, cell.Yg, torch.tensor(W0, device=B.DEV), delta, tol, CHUNK,
                           lam=lam, pen=cell.Ninv if lam else None, patience=patience, quiet=True)
        r.pop("trace")
        r.pop("per_iter_seconds")
        m = metrics64(W, cell.d, delta)
        print(f"  [{loss} tol={tol:.0e} p={patience} {mode}] λ = {lam}: {r['n_iter']} vòng, {r['seconds']:.1f} s, "
              f"MAE val {m['mae_va']:.8f}, MSE val {m['mse_va']:.8f}", flush=True)
        return {"lam": lam, "mode": mode, **r, "metrics": m}, W

    lams = list(LAMS)
    rows = {"cold": [], "warm": []}
    W_warm, i, added = None, 0, 0
    while i < len(lams):
        lam = lams[i]
        rc, W_cold = one(lam, cell.cold_init(lam, delta), "cold")
        rows["cold"].append(rc)
        if lam == 0:
            rows["warm"].append({**rc, "mode": "warm"})
            W_warm = W_cold
        else:
            rw, W_warm = one(lam, W_warm, "warm")
            rows["warm"].append(rw)
        i += 1
        if i == len(lams) and added < extend:
            top = lams[-1]
            if any(argmin(rows[mo], key) == top for mo in rows for key in (own, "mse_va")):
                lams.append(B.next_lam(top))
                added += 1
    summ = {mo: {"total_iter": sum(r["n_iter"] for r in rows[mo]),
                 "total_seconds": sum(r["seconds"] for r in rows[mo]),
                 "best_lam_own": argmin(rows[mo], own), "best_lam_mse": argmin(rows[mo], "mse_va")}
            for mo in rows}
    save_json(f, {"cell": "ETTh1 H=96", "model": f"DLinear (pen = N⁻¹, k = {K})", "loss": loss,
                  "delta": delta, "tol": tol, "patience": patience, "chunk": CHUNK, "lams": lams,
                  "extended": added, "own_metric": own,
                  "cold_init": "ridge MSE dạng đóng, Pen = 2δλ·N⁻¹ (mục 1.5 NHAT_KY); λ = 0 là OLS",
                  "warm_init": "nghiệm IRLS của λ liền trước; λ = 0 chép từ lạnh",
                  "summary": summ, "rows": rows})


def argmin(rows, key, lams=None):
    rows = [r for r in rows if lams is None or r["lam"] in lams]
    return min(rows, key=lambda r: r["metrics"][key])["lam"]


# ---------------------------------------------------------------------------
# A.4: tiêu chí chấp nhận
# ---------------------------------------------------------------------------

def criteria(sw, ref):
    """Ba tiêu chí của A.4 cho một lần quét MAE. ref: {λ: metrics của nghiệm tham chiếu}."""
    cold = {r["lam"]: r["metrics"] for r in sw["rows"]["cold"]}
    warm = {r["lam"]: r["metrics"] for r in sw["rows"]["warm"]}
    ref_lams = sorted(ref)
    out = {}
    for key in ("mae_va", "mse_va"):
        # 1. λ* ổn định: lạnh = warm trên cả lưới; và trên các λ có tham chiếu, cả hai = tham chiếu
        lc, lw = argmin(sw["rows"]["cold"], key), argmin(sw["rows"]["warm"], key)
        lref = min(ref_lams, key=lambda l: ref[l][key])
        lc_r = argmin(sw["rows"]["cold"], key, ref_lams)
        lw_r = argmin(sw["rows"]["warm"], key, ref_lams)
        c1 = lc == lw and lc_r == lw_r == lref
        # 2. khoảng cách: ở mỗi kiểu, val(λ tốt thứ hai) − val(λ*) > max |warm − lạnh| ở hai λ đó
        c2_detail = {}
        for mode, vals in (("cold", cold), ("warm", warm)):
            order = sorted(vals, key=lambda l: vals[l][key])
            l1, l2 = order[0], order[1]
            gap = vals[l2][key] - vals[l1][key]
            noise = max(abs(warm[l][key] - cold[l][key]) for l in (l1, l2))
            c2_detail[mode] = {"lam_best": l1, "lam_second": l2, "gap": gap, "warm_minus_cold_max": noise,
                               "ok": gap > noise}
        c2 = all(v["ok"] for v in c2_detail.values())
        # 3. warm gần tham chiếu ở mọi λ có tham chiếu (cả MAE val và MSE val)
        dev = {l: warm[l][key] - ref[l][key] for l in ref_lams}
        c3 = all(abs(v) <= GAP_TOL for v in dev.values())
        out[key] = {"lam_cold": lc, "lam_warm": lw, "lam_ref": lref, "lam_cold_on_ref": lc_r,
                    "lam_warm_on_ref": lw_r, "c1": c1, "c2": c2, "c2_detail": c2_detail,
                    "warm_minus_ref": dev, "c3": c3}
    ok = {c: all(out[k][c] for k in out) for c in ("c1", "c2", "c3")}
    return out, ok


def step_analyze():
    ref = {}
    for lam in REF_LAMS:
        r = load_json(OUT / f"reference_mae_lam{lam}.json")
        if r:
            ref[lam] = r["metrics"]
    if len(ref) < len(REF_LAMS):
        sys.exit(f"thiếu nghiệm tham chiếu: có {sorted(ref)}")
    print("[tham chiếu] λ: MAE val, MSE val, Huber val (trôi 500 vòng cuối MAE val / MSE val)")
    for lam in REF_LAMS:
        r = load_json(OUT / f"reference_mae_lam{lam}.json")
        m, dr = r["metrics"], r["drift_last_500"]
        print(f"  {lam:>5}: {m['mae_va']:.8f} {m['mse_va']:.8f} {m['huber1_va']:.8f}  "
              f"({dr['mae_va']:.1e} / {dr['mse_va']:.1e}), {r['n_iter']} vòng, {r['seconds']:.0f} s")

    configs = []
    for tol, pat in MAE_CONFIGS:
        sw = load_json(sweep_name("mae", tol, pat))
        if not sw:
            print(f"thiếu sweep mae tol={tol:.0e} p={pat}")
            continue
        det, ok = criteria(sw, ref)
        s = sw["summary"]
        configs.append({"tol": tol, "patience": pat, "iters_warm": s["warm"]["total_iter"],
                        "iters_cold": s["cold"]["total_iter"], "seconds_warm": s["warm"]["total_seconds"],
                        "seconds_cold": s["cold"]["total_seconds"], "criteria": ok, "detail": det,
                        "pass": all(ok.values())})
        print(f"\n[mae tol={tol:.0e} p={pat}] vòng warm {s['warm']['total_iter']}, lạnh {s['cold']['total_iter']}; "
              f"tiêu chí 1/2/3: {ok['c1']}/{ok['c2']}/{ok['c3']}")
        for key, d in det.items():
            print(f"  {key}: λ* lạnh {d['lam_cold']}, warm {d['lam_warm']}, tham chiếu {d['lam_ref']} "
                  f"(trên λ tham chiếu: lạnh {d['lam_cold_on_ref']}, warm {d['lam_warm_on_ref']})")
            for mode, c in d["c2_detail"].items():
                print(f"    {mode}: λ* {c['lam_best']}, thứ hai {c['lam_second']}: chênh {c['gap']:.2e} "
                      f"vs |warm − lạnh| {c['warm_minus_cold_max']:.2e} → {c['ok']}")
            print("    warm − tham chiếu: " + ", ".join(f"{l}: {v:+.1e}" for l, v in d["warm_minus_ref"].items()))
    passed = [c for c in configs if c["pass"]]
    chosen = min(passed, key=lambda c: c["iters_warm"]) if passed else None
    huber = {}
    for tol, pat in HUBER_CONFIGS:
        sw = load_json(sweep_name("huber", tol, pat))
        if sw:
            huber[f"tol{tol:.0e}_p{pat}"] = {"summary": sw["summary"]}
            print(f"\n[huber tol={tol:.0e} p={pat}] " + "; ".join(
                f"{mo}: {v['total_iter']} vòng, λ* {v['best_lam_own']} / {v['best_lam_mse']}" for mo, v in sw["summary"].items()))
    print("\n→ cấu hình chọn: " + (f"tol = {chosen['tol']:.0e}, patience = {chosen['patience']}" if chosen
                                  else "KHÔNG cấu hình nào đạt"))
    save_json(OUT / "acceptance.json", {"reference_lams": REF_LAMS, "gap_tol": GAP_TOL, "configs": configs,
                                        "chosen": None if not chosen else {"tol": chosen["tol"], "patience": chosen["patience"]},
                                        "rule": "rẻ nhất theo tổng vòng warm trong các cấu hình đạt cả ba tiêu chí",
                                        "huber": huber})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["reference", "sweep", "analyze"], required=True)
    ap.add_argument("--loss", choices=list(LOSSES), default="mae")
    ap.add_argument("--tol", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--lam", type=float, nargs="*", help="reference: chỉ chạy các λ này")
    ap.add_argument("--all", action="store_true", help="sweep: mọi cấu hình của A.3")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "analyze":
        step_analyze()
    else:
        cell = Cell()
        if a.step == "reference":
            step_reference(cell, [int(x) for x in a.lam] if a.lam else REF_LAMS, a.force)
        elif a.all:
            for loss, cfgs in (("mae", MAE_CONFIGS), ("huber", HUBER_CONFIGS)):
                for tol, pat in cfgs:
                    step_sweep(cell, loss, tol, pat, a.force)
        else:
            step_sweep(cell, a.loss, a.tol, a.patience, a.force)
