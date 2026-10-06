"""Trình tối ưu toàn batch trên Linear, ETTh2, H = 720 (docs/NHIEM_VU_5.md, Phần A).

Ba cách, so với ridge λ* và Adam mini-batch 20 seed đã có (results/sgd_etth2_20seed/):
    gd_zero    GD toàn batch, lr = 1/λ_max, W = 0, b = 0 (tất định)
    gd_init    GD toàn batch, lr = 1/λ_max, khởi tạo nn.Linear mặc định (seed 2021–2030)
    adam_full  Adam toàn batch, lr chọn trong {0,05; 0,005; 0,0005} theo val của seed 2021, rồi seed 2021–2030

Hàm mất mát giống scripts/sgd_etth2.py: MSE trung bình trên mọi phần tử, có bias, J = ‖Y − Xt Wtᵀ‖² / (n·H),
với Xt = [X, 1], Wt = [W, b]. Mọi thứ tính bằng float64 trên CPU qua ma trận Gram G = Xtᵀ Xt, C = Xtᵀ Y:
gradient đúng của toàn batch là (2/(nH))(Wt G − Cᵀ), MSE = (‖Y‖² − 2 tr(Wt C) + tr(Wt G Wtᵀ)) / (nH).
Hessian theo mỗi hàng của Wt là 2G/(nH); λ_max tính bằng power iteration (kiểm chéo bằng eigvalsh).

Đánh giá MSE val mỗi k bước (GD: k = 20 trên tối đa 20 000 bước; Adam: k = 5 trên tối đa 5 000 bước), giữ Wt
ở bước có val tốt nhất, dừng khi val không cải thiện trong 25% số bước tối đa. Test chỉ ghi, không dùng để chọn.
MSE val/test của Wt tốt nhất tính lại bằng phần dư trực tiếp (như dạng đóng).

    python scripts/optim_etth2.py            # chạy (bỏ qua file đã có) rồi lập bảng
    python scripts/optim_etth2.py --report   # chỉ lập bảng

Kết quả: results/optim_etth2/{cách}_seed{s}.json (+ .npy: Wt tốt nhất [H, L + 1]), tables.md, summary.json.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import load_cell, save_json                   # noqa: E402  (tắt TF32 khi import)
from src.models import MODELS, effective_weights                # noqa: E402
from src.solvers import ridge_path                              # noqa: E402

OUT = ROOT / "results" / "optim_etth2"
MB_DIR = ROOT / "results" / "sgd_etth2_20seed"
DS, MODEL, L, H = "ETTh2", "Linear", 336, 720
SEEDS = list(range(2021, 2031))
ADAM_LRS = [0.05, 0.005, 0.0005]
CFG = {"gd": {"max_steps": 20_000, "k": 20}, "adam": {"max_steps": 5_000, "k": 5}}
DT = torch.float64


class Quad:
    """Thống kê Gram của một tập: MSE của Wt bất kỳ qua G, C."""

    def __init__(self, Xt, Y):
        Xt, Y = torch.as_tensor(Xt, dtype=DT), torch.as_tensor(Y, dtype=DT)
        self.G, self.C, self.yy, self.N = Xt.T @ Xt, Xt.T @ Y, float((Y ** 2).sum()), Y.numel()

    def mse(self, Wt):
        return float((self.yy - 2 * (Wt * self.C.T).sum() + ((Wt @ self.G) * Wt).sum()) / self.N)

    def grad(self, Wt):
        return (2.0 / self.N) * (Wt @ self.G - self.C.T)


def power_iteration(A, iters=1000, tol=1e-12, seed=0):
    v = torch.randn(A.shape[0], dtype=DT, generator=torch.Generator().manual_seed(seed))
    v /= v.norm()
    lam = 0.0
    for i in range(iters):
        w = A @ v
        lam_new = float(v @ w)
        v = w / w.norm()
        if abs(lam_new - lam) <= tol * abs(lam_new):
            return lam_new, i + 1
        lam = lam_new
    return lam, iters


def init_wt(seed):
    """Khởi tạo như LTSF-Linear: manual_seed(seed) rồi tạo mô hình (giống scripts/sgd_etth2.run_one)."""
    if seed is None:
        return torch.zeros(H, L + 1, dtype=DT)
    torch.manual_seed(seed)
    W, b = effective_weights(MODELS[MODEL](L, H))
    return torch.as_tensor(np.concatenate([W, b[:, None]], axis=1), dtype=DT)


def direct(Wt, Xt, Y):
    R = Y - Xt @ Wt.T
    return float((R ** 2).mean()), float(np.abs(R).mean())


def run(method, seed, lr, d, q, kind):
    max_steps, k = CFG[kind]["max_steps"], CFG[kind]["k"]
    patience = max_steps // 4
    Wt = init_wt(seed).clone().requires_grad_(kind == "adam")
    opt = torch.optim.Adam([Wt], lr=lr) if kind == "adam" else None
    traj, best = [], (float("inf"), -1, None)
    t0 = time.perf_counter()
    s = 0
    while True:
        if s % k == 0:
            with torch.no_grad():
                row = {"step": s, "train_mse": q["tr"].mse(Wt), "val_mse": q["va"].mse(Wt), "test_mse": q["te"].mse(Wt)}
            traj.append(row)
            if row["val_mse"] < best[0]:
                best = (row["val_mse"], s, Wt.detach().clone())
            if s - best[1] >= patience or s >= max_steps:
                break
        with torch.no_grad():
            g = q["tr"].grad(Wt)
        if kind == "gd":
            Wt = Wt - lr * g
        else:
            Wt.grad = g
            opt.step()
        s += 1
    sec = time.perf_counter() - t0
    Wb = best[2].numpy()
    res = {"dataset": DS, "H": H, "model": MODEL, "method": method, "seed": seed, "lr": lr, "optimizer": kind,
           "init": "W = 0, b = 0" if seed is None else "nn.Linear mặc định (manual_seed(seed) rồi tạo mô hình)",
           "dtype": "float64", "device": "cpu", "num_threads": torch.get_num_threads(),
           "max_steps": max_steps, "eval_every": k, "patience_steps": patience, "steps_run": s,
           "best_step": best[1], "best_at_last_step": best[1] == s, "stopped_by_patience": s - best[1] >= patience,
           "seconds": sec, "trajectory": traj}
    for tag, xs, ys in (("train", "Xt", "Y"), ("val", "Xt_va", "Y_va"), ("test", "Xt_te", "Y_te")):
        res[f"{tag}_mse"], res[f"{tag}_mae"] = direct(Wb, d[xs], d[ys])
    name = f"{method}_lr{lr:g}" if method == "adam_lrsel" else method
    stem = f"{name}_seed{seed if seed is not None else 'none'}"
    np.save(OUT / f"{stem}.npy", Wb)
    save_json(OUT / f"{stem}.json", res)
    print(f"  {stem}: bước tốt nhất {best[1]}/{s}, val {res['val_mse']:.6f}, test {res['test_mse']:.6f}, {sec:.0f} s",
          flush=True)
    return res


def lambda_grid():
    with (ROOT / "results" / "lambda_path.csv").open(encoding="utf-8", newline="") as f:
        lams = sorted({float(r["lam"]) for r in csv.DictReader(f)
                       if r["dataset"] == DS and int(r["H"]) == H and r["model"] == MODEL and r["objective"] == "MSE"})
    return [x for x in lams if x > 0]


def ridge_star():
    with (ROOT / "results" / "results.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r["dataset"] == DS and int(r["H"]) == H and r["model"] == MODEL and r["objective"] == "MSE"
                    and r["selection"] == "val_objective" and r["level"] == "lam_star"):
                return {"lam": float(r["lam"]), "val_mse": float(r["val_mse"]), "test_mse": float(r["test_mse"])}


def load(stem):
    f = OUT / f"{stem}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def run_all(d, q, lr_gd):
    def get(method, seed, lr, kind):
        stem = f"{method}_lr{lr:g}" if method == "adam_lrsel" else method
        r = load(f"{stem}_seed{seed if seed is not None else 'none'}")
        return r if r is not None else run(method, seed, lr, d, q, kind)

    gz, star = get("gd_zero", None, lr_gd, "gd"), ridge_star()
    if gz["val_mse"] < star["val_mse"] - 0.005:          # NHIEM_VU_5 A.2 và mục 5: dừng, kiểm lại code
        sys.exit(f"DỪNG: gd_zero val {gz['val_mse']:.6f} tốt hơn ridge λ* {star['val_mse']:.6f} quá 0,005")
    sel = {lr: get("adam_lrsel", 2021, lr, "adam")["val_mse"] for lr in ADAM_LRS}
    lr_adam = min(sel, key=sel.get)
    print(f"Adam toàn batch, seed 2021: val tốt nhất theo lr {sel} → chọn lr = {lr_adam}")
    for seed in SEEDS:
        get("gd_init", seed, lr_gd, "gd")
        get("adam_full", seed, lr_adam, "adam")
    return sel, lr_adam


# ---------------------------------------------------------------------------
# bảng
# ---------------------------------------------------------------------------

def ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def side(m, h):
    """thắng / thua / hòa theo CI 95% của Δ = SGD − dạng đóng (docs/NHIEM_VU_5.md, B.4)."""
    if m + h < 0:
        return "thắng"
    if m - h > 0:
        return "thua"
    return "hòa"


def label(v, t):
    if v == "thắng" and t == "thắng":
        return "SGD tốt hơn thật"
    if v == "thua" and t == "thua":
        return "dạng đóng tốt hơn thật"
    if v == "hòa" and t != "hòa":
        return "khác biệt do val khác test"
    if v == "thắng":
        return "SGD có dấu hiệu khớp val quá mức"
    if v == "thua":
        return "dạng đóng có dấu hiệu khớp val quá mức"
    return "không phân biệt được"


def num(x, p=4):
    return f"{x:.{p}f}".replace(".", ",") if x >= 0 else f"−{abs(x):.{p}f}".replace(".", ",")


def report(d, lam_info, sel, lr_adam):
    star = ridge_star()
    md, summ = [], {"lambda_max": lam_info, "adam_lr_selection_seed2021": {str(k): v for k, v in sel.items()},
                    "adam_lr": lr_adam, "ridge_star": star}

    # ---------- A.2: gd_zero so với đường ridge ----------
    gz = load("gd_zero_seednone")
    Wg = np.load(OUT / "gd_zero_seednone.npy")[:, :-1]
    X, Y = d["X"], d["Y"]
    W_at = ridge_path(X - X.mean(0), Y - Y.mean(0))
    lams = lambda_grid()
    rel = [float(np.linalg.norm(Wg - W_at(lam)) / np.linalg.norm(W_at(lam))) for lam in lams]
    i = int(np.argmin(rel))
    # kiểm: ridge λ* của ridge_path cho lại val/test của results.csv
    Ws = W_at(star["lam"])
    bs = Y.mean(0) - Ws @ X.mean(0)
    Wst = np.concatenate([Ws, bs[:, None]], axis=1)
    chk = {"val": direct(Wst, d["Xt_va"], d["Y_va"])[0], "test": direct(Wst, d["Xt_te"], d["Y_te"])[0]}
    rel_star = float(np.linalg.norm(Wg - Ws) / np.linalg.norm(Ws))
    summ["a2"] = {"gd_zero_val": gz["val_mse"], "gd_zero_test": gz["test_mse"], "gd_zero_best_step": gz["best_step"],
                  "d_val_vs_star": gz["val_mse"] - star["val_mse"], "d_test_vs_star": gz["test_mse"] - star["test_mse"],
                  "min_rel_dist": rel[i], "lam_at_min_rel_dist": lams[i], "n_lams": len(lams),
                  "rel_dist_at_lam_star": rel_star, "ridge_star_recomputed": chk,
                  "gd_zero_beats_star_val_by_more_than_0.005": gz["val_mse"] < star["val_mse"] - 0.005,
                  "rel_dist_path": [{"lam": a, "rel": b} for a, b in zip(lams, rel)]}
    md.append("## A.2. `gd_zero` so với đường ridge\n")
    md.append(f"- λ_max của Hessian 2G/(nH): power iteration {lam_info['power']:.6g} ({lam_info['power_iters']} vòng), "
              f"eigvalsh {lam_info['eigvalsh']:.6g}; lr của GD = 1/λ_max = {lam_info['lr']:.6g}.")
    md.append(f"- Ridge λ* = {star['lam']:.4g}: val {num(star['val_mse'])}, test {num(star['test_mse'])} "
              f"(tính lại bằng ridge_path: {num(chk['val'], 6)} / {num(chk['test'], 6)}).")
    md.append(f"- `gd_zero`, bước tốt nhất {gz['best_step']}: val {num(gz['val_mse'])}, test {num(gz['test_mse'])}; "
              f"Δ so với λ*: val {num(summ['a2']['d_val_vs_star'])}, test {num(summ['a2']['d_test_vs_star'])}.")
    md.append(f"- ‖W_gd − W_ridge(λ)‖ / ‖W_ridge(λ)‖ (chỉ phần W, không gồm bias) nhỏ nhất trên lưới {len(lams)} giá trị: "
              f"{num(rel[i])} tại λ = {lams[i]:.4g}; tại λ*: {num(rel_star)}.\n")

    # ---------- A.3 ----------
    mb = sorted((json.loads(f.read_text(encoding="utf-8")) for f in MB_DIR.glob(f"{MODEL}_H{H}_seed*.json")),
                key=lambda r: r["seed"])
    rows = {"gd_zero": [gz], "gd_init": [load(f"gd_init_seed{s}") for s in SEEDS],
            "adam_full": [load(f"adam_full_seed{s}") for s in SEEDS], "Adam mini-batch": mb}
    md.append("## A.3. Bảng\n")
    md.append(f"Δ = cách − ridge λ*. CI 95% theo t với n − 1 bậc tự do. `gd_zero` tất định (n = 1): Δ là một số, "
              f"nhãn theo dấu của Δ. lr của Adam toàn batch: {lr_adam:g} (seed 2021, val tốt nhất: "
              + ", ".join(f"lr {k:g} → {num(v, 6)}" for k, v in sel.items()) + ").\n")
    md.append("| Cách | n | MSE val (TB ± sd) | MSE test (TB ± sd) | Δ val so với λ* [CI] | Δ test so với λ* [CI] | "
              "bước tốt nhất (TB; min–max) | dừng ở bước cuối | nhãn |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    md.append(f"| ridge λ* (λ = {star['lam']:.4g}) | — | {num(star['val_mse'])} | {num(star['test_mse'])} | — | — | — | — | — |")
    summ["a3"] = {}
    for name, rs in rows.items():
        v = np.array([r["val_mse"] for r in rs])
        t = np.array([r["test_mse"] for r in rs])
        if name == "Adam mini-batch":
            steps = np.array([r["best_epoch"] * r["n_train_batches"] for r in rs])
            last = "—"
        else:
            steps = np.array([r["best_step"] for r in rs])
            last = f"{sum(r['best_at_last_step'] for r in rs)}/{len(rs)}"
        mv, hv = ci(v - star["val_mse"])
        mt, ht = ci(t - star["test_mse"])
        sv, st = side(mv, hv), side(mt, ht)
        lab = label(sv, st)
        sd = (lambda x: f" ± {num(x.std(ddof=1))}" if len(x) > 1 else "")
        cis = (lambda m, h: f"{num(m)} [{num(m - h)}; {num(m + h)}]" if len(rs) > 1 else num(m))
        md.append(f"| {'`' + name + '`' if name != 'Adam mini-batch' else name + ' (20 seed, đã có)'} | {len(rs)} | "
                  f"{num(v.mean())}{sd(v)} | {num(t.mean())}{sd(t)} | {cis(mv, hv)} ({sv}) | {cis(mt, ht)} ({st}) | "
                  f"{steps.mean():.0f}; {steps.min()}–{steps.max()} | {last} | {lab} |")
        summ["a3"][name] = {"n": len(rs), "val_mean": float(v.mean()), "test_mean": float(t.mean()),
                            "val_sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                            "test_sd": float(t.std(ddof=1)) if len(t) > 1 else None,
                            "d_val": mv, "d_val_ci": hv, "d_test": mt, "d_test_ci": ht, "val_side": sv,
                            "test_side": st, "label": lab, "best_step_mean": float(steps.mean()),
                            "best_step_min": int(steps.min()), "best_step_max": int(steps.max())}
    md.append("\nBước tốt nhất của Adam mini-batch = epoch tốt nhất × số batch mỗi epoch (bước ở cuối epoch đó).\n")
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_json(OUT / "summary.json", summ)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="chỉ lập bảng từ file đã có")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cell(DS, H)
    q = {"tr": Quad(d["Xt"], d["Y"]), "va": Quad(d["Xt_va"], d["Y_va"]), "te": Quad(d["Xt_te"], d["Y_te"])}
    Hs = 2.0 * q["tr"].G / q["tr"].N
    lam_pow, iters = power_iteration(Hs)
    lam_eig = float(torch.linalg.eigvalsh(Hs)[-1])
    lam_info = {"power": lam_pow, "power_iters": iters, "eigvalsh": lam_eig, "lr": 1.0 / lam_pow,
                "hessian": "2·Xtᵀ Xt / (n·H), Xt = [X, 1], n·H = số phần tử của Y train"}
    print(f"λ_max: power iteration {lam_pow:.8g} ({iters} vòng), eigvalsh {lam_eig:.8g}; lr GD = {1 / lam_pow:.6g}")
    if args.report:
        sel = {lr: load(f"adam_lrsel_lr{lr:g}_seed2021")["val_mse"] for lr in ADAM_LRS}
        lr_adam = min(sel, key=sel.get)
    else:
        sel, lr_adam = run_all(d, q, 1.0 / lam_pow)
    report(d, lam_info, sel, lr_adam)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
