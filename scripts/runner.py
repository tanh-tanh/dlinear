"""Chạy grid: đường λ của Linear, DLinear, NLinear × MSE, MAE, Huber × ETTh1, ETTh2, ETTm1 × H.

Không chọn λ* ở đây: mỗi (dataset, H, mô hình, mục tiêu, λ) là một dòng của results/lambda_path.csv.
Chọn λ* bằng scripts/select_lambda.py.

    python scripts/runner.py                                   # toàn bộ grid; chạy lại = chạy tiếp
    python scripts/runner.py --data ETTh1 --H 96               # lọc
    python scripts/runner.py --dry-run                         # danh sách việc và ước lượng thời gian

Cách giải
    MSE: dạng đóng trên λ ∈ {0} ∪ logspace(−2, 5, 141) (20 giá trị mỗi bậc; lưới cũ 36 giá trị là tập con). Linear: ridge; DLinear: ridge với phạt λ·N⁻¹;
        NLinear: ridge trên chuỗi đã trừ giá trị cuối (nlinear_transform), bằng NLinear ràng buộc với
        phạt diag(1, …, 1, 0) (thế w_L = 1 − Σ_{j<L} w_j); λ = 0 là nlinear_constrained.
    MAE, Huber: irls chế độ pha, chunk 16, λ ∈ IRLS_LAMS tăng dần, warm start; λ = 0 khởi tạo từ
        nghiệm MSE λ = 0 của cùng mô hình. Tự mở rộng tối đa 2 giá trị (×3 ở đầu trên, ÷3 ở đầu dưới)
        nếu λ* theo bất kỳ metric val nào nằm ở đầu mút.
    DLinear ở λ = 0 ≡ Linear với mọi mục tiêu: không giải, chép dòng của Linear (derived_from = Linear).

Phạt: Linear pen = I; DLinear pen = N⁻¹ (k = 25); NLinear constrained=True, pen = diag(1, …, 1, 0): phạt
L − 1 hệ số đầu của W_eff, không phạt lag cuối. Đây là weight decay thật của NLinear (Linear trên x − x_L:
trọng số của x_L − x_L = 0 không ảnh hưởng dự báo nên về 0), giống W_nlinear_wd của notebook 01.
Không phạt bias. Quy ước λ ghi trong cột lam_convention.

Chạy tiếp: khóa (dataset, H, model, objective, lam); W mỗi dòng ở results/weights/*.npy (float64).
"""
import argparse
import csv
import datetime
import json
import os
import sys

# Chống phân mảnh bộ nhớ CUDA: ở ETTm1 H = 720, cấp phát thật ~11,6 GiB nhưng dự trữ tới 15,2 GiB, cộng
# màn hình là tràn VRAM sang RAM hệ thống (chạy chậm hẳn). Phải đặt trước khi import torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_5060ti as B                    # noqa: E402  (tắt TF32 khi import)
from bench_5060ti import HUBER_DELTA, MAE_DELTA, ROOT, git_commit, load_cell  # noqa: E402
from src.operators import build_N, build_P, nlinear_effective, nlinear_transform  # noqa: E402
from src.solvers import fit_with_bias, irls, nlinear_constrained, ols  # noqa: E402

L, K = 336, 25
DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["Linear", "DLinear", "NLinear"]
OBJECTIVES = ["MSE", "MAE", "Huber"]
MSE_LAMS = [0.0] + [float(f"{x:.6g}") for x in np.logspace(-2, 5, 141)]   # NHIEM_VU_3 C.2; cũ: 36 giá trị
IRLS_LAMS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
MAX_EXTEND = 2
CHUNK = 16
MAX_ITER = 1000
# Tiêu chí dừng theo docs/BAO_CAO_TIEU_CHI_DUNG.md (Phần A của NHIEM_VU_2, phương án a′): λ = 0 khởi tạo
# từ nghiệm MSE, mặt MAE phẳng nhất ở đó nên cần tol chặt hơn (tol_lam0); λ > 0 warm start.
STOP = {"MAE": {"tol": 1e-10, "patience": 1, "tol_lam0": 3e-11},
        "Huber": {"tol": 1e-9, "patience": 1, "tol_lam0": 1e-9}}
DELTAS = {"MSE": None, "MAE": MAE_DELTA, "Huber": HUBER_DELTA}
LAM_CONVENTION = {
    "MSE": "Σr² + λ·Σ_h w_hᵀPen w_h (tổng, không phạt bias)",
    "MAE": "Σ|r| + λ·Σ_h w_hᵀPen w_h (irls δ = 1e-6, tổng, không phạt bias)",
    "Huber": "Σρ_δ(r)/δ + λ·Σ_h w_hᵀPen w_h, δ = 1, ρ = r²/2 trong ngưỡng (tổng, không phạt bias)",
}
PATH_CSV = ROOT / "results" / "lambda_path.csv"
WEIGHTS = ROOT / "results" / "weights"
COLUMNS = ["dataset", "H", "model", "objective", "lam", "lam_convention", "delta", "k",
           "train_obj", "val_mse", "val_mae", "val_huber", "test_mse", "test_mae",
           "n_iter", "converged", "seconds", "init", "derived_from",
           "tol", "patience", "chunk", "gram_dtype", "gpu", "torch_version", "git_commit", "timestamp",
           "n_fallback"]


def lam_key(lam):
    return f"{float(lam):.6g}"


def weight_path(ds, H, model, obj, lam):
    return WEIGHTS / f"{ds}_H{H}_{model}_{obj}_lam{lam_key(lam)}.npy"


# ---------------------------------------------------------------------------
# bảng kết quả: đọc để chạy tiếp, ghi từng dòng
# ---------------------------------------------------------------------------

def read_done():
    if not PATH_CSV.exists():
        return {}
    with PATH_CSV.open(encoding="utf-8", newline="") as f:
        return {(r["dataset"], int(r["H"]), r["model"], r["objective"], lam_key(r["lam"])): r
                for r in csv.DictReader(f)}


class Writer:
    def __init__(self):
        PATH_CSV.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        new = not PATH_CSV.exists()
        # file đã có: giữ đúng header của nó (lambda_path.csv chưa có cột n_fallback), bỏ cột thừa
        fields = COLUMNS
        if not new:
            with PATH_CSV.open(encoding="utf-8", newline="") as f:
                fields = next(csv.reader(f))
        self.f = PATH_CSV.open("a", encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.env = {"gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                    "git_commit": git_commit(), "chunk": CHUNK}

    def write(self, row, W):
        np.save(weight_path(row["dataset"], row["H"], row["model"], row["objective"], row["lam"]), W)
        row = {**{c: "" for c in COLUMNS}, **self.env, **row,
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
        self.w.writerow(row)
        self.f.flush()


# ---------------------------------------------------------------------------
# một ô (dataset, H)
# ---------------------------------------------------------------------------

class Cell:
    def __init__(self, ds, H):
        self.ds, self.H = ds, H
        self.d = load_cell(ds, H)
        self.n = len(self.d["X"])
        # ô lớn (chỉ ETTm1 H = 720): chunk nhỏ hơn để khối einsum ~2,4 GB thay vì ~4,7 GB, tránh tràn VRAM
        self.chunk = CHUNK if self.n * H <= 1e8 else CHUNK // 2
        P = build_P(L, K)
        self.Ninv = np.linalg.inv(build_N(P))
        self.pens = {"Linear": np.eye(L), "DLinear": self.Ninv, "NLinear": np.diag(np.r_[np.ones(L - 1), 0.0])}
        self._gpu = None
        # MSE dạng đóng trên dữ liệu đã trừ trung bình
        X, Y = self.d["X"], self.d["Y"]
        self.x_bar, self.y_bar = X.mean(0), Y.mean(0)
        Xc = X - self.x_bar
        self.A = Xc.T @ Xc
        self.C = Xc.T @ (Y - self.y_bar)
        # NLinear: X_n = (X − x_L)[:, :L−1], Y_n = Y − x_L, đã trừ trung bình
        Xn, Yn, _ = nlinear_transform(X, Y)
        self.xn_bar, self.yn_bar = Xn.mean(0), Yn.mean(0)
        Xn = Xn - self.xn_bar
        self.An = Xn.T @ Xn
        self.Cn = Xn.T @ (Yn - self.yn_bar)

    @property
    def gpu(self):
        if self._gpu is None:
            Xg, Yg = B.to_gpu(self.d["Xt"], self.d["Y"])
            B.warmup(Xg, Yg, torch.zeros(self.H, L + 1, dtype=torch.float64, device=B.DEV), self.chunk)
            self._gpu = (Xg, Yg)
        return self._gpu

    def free(self):
        self._gpu = None
        torch.cuda.empty_cache()

    def closed_form(self, model, lam):
        """MSE dạng đóng; trả W [H, L + 1] (W_eff, bias ở cột cuối)."""
        if model == "NLinear" and lam == 0:
            W, b = fit_with_bias(nlinear_constrained, self.d["X"], self.d["Y"])
        elif model == "NLinear":
            # dự báo = X_n W_n + b + x_L = W_eff x + b, với W_eff = [W_n, 1 − W_n 1]; phạt ‖W_n‖²
            Wn = np.linalg.solve(self.An + lam * np.eye(L - 1), self.Cn).T
            b = self.yn_bar - Wn @ self.xn_bar
            W = nlinear_effective(Wn)
        elif lam == 0:
            W, b = fit_with_bias(ols, self.d["X"], self.d["Y"])
        else:
            W = np.linalg.solve(self.A + lam * self.pens[model], self.C).T
            b = self.y_bar - W @ self.x_bar
        return np.concatenate([W, b[:, None]], axis=1)

    def evaluate(self, W, model, obj, lam):
        """Sai số float64 trên CPU. train_obj = hàm mục tiêu dạng tổng (có phạt) chia n·H."""
        d = self.d
        R = d["Y"] - d["Xt"] @ W.T
        if obj == "MSE":
            loss = (R ** 2).sum()
        else:
            delta = DELTAS[obj]
            a = np.abs(R)
            q = np.minimum(a, delta)
            loss = (q * (a - q / 2)).sum() / delta
        pen = lam * np.einsum("hi,ij,hj->", W[:, :-1], self.pens[model], W[:, :-1]) if lam else 0.0
        out = {"train_obj": (loss + pen) / R.size}
        for tag, name in (("va", "val"), ("te", "test")):
            Rs = d[f"Y_{tag}"] - d[f"Xt_{tag}"] @ W.T
            out[f"{name}_mse"] = float((Rs ** 2).mean())
            out[f"{name}_mae"] = float(np.abs(Rs).mean())
            if name == "val":
                out["val_huber"] = B.huber_np(Rs, HUBER_DELTA)
        return out

    def run_irls(self, model, W0, delta, lam_irls, tol, patience, max_iter=MAX_ITER):
        Xg, Yg = self.gpu
        t0 = B.sync_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            W, n = irls(Xg, Yg, delta, torch.as_tensor(W0, device=B.DEV), lam=lam_irls,
                        pen=self.pens[model] if lam_irls else None, constrained=(model == "NLinear"),
                        max_iter=max_iter, tol=tol, patience=patience, gram_dtype=torch.float32,
                        chunk=self.chunk)
        return W.cpu().numpy(), n, not caught, B.sync_time() - t0


# ---------------------------------------------------------------------------
# đường λ
# ---------------------------------------------------------------------------

def base_row(cell, model, obj, lam):
    return {"dataset": cell.ds, "H": cell.H, "model": model, "objective": obj, "lam": lam,
            "lam_convention": LAM_CONVENTION[obj], "delta": DELTAS[obj] if DELTAS[obj] else "",
            "k": K if model == "DLinear" else ""}


def derive_dlinear_lam0(cell, obj, done, writer):
    """DLinear λ = 0 ≡ Linear: chép dòng (và W) của Linear."""
    key = (cell.ds, cell.H, "DLinear", obj, lam_key(0))
    if key in done:
        return
    src = done[(cell.ds, cell.H, "Linear", obj, lam_key(0))]
    W = np.load(weight_path(cell.ds, cell.H, "Linear", obj, 0))
    row = {c: src[c] for c in COLUMNS if c in src}
    row.update(model="DLinear", k=K, derived_from="Linear", seconds=0)
    writer.write(row, W)
    done[key] = row


def run_mse(cell, model, done, writer):
    for lam in MSE_LAMS:
        key = (cell.ds, cell.H, model, "MSE", lam_key(lam))
        if key in done:
            continue
        if model == "DLinear" and lam == 0:
            derive_dlinear_lam0(cell, "MSE", done, writer)
            continue
        row = base_row(cell, model, "MSE", lam)
        t0 = time.perf_counter()
        W = cell.closed_form(model, lam)
        row.update(init="closed_form", n_iter=0, converged=True, seconds=time.perf_counter() - t0,
                   gram_dtype="float64")
        row.update(cell.evaluate(W, model, "MSE", lam))
        writer.write(row, W)
        done[key] = row
    print(f"  {cell.ds} H={cell.H} {model} MSE: xong", flush=True)


def lam_grid(cell, model, obj, done):
    """Lưới λ hiện tại của một đường: lưới gốc cộng các giá trị mở rộng đã chạy."""
    have = sorted(float(k[4]) for k in done if k[:4] == (cell.ds, cell.H, model, obj))
    return sorted(set(IRLS_LAMS) | set(have))


def edge_extension(cell, model, obj, done):
    """λ mới cần thêm (hoặc None) nếu λ* theo một metric val nằm ở đầu mút lưới."""
    lams = lam_grid(cell, model, obj, done)
    rows = [done.get((cell.ds, cell.H, model, obj, lam_key(l))) for l in lams]
    if any(r is None for r in rows):
        return None
    n_ext = len(lams) - len(IRLS_LAMS)
    if n_ext >= MAX_EXTEND:
        return None
    for metric in ("val_mse", "val_mae", "val_huber"):
        best = min(zip(lams, rows), key=lambda t: float(t[1][metric]))[0]
        if best == lams[-1]:
            return lams[-1] * 3
        if best == lams[0]:
            pos = [l for l in lams if l > 0]
            return pos[0] / 3
    return None


def run_irls_path(cell, model, obj, done, writer):
    delta = DELTAS[obj]
    tol, patience = STOP[obj]["tol"], STOP[obj]["patience"]

    def one(lam):
        key = (cell.ds, cell.H, model, obj, lam_key(lam))
        if key in done:
            return
        if model == "DLinear" and lam == 0:
            derive_dlinear_lam0(cell, obj, done, writer)
            return
        lams = lam_grid(cell, model, obj, done)
        below = [l for l in lams if l < lam and (cell.ds, cell.H, model, obj, lam_key(l)) in done]
        if lam == 0:
            W0, init = np.load(weight_path(cell.ds, cell.H, model, "MSE", 0)), "ols"
        else:
            W0, init = np.load(weight_path(cell.ds, cell.H, model, obj, max(below))), "warm"
        t = STOP[obj]["tol_lam0"] if lam == 0 else tol
        W, n, conv, sec = cell.run_irls(model, W0, delta, lam, t, patience)
        row = base_row(cell, model, obj, lam)
        row.update(init=init, n_iter=n, converged=conv, seconds=sec, tol=t, patience=patience, chunk=cell.chunk,
                   gram_dtype="float32", n_fallback=irls.last_fallbacks, **cell.evaluate(W, model, obj, lam))
        writer.write(row, W)
        done[key] = row
        print(f"  {cell.ds} H={cell.H} {model} {obj} λ = {lam:g}: {n} vòng, {sec:.1f} s, "
              f"val MSE {row['val_mse']:.6f} MAE {row['val_mae']:.6f}"
              + (f", fallback float64 {irls.last_fallbacks} bước h" if irls.last_fallbacks else ""), flush=True)

    for lam in IRLS_LAMS:
        one(lam)
    while (lam := edge_extension(cell, model, obj, done)) is not None:
        print(f"  λ* ở đầu mút: mở rộng λ = {lam:g}", flush=True)
        # mở rộng ở đầu dưới chèn vào giữa 0 và λ dương nhỏ nhất: warm từ λ lớn nhất nhỏ hơn nó (λ = 0)
        one(lam)


# ---------------------------------------------------------------------------
# lập kế hoạch, ước lượng
# ---------------------------------------------------------------------------

def plan(args):
    return [(ds, H, m, o) for ds in DATASETS if ds in args.data for H in HORIZONS if H in args.H
            for o in OBJECTIVES if o in args.objective for m in MODELS if m in args.model]


def estimate(jobs, done):
    """Giây ước lượng cho các việc chưa xong, từ số đo của docs/BAO_CAO_5060TI.md và BAO_CAO_TIEU_CHI_DUNG.md."""
    s_iter = {}
    for ds in DATASETS:
        for H in HORIZONS:
            c = B.load_json(f"cell_{ds}_H{H}")
            s_iter[(ds, H)] = c["s_per_iter_median"] if c else None
    iters = {}
    for obj in ("MAE", "Huber"):
        tol, pat, tol0 = STOP[obj]["tol"], STOP[obj]["patience"], STOP[obj]["tol_lam0"]
        suffix = f"_lam0tol{tol0:.0e}" if tol0 != tol else ""
        f = ROOT / "results" / "stopping" / f"sweep_{obj.lower()}_tol{tol:.0e}_p{pat}{suffix}.json"
        sw = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        iters[obj] = sw["summary"]["warm"]["total_iter"] if sw else None
    total, lines = 0.0, []
    for ds, H, m, o in jobs:
        n_left = sum((ds, H, m, o, lam_key(l)) not in done for l in (MSE_LAMS if o == "MSE" else IRLS_LAMS))
        if not n_left:
            continue
        s = s_iter[(ds, H)] or 0.0
        if o == "MSE":
            sec = n_left * 0.05                                  # dạng đóng
        else:
            sec = iters[o] * s * n_left / len(IRLS_LAMS)
            if m == "DLinear":
                sec *= (len(IRLS_LAMS) - 1) / len(IRLS_LAMS)
        total += sec
        lines.append((ds, H, m, o, n_left, sec))
    return total, lines, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--H", nargs="+", type=int, default=HORIZONS, choices=HORIZONS)
    ap.add_argument("--model", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--objective", nargs="+", default=OBJECTIVES, choices=OBJECTIVES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if any(v is None for s in STOP.values() for v in s.values()):
        sys.exit("STOP chưa điền: cần kết quả Phần A (docs/BAO_CAO_TIEU_CHI_DUNG.md)")
    # DLinear λ = 0 chép từ Linear, MAE/Huber λ = 0 khởi tạo từ MSE λ = 0 của cùng mô hình
    if "DLinear" in args.model and "Linear" not in args.model:
        args.model = ["Linear"] + args.model
    if set(args.objective) & {"MAE", "Huber"} and "MSE" not in args.objective:
        args.objective = ["MSE"] + args.objective
    args.objective = [o for o in OBJECTIVES if o in args.objective]
    args.model = [m for m in MODELS if m in args.model]
    jobs = plan(args)
    done = read_done()
    total, lines, iters = estimate(jobs, done)
    if args.dry_run:
        print(f"{len(jobs)} đường λ, {len(lines)} chưa xong; số vòng mỗi đường (warm, cả lưới): {iters}")
        per = {}
        for ds, H, m, o, n_left, sec in lines:
            per[(ds, H)] = per.get((ds, H), 0) + sec
            per[ds] = per.get(ds, 0) + sec
            per[o] = per.get(o, 0) + sec
        for ds in DATASETS:
            if ds in per:
                print(f"  {ds}: {per[ds] / 3600:.2f} h  (" + ", ".join(
                    f"H={H} {per[(ds, H)] / 60:.0f} phút" for H in HORIZONS if (ds, H) in per) + ")")
        print("  theo mục tiêu: " + ", ".join(f"{o} {per[o] / 3600:.2f} h" for o in OBJECTIVES if o in per))
        print(f"  tổng ước lượng: {total / 3600:.1f} giờ (chưa tính nạp dữ liệu, OLS, mở rộng lưới)")
        return

    writer = Writer()
    print(f"ước lượng {total / 3600:.1f} giờ cho phần chưa xong", flush=True)
    for ds in DATASETS:
        for H in HORIZONS:
            cell_jobs = [(m, o) for d_, h_, m, o in jobs if (d_, h_) == (ds, H)]
            if not cell_jobs:
                continue
            print(f"[{ds} H={H}]", flush=True)
            cell = Cell(ds, H)
            for obj in OBJECTIVES:
                for model in MODELS:
                    if (model, obj) not in cell_jobs:
                        continue
                    if obj == "MSE":
                        run_mse(cell, model, done, writer)
                    else:
                        run_irls_path(cell, model, obj, done, writer)
            cell.free()
            del cell
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
