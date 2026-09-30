"""Nhiệm vụ 2, Phần B: các phép kiểm trên dữ liệu thật (ETTh1, L = 336, H = 96), chế độ pha, chunk 16.

    python scripts/checks_mae.py --check unique        # B.1: MAE từ ba điểm khởi tạo có cùng nghiệm?
    python scripts/checks_mae.py --check dlinear       # B.2: DLinear 2L chiều ≡ L chiều với pen = N⁻¹

Kết quả: results/checks_mae/*.json. Test trên dữ liệu giả của cùng các phép kiểm ở tests/test_irls.py.
"""
import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_5060ti as B                    # noqa: E402  (tắt TF32 khi import)
from bench_5060ti import MAE_DELTA, ROOT, load_cell, metrics64, ols_init, save_json, to_gpu, warmup  # noqa: E402
from src.operators import build_N, build_P, dlinear_effective, make_Z  # noqa: E402
from src.solvers import huber_objective, irls, weight_gram  # noqa: E402

OUT = ROOT / "results" / "checks_mae"
CHUNK = 16
SEED = 0


def solve(Xg, Yg, W0, delta, tol, patience, **kw):
    """irls chế độ pha, bắt cảnh báo; trả (W numpy, số vòng, J cuối của irls, giây, cảnh báo)."""
    Js = []
    t0 = B.sync_time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        W, n = irls(Xg, Yg, delta, torch.as_tensor(W0, device=B.DEV), tol=tol, patience=patience,
                    callback=lambda it, W, J: Js.append(J.item()), gram_dtype=torch.float32, chunk=CHUNK, **kw)
    return W.cpu().numpy(), n, Js[-1], B.sync_time() - t0, [str(w.message) for w in caught]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# B.1
# ---------------------------------------------------------------------------

def check_unique(tol=1e-10, patience=5):
    d = load_cell("ETTh1", 96)
    Xg, Yg = to_gpu(d["Xt"], d["Y"])
    W_ols = ols_init(d["X"], d["Y"])
    warmup(Xg, Yg, torch.tensor(W_ols, device=B.DEV), CHUNK)
    rng = np.random.default_rng(SEED)
    # nhiễu Gauss, độ lệch chuẩn mỗi phần tử = 10% chuẩn của hàng tương ứng của W_OLS
    noise = rng.standard_normal(W_ols.shape) * 0.1 * np.linalg.norm(W_ols, axis=1, keepdims=True)
    inits = {"ols": W_ols, "zero": np.zeros_like(W_ols), "ols_noise": W_ols + noise}
    runs, Ws = {}, {}
    for name, W0 in inits.items():
        print(f"[unique] khởi tạo {name}", flush=True)
        W, n, J, sec, warn = solve(Xg, Yg, W0, MAE_DELTA, tol, patience)
        m = metrics64(W, d, MAE_DELTA)
        runs[name] = {"n_iter": n, "seconds": sec, "warnings": warn, "J_final_irls": J, "metrics": m,
                      "init_rel_dist_to_ols": rel(W0, W_ols) if name != "ols" else 0.0}
        Ws[name] = W
        np.save(OUT / f"unique_{name}.npy", W)
        print(f"  {n} vòng, {sec:.0f} s, J {m['J_train']:.12e}, MAE val {m['mae_va']:.8f}, MSE val {m['mse_va']:.8f}",
              flush=True)
    pairs = {}
    for a, b in itertools.combinations(inits, 2):
        ma, mb = runs[a]["metrics"], runs[b]["metrics"]
        pairs[f"{a}|{b}"] = {
            "J_rel": (ma["J_train"] - mb["J_train"]) / mb["J_train"],
            "W_rel": rel(Ws[a], Ws[b]),
            **{f"d_{k}": ma[k] - mb[k] for k in ("mse_va", "mae_va", "mse_te", "mae_te")}}
    max_J = max(abs(p["J_rel"]) for p in pairs.values())
    save_json(OUT / "unique.json", {
        "cell": "ETTh1 H=96", "model": "Linear", "lam": 0, "loss": "mae", "delta": MAE_DELTA,
        "tol": tol, "patience": patience, "chunk": CHUNK, "mode": "mixed", "seed": SEED,
        "noise": "Gauss, độ lệch chuẩn mỗi phần tử = 0,1·‖hàng h của W_OLS‖",
        "runs": runs, "pairs": pairs, "max_abs_J_rel": max_J, "J_within_1e-8": max_J <= 1e-8})
    for k, p in pairs.items():
        print(f"  {k}: J tương đối {p['J_rel']:+.2e}, ‖ΔW‖/‖W‖ {p['W_rel']:.2e}, ΔMAE val {p['d_mae_va']:+.2e}, "
              f"ΔMSE val {p['d_mse_va']:+.2e}, ΔMSE test {p['d_mse_te']:+.2e}")


# ---------------------------------------------------------------------------
# B.2
# ---------------------------------------------------------------------------

def diagnose_2L(Zg, Yg, G0, delta, lam, iters=2):
    """Vì sao irls chế độ pha lỗi trong 2L chiều: Z có hạng L nên ZᵀWZ suy biến, còn sai số float32 của A_h
    cỡ phạt 2λδ. So trị riêng nhỏ nhất của A_0 (h = 0) lập bằng float64 và float32, và chạy vài vòng
    với A_h float64 (gram_dtype=None) để xem J có giảm không."""
    R = Yg - Zg @ torch.as_tensor(G0, device=B.DEV).T
    w = delta / R.abs().clamp(min=delta)
    A64 = weight_gram(Zg, w[:, :1], 1)[0]
    A32 = weight_gram(Zg.float(), w[:, :1].float(), 1)[0].double()
    ev64, ev32 = torch.linalg.eigvalsh(A64), torch.linalg.eigvalsh(A32)
    Js = []
    t0 = B.sync_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irls(Zg, Yg, delta, torch.as_tensor(G0, device=B.DEV), lam=lam, max_iter=iters, tol=0.0,
             gram_dtype=None, chunk=CHUNK, callback=lambda it, W, J: Js.append(J.item()))
    J0 = huber_objective(R, delta).item()
    return {"norm_A0": ev64[-1].item(), "eig_min_A0_float64": ev64[0].item(),
            "eig_min_A0_float32": ev32[0].item(), "norm_A0_float32_minus_float64": torch.linalg.matrix_norm(A32 - A64, 2).item(),
            "penalty_2_lam_delta": 2 * lam * delta, "J0": J0, "J_float64_gram": Js,
            "s_per_iter_float64_gram": (B.sync_time() - t0) / iters}


def check_dlinear(lam=100.0, k=25):
    d = load_cell("ETTh1", 96)
    L = B.L
    P = build_P(L, k)
    Ninv = np.linalg.inv(build_N(P))
    ones = lambda n: np.ones((n, 1))
    Zt = np.concatenate([make_Z(d["X"], P), ones(len(d["X"]))], axis=1)       # [n, 2L + 1]
    Xg, Yg, Zg = to_gpu(d["Xt"], d["Y"], Zt)
    del Zt
    W_ols = ols_init(d["X"], d["Y"])
    warmup(Xg, Yg, torch.tensor(W_ols, device=B.DEV), CHUNK)
    # khởi tạo tương ứng: W_OLS trong L chiều; trong 2L chiều W_t = W_s = W_OLS cho cùng W_eff
    G0 = np.concatenate([W_ols[:, :-1], W_ols[:, :-1], W_ols[:, -1:]], axis=1)

    def F(W, delta):
        """Hàm mục tiêu có phạt, dạng tổng, L chiều: Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ N⁻¹ w_h."""
        R = d["Y"] - d["Xt"] @ W.T
        a = np.abs(R)
        q = np.minimum(a, delta)
        return float((q * (a - q / 2)).sum() / delta + lam * np.einsum("hi,ij,hj->", W[:, :-1], Ninv, W[:, :-1]))

    out = {}
    for loss, delta, tol in (("huber", 1.0, 1e-9), ("mae", MAE_DELTA, 1e-10)):
        print(f"[dlinear {loss}] λ = {lam}, k = {k}, tol = {tol}, patience = 5", flush=True)
        try:
            G, n1, _, s1, w1 = solve(Zg, Yg, G0, delta, tol, 5, lam=lam)
        except RuntimeError as e:
            out[loss] = {"delta": delta, "tol": tol, "error_direct_2L": str(e),
                         "diagnosis": diagnose_2L(Zg, Yg, G0, delta, lam)}
            print(f"  2L chiều lỗi: {e}; chẩn đoán: {out[loss]['diagnosis']}", flush=True)
            continue
        W2, n2, _, s2, w2 = solve(Xg, Yg, W_ols, delta, tol, 5, lam=lam, pen=Ninv)
        W1 = np.concatenate([dlinear_effective(G[:, :L], G[:, L:2 * L], P), G[:, -1:]], axis=1)
        F1, F2 = F(W1, delta), F(W2, delta)
        pen_2L = lam * float((G[:, :-1] ** 2).sum())
        pen_L1 = lam * float(np.einsum("hi,ij,hj->", W1[:, :-1], Ninv, W1[:, :-1]))
        m1, m2 = metrics64(W1, d, delta), metrics64(W2, d, delta)
        out[loss] = {
            "delta": delta, "tol": tol, "patience": 5,
            "direct_2L": {"n_iter": n1, "seconds": s1, "warnings": w1, "F": F1, "F_with_2L_penalty": F1 - pen_L1 + pen_2L,
                          "metrics": m1},
            "ninv_L": {"n_iter": n2, "seconds": s2, "warnings": w2, "F": F2, "metrics": m2},
            "W_eff_rel": rel(W1[:, :-1], W2[:, :-1]), "bias_max_abs": float(np.abs(W1[:, -1] - W2[:, -1]).max()),
            "F_rel": (F1 - F2) / F2, "F_2Lpen_rel": (F1 - pen_L1 + pen_2L - F2) / F2,
            **{f"d_{k_}": m1[k_] - m2[k_] for k_ in ("mse_va", "mae_va", "mse_te", "mae_te")}}
        o = out[loss]
        print(f"  2L: {n1} vòng {s1:.0f} s; N⁻¹: {n2} vòng {s2:.0f} s; ‖ΔW_eff‖/‖W‖ {o['W_eff_rel']:.2e}, "
              f"bias {o['bias_max_abs']:.2e}, ΔF/F {o['F_rel']:+.2e} (phạt 2L: {o['F_2Lpen_rel']:+.2e}), "
              f"MSE test {m1['mse_te']:.8f} / {m2['mse_te']:.8f}, MAE test {m1['mae_te']:.8f} / {m2['mae_te']:.8f}",
              flush=True)
        np.save(OUT / f"dlinear_{loss}_2L_Weff.npy", W1)
        np.save(OUT / f"dlinear_{loss}_Ninv.npy", W2)
    save_json(OUT / "dlinear.json", {"cell": "ETTh1 H=96", "lam": lam, "k": k, "chunk": CHUNK, "mode": "mixed",
                                     "init": "L chiều: W_OLS; 2L chiều: W_t = W_s = W_OLS (cùng W_eff)",
                                     "results": out})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["unique", "dlinear"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"unique": check_unique, "dlinear": check_dlinear}[a.check]()
