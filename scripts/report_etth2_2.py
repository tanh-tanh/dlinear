"""Lập các bảng của docs/BAO_CAO_ETTH2_2.md (docs/NHIEM_VU_4.md, Phần E).

    python scripts/report_etth2_2.py

Đọc results/sgd_etth2_20seed/ (A), results/sgd_etth2_filtered/ (B), results/constant_windows/ (dạng đóng trên
train đã lọc, NHIEM_VU_3 C.3), results/mae_filtered/ (C), results/official_ltsf/ (D), results/results.csv và
results/lambda_path.csv (train đủ), results/weights/ (W dạng đóng, cho B.3).
Ghi results/etth2_report_2/tables.md và summary.json. Không chọn gì trên test.

Khoảng tin cậy 95%: phân phối t với n − 1 bậc tự do. λ* là hằng số (không ngẫu nhiên theo seed), nên CI của
Δ = trung bình SGD − λ* là CI của trung bình SGD dịch đi λ*. Δ do lọc của SGD so cặp theo seed: cùng seed thì
cùng khởi tạo và cùng thứ tự xáo, chỉ khác hàm mất mát bỏ các cặp (cửa sổ, kênh) hằng.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import load_cell, save_json      # noqa: E402
from constant_windows import EPS, constant_mask    # noqa: E402

OUT = ROOT / "results" / "etth2_report_2"
A_DIR = ROOT / "results" / "sgd_etth2_20seed"
B_DIR = ROOT / "results" / "sgd_etth2_filtered"
NV3_DIR = ROOT / "results" / "sgd_etth2"
HORIZONS = [96, 192, 336, 720]
PUB = {("Linear", 96): 0.288, ("Linear", 192): 0.377, ("Linear", 336): 0.452, ("Linear", 720): 0.698,
       ("DLinear", 96): 0.289, ("DLinear", 192): 0.383, ("DLinear", 336): 0.448, ("DLinear", 720): 0.605,
       ("NLinear", 96): 0.277, ("NLinear", 192): 0.344, ("NLinear", 336): 0.357, ("NLinear", 720): 0.394}


def f4(x):
    return f"{x:+.4f}".replace(".", ",").replace("+", "") if x >= 0 else f"−{abs(x):.4f}".replace(".", ",")


def ci(x):
    """(trung bình, nửa độ rộng CI 95% theo t)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fci(m, h):
    return f"{f4(m)} [{f4(m - h)}; {f4(m + h)}]"


def lamf(x):
    return f"{x:.4g}".replace(".", ",")


def load_runs(d, model, H):
    rs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob(f"{model}_H{H}_seed*.json"))]
    return sorted(rs, key=lambda r: r["seed"])


def results_star():
    out = {}
    with (ROOT / "results" / "results.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r["dataset"] == "ETTh2" and r["objective"] == "MSE" and r["selection"] == "val_objective"
                    and r["level"] == "lam_star"):
                out[(r["model"], int(r["H"]))] = r
    return out


def broken(rs):
    """Seed "hỏng": dừng ở epoch 1, hoặc MSE test > trung vị + 3·MAD (MAD = trung vị |x − trung vị|, không nhân 1,4826)."""
    t = np.array([r["test_mse"] for r in rs])
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    return [r["seed"] for r in rs if r["best_epoch"] == 1 or r["test_mse"] > med + 3 * mad]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, summ = [], {}
    star = results_star()

    # ---------- kiểm tất định: seed 2021–2023 phải trùng nhiệm vụ 3 ----------
    n_same, n_all = 0, 0
    for f in A_DIR.glob("*_seed202[123].npy"):
        n_all += 1
        n_same += bool(np.array_equal(np.load(f), np.load(NV3_DIR / f.name)))
    summ["determinism"] = {"n_files": n_all, "n_bit_identical": n_same}
    md.append(f"Kiểm tất định: {n_same}/{n_all} file W của seed 2021–2023 trùng từng bit với nhiệm vụ 3.\n")

    # ---------- A ----------
    md.append("## A. SGD 20 seed so với λ*, ETTh2\n")
    md.append("| Mô hình | H | n | MSE test: TB ± sd (trung vị) | MSE val: TB ± sd (trung vị) | λ* test / val | "
              "Δ test [CI 95%] | Δ val [CI 95%] | seed tốt hơn λ* | seed hỏng | CI dưới 0 | tiêu chí cũ (2 sd) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    a_rows, a_drop = [], []
    for model in ["Linear", "DLinear", "NLinear"]:
        for H in HORIZONS:
            rs = load_runs(A_DIR, model, H)
            if len(rs) < 2:
                continue
            s = star[(model, H)]
            cf_t, cf_v = float(s["test_mse"]), float(s["val_mse"])
            bad = broken(rs)
            for tag, sub in (("giữ", rs), ("bỏ", [r for r in rs if r["seed"] not in bad])):
                t = np.array([r["test_mse"] for r in sub])
                v = np.array([r["val_mse"] for r in sub])
                mt, ht = ci(t - cf_t)
                mv, hv = ci(v - cf_v)
                row = {"model": model, "H": H, "n": len(sub), "keep_broken": tag == "giữ",
                       "test_mean": float(t.mean()), "test_sd": float(t.std(ddof=1)), "test_median": float(np.median(t)),
                       "val_mean": float(v.mean()), "val_sd": float(v.std(ddof=1)), "val_median": float(np.median(v)),
                       "cf_test": cf_t, "cf_val": cf_v, "lam_star": float(s["lam"]),
                       "d_test": mt, "d_test_ci": ht, "d_val": mv, "d_val_ci": hv,
                       "frac_better": float((t < cf_t).mean()), "broken": bad,
                       "ci_below_0": bool(mt + ht < 0), "old_2sd": bool(mt < 0 and -mt > 2 * t.std(ddof=1)),
                       "vs_pub": (t.mean() - PUB[(model, H)]) / PUB[(model, H)]}
                (a_rows if tag == "giữ" else a_drop).append(row)
                if tag == "giữ" or bad:
                    md.append(f"| {model}{' (bỏ hỏng)' if tag == 'bỏ' else ''} | {H} | {len(sub)} | "
                              f"{f4(row['test_mean'])} ± {f4(row['test_sd'])} ({f4(row['test_median'])}) | "
                              f"{f4(row['val_mean'])} ± {f4(row['val_sd'])} ({f4(row['val_median'])}) | "
                              f"{f4(cf_t)} / {f4(cf_v)} | {fci(mt, ht)} | {fci(mv, hv)} | "
                              f"{int(round(row['frac_better'] * len(sub)))}/{len(sub)} | "
                              f"{', '.join(map(str, bad)) if tag == 'giữ' and bad else '—'} | "
                              f"{'có' if row['ci_below_0'] else 'không'} | {'có' if row['old_2sd'] else 'không'} |")
    verdict = {}
    for model in ["Linear", "DLinear", "NLinear"]:
        for label, rows in (("giữ", a_rows), ("bỏ", a_drop)):
            rr = [r for r in rows if r["model"] == model]
            if rr:
                verdict[f"{model}_{label}"] = {"ci_below_0": sum(r["ci_below_0"] for r in rr),
                                               "old_2sd": sum(r["old_2sd"] for r in rr), "n_cells": len(rr)}
    md.append("\n" + "; ".join(f"{k}: CI dưới 0 ở {v['ci_below_0']}/{v['n_cells']} ô, 2 sd ở {v['old_2sd']}/{v['n_cells']}"
                               for k, v in verdict.items()) + "\n")
    md.append("SGD 20 seed so với công bố (trung bình, giữ seed hỏng): " + "; ".join(
        f"{r['model']} H={r['H']} {100 * r['vs_pub']:+.1f} %".replace(".", ",") for r in a_rows) + "\n")
    summ["A"] = {"keep": a_rows, "drop_broken": a_drop, "verdict": verdict}

    # ---------- B ----------
    cw = json.loads((ROOT / "results" / "constant_windows" / "summary.json").read_text(encoding="utf-8"))
    cf_f = {(c["model"], c["H"]): c for c in cw["cells"]}
    # kiểm lại nghiệm dạng đóng đã lọc của C.3: số dòng và λ* từ path.csv
    with (ROOT / "results" / "constant_windows" / "path.csv").open(encoding="utf-8", newline="") as f:
        path = list(csv.DictReader(f))
    checks = []
    for (model, H), c in cf_f.items():
        rows = [r for r in path if r["model"] == model and int(r["H"]) == H]
        best = min(rows, key=lambda r: float(r["val_mse"]))
        checks.append({"model": model, "H": H, "n_lams": len(rows), "lam_star_path": float(best["lam"]),
                       "lam_star_summary": c["after"]["lam_star"],
                       "ok": bool(len(rows) == 142 and float(best["lam"]) == c["after"]["lam_star"])})
    summ["B_cf_check"] = checks
    md.append("## B. SGD trên train đã lọc\n")
    md.append(f"Kiểm nghiệm dạng đóng đã lọc (C.3 của nhiệm vụ 3): {sum(c['ok'] for c in checks)}/{len(checks)} "
              f"(mô hình, H) có đủ 142 λ và λ* khớp summary.json.\n")
    b_rows = []
    for metric, key_cf in (("test", "test_mse"), ("val", "val_mse")):
        md.append(f"### MSE {metric}\n")
        md.append("| Mô hình | H | dạng đóng: đủ → lọc (Δ_CF) | SGD: đủ → lọc | Δ_SGD [CI] | Δ_SGD − Δ_CF [CI] | "
                  "SGD − CF, đủ [CI] | SGD − CF, lọc [CI] |\n|---|---|---|---|---|---|---|---|")
        for model in ["Linear", "DLinear"]:
            for H in HORIZONS:
                full, filt = load_runs(A_DIR, model, H), load_runs(B_DIR, model, H)
                if len(filt) < 2:
                    continue
                seeds = sorted(set(r["seed"] for r in full) & set(r["seed"] for r in filt))
                xf = np.array([next(r for r in full if r["seed"] == s)[f"{metric}_mse"] for s in seeds])
                xl = np.array([next(r for r in filt if r["seed"] == s)[f"{metric}_mse"] for s in seeds])
                cf_full = float(star[(model, H)][key_cf])
                cf_filt = cf_f[(model, H)]["after"][key_cf] if key_cf in cf_f[(model, H)]["after"] else None
                if cf_filt is None:      # summary.json của C.3 có val_mse, test_mse
                    continue
                d_cf = cf_filt - cf_full
                m_s, h_s = ci(xl - xf)
                g_f, hg_f = ci(xf - cf_full)
                g_l, hg_l = ci(xl - cf_filt)
                row = {"metric": metric, "model": model, "H": H, "n": len(seeds), "cf_full": cf_full, "cf_filt": cf_filt,
                       "d_cf": d_cf, "sgd_full": float(xf.mean()), "sgd_filt": float(xl.mean()),
                       "d_sgd": m_s, "d_sgd_ci": h_s, "d_sgd_minus_d_cf": m_s - d_cf,
                       "gap_full": g_f, "gap_full_ci": hg_f, "gap_filt": g_l, "gap_filt_ci": hg_l}
                if metric == "test":
                    row["mechanism"] = bool(d_cf < 0 and (m_s - d_cf) - h_s > 0 and g_l + hg_l >= 0)
                b_rows.append(row)
                md.append(f"| {model} | {H} | {f4(cf_full)} → {f4(cf_filt)} ({f4(d_cf)}) | {f4(xf.mean())} → "
                          f"{f4(xl.mean())} | {fci(m_s, h_s)} | {fci(m_s - d_cf, h_s)} | {fci(g_f, hg_f)} | {fci(g_l, hg_l)} |")
        md.append("")
    md.append("Tiêu chí B.4 (H ∈ {336, 720}, MSE test): " + "; ".join(
        f"{r['model']} H={r['H']}: {'ủng hộ' if r['mechanism'] else 'không'}" for r in b_rows
        if r["metric"] == "test" and r["H"] in (336, 720)) + "\n")
    summ["B"] = b_rows

    # ---------- B.3: phần dư trên cửa sổ hằng của train ----------
    md.append("## B.3. MSE phần dư trên các cửa sổ hằng của train (train đủ)\n")
    md.append("| H | Mô hình | nhóm | số cửa sổ | dạng đóng λ* | SGD (TB ± sd theo seed) | SGD / dạng đóng |\n"
              "|---|---|---|---|---|---|---|")
    b3 = []
    for H in HORIZONS:
        cell = load_cell("ETTh2", H, need_eval=False)
        X, Y = cell["X"], cell["Y"]
        cx, cy = np.ptp(X, 1) < EPS, np.ptp(Y, 1) < EPS
        assert ((cx | cy) == constant_mask(X, Y)).all()
        groups = {"x hằng, y không": cx & ~cy, "chỉ y hằng": ~cx & cy, "cả hai": cx & cy,
                  "không hằng": ~(cx | cy)}
        for model in ["Linear", "DLinear"]:
            lam = float(star[(model, H)]["lam"])
            Wcf = np.load(ROOT / "results" / "weights" / f"ETTh2_H{H}_{model}_MSE_lam{lam:.6g}.npy")
            Rcf = Y - (X @ Wcf[:, :-1].T + Wcf[:, -1])
            Rs = []
            for r in load_runs(A_DIR, model, H):
                Ws = np.load(A_DIR / f"{model}_H{H}_seed{r['seed']}.npy")
                Rs.append(Y - (X @ Ws[:, :-1].T + Ws[:, -1]))
            for g, m in groups.items():
                if not m.any():
                    continue
                cf = float((Rcf[m] ** 2).mean())
                sg = np.array([float((R[m] ** 2).mean()) for R in Rs])
                b3.append({"H": H, "model": model, "group": g, "n": int(m.sum()), "cf": cf,
                           "sgd_mean": float(sg.mean()), "sgd_sd": float(sg.std(ddof=1)), "ratio": float(sg.mean() / cf)})
                md.append(f"| {H} | {model} | {g} | {int(m.sum())} | {f4(cf)} | {f4(sg.mean())} ± {f4(sg.std(ddof=1))} | "
                          f"{sg.mean() / cf:.2f}".replace(".", ",") + " |")
    md.append("")
    summ["B3"] = b3

    # ---------- C ----------
    md.append("## C. MAE trên train đã lọc (λ = 0)\n")
    md.append("| Mô hình | H | train | MSE test: MSE λ = 0 | MSE test: MAE λ = 0 | lợi của MAE | MAE test: MSE / MAE λ = 0 | "
              "vòng | fallback |\n|---|---|---|---|---|---|---|---|---|")
    lp = {}
    with (ROOT / "results" / "lambda_path.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "ETTh2" and float(r["lam"]) == 0:
                lp[(r["model"], int(r["H"]), r["objective"])] = r
    c_rows = []
    for model in ["Linear", "NLinear"]:
        for H in HORIZONS:
            fj = ROOT / "results" / "mae_filtered" / f"{model}_H{H}.json"
            if not fj.exists():
                continue
            j = json.loads(fj.read_text(encoding="utf-8"))
            full_mse, full_mae = lp[(model, H, "MSE")], lp[(model, H, "MAE")]
            gain_full = float(full_mse["test_mse"]) - float(full_mae["test_mse"])
            gain_filt = j["mse_lam0"]["test_mse"] - j["mae"]["test_mse"]
            c_rows.append({"model": model, "H": H, "full_mse_obj": float(full_mse["test_mse"]),
                           "full_mae_obj": float(full_mae["test_mse"]), "gain_full": gain_full,
                           "filt_mse_obj": j["mse_lam0"]["test_mse"], "filt_mae_obj": j["mae"]["test_mse"],
                           "gain_filt": gain_filt, "ratio": gain_filt / gain_full if gain_full else None,
                           "full_mae_test": [float(full_mse["test_mae"]), float(full_mae["test_mae"])],
                           "filt_mae_test": [j["mse_lam0"]["test_mae"], j["mae"]["test_mae"]],
                           "n_iter": j["n_iter"], "n_fallback": j["n_fallback"], "seconds": j["seconds"],
                           "cf_filt_lam_star": cf_f[(model, H)]["after"]["test_mse"]})
            md.append(f"| {model} | {H} | đủ | {f4(float(full_mse['test_mse']))} | {f4(float(full_mae['test_mse']))} | "
                      f"{f4(gain_full)} | {f4(float(full_mse['test_mae']))} / {f4(float(full_mae['test_mae']))} | "
                      f"{full_mae['n_iter']} | không ghi |")
            md.append(f"| | | đã lọc | {f4(j['mse_lam0']['test_mse'])} | {f4(j['mae']['test_mse'])} | {f4(gain_filt)} "
                      f"({100 * gain_filt / gain_full:.0f} % của lợi cũ) | {f4(j['mse_lam0']['test_mae'])} / "
                      f"{f4(j['mae']['test_mae'])} | {j['n_iter']} | {j['n_fallback']} |")
    md.append("\nSo MAE λ = 0 trên train đủ với MSE λ* trên train đã lọc (MSE test): " + "; ".join(
        f"{r['model']} H={r['H']}: {f4(r['full_mae_obj'])} so với {f4(r['cf_filt_lam_star'])}" for r in c_rows) + "\n")
    summ["C"] = c_rows

    # ---------- D ----------
    md.append("## D. Code gốc của LTSF-Linear\n")
    md.append("| Mô hình | H | seed | code gốc: MSE / MAE test | chấm lại bằng repo: MSE / MAE test | chênh MSE | "
              "epoch tốt nhất |\n|---|---|---|---|---|---|---|")
    d_rows = []
    for f in sorted((ROOT / "results" / "official_ltsf").glob("*_seed*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        d_rows.append({k: j[k] for k in ("model", "H", "seed", "official_test_mse", "official_test_mae",
                                         "repo_test_mse", "repo_test_mae", "repo_val_mse", "best_epoch", "n_epochs_run")})
    d_rows.sort(key=lambda r: (r["H"], r["model"], r["seed"]))
    for r in d_rows:
        md.append(f"| {r['model']} | {r['H']} | {r['seed']} | {r['official_test_mse']:.4f} / {r['official_test_mae']:.4f} | "
                  f"{r['repo_test_mse']:.4f} / {r['repo_test_mae']:.4f} | {r['repo_test_mse'] - r['official_test_mse']:+.1e} | "
                  f"{r['best_epoch']}/{r['n_epochs_run']} |".replace(".", ","))
    md.append("\n| Mô hình | H | code gốc: TB ± sd (n) | SGD repo 20 seed: TB ± sd | công bố |\n|---|---|---|---|---|")
    d_sum = []
    for (model, H) in sorted({(r["model"], r["H"]) for r in d_rows}, key=lambda t: (t[1], t[0])):
        o = np.array([r["official_test_mse"] for r in d_rows if (r["model"], r["H"]) == (model, H)])
        rs = load_runs(A_DIR, model, H)
        a = np.array([r["test_mse"] for r in rs])
        d_sum.append({"model": model, "H": H, "official_mean": float(o.mean()),
                      "official_sd": float(o.std(ddof=1)) if len(o) > 1 else None, "n": len(o),
                      "repo_mean": float(a.mean()) if len(a) else None, "pub": PUB[(model, H)]})
        md.append(f"| {model} | {H} | {f4(o.mean())} ± {f4(o.std(ddof=1)) if len(o) > 1 else '—'} ({len(o)}) | "
                  f"{f4(a.mean())} ± {f4(a.std(ddof=1))} | {f4(PUB[(model, H)])} |")
    summ["D"] = {"runs": d_rows, "summary": d_sum}

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    save_json(OUT / "summary.json", summ)
    print(f"→ {(OUT / 'tables.md').relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
