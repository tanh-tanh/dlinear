"""Lập bảng B1, B4, C.3 và hình H3 của docs/BAO_CAO_SGD_B1_B4.md (docs/NHIEM_VU_5.md, Phần C).

    python scripts/report_sgd_b1_b4.py

Đọc SGD: results/sgd_b1_b4/{ETTh1,ETTm1}/ (Phần B) và results/sgd_etth2_20seed/ (ETTh2, nhiệm vụ 4); dạng đóng λ*:
results/results.csv (mục tiêu MSE, λ* chọn theo val MSE). Ghi results/sgd_b1_b4/{tables.md, summary.json,
H3_sgd_vs_optimum.png}. Không chọn gì trên test.

- Mọi bảng chính dùng đủ seed (giữ seed hỏng). Cột "bỏ seed hỏng" chỉ để tham khảo: tiêu chí hỏng dùng test.
- CI 95%: phân phối t, n − 1 bậc tự do (λ* là hằng số nên CI của Δ là CI của trung bình SGD dịch đi λ*).
- C.3: DLinear − Linear, CI 95% Welch (hai nhóm seed độc lập theo mô hình).
- "Phân vị của số công bố": tỷ lệ seed có MSE test nhỏ hơn số công bố, nhân 100.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import save_json                          # noqa: E402
from optim_etth2 import label, side                         # noqa: E402

OUT = ROOT / "results" / "sgd_b1_b4"
DIRS = {"ETTh1": OUT / "ETTh1", "ETTh2": ROOT / "results" / "sgd_etth2_20seed", "ETTm1": OUT / "ETTm1"}
DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["Linear", "DLinear", "NLinear"]
# NHIEM_VU_3.md mục 0.1 (Bảng 2 của Zeng et al., MSE), thứ tự Linear, NLinear, DLinear
_PUB = {
    "ETTh1": [(0.375, 0.374, 0.375), (0.418, 0.408, 0.405), (0.479, 0.429, 0.439), (0.624, 0.440, 0.472)],
    "ETTh2": [(0.288, 0.277, 0.289), (0.377, 0.344, 0.383), (0.452, 0.357, 0.448), (0.698, 0.394, 0.605)],
    "ETTm1": [(0.308, 0.306, 0.299), (0.340, 0.349, 0.335), (0.376, 0.375, 0.369), (0.440, 0.433, 0.425)],
}
PUB = {(ds, H, m): v[i] for ds, rows in _PUB.items() for H, v in zip(HORIZONS, rows)
       for i, m in enumerate(["Linear", "NLinear", "DLinear"])}


def num(x, p=4):
    return f"{x:.{p}f}".replace(".", ",") if x >= 0 else f"−{abs(x):.{p}f}".replace(".", ",")


def ci(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def welch(a, b):
    """(trung bình a − trung bình b, nửa độ rộng CI 95% Welch, bậc tự do)."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * np.sqrt(va + vb)), float(df)


def fci(m, h):
    return f"{num(m)} [{num(m - h)}; {num(m + h)}]"


def load_runs(ds, model, H):
    rs = [json.loads(f.read_text(encoding="utf-8")) for f in DIRS[ds].glob(f"{model}_H{H}_seed*.json")]
    return sorted(rs, key=lambda r: r["seed"])


def results_star():
    out = {}
    with (ROOT / "results" / "results.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["objective"] == "MSE" and r["selection"] == "val_objective" and r["level"] == "lam_star":
                out[(r["dataset"], int(r["H"]), r["model"])] = r
    return out


def broken(rs):
    """Như report_etth2_2.broken: dừng ở epoch 1, hoặc MSE test > trung vị + 3·MAD."""
    t = np.array([r["test_mse"] for r in rs])
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    return [r["seed"] for r in rs if r["best_epoch"] == 1 or r["test_mse"] > med + 3 * mad]


def cell_label(rs, cf_v, cf_t):
    v = np.array([r["val_mse"] for r in rs])
    t = np.array([r["test_mse"] for r in rs])
    mv, hv = ci(v - cf_v)
    mt, ht = ci(t - cf_t)
    sv, st = side(mv, hv), side(mt, ht)
    return {"d_val": mv, "d_val_ci": hv, "d_test": mt, "d_test_ci": ht, "val_side": sv, "test_side": st,
            "label": label(sv, st)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    star = results_star()
    runs = {(ds, H, m): load_runs(ds, m, H) for ds in DATASETS for H in HORIZONS for m in MODELS}
    runs = {k: v for k, v in runs.items() if len(v) >= 2}
    md, summ = [], {"n_seeds": {f"{k[0]} H={k[1]} {k[2]}": len(v) for k, v in runs.items()}}

    # ---------- thời gian, số seed ----------
    md.append("## Số seed và thời gian chạy (giây mỗi lần, CPU)\n")
    md.append("| Dataset | Mô hình | n seed mỗi H | giây/lần: TB (min–max) | epoch chạy: TB | epoch tốt nhất: TB | tổng giờ CPU |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for m in MODELS:
            rs = [r for H in HORIZONS for r in runs.get((ds, H, m), [])]
            if not rs:
                continue
            sec = np.array([r["seconds"] for r in rs])
            ns = sorted({len(runs[(ds, H, m)]) for H in HORIZONS if (ds, H, m) in runs})
            md.append(f"| {ds} | {m} | {', '.join(map(str, ns))} | {sec.mean():.0f} ({sec.min():.0f}–{sec.max():.0f}) | "
                      f"{np.mean([r['n_epochs_run'] for r in rs]):.1f} | {np.mean([r['best_epoch'] for r in rs]):.1f} | "
                      f"{sec.sum() / 3600:.2f} |")

    # ---------- B1 ----------
    md.append("\n## B1. SGD so với số công bố (MSE test)\n")
    md.append("Phân vị = % seed có MSE test nhỏ hơn số công bố.\n")
    md.append("| Dataset | H | Mô hình | SGD: TB ± sd (n) | trung vị | [min; max] | công bố | phân vị của công bố | "
              "công bố trong [min, max]? |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    b1 = []
    for (ds, H, m), rs in runs.items():
        t = np.array([r["test_mse"] for r in rs])
        p = PUB[(ds, H, m)]
        inside = bool(t.min() <= p <= t.max())
        row = {"dataset": ds, "H": H, "model": m, "n": len(t), "mean": float(t.mean()), "sd": float(t.std(ddof=1)),
               "median": float(np.median(t)), "min": float(t.min()), "max": float(t.max()), "pub": p,
               "pub_percentile": float((t < p).mean() * 100), "pub_inside": inside}
        b1.append(row)
        md.append(f"| {ds} | {H} | {m} | {num(row['mean'])} ± {num(row['sd'])} ({len(t)}) | {num(row['median'])} | "
                  f"[{num(row['min'])}; {num(row['max'])}] | {num(p, 3)} | {row['pub_percentile']:.0f} | "
                  f"{'có' if inside else '**không**'} |")
    n_in = sum(r["pub_inside"] for r in b1)
    outside = [r for r in b1 if not r["pub_inside"]]
    md.append(f"\nSố công bố nằm trong [min, max] của các seed: **{n_in}/{len(b1)} ô**. Ngoài: "
              + ("; ".join(f"{r['dataset']} H={r['H']} {r['model']} (công bố {num(r['pub'], 3)}, "
                           f"{'thấp hơn' if r['pub'] < r['min'] else 'cao hơn'} mọi seed)" for r in outside) or "không có")
              + ".\n")
    summ["B1"] = {"rows": b1, "n_inside": n_in, "n_cells": len(b1)}

    # ---------- B4 ----------
    md.append("## B4. SGD so với dạng đóng λ*\n")
    md.append("Δ = trung bình SGD − dạng đóng λ*; âm là SGD tốt hơn. Nhãn theo quy tắc B.4 của NHIEM_VU_5. "
              "Cột cuối bỏ seed hỏng (dừng ở epoch 1 hoặc test > trung vị + 3·MAD), **chỉ để tham khảo**.\n")
    md.append("| Dataset | H | Mô hình | λ* | dạng đóng: val / test | SGD: val / test (TB, n) | Δ val [CI] | Δ test [CI] | "
              "nhãn | seed hỏng | nhãn khi bỏ seed hỏng (tham khảo) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    b4 = []
    for (ds, H, m), rs in runs.items():
        s = star[(ds, H, m)]
        cf_v, cf_t = float(s["val_mse"]), float(s["test_mse"])
        lab = cell_label(rs, cf_v, cf_t)
        bad = broken(rs)
        lab_drop = cell_label([r for r in rs if r["seed"] not in bad], cf_v, cf_t)["label"] if bad else lab["label"]
        v = np.mean([r["val_mse"] for r in rs])
        t = np.mean([r["test_mse"] for r in rs])
        row = {"dataset": ds, "H": H, "model": m, "n": len(rs), "lam_star": float(s["lam"]), "cf_val": cf_v,
               "cf_test": cf_t, "sgd_val": float(v), "sgd_test": float(t), **lab, "broken": bad,
               "label_drop_broken": lab_drop}
        b4.append(row)
        md.append(f"| {ds} | {H} | {m} | {float(s['lam']):.4g} | {num(cf_v)} / {num(cf_t)} | {num(v)} / {num(t)} ({len(rs)}) | "
                  f"{fci(lab['d_val'], lab['d_val_ci'])} | {fci(lab['d_test'], lab['d_test_ci'])} | {lab['label']} | "
                  f"{', '.join(map(str, bad)) or '—'} | {lab_drop if bad else '(như cột nhãn)'} |")
    labels = ["SGD tốt hơn thật", "dạng đóng tốt hơn thật", "khác biệt do val khác test",
              "SGD có dấu hiệu khớp val quá mức", "dạng đóng có dấu hiệu khớp val quá mức", "không phân biệt được"]
    md.append("\n### Đếm theo nhãn (đủ seed)\n")
    md.append("| Nhãn | " + " | ".join(DATASETS) + " | " + " | ".join(MODELS) + " | tổng |")
    md.append("|---|" + "---|" * (len(DATASETS) + len(MODELS) + 1))
    counts = {}
    for lb in labels:
        c_ds = [sum(r["label"] == lb and r["dataset"] == ds for r in b4) for ds in DATASETS]
        c_m = [sum(r["label"] == lb and r["model"] == m for r in b4) for m in MODELS]
        counts[lb] = {"by_dataset": dict(zip(DATASETS, c_ds)), "by_model": dict(zip(MODELS, c_m)), "total": sum(c_ds)}
        md.append(f"| {lb} | " + " | ".join(map(str, c_ds)) + " | " + " | ".join(map(str, c_m)) + f" | {sum(c_ds)} |")
    md.append("\nChéo dataset × mô hình (đủ seed):\n")
    md.append("| Nhãn | " + " | ".join(f"{ds} {m}" for ds in DATASETS for m in MODELS) + " |")
    md.append("|---|" + "---|" * (len(DATASETS) * len(MODELS)))
    for lb in labels:
        md.append(f"| {lb} | " + " | ".join(str(sum(r["label"] == lb and r["dataset"] == ds and r["model"] == m
                                                    for r in b4)) for ds in DATASETS for m in MODELS) + " |")
    sgd_better_outside = [f"{r['dataset']} H={r['H']} {r['model']}" for r in b4
                          if r["label"] == "SGD tốt hơn thật" and r["dataset"] != "ETTh2"]
    md.append(f"\nÔ ngoài ETTh2 có nhãn \"SGD tốt hơn thật\": {', '.join(sgd_better_outside) or 'không có'}.\n")
    summ["B4"] = {"rows": b4, "counts": counts, "sgd_better_outside_etth2": sgd_better_outside}

    # ---------- C.3 ----------
    md.append("## C.3. DLinear − Linear khi huấn luyện bằng SGD\n")
    md.append("Âm là DLinear tốt hơn. SGD: CI 95% Welch. Dạng đóng: λ* của mỗi mô hình (một số). Công bố: một lần chạy.\n")
    md.append("| Dataset | H | n (DLinear, Linear) | SGD Δ val [CI Welch] | SGD Δ test [CI Welch] | dạng đóng Δ val / Δ test | "
              "công bố Δ test | DLinear tốt hơn có ý nghĩa (val / test) |")
    md.append("|---|---|---|---|---|---|---|---|")
    c3 = []
    for ds in DATASETS:
        for H in HORIZONS:
            if (ds, H, "DLinear") not in runs or (ds, H, "Linear") not in runs:
                continue
            a, b = runs[(ds, H, "DLinear")], runs[(ds, H, "Linear")]
            dv = welch(np.array([r["val_mse"] for r in a]), np.array([r["val_mse"] for r in b]))
            dt = welch(np.array([r["test_mse"] for r in a]), np.array([r["test_mse"] for r in b]))
            sd_, sl = star[(ds, H, "DLinear")], star[(ds, H, "Linear")]
            cfv = float(sd_["val_mse"]) - float(sl["val_mse"])
            cft = float(sd_["test_mse"]) - float(sl["test_mse"])
            pub = PUB[(ds, H, "DLinear")] - PUB[(ds, H, "Linear")]
            sig_v, sig_t = dv[0] + dv[1] < 0, dt[0] + dt[1] < 0
            row = {"dataset": ds, "H": H, "n_dlinear": len(a), "n_linear": len(b), "d_val": dv[0], "d_val_ci": dv[1],
                   "d_val_df": dv[2], "d_test": dt[0], "d_test_ci": dt[1], "d_test_df": dt[2], "cf_d_val": cfv,
                   "cf_d_test": cft, "pub_d_test": pub, "dlinear_better_val": bool(sig_v),
                   "dlinear_better_test": bool(sig_t), "linear_better_val": bool(dv[0] - dv[1] > 0),
                   "linear_better_test": bool(dt[0] - dt[1] > 0)}
            c3.append(row)
            md.append(f"| {ds} | {H} | {len(a)}, {len(b)} | {fci(dv[0], dv[1])} | {fci(dt[0], dt[1])} | "
                      f"{num(cfv, 5)} / {num(cft, 5)} | {num(pub, 3)} | {'có' if sig_v else 'không'} / "
                      f"{'có' if sig_t else 'không'} |")
    nv = sum(r["dlinear_better_val"] for r in c3)
    nt = sum(r["dlinear_better_test"] for r in c3)
    nb = sum(r["dlinear_better_val"] and r["dlinear_better_test"] for r in c3)
    lv = sum(r["linear_better_val"] for r in c3)
    lt = sum(r["linear_better_test"] for r in c3)
    md.append(f"\nDLinear tốt hơn Linear có ý nghĩa (CI Welch nằm hẳn dưới 0): val {nv}/{len(c3)} ô, test {nt}/{len(c3)} ô, "
              f"cả hai {nb}/{len(c3)} ô. Ngược lại (Linear tốt hơn có ý nghĩa): val {lv}, test {lt} ô.\n")
    summ["C3"] = {"rows": c3, "n_dlinear_better_val": nv, "n_dlinear_better_test": nt, "n_both": nb,
                  "n_linear_better_val": lv, "n_linear_better_test": lt}

    figure(runs, star)
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_json(OUT / "summary.json", summ)
    print("\n".join(md))


def figure(runs, star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Linear": "#2a78d6", "DLinear": "#eb6834", "NLinear": "#1baf7a"}   # 3 ô đầu của bảng màu phân loại
    markers = {"Linear": "o", "DLinear": "s", "NLinear": "^"}
    fig, axes = plt.subplots(len(DATASETS), len(HORIZONS), figsize=(14, 9.5))
    rng = np.random.default_rng(0)                    # chỉ để rải điểm theo chiều ngang
    for i, ds in enumerate(DATASETS):
        for j, H in enumerate(HORIZONS):
            ax = axes[i, j]
            for x, m in enumerate(MODELS):
                rs = runs.get((ds, H, m), [])
                if not rs:
                    continue
                t = np.array([r["test_mse"] for r in rs])
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(t)), t, s=16, color=colors[m], marker=markers[m],
                           alpha=0.75, linewidths=0, zorder=3, label=f"{m}: SGD, từng seed" if i == j == 0 else None)
                cf = float(star[(ds, H, m)]["test_mse"])
                ax.hlines(cf, x - 0.32, x + 0.32, color="#222222", lw=2, zorder=4,
                          label="dạng đóng λ*" if i == j == 0 and x == 0 else None)
                ax.scatter([x + 0.38], [PUB[(ds, H, m)]], marker="*", s=90, color="#222222", zorder=5,
                           edgecolors="white", linewidths=0.5,
                           label="số công bố (Zeng et al.)" if i == j == 0 and x == 0 else None)
            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels(MODELS, fontsize=8)
            ax.set_xlim(-0.6, 2.7)
            ax.set_title(f"{ds}, H = {H}", fontsize=10)
            ax.grid(axis="y", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel("MSE test (dữ liệu chuẩn hóa)")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("MSE test của SGD theo seed so với nghiệm dạng đóng λ* và số công bố (L = 336; mỗi điểm là một seed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "H3_sgd_vs_optimum.png", dpi=150)
    print(f"→ {(OUT / 'H3_sgd_vs_optimum.png').relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
